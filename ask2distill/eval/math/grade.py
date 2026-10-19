"""Rule-grade one checkpoint's math eval directory into ``grading_results.json``.

Every ``<task>_t<T>_p<P>_n<N>-MNT<M>.jsonl`` written by ``generate`` becomes one
entry of a JSON list, with the original schema::

    {"hyperparameters": {"temperature", "top_p", "n", "max_tokens", "task_name"},
     "mean_score", "distinct_4gram", "best_score", "solve_none", "solve_all",
     "avg_output_length", "format_error_rollouts"}

``mean_score`` is avg@n: per problem the fraction of its n responses that are
correct (``grader_utils.grade_answer_verl``), averaged over problems. The
hyperparameter values are the strings parsed from the file name.

``avg_output_length`` is measured with ``len_tokenizer`` (the final runs used the
DeepSeek-R1-Distill-Qwen-1.5B tokenizer for every pair); when it cannot be
loaded the length is counted in characters, as the original fell back to.
Grading is CPU-only. Run it once per directory after all tasks are generated:
two graders writing the same ``grading_results.json`` would race.

Not ported: the CompassVerifier LLM judge (off in every final run) and the
parquet input branch (generation writes jsonl only). Files are graded in sorted
order, so the list order no longer depends on the filesystem.

Usage::

    python -m ask2distill.eval.math.grade <eval_dir> [--len-tokenizer DIR] [--strict-boxed]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Sequence

from ask2distill.common.io import atomic_write_text
from ask2distill.eval.math.grader_utils import grade_answer_verl

RESULT_NAME = "grading_results.json"
_HPARAMS = re.compile(r"_t(?P<temperature>[\d.]+)_p(?P<top_p>[\d.]+)_n(?P<n>\d+)-MNT(?P<max_tokens>\d+)")
_SUFFIX = re.compile(r"_t[\d.]+_p[\d.]+_n\d+-MNT\d+$")


def length_function(len_tokenizer: str | Path | None) -> Callable[[str], int]:
    """Token counter for ``avg_output_length``; character count if the tokenizer is unavailable."""
    tok = None
    if len_tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(len_tokenizer))
        except Exception:
            tok = None
    if tok is None:
        return len
    return lambda seq: len(tok.encode(seq))


def get_diverse_score(sequences: Sequence[str], n: int = 4) -> float:
    """Distinct whitespace n-grams over all n-grams of ``sequences``."""
    distinct_ngrams = set()
    total_ngrams = 0
    for seq in sequences:
        tokens = seq.split()
        for i in range(len(tokens) - n + 1):
            distinct_ngrams.add(tuple(tokens[i:i + n]))
            total_ngrams += 1
    return len(distinct_ngrams) / total_ngrams if total_ngrams > 0 else 0


def process_jsonl_file(file_name: str | Path) -> list[dict]:
    """Group rows by ``example_id``: ``[{"gt": answer, "responses": [...]}, ...]``."""
    results: list[dict] = []
    with open(file_name) as f:
        for line in f:
            data = json.loads(line)
            idx = int(data["example_id"])
            while len(results) <= idx:
                results.append({"gt": None, "responses": []})
            results[idx]["gt"] = data["answer"]
            results[idx]["responses"].append(data["response"])
    return results


def parse_hyperparameters_from_filename(filename: str) -> dict:
    match = _HPARAMS.search(filename)
    return match.groupdict() if match else {}


def task_name_of(path: Path) -> str:
    """``aime24_t0.7_p0.95_n16-MNT31744.jsonl`` -> ``aime24``.

    Everything before the settings suffix: cutting at the first "_" instead
    collapsed hmmt_feb25 and hmmt_feb26 into two rows both labelled "hmmt".
    """
    return _SUFFIX.sub("", path.stem)


def grade_file(file_path: Path, get_len: Callable[[str], int] = len,
               strict_boxed: bool = False) -> dict | None:
    """Grade one generation file; ``None`` for a file name without the settings suffix."""
    hyperparams = parse_hyperparameters_from_filename(file_path.name)
    if not hyperparams:
        print(f"Skipping file with unrecognized format: {file_path}")
        return None
    hyperparams["task_name"] = task_name_of(file_path)

    df = process_jsonl_file(file_path)
    # The chunk size is the response count of the FIRST problem (as run).
    num_pred = len(df[0]["responses"])

    results = {
        "hyperparameters": hyperparams,
        "mean_score": 0,
        "distinct_4gram": 0,
        "best_score": 0,
        "solve_none": 0,
        "solve_all": 0,
        "avg_output_length": 0,
        "format_error_rollouts": 0,
    }
    diverse = []
    without_boxed = 0
    response_lengths = []
    final_scores = []
    for row in df:
        responses_list = [str(r) for r in row["responses"]]
        response_lengths += [get_len(r) for r in responses_list]
        without_boxed += sum("boxed" not in r for r in responses_list)
        for response in responses_list:
            final_scores.append(bool(grade_answer_verl(response, row["gt"], strict_boxed)))
        diverse.append(get_diverse_score(responses_list))

    if final_scores:
        avg_scores = [sum(final_scores[i:i + num_pred]) / num_pred
                      for i in range(0, len(final_scores), num_pred)]
        best = [max(final_scores[i:i + num_pred]) for i in range(0, len(final_scores), num_pred)]
    else:
        avg_scores = [0]
        best = [0]

    results["mean_score"] = sum(avg_scores) / len(avg_scores) if avg_scores else 0
    results["distinct_4gram"] = sum(diverse) / len(diverse) if diverse else 0
    results["best_score"] = sum(best) / len(best) if best else 0
    results["solve_none"] = sum(1 for s in avg_scores if s == 0)
    results["solve_all"] = sum(1 for s in avg_scores if s == 1)
    results["avg_output_length"] = (sum(response_lengths) / len(response_lengths)
                                    if response_lengths else 0)
    results["format_error_rollouts"] = without_boxed
    return results


def grade_dir(out_dir: Path, *, len_tokenizer: str | Path | None = None,
              file_glob: str = "*.jsonl", result_name: str = RESULT_NAME,
              strict_boxed: bool = False) -> Path:
    """Grade every generation file in ``out_dir``; returns the results path.

    ``file_glob`` / ``result_name`` select one generation setting when runs at
    different max_tokens share a directory (each file name encodes its own).
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(f"no eval directory {out_dir}")
    get_len = length_function(len_tokenizer)
    all_results = []
    for file_path in sorted(out_dir.glob(file_glob)):
        print(f"Processing file: {file_path}", flush=True)
        res = grade_file(file_path, get_len, strict_boxed)
        if res:
            all_results.append(res)
    out = out_dir / result_name
    atomic_write_text(out, json.dumps(all_results, indent=4))
    print(f"Grading results saved to {out}", flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("eval_dir")
    ap.add_argument("--len-tokenizer", default=None,
                    help="tokenizer for avg_output_length (default: count characters)")
    ap.add_argument("--file-glob", default="*.jsonl")
    ap.add_argument("--result-name", default=RESULT_NAME)
    ap.add_argument("--strict-boxed", action="store_true",
                    help="no 'Answer:' fallback; only \\boxed{} answers count")
    a = ap.parse_args(argv)
    grade_dir(Path(a.eval_dir), len_tokenizer=a.len_tokenizer, file_glob=a.file_glob,
              result_name=a.result_name, strict_boxed=a.strict_boxed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
