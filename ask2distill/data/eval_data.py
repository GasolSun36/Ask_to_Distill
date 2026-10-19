"""Every math and code evaluation set from one release: ``gasolsun/Ask_to_Distill-eval``.

    a2d data eval [--revision SHA] [--from-dir DIR]

The dataset repo holds the evaluation sets in one record format (problem / answer / tests plus the
exact prompt). This command fetches it at a pinned revision and writes the files the evaluators
read, so ``a2d eval math|code`` runs right after it:

=====================================================  ================================================
``$A2D_DATA/eval/math/<task>/test.parquet``            AIME24, AIME25, AIME26, BRUMO25, MATH-500 (hosted)
                                                       and AMC23 (``math-ai/amc23`` @``80815d3``)
``$A2D_DATA/eval/code/extra_bench/<bench>.jsonl``      APPS introductory / interview1k / competition,
                                                       CodeContests
``$A2D_CACHE/evalplus/{HumanEvalPlus,MbppPlus}*.jsonl``  the official EvalPlus release files, verbatim
``$A2D_CACHE/hf``                                      LiveCodeBench ``test5`` / ``test6`` @``0fe84c3``
=====================================================  ================================================

AMC23 and LiveCodeBench are not rehosted (their sources declare no redistribution license); they
come from the original repositories at the pinned revisions the release names. Written with the
locked pandas / pyarrow, every file is byte-identical to the copy the final runs evaluated on; the
command checks each against the md5 recorded under ``reference`` in ``data/MANIFEST.json``.

The tau2 test tasks are part of ``a2d data build tau2``, which also builds the environment the tau2
evaluation runs in. The HMMT sets (``eval.optional_tasks``) are not in the release; ``a2d data
build math`` builds them.

``--from-dir DIR`` works without network: ``DIR/Ask_to_Distill-eval`` (a snapshot of the release),
``DIR/AMC23`` (a snapshot of ``math-ai/amc23``) and ``DIR/hf`` (an HF cache holding the two
LiveCodeBench files, e.g. a copy of ``$A2D_CACHE/hf`` made where the hub is reachable).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from ask2distill.paths import get_paths

EVAL_REPO = "gasolsun/Ask_to_Distill-eval"
EVAL_REVISION = "69d5e7fdebe7ebc0289a55cce36515734bdd8b1e"
EVAL_PATTERNS = ["math/*.jsonl", "code/*.jsonl", "code/evalplus_native/*.jsonl"]
AMC23 = ("math-ai/amc23", "80815d37005feb82cd7f8fbc6901d5d3eff43057", "test-00000-of-00001.parquet")
# release file -> eval/math/<dir>; AIME24 keeps the row layout of DAPO's own AIME-2024 parquet
MATH = {"aime24": "AIME24", "aime25": "AIME25", "aime26": "AIME26", "brumo25": "BRUMO25",
        "math500": "MATH-500"}
EXTRA_BENCH = ("apps_introductory", "apps_interview1k", "apps_competition", "codecontests")


def fetch_release(raw_root: Path, revision: str, from_dir: Path | None) -> Path:
    """Local snapshot of the evaluation release."""
    if from_dir is not None:
        p = from_dir / "Ask_to_Distill-eval"
        if not p.is_dir():
            raise FileNotFoundError(f"{p} not found (expected a snapshot of {EVAL_REPO} under --from-dir)")
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=EVAL_REPO, repo_type="dataset", revision=revision,
                                  allow_patterns=EVAL_PATTERNS,
                                  local_dir=str(raw_root / "Ask_to_Distill-eval")))


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def math_frame(rows: list[dict], name: str):
    """verl rows of one math set (the schema ``build_math_data`` writes)."""
    import pandas as pd

    from ask2distill.data.build_math_data import to_verl
    if name != "aime24":
        return to_verl([r["problem"] for r in rows], [r["answer"] for r in rows], MATH[name].lower())
    from ask2distill.common.prompts_math import DAPO_PREFIX, DAPO_TAIL
    return pd.DataFrame([{
        "data_source": "math_dapo",
        "prompt": [{"content": DAPO_PREFIX + r["problem"] + DAPO_TAIL, "role": "user"}],
        "ability": "MATH",
        "reward_model": {"ground_truth": r["answer"], "style": "rule-lighteval/MATH_v2"},
        "extra_info": {"index": int(r["source_index"]), "raw_problem": r["problem"], "split": None},
    } for r in rows])


def extra_bench_line(r: dict) -> str:
    """One row in LiveCodeBench's format (``build_code_eval_data``): starter code joins the question."""
    q = r["question"]
    if r.get("starter_code", "").strip():
        q = f"{q}\n\nUse this starter code:\n```python\n{r['starter_code'].strip()}\n```"
    return json.dumps({"id": r["id"], "question": q, "difficulty": r["difficulty"],
                       "input_output": r["tests"]}) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def amc23(data_dir: Path, raw_root: Path, from_dir: Path | None) -> str:
    import pandas as pd

    from ask2distill.data.build_math_data import to_verl, write
    repo, revision, fname = AMC23
    if from_dir is not None:
        src = from_dir / "AMC23" / fname
        if not src.is_file():
            raise FileNotFoundError(f"{src} not found (expected a snapshot of {repo} under --from-dir)")
    else:
        from huggingface_hub import hf_hub_download
        src = Path(hf_hub_download(repo, fname, repo_type="dataset", revision=revision,
                                   local_dir=str(raw_root / "AMC23")))
    df = pd.read_parquet(src)
    rel = "eval/math/AMC23/test.parquet"
    write(to_verl(df["question"], df["answer"], "amc23"), data_dir, rel)
    return rel


def build(data_dir: Path, cache: Path, revision: str = EVAL_REVISION, from_dir: Path | None = None) -> list[str]:
    from ask2distill.data import manifest
    from ask2distill.data.build_code_eval_data import (
        EVALPLUS_FILES, LCB_FILES, LCB_REPO, LCB_REVISION, hub_file, import_caches,
    )
    from ask2distill.data.build_math_data import write
    raw_root = cache / "raw"
    if from_dir is not None:
        import_caches(from_dir, cache)
        os.environ["HF_HUB_OFFLINE"] = "1"
    rel = fetch_release(raw_root, revision, from_dir)
    print(f"== {EVAL_REPO} @ {revision[:7]} -> {rel} ==", flush=True)
    written = []

    print("== math ==", flush=True)
    for name, out in MATH.items():
        written.append(f"eval/math/{out}/test.parquet")
        write(math_frame(read_jsonl(rel / "math" / f"{name}.jsonl"), name), data_dir, written[-1])
    written.append(amc23(data_dir, raw_root, from_dir))

    print("== code ==", flush=True)
    for bench in EXTRA_BENCH:
        rows = read_jsonl(rel / "code" / f"{bench}.jsonl")
        written.append(f"eval/code/extra_bench/{bench}.jsonl")
        write_text(data_dir / written[-1], "".join(extra_bench_line(r) for r in rows))
        print(f"  wrote {data_dir / written[-1]}  ({len(rows)} rows)", flush=True)
    for f in EVALPLUS_FILES:
        dst = cache / "evalplus" / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(rel / "code" / "evalplus_native" / f, dst)
        print(f"  wrote {dst}", flush=True)
    for f in LCB_FILES:
        p = hub_file(LCB_REPO, f, cache, LCB_REVISION)
        print(f"  LiveCodeBench {f} @ {LCB_REVISION[:7]} -> {p}", flush=True)

    print("== check against the files the final runs used ==", flush=True)
    manifest.report_reference(data_dir, written)
    print("\nEVAL_DATA_READY", flush=True)
    return written


def main(cfg=None, args: argparse.Namespace | None = None) -> int:
    """``a2d data eval``."""
    paths = get_paths()
    from_dir = getattr(args, "from_dir", None) if args is not None else None
    revision = (getattr(args, "revision", None) if args is not None else None) or EVAL_REVISION
    build(paths.data, paths.cache, revision, Path(from_dir) if from_dir else None)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m ask2distill.data.eval_data")
    ap.add_argument("--revision")
    ap.add_argument("--from-dir")
    sys.exit(main(None, ap.parse_args()))
