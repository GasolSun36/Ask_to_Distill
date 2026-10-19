"""Build DAPO-Math-17k (the math Vanilla arm) and the eight math test sets.

    a2d data build math [--from-dir DIR]

Writes, under ``$A2D_DATA``::

    real/DAPO-Math-17k.parquet          training set (17,398 rows)
    eval/math/AIME24/test.parquet       30     eval/math/AIME25/test.parquet     30
    eval/math/AIME26/test.parquet       30     eval/math/AMC23/test.parquet      40
    eval/math/BRUMO25/test.parquet      30     eval/math/MATH-500/test.parquet   500
    eval/math/HMMT_FEB25/test.parquet   30     eval/math/HMMT_FEB26/test.parquet 33

``MATH-500`` keeps its hyphen: that is the directory the eval reads
(``eval.task_dirs`` in ``configs/base/eval_math.yaml``).

Two things worth knowing about the upstream files:

* BytedTsinghua-SIA/DAPO-Math-17k ships 1,791,700 rows -- the 17,398 real
  prompts repeated 103x. Training on that makes "1 epoch" ~28,000 steps instead
  of ~279, so it is deduplicated here.
* BytedTsinghua-SIA/AIME-2024 ships 960 rows = 30 problems x 32, for DAPO's
  avg@32 protocol. Deduplicated to 30; the eval samples 16 per problem itself.

AIME25/AMC23 and the MathArena sets have no ready-made verl-format release, so
they are converted into exactly the schema the DAPO files use (verified field
by field against aime-2024.parquet), including the same prompt wrapper.

Raw downloads go to ``$A2D_CACHE/raw/<name>`` (``snapshot_download``; the
original fetched the hub's ``main`` and recorded no revision). ``--from-dir
DIR`` reads them from ``DIR/<name>/`` instead -- a copy of that raw directory
made where the hub is reachable. Each built file is compared with the md5 the
final runs' copy had (recorded under ``reference`` in ``data/MANIFEST.json``;
neither DAPO-Math-17k nor the test sets are redistributed); a rebuild is
content-equal but its bytes may differ with the upstream revision or the
parquet writer version, which this command reports.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ask2distill.common.prompts_math import DAPO_PREFIX, DAPO_TAIL
from ask2distill.paths import get_paths

TRAIN = ("BytedTsinghua-SIA/DAPO-Math-17k", "DAPO-Math-17k", "data/dapo-math-17k.parquet",
         "real/DAPO-Math-17k.parquet")
AIME24 = ("BytedTsinghua-SIA/AIME-2024", "AIME-2024", "data/aime-2024.parquet",
          "eval/math/AIME24/test.parquet")
# (repo_id, local raw name, file inside the repo, problem column, answer column, out dir)
CONVERTED = [
    ("math-ai/aime25",          "AIME25",     "test.jsonl",                        "problem",  "answer", "AIME25"),
    ("math-ai/amc23",           "AMC23",      "test-00000-of-00001.parquet",       "question", "answer", "AMC23"),
    ("MathArena/aime_2026",     "AIME26",     "data/train-00000-of-00001.parquet", "problem",  "answer", "AIME26"),
    ("HuggingFaceH4/MATH-500",  "MATH-500",   "test.jsonl",                        "problem",  "answer", "MATH-500"),
    ("MathArena/hmmt_feb_2025", "HMMT_FEB25", "data/train-00000-of-00001.parquet", "problem",  "answer", "HMMT_FEB25"),
    ("MathArena/hmmt_feb_2026", "HMMT_FEB26", "data/train-00000-of-00001.parquet", "problem",  "answer", "HMMT_FEB26"),
    # MathArena only publishes brumo_2025; there is no brumo_2026 repo.
    ("MathArena/brumo_2025",    "BRUMO25",    "data/train-00000-of-00001.parquet", "problem",  "answer", "BRUMO25"),
]
EVAL_DIRS = ["AIME24", "AIME25", "AMC23", "AIME26", "MATH-500", "HMMT_FEB25", "HMMT_FEB26", "BRUMO25"]


def fetch(repo_id: str, raw_name: str, raw_root: Path, from_dir: Path | None) -> Path:
    """Local directory holding ``repo_id``'s files."""
    if from_dir is not None:
        p = from_dir / raw_name
        if not p.is_dir():
            raise FileNotFoundError(f"{p} not found (expected a copy of {repo_id} under --from-dir)")
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset",
                                  local_dir=str(raw_root / raw_name)))


def dedup(df, label: str):
    """Drop the repeated copies, keeping first occurrence (and thus the original order)."""
    key = df["prompt"].apply(lambda m: m[0]["content"])
    out = df[~key.duplicated(keep="first")].reset_index(drop=True)
    print(f"  {label}: {len(df)} -> {len(out)} rows after dedup")
    return out


def as_answer_str(a) -> str:
    """AMC23 stores answers as floats (27.0); the grader compares strings."""
    if isinstance(a, float) and a.is_integer():
        return str(int(a))
    return str(a)


def to_verl(problems, answers, source_note: str):
    import pandas as pd
    rows = []
    for i, (prob, ans) in enumerate(zip(problems, answers)):
        rows.append({
            "data_source": "math_dapo",
            "prompt": [{"role": "user", "content": DAPO_PREFIX + prob + DAPO_TAIL}],
            "ability": "MATH",
            "reward_model": {"ground_truth": as_answer_str(ans), "style": "rule-lighteval/MATH_v2"},
            "extra_info": {"index": i, "raw_problem": prob, "split": source_note},
        })
    return pd.DataFrame(rows)


def write(df, data_dir: Path, rel: str) -> Path:
    path = data_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    print(f"  wrote {path}  ({len(df)} rows)")
    return path


def sanity_check(data_dir: Path) -> None:
    """Every file loads and matches the DAPO schema."""
    import pandas as pd
    ref = None
    for rel in [TRAIN[3]] + [f"eval/math/{d}/test.parquet" for d in EVAL_DIRS]:
        d = pd.read_parquet(data_dir / rel)
        cols = list(d.columns)
        ref = ref or cols
        assert cols == ref, f"{rel} columns {cols} != {ref}"
        r = d.iloc[0]
        assert r["prompt"][0]["role"] == "user"
        assert r["reward_model"]["ground_truth"]
        print(f"  OK {rel:34s} {len(d):>6} rows  gt[0]={r['reward_model']['ground_truth']!r}")


def compare_with_manifest(data_dir: Path) -> None:
    """Report rebuilt files whose bytes differ from the copies the final runs used."""
    from ask2distill.data import manifest as M
    try:
        man = M.load()
    except M.ManifestError:
        return
    rels = {TRAIN[3]} | {f"eval/math/{d}/test.parquet" for d in EVAL_DIRS}
    for e in M.compare_entries(man):
        if e["path"] in rels:
            status, detail = M.check_entry(data_dir, e)
            if status == "mismatch":
                print(f"  note: {detail} -- content-equal rebuilds can differ in parquet bytes "
                      "(upstream revision or parquet writer version)")
            elif status == "ok":
                print(f"  {e['path']}: identical to the file the final runs used")


def build(data_dir: Path, raw_root: Path, from_dir: Path | None = None) -> None:
    import pandas as pd
    raw_root.mkdir(parents=True, exist_ok=True)

    print("== train: DAPO-Math-17k ==")
    p = fetch(TRAIN[0], TRAIN[1], raw_root, from_dir)
    write(dedup(pd.read_parquet(p / TRAIN[2]), "DAPO"), data_dir, TRAIN[3])

    print("== eval: AIME24 ==")
    p = fetch(AIME24[0], AIME24[1], raw_root, from_dir)
    write(dedup(pd.read_parquet(p / AIME24[2]), "AIME24"), data_dir, AIME24[3])

    for repo, raw_name, fname, pcol, acol, out_dir in CONVERTED:
        print(f"== eval: {out_dir} ==")
        p = fetch(repo, raw_name, raw_root, from_dir)
        f = p / fname
        df = pd.read_json(f, lines=True) if f.suffix == ".jsonl" else pd.read_parquet(f)
        write(to_verl(df[pcol], df[acol], out_dir.lower()), data_dir, f"eval/math/{out_dir}/test.parquet")

    print("\n== sanity check: every file loads and matches the DAPO schema ==")
    sanity_check(data_dir)
    compare_with_manifest(data_dir)
    print("\nDATASETS_READY")


def main(cfg=None, args: argparse.Namespace | None = None) -> int:
    """``a2d data build math``."""
    paths = get_paths()
    from_dir = getattr(args, "from_dir", None) if args is not None else None
    build(paths.data, paths.cache / "raw", Path(from_dir) if from_dir else None)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m ask2distill.data.build_math_data")
    ap.add_argument("--from-dir")
    sys.exit(main(None, ap.parse_args()))
