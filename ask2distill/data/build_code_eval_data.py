"""Code evaluation data: LiveCodeBench and EvalPlus caches, APPS and CodeContests.

    a2d data build code-eval [--from-dir DIR]

1. **LiveCodeBench v5_v6** -- the ``test5.jsonl`` / ``test6.jsonl`` shards of
   ``livecodebench/code_generation_lite`` at revision ``0fe84c3``, prefetched
   into ``$A2D_CACHE/hf`` (the ``HF_HOME`` every stage runs with). The vendored
   LCB loader asks for the same revision (``LCB_DATASET_REVISION``), so the eval
   finds them offline.
2. **EvalPlus** -- MBPP+ v0.2.0 and HumanEval+ v0.1.10, fetched by evalplus
   itself into ``$A2D_CACHE/evalplus`` (``XDG_CACHE_HOME=$A2D_CACHE``).
3. **extra_bench** -- APPS introductory / competition (1000 each), APPS
   interview1k and CodeContests (165) in LiveCodeBench's evaluation format,
   written to ``$A2D_DATA/eval/code/extra_bench/``. APPS interview1k is
   ``random.Random(0).shuffle`` of the 3000 interview problems, first 1000
   (first id ``apps_2639``); the full interview file stays in the cache.

``--from-dir DIR`` is for clusters whose HF file CDN is blocked: run this
command where the hub is reachable, rsync ``$A2D_CACHE/hf`` and
``$A2D_CACHE/evalplus`` into ``DIR/hf`` and ``DIR/evalplus``, then run it here
with ``--from-dir DIR``. The caches are merged in and everything else runs
with ``HF_HUB_OFFLINE=1``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

from ask2distill.paths import get_paths

LCB_REPO = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_FILES = ("test5.jsonl", "test6.jsonl")          # release_v5 + v6 = the 342-problem v5_v6 set
APPS_REPO, APPS_FILE = "codeparrot/apps", "test.jsonl"
CC_REPO, CC_FILE = "deepmind/code_contests", "data/test-00000-of-00001-9c49eeff30aacaa8.parquet"
MAX_TESTS = 50                                       # keep grading bounded (EXTRA_MAX_TESTS)
INTERVIEW_1K = 1000
INTERVIEW_1K_SEED = 0
INTERVIEW_1K_FIRST_ID = "apps_2639"
EVALPLUS_FILES = ("MbppPlus-v0.2.0.jsonl", "HumanEvalPlus-v0.1.10.jsonl")


def _trim(io: dict, limit: int) -> dict:
    """Cap the test count. CodeContests ships 200 generated tests per problem;
    grading all of them for 165 problems x n samples dominates the run and adds
    nothing -- a wrong program fails on the early cases."""
    ins, outs = io.get("inputs", []), io.get("outputs", [])
    if limit and len(ins) > limit:
        ins, outs = ins[:limit], outs[:limit]
    out = {"inputs": ins, "outputs": outs}
    if io.get("fn_name"):
        out["fn_name"] = io["fn_name"]
    return out


def apps(src: str | os.PathLike, out_path: Path, tiers: set[str]) -> int:
    """Convert APPS ``test.jsonl`` rows of ``tiers`` into the LCB format."""
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f, open(src) as fi:
        for line in fi:
            r = json.loads(line)
            if r["difficulty"] not in tiers or not r["input_output"]:
                continue
            io = json.loads(r["input_output"])
            if not io.get("inputs"):
                continue
            q = r["question"]
            # The 1% call-based problems need their starter code shown, or the
            # model cannot know which function name the tests will call.
            if r.get("starter_code", "").strip():
                q = f"{q}\n\nUse this starter code:\n```python\n{r['starter_code'].strip()}\n```"
            f.write(json.dumps({"id": f"apps_{r['id']}", "question": q,
                                "difficulty": r["difficulty"],
                                "input_output": json.dumps(_trim(io, MAX_TESTS))}) + "\n")
            n += 1
    print(f"[apps] {sorted(tiers)} -> {out_path}: {n}", flush=True)
    return n


def code_contests(src: str | os.PathLike, out_path: Path) -> int:
    """Convert the CodeContests test parquet (public + private + generated tests)."""
    import pandas as pd
    d = pd.read_parquet(src)
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for _, r in d.iterrows():
            ins, outs = [], []
            for key in ("public_tests", "private_tests", "generated_tests"):
                t = r.get(key)
                if t is not None and len(t["input"]):
                    ins += list(t["input"])
                    outs += list(t["output"])
            if not ins:
                continue
            f.write(json.dumps({"id": f"cc_{r['name'][:60]}", "question": r["description"],
                                "difficulty": f"cf{int(r['difficulty'])}",
                                "input_output": json.dumps(
                                    _trim({"inputs": ins, "outputs": outs}, MAX_TESTS))}) + "\n")
            n += 1
    print(f"[codecontests] -> {out_path}: {n}", flush=True)
    return n


def interview_1k(full: Path, out_path: Path) -> int:
    """``Random(0).shuffle`` of the interview rows, first 1000, lines kept byte for byte."""
    with open(full) as fh:
        rows = fh.readlines()
    random.Random(INTERVIEW_1K_SEED).shuffle(rows)
    rows = rows[:INTERVIEW_1K]
    first = json.loads(rows[0])["id"]
    if first != INTERVIEW_1K_FIRST_ID:
        raise RuntimeError(f"apps_interview1k starts with {first}, expected {INTERVIEW_1K_FIRST_ID}: "
                           "the APPS interview file differs from the one the paper used")
    with open(out_path, "w") as fh:
        fh.writelines(rows)
    print(f"[apps_interview1k] -> {out_path}: {len(rows)} (first {first})", flush=True)
    return len(rows)


def merge_tree(src: Path, dst: Path) -> None:
    """Copy ``src`` into ``dst``, keeping files that already exist."""
    for p in src.rglob("*"):
        q = dst / p.relative_to(src)
        if p.is_symlink() and not p.exists():
            continue
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        elif not q.exists():
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q, follow_symlinks=True)


def import_caches(from_dir: Path, cache: Path) -> None:
    """Merge ``DIR/hf`` (or ``DIR/hub``) and ``DIR/evalplus`` into ``$A2D_CACHE``."""
    hub = from_dir / "hf" / "hub" if (from_dir / "hf" / "hub").is_dir() else from_dir / "hub"
    if hub.is_dir():
        merge_tree(hub, cache / "hf" / "hub")
        print(f"[cache] merged {hub} -> {cache / 'hf' / 'hub'}")
    if (from_dir / "evalplus").is_dir():
        merge_tree(from_dir / "evalplus", cache / "evalplus")
        print(f"[cache] merged {from_dir / 'evalplus'} -> {cache / 'evalplus'}")


def hub_file(repo: str, filename: str, cache: Path, revision: str | None = None) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                           cache_dir=str(cache / "hf" / "hub"))


def prefetch_evalplus(cache: Path) -> None:
    """Let evalplus download its own datasets into ``$A2D_CACHE/evalplus``."""
    code = ("from evalplus.data import get_human_eval_plus, get_mbpp_plus; "
            "get_human_eval_plus(); get_mbpp_plus()")
    env = dict(os.environ, XDG_CACHE_HOME=str(cache))
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
    missing = [f for f in EVALPLUS_FILES if not (cache / "evalplus" / f).is_file()]
    if missing:
        raise RuntimeError(f"evalplus did not produce {missing} under {cache / 'evalplus'}")
    print(f"[evalplus] {', '.join(EVALPLUS_FILES)} in {cache / 'evalplus'}")


def build(data_dir: Path, cache: Path, from_dir: Path | None = None) -> None:
    if from_dir is not None:
        import_caches(from_dir, cache)
        os.environ["HF_HUB_OFFLINE"] = "1"
    for f in LCB_FILES:
        p = hub_file(LCB_REPO, f, cache, LCB_REVISION)
        print(f"[lcb] {f} @ {LCB_REVISION[:7]} -> {p}")
    prefetch_evalplus(cache)

    xb = data_dir / "eval" / "code" / "extra_bench"
    apps_src = hub_file(APPS_REPO, APPS_FILE, cache)
    apps(apps_src, xb / "apps_introductory.jsonl", {"introductory"})
    apps(apps_src, xb / "apps_competition.jsonl", {"competition"})
    full = cache / "extra_bench" / "apps_interview.jsonl"
    apps(apps_src, full, {"interview"})
    interview_1k(full, xb / "apps_interview1k.jsonl")
    code_contests(hub_file(CC_REPO, CC_FILE, cache), xb / "codecontests.jsonl")


def main(cfg=None, args: argparse.Namespace | None = None) -> int:
    """``a2d data build code-eval``."""
    paths = get_paths()
    from_dir = getattr(args, "from_dir", None) if args is not None else None
    build(paths.data, paths.cache, Path(from_dir) if from_dir else None)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m ask2distill.data.build_code_eval_data")
    ap.add_argument("--from-dir")
    sys.exit(main(None, ap.parse_args()))
