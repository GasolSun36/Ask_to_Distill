"""pass^1 on the official tau2 test split: per domain and the unweighted mean of three.

For each domain, results are grouped by ``task_id``; only tasks with at least
four trials count, and pass^1 is the mean of ``reward >= 0.999`` over their
first four trials, in percent, rounded to one decimal. ``mean3`` is the sum of
the three domain numbers over 3 (``None`` unless all three exist). pass^4 is
reported alongside. Pure JSON -- no tau2 import.

Caveat (``EVAL_TEST_ALL_DONE``): while an evaluation is still running, tasks
with fewer than four trials are left out, so a partial summary under-reports
task counts (shown as "(<n>t)" in the table) and is not comparable.

    a2d tau2 summarize --outputs-dir $A2D_OUT/tau2/outputs/test4 --models Agent_Qwen3-4B_Qwen3-4B-Thinking-2507-ours-s<N> [--out PREFIX]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics as st
from pathlib import Path

DOMAINS = ("airline", "retail", "telecom")
TRIALS = 4
PASS_THRESHOLD = 0.999
# full task counts of the official test split
FULL_TASKS = {"airline": 20, "retail": 40, "telecom": 40}


def record(outputs_dir: Path, served_name: str, *, trials: int = TRIALS,
           threshold: float = PASS_THRESHOLD) -> dict:
    """``{domain: {pass1, pass4, tasks}, ..., "mean3": float | None}`` (the format of the reported summaries).

    ``outputs_dir`` is the tag directory holding ``<served_name>__<domain>/*.json``.
    """
    r: dict = {}
    for d in DOMAINS:
        by = collections.defaultdict(list)
        for f in glob.glob(f"{outputs_dir}/{served_name}__{d}/*.json"):
            with open(f) as fh:
                x = json.load(fh)
            by[x["task_id"]].append(float(x["reward"] >= threshold))
        full = [v for v in by.values() if len(v) >= trials]
        r[d] = {"pass1": round(100 * st.mean(x for v in full for x in v[:trials]), 1) if full else None,
                "pass4": round(100 * st.mean(all(v[:trials]) for v in full), 1) if full else None,
                "tasks": len(full)}
    ok = [r[d]["pass1"] for d in r if r[d]["pass1"] is not None]
    r["mean3"] = round(sum(ok) / 3, 1) if len(ok) == 3 else None
    return r


def summarize(outputs_dir: Path, served_name: str, *, trials: int = TRIALS,
              threshold: float = PASS_THRESHOLD) -> dict:
    """Flat pass^1 summary for one served model.

    ``{"airline", "retail", "telecom": pass^1 or None, "mean": mean3, "tasks": {domain: n},
    "detail": record(...)}`` -- the shape ``a2d collect tau2`` tabulates.
    """
    r = record(outputs_dir, served_name, trials=trials, threshold=threshold)
    return {**{d: r[d]["pass1"] for d in DOMAINS}, "mean": r["mean3"],
            "tasks": {d: r[d]["tasks"] for d in DOMAINS}, "detail": r}


def table(rows: dict[str, dict]) -> str:
    """The markdown table of the reported summaries."""
    lines = ["| model | airline | retail | telecom | mean |", "|---|---:|---:|---:|---:|"]
    for m, r in rows.items():
        def c(d):
            return f"{r[d]['pass1']}" + ("" if r[d]["tasks"] == FULL_TASKS[d] else f" ({r[d]['tasks']}t)")
        lines.append(f"| {m} | {c('airline')} | {c('retail')} | {c('telecom')} | {r['mean3']} |")
    return "\n".join(lines) + "\n"


def main(cfg, args: argparse.Namespace) -> int:
    """``a2d tau2 summarize --models M [M ...] [--outputs-dir DIR] [--out PREFIX]``."""
    models = getattr(args, "models", None)
    if not models:
        raise SystemExit("a2d tau2 summarize needs --models")
    outputs_dir = getattr(args, "outputs_dir", None) or (Path(str(cfg.eval.out_dir)) / str(cfg.eval.tag))
    trials = int(cfg.eval.trials) if cfg is not None else TRIALS
    threshold = float(cfg.eval.pass_threshold) if cfg is not None else PASS_THRESHOLD
    rows = {m: record(Path(outputs_dir), m, trials=trials, threshold=threshold) for m in models}
    text = table(rows)
    out = getattr(args, "out", None)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(str(out) + ".json").write_text(json.dumps(rows, indent=1))
        Path(str(out) + ".md").write_text(text)
    print(text, end="", flush=True)
    return 0
