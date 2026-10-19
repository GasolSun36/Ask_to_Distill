"""Stage 2b: pick a synthesizer checkpoint, reward against diversity, on sampled output.

Reward comes from the training dump (64 scored questions per step, averaged
over the reward ranks and over +-``reward_window`` steps around the
checkpoint) rather than a separate scoring pass -- an earlier selection used 5
questions per checkpoint, which is noise. It is ``synrm/R_raw`` = sum s_t / L^alpha,
without the length penalty or the diversity multiplier. Diversity has to come
from fresh sampling (``select.sweep``): collapse is only visible at sampling
time.

Per swept checkpoint (files with fewer than ``min_rows`` questions are skipped):

* mean off-diagonal cosine of the question embeddings;
* effective clusters: inverse Simpson index over a greedy grouping at cosine
  ``eff_cluster_cos`` (0.9);
* a topic-share column (diagnostic only; one question counts once, toward its
  first matching keyword).

Gates, in order: diversity (mean cosine < ``cos_max`` and effective clusters >=
``min_eff_clusters``) -> reward (>= ``reward_frac`` of the best passing
reward) -> lowest mean cosine wins. The thresholds sit between the mean cosine
of a healthy and of a collapsed synthesizer, and the cluster floor below that of
an untrained synthesizer. A pair may additionally compare saturation yield over
candidate steps (``select.yield_compare``); the step used is ``select.chosen_step``.

The reward worker numbers its steps from 1 in every 4-hour window
(``reward.step_counter: local``); ``windows`` maps that back onto global steps.
A dump row that carries a ``global_step`` field is used as is.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ask2distill import config as a2d_config
from ask2distill.common.io import atomic_write_text

# Only a diagnostic column (the gate is cos_max), but a math vocabulary scores
# every code question as "no topic" and the column stops meaning anything.
TOPICS = {
    "math": ("grid", "cube", "rectangle", "square", "triangle", "circle",
             "polynomial", "sequence", "prime", "probability", "function", "matrix"),
    "code": ("array", "string", "tree", "graph", "query", "substring", "subarray",
             "permutation", "matrix", "interval", "prefix", "modulo", "game",
             "shortest path", "binary search", "dynamic programming"),
}
_STEP_RE = re.compile(r"step(\d+)")


def windows(local_steps: Sequence[int], save_freq: int) -> list[int]:
    """Map a worker's local step counter back onto global steps.

    The reward worker numbers its steps from 1 and resets on every restart, so a
    run spanning three 4h windows writes 1..23, 1..22, 1..20 for what are really
    global steps 1..23, 21..42, 41..60. Left uncorrected the three windows pile
    onto the same keys and every step past the first window's length reads as
    missing -- on the 7B run that meant steps 30-60 had no reward at all and the
    Pareto layer could only choose from the first 25.

    A restart resumes from the last saved checkpoint, so the offset of the next
    window is the largest multiple of ``save_freq`` at or below the global step
    the previous window died on.
    """
    out, off, prev = [], 0, 0
    for k in local_steps:
        if k <= prev:                      # counter reset -> new window
            off = ((off + prev) // save_freq) * save_freq
        out.append(off + k)
        prev = k
    return out


def _metric_rows(path: str) -> Iterable[dict]:
    with open(path, errors="ignore") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except Exception:
                continue


def reward_by_step(dump_prefix: str | os.PathLike, save_freq: int) -> dict[int, float]:
    """Mean ``synrm/R_raw`` per global step over every ``<dump_prefix>.metrics.r*`` file.

    Keys are (global step, window index) first, so a step replayed after a
    rollback is resolved in favour of the window that actually continued.
    """
    by: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
    for f in sorted(glob.glob(os.fspath(dump_prefix) + ".metrics.r*")):
        local, given, vals = [], [], []
        for d in _metric_rows(f):
            if "synrm/R_raw" in d:
                local.append(d["step"])
                given.append(d.get("global_step"))
                vals.append(d["synrm/R_raw"])
        gl = given if local and all(g is not None for g in given) else windows(local, save_freq)
        win, prev = 0, 0
        for k, g, v in zip(local, gl, vals):
            if k <= prev:
                win += 1
            by[(g, win)].append(v)
            prev = k
    best: dict[int, tuple[int, float]] = {}
    for (g, win), v in by.items():
        if g not in best or win > best[g][0]:
            best[g] = (win, sum(v) / len(v))
    return {g: v for g, (_, v) in best.items()}


def sweep_files(sweep_dir: str | os.PathLike) -> list[tuple[int, Path]]:
    """``(step, path)`` for every ``sweep_step<N>.jsonl``, in numeric step order."""
    files = glob.glob(os.path.join(os.fspath(sweep_dir), "sweep_step*.jsonl"))
    steps = [(int(_STEP_RE.search(os.path.basename(p)).group(1)), Path(p)) for p in files]
    return sorted(steps, key=lambda x: x[0])


def diversity_stats(E: np.ndarray, eff_cluster_cos: float) -> tuple[float, float]:
    """``(mean off-diagonal cosine, effective clusters)`` of unit-row embeddings ``E``."""
    n = len(E)
    C = E @ E.T
    off = C[~np.eye(n, dtype=bool)]
    # effective clusters: inverse Simpson index over a greedy cosine grouping
    seen: list[int] = []
    lab: list[int] = []
    for i in range(n):
        hit = next((j for j, k in enumerate(seen) if float(E[i] @ E[k]) > eff_cluster_cos), None)
        if hit is None:
            seen.append(i)
            lab.append(len(seen) - 1)
        else:
            lab.append(hit)
    cnt = np.bincount(lab).astype(float) / n
    return float(off.mean()), 1.0 / float((cnt ** 2).sum())


def topic_share(qs: Sequence[str], topics: Sequence[str]) -> float:
    """Share of questions in the four most common topics (excluding "no topic")."""
    # One question counts once, toward its first matching topic. Counting every
    # keyword hit lets a question about a square grid land in two buckets,
    # which pushed this "share" over 100%.
    tc: collections.Counter = collections.Counter()
    for q in qs:
        low = q.lower()
        hit = next((t for t in topics if t in low), None)
        tc[hit or "_other"] += 1
    return sum(c for t, c in tc.most_common(4) if t != "_other") / max(len(qs), 1)


def _pcfg(pcfg: Any, key: str) -> Any:
    return pcfg[key] if isinstance(pcfg, Mapping) else getattr(pcfg, key)


def pick(dump_prefix: str | os.PathLike, sweep_dir: str | os.PathLike, pcfg: Any, embedder: Any,
         *, domain: str = "math") -> dict:
    """Run the three gates over every sweep file; ``{"step": int | None, "table": [...], ...}``.

    ``pcfg`` is the ``select.pick`` section. ``step`` is ``None`` when no
    checkpoint passes the diversity gate, unless ``allow_empty`` (smoke only),
    which falls back to the last swept step and sets ``fallback``.
    """
    cos_max = float(_pcfg(pcfg, "cos_max"))
    min_eff = float(_pcfg(pcfg, "min_eff_clusters"))
    eff_cos = float(_pcfg(pcfg, "eff_cluster_cos"))
    frac = float(_pcfg(pcfg, "reward_frac"))
    window = int(_pcfg(pcfg, "reward_window"))
    min_rows = int(_pcfg(pcfg, "min_rows"))
    allow_empty = bool(_pcfg(pcfg, "allow_empty"))
    topics = TOPICS[domain]

    rw = reward_by_step(dump_prefix, int(_pcfg(pcfg, "save_freq")))
    table: list[dict] = []
    for st, p in sweep_files(sweep_dir):
        with open(p, encoding="utf-8") as fh:
            qs = [json.loads(line)["problem"] for line in fh]
        if len(qs) < min_rows:
            continue
        E = embedder.embed(qs)
        E = E.numpy() if hasattr(E, "numpy") else np.asarray(E)
        mean_cos, eff = diversity_stats(E, eff_cos)
        # reward near this checkpoint: the training steps it was saved from
        near = [rw[s] for s in rw if abs(s - st) <= window]
        reward = sum(near) / len(near) if near else float("nan")
        passed = mean_cos < cos_max and eff >= min_eff
        table.append({"step": st, "reward": reward, "mean_cos": mean_cos, "eff_clusters": eff,
                      "top_topics": topic_share(qs, topics), "n": len(qs), "passed": passed,
                      "reason": None if passed else
                      ("cos_over_max" if mean_cos >= cos_max else "too_few_clusters")})

    ok = [r for r in table if r["passed"]]
    if not ok:
        if allow_empty and table:
            last = table[-1]
            print(f"[pick] WARNING: no checkpoint passed the diversity gate; allow_empty -> "
                  f"falling back to the last swept step {last['step']}", flush=True)
            return {"step": last["step"], "reward": last["reward"], "mean_cos": last["mean_cos"],
                    "eff_clusters": last["eff_clusters"], "n_passing": 0,
                    "best_reward": float("nan"), "fallback": True, "table": table}
        return {"step": None, "n_passing": 0, "fallback": False, "table": table}
    best_r = max(r["reward"] for r in ok)
    # among those within reward_frac of the best reward, take the most diverse
    cand = [r for r in ok if r["reward"] >= frac * best_r]
    chosen = min(cand, key=lambda r: r["mean_cos"])
    return {"step": chosen["step"], "reward": chosen["reward"], "mean_cos": chosen["mean_cos"],
            "eff_clusters": chosen["eff_clusters"], "n_passing": len(ok), "best_reward": best_r,
            "n_candidates": len(cand), "fallback": False, "table": table}


def format_table(result: dict) -> str:
    """The per-checkpoint table and the decision, as the original printed them."""
    lines = [f"{'step':>5}{'reward':>9}{'mean_cos':>10}{'eff_clu':>9}{'top_topic':>10}{'n':>6}  verdict"]
    for r in result["table"]:
        verdict = "pass" if r["passed"] else f"FAIL {r['reason']}"
        lines.append(f"{r['step']:>5}{r['reward']:>9.4f}{r['mean_cos']:>10.3f}{r['eff_clusters']:>9.1f}"
                     f"{100 * r['top_topics']:>9.0f}%{r['n']:>6}  {verdict}")
    if result["step"] is None:
        lines.append("\nno checkpoint passed the diversity gate -- an earlier step or "
                     "different training is needed")
    elif result.get("fallback"):
        lines.append(f"\nfallback (allow_empty) -> step {result['step']}")
    else:
        best = result["best_reward"]
        lines.append(f"\n{result['n_passing']} passed the gates; reward peak {best:.4f}, "
                     f"{result['n_candidates']} within the reward band")
        pct = 100 * result["reward"] / best if best else float("nan")
        lines.append(f"pareto pick -> step {result['step']}  (reward {result['reward']:.4f} = "
                     f"{pct:.0f}% of peak, cos {result['mean_cos']:.3f}, "
                     f"eff clusters {result['eff_clusters']:.1f})")
    return "\n".join(lines)


def _json_safe(x: Any) -> Any:
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    return x


def main(cfg, args: argparse.Namespace | None = None) -> int:
    """``a2d select pick --pair P``: write ``select.pick.out``."""
    a2d_config.require(cfg, "select.pick")
    a2d_config.require(cfg, "select.sweep")
    p, s = cfg.select.pick, cfg.select.sweep
    dump_prefix = str(cfg.reward.dump)
    # Count, don't just test for existence (see select.sweep): every expected
    # step must have its sweep file before a pick means anything.
    missing = [int(st) for st in s.steps
               if not (Path(str(s.out_dir)) / f"sweep_step{int(st)}.jsonl").is_file()]
    problems = []
    if missing:
        problems.append(f"sweep incomplete: no sweep file for step(s) {missing} in {s.out_dir}; "
                        f"run `a2d select sweep --pair {cfg.pair}` first")
    if not glob.glob(dump_prefix + ".metrics.r*"):
        problems.append(f"no reward metrics at {dump_prefix}.metrics.r*")
    if getattr(args, "dry_run", False):
        print(f"[pick] would pick from {s.out_dir} with rewards {dump_prefix}.metrics.r* "
              f"(save_freq {p.save_freq}) -> {p.out}", flush=True)
        for msg in problems:
            print(f"[pick] not ready: {msg}", flush=True)
        return 0
    if problems:
        raise SystemExit("[pick] " + "; ".join(problems))
    from ask2distill.common.embed import Embedder
    from ask2distill.harvest.sampling import export_stage_env
    export_stage_env("select")
    embedder = Embedder(str(cfg.models.embedder), device="cuda")
    result = pick(dump_prefix, str(s.out_dir), p, embedder, domain=str(cfg.domain))
    print(format_table(result), flush=True)
    if result["step"] is None:
        return 1
    if int(result["step"]) != int(cfg.select.chosen_step):
        print(f"[pick] note: select.chosen_step={cfg.select.chosen_step} (frozen) differs from "
              f"this pick ({result['step']}); see the pair config for why", flush=True)
    out = Path(str(p.out))
    atomic_write_text(out, json.dumps(_json_safe(result), indent=1) + "\n")
    print(f"[pick] -> {out}", flush=True)
    return 0
