"""Pick the tau2 synthesizer checkpoint of a version from its reward dumps.

Rule: for every saved step N, take
the dump rows of the trailing window of ``select.rule.window`` steps
(N-3 .. N) and score

    valid fraction x mean R_raw over valid cards  ( = sum of R_raw over valid rows / rows )

where a row is valid when the reward rejected nothing. The highest score wins
(ties: the earliest step). R_raw is the service's reward before the diversity
multiplier. The tau2 worker stamps rows with verl's global step, so no window
correction is needed (unlike the math/code worker's restarting counter).

The steps used for reproduction are ``synrl.versions.<v>.chosen_step``.
``a2d tau2 select`` prints the table, the rule's pick and that step, and warns
when they differ.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

from omegaconf import DictConfig


def load_dump(dump_prefix: str | Path) -> dict[int, list[dict]]:
    """Rows of every reward rank (``<prefix>.r<rank>``), grouped by step."""
    by: dict[int, list[dict]] = defaultdict(list)
    for f in sorted(glob.glob(f"{dump_prefix}.r[0-9]*")):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    by[int(r["step"])].append(r)
    return dict(by)


def window_score(by_step: dict[int, list[dict]], step: int, window: int) -> tuple[float, int]:
    """(score, rows) over steps ``step-window+1 .. step``."""
    rows = [r for s in range(step - window + 1, step + 1) for r in by_step.get(s, [])]
    if not rows:
        return 0.0, 0
    valid = [r for r in rows if not r.get("rejected")]
    return sum(float(r["R_raw"]) for r in valid) / len(rows), len(rows)


def score_table(by_step: dict[int, list[dict]], save_freq: int, total_steps: int,
                window: int) -> list[dict]:
    """One record per saved step: ``{step, score, rows}``."""
    out = []
    for step in range(save_freq, total_steps + 1, save_freq):
        if step not in by_step:
            continue
        s, n = window_score(by_step, step, window)
        out.append({"step": step, "score": s, "rows": n})
    return out


def pick(table: list[dict]) -> int:
    """Step with the highest score (earliest on ties)."""
    if not table:
        raise ValueError("no saved step has dump rows")
    best = max(r["score"] for r in table)
    return min(r["step"] for r in table if r["score"] == best)


def main(cfg: DictConfig, args: argparse.Namespace) -> int:
    """``a2d tau2 select --version vX``."""
    version = getattr(args, "version", None)
    if not version:
        raise SystemExit("a2d tau2 select needs --version")
    v = cfg.synrl.versions[version]
    rule = cfg.select.rule
    if str(rule.score) != "valid_frac_x_mean_R_valid":
        raise SystemExit(f"unknown select.rule.score {rule.score!r}")
    by = load_dump(str(v.dump))
    if not by:
        raise SystemExit(f"no dump rows at {v.dump}.r*")
    table = score_table(by, int(cfg.synrl.save_freq), int(cfg.synrl.total_steps), int(rule.window))
    for r in table:
        print(f"step {r['step']:>4}  score {r['score']:.4f}  rows {r['rows']}", flush=True)
    rule_step, frozen = pick(table), int(v.chosen_step)
    print(f"rule pick: {version}@{rule_step}   frozen pick: {version}@{frozen}", flush=True)
    if rule_step != frozen:
        print(f"WARNING: the window rule picks {rule_step}, the chain used {frozen} "
              "(synrl.versions.<v>.chosen_step); merge the frozen step to reproduce", flush=True)
    out = Path(str(cfg.paths.out)) / str(cfg.pair) / "synrl" / f"select_{version}.json"
    from ask2distill.common.io import atomic_write_json
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, {"version": version, "rule_step": rule_step, "chosen_step": frozen,
                            "table": table})
    print(f"-> {out}", flush=True)
    return 0
