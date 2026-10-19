"""Stage 3d: turn a filtered question pool into a verl OPD training parquet.

Arms must be budget-matched exactly: the comparison is about which questions
were generated, not how many. The first ``bank.n`` questions are taken in
file order -- the shards already interleave and the questions were deduped as
they were written -- so no extra selection step sneaks a second variable in.

Row format (no ground truth: OPD needs none; the reward function is only
logged):

* math (``template: dapo_prefix_only``): ``data_source "math_dapo"``, one user
  turn ``DAPO_PREFIX + q`` -- WITHOUT the "Remember to put your answer..."
  tail that the DAPO-17k Vanilla rows carry (docs/DEVIATIONS.md), ``ability
  "MATH"``;
* code (``template: lcb_no_system``): ``data_source "code_stdio"``, one user
  turn ``solve_prompt(q)`` (the LiveCodeBench template, byte-identical to
  the real-question arm OpenCodeReasoning-10k and to the eval), no system turn, ``ability "CODE"``;
* both: ``reward_model {"ground_truth": "N/A", "style": "rule-lighteval/MATH_v2"}``,
  ``extra_info {"index": "<bank.row_id_prefix>_<i>"}``. The prefix is a config
  value, not derived from the bank name, so a rebuilt bank keeps the row ids
  of the released bank byte for byte (e.g. ``q3armg9_0``; do not change).

A pool shorter than ``n`` is an error unless ``bank.allow_short``: Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500
is the one final bank that is short (2,889 < 3,200; its filter pass was dedup-limited)
and trained on all of it; smoke configs also allow it.

Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507 first re-filters its pool
(config ``bank.refilter``, module ``ask2distill.bank.refilter``).
After writing, the step count after verl's overlong-prompt filter is reported
(``arm_steps``) when the student's tokenizer is available.

Output ``bank.out`` (``$A2D_OUT/<pair>/banks/<bank.name>.parquet``); the released
banks under ``$A2D_DATA/banks`` are what ``a2d opd`` uses by default.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import DictConfig

from ask2distill import config as a2d_config
from ask2distill.common.io import iter_jsonl
from ask2distill.common.prompts_code import solve_prompt
from ask2distill.common.prompts_math import bank_prompt

REWARD_MODEL = {"ground_truth": "N/A", "style": "rule-lighteval/MATH_v2"}
TEMPLATES = {"math": "dapo_prefix_only", "code": "lcb_no_system"}


def bank_rows(questions: list[str], domain: str, row_id_prefix: str) -> list[dict]:
    """verl rows for ``questions`` (already cut to N); row i gets index ``<row_id_prefix>_<i>``."""
    if domain == "math":
        source, ability, content = "math_dapo", "MATH", bank_prompt
    elif domain == "code":
        source, ability, content = "code_stdio", "CODE", solve_prompt
    else:
        raise ValueError(f"domain must be 'math' or 'code', got {domain!r}")
    return [{"data_source": source,
             "prompt": [{"role": "user", "content": content(q)}],
             "ability": ability,
             "reward_model": dict(REWARD_MODEL),
             "extra_info": {"index": f"{row_id_prefix}_{i}"}}
            for i, q in enumerate(questions)]


def build(pool: Path, out: Path, n: int, domain: str, row_id_prefix: str,
          allow_short: bool = False) -> int:
    """Write the first ``n`` questions of ``pool`` to ``out``; returns the row count."""
    import pandas as pd

    from ask2distill.common.io import write_parquet
    qs = [r["problem"] for r in iter_jsonl(pool)]
    if len(qs) < n:
        msg = f"[build] only {len(qs)} questions in {pool}, wanted {n}"
        if not allow_short:
            raise ValueError(msg + " (set bank.allow_short only for a documented short yield)")
        print("WARNING " + msg, flush=True)
    rows = bank_rows(qs[:n], domain, row_id_prefix)
    # pandas' default index handling (index=None), as the original to_parquet call
    write_parquet(Path(out), pd.DataFrame(rows), index=None)  # type: ignore[arg-type]
    print(f"BUILD_DONE {out} rows={len(rows)} steps={len(rows) // 64}", flush=True)
    return len(rows)


def report_steps(cfg: DictConfig, out: Path) -> None:
    """Print the OPD step count after verl's prompt-length filter, if the student is local."""
    student = Path(str(cfg.opd.student))
    if not (student / "tokenizer_config.json").is_file():
        print(f"[bank] step count skipped: no tokenizer at {student}", flush=True)
        return
    from ask2distill.bank.arm_steps import count_steps
    r = count_steps(out, student, int(cfg.opd.max_prompt_length), int(cfg.opd.mini_batch_size))
    print(f"[bank] {r['kept']}/{r['rows']} prompts fit max_prompt_length="
          f"{cfg.opd.max_prompt_length} -> {r['steps']} steps per epoch", flush=True)


def main(cfg: DictConfig, args: argparse.Namespace | None = None) -> int:
    """``a2d bank --pair P`` (a pair with ``bank.refilter`` re-filters its pool first)."""
    a2d_config.require(cfg, "bank")
    b = cfg.bank
    domain = str(cfg.domain)
    if str(b.template) != TEMPLATES.get(domain):
        raise a2d_config.ConfigError(f"bank.template={b.template} is not the {domain} template "
                                     f"{TEMPLATES.get(domain)!r}")
    allow_short = bool(b.allow_short) or bool(getattr(args, "allow_short", False))
    refilter = b.get("refilter")
    if getattr(args, "dry_run", False):
        if refilter is not None:
            print(f"[bank] would re-filter {refilter.input} -> {refilter.output}", flush=True)
        print(f"[bank] would build {b.name} (n={b.n}, {b.template}) from {b.pool} -> {b.out}",
              flush=True)
        return 0
    if refilter is not None:
        from ask2distill.bank.refilter import refilter as run_refilter
        if str(refilter.kind) != "comp_strong_salvage_exact_dedup":
            raise a2d_config.ConfigError(f"unknown bank.refilter.kind {refilter.kind!r}")
        run_refilter(str(refilter.input), str(refilter.output))
    pool = Path(str(b.pool))
    if not pool.is_file():
        raise SystemExit(f"[bank] pool {pool} not found (run `a2d filter` / `a2d aggregate` first)")
    out = Path(str(b.out))
    os.makedirs(out.parent, exist_ok=True)
    rows = build(pool, out, int(b.n), domain, str(b.row_id_prefix), allow_short=allow_short)
    if b.expected_rows is not None and rows != int(b.expected_rows):
        print(f"[bank] note: {rows} rows, the final run's bank had {b.expected_rows}", flush=True)
    report_steps(cfg, out)
    return 0
