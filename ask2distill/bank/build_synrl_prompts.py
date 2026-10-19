"""Build the synthesizer-RL prompt parquet (``synrl.prompts``).

The rows are the instructions S_θ is optimised on; there is no ground truth -- the
reward worker supplies the whole signal. A run consumes
``synrl.total_steps x synrl.train_batch_size`` rows in order (480 for 60 steps
of 8 prompts, ``data.shuffle=False``); verl repeats every row
``synrl.rollout_n`` times, so the 8 rollouts of a GRPO group share one prompt
and therefore one topic.

Rows: ``synrl.prompt_rows`` prompts, topics assigned by the seeded rotation of
``common.topics.rotation`` over ``synrl.topics`` with
``random.Random(synrl.prompt_seed)``, rendered by
``prompts_math.synthesizer_messages`` / ``prompts_code.synthesizer_messages``;
``data_source "synrl"``, ``ability "MATH"`` / ``"CODE"``,
``reward_model {"ground_truth": "N/A", "style": "rule-lighteval/MATH_v2"}``,
``extra_info {"index": "synrl_<domain>_<i>", "topic": <topic>}``. The same
config always rebuilds the same file, so the parquet is not part of the data
release: ``a2d prompts build --pair P`` writes it to ``synrl.prompts``
(by default ``$A2D_OUT/<pair>/prompts/SYNRL_TOPICS_<DOMAIN>.parquet``), and
``a2d synrl`` builds it the same way when the file is absent.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

from omegaconf import DictConfig

from ask2distill import config as a2d_config
from ask2distill.bank.build_bank import REWARD_MODEL
from ask2distill.common import prompts_code, prompts_math
from ask2distill.common.topics import rotation

ABILITY = {"code": "CODE", "math": "MATH"}
MESSAGES = {"code": prompts_code.synthesizer_messages, "math": prompts_math.synthesizer_messages}


def prompt_rows(domain: str, topics: Sequence[str], n: int, seed: int) -> list[dict]:
    """``n`` prompt rows; topic ``i`` is ``rotation(topics, n, random.Random(seed))[i]``."""
    if domain not in ABILITY:
        raise ValueError(f"domain must be 'math' or 'code', got {domain!r}")
    rows = []
    for i, topic in enumerate(rotation(topics, n, random.Random(seed))):
        rows.append({"data_source": "synrl",
                     "prompt": MESSAGES[domain](topic),
                     "ability": ABILITY[domain],
                     "reward_model": dict(REWARD_MODEL),
                     "extra_info": {"index": f"synrl_{domain}_{i}", "topic": topic}})
    return rows


def output_path(cfg: DictConfig) -> Path:
    """``synrl.prompts`` (the file ``a2d synrl`` trains on)."""
    return Path(str(cfg.synrl.prompts))


def check(cfg: DictConfig) -> tuple[int, int, list[str]]:
    """``(rows, seed, topics)`` of ``cfg.synrl``; refuses a parquet too short for the run."""
    from ask2distill.common.topics import load_topics
    if str(cfg.domain) not in ABILITY:
        raise SystemExit(f"[prompts] {cfg.pair}: tau2 prompts are built by `a2d tau2 build-prompts`")
    g = cfg.synrl
    n, seed = int(g.prompt_rows), int(g.prompt_seed)
    need = int(g.total_steps) * int(g.train_batch_size)
    if n < need:
        raise a2d_config.ConfigError(f"synrl.prompt_rows={n} < total_steps x train_batch_size = "
                                     f"{need}: the run would run out of prompts")
    return n, seed, load_topics(str(g.topics))


def build(cfg: DictConfig) -> Path:
    """Write the prompt parquet of ``cfg`` to ``synrl.prompts``."""
    import pandas as pd

    from ask2distill.common.io import write_parquet
    n, seed, topics = check(cfg)
    out = output_path(cfg)
    rows = prompt_rows(str(cfg.domain), topics, n, seed)
    write_parquet(out, pd.DataFrame(rows), index=None)  # type: ignore[arg-type]
    print(f"PROMPTS_DONE {out} rows={len(rows)} topics={len(topics)}", flush=True)
    return out


def main(cfg: DictConfig, args: argparse.Namespace | None = None) -> int:
    """``a2d prompts build --pair P``."""
    a2d_config.require(cfg, "synrl")
    n, seed, topics = check(cfg)
    if getattr(args, "dry_run", False):
        print(f"[prompts] would build {n} {cfg.domain} synthesizer prompts over {len(topics)} topics "
              f"of {cfg.synrl.topics} (seed {seed}) -> {output_path(cfg)}", flush=True)
        return 0
    build(cfg)
    return 0
