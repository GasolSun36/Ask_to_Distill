"""Stage 3a: sample the chosen synthesizer into raw question shards.

``a2d harvest --pair P [--shard i]`` writes ``harvest.out_dir/raw.s<i>.jsonl``
for one shard or for every shard that is not on disk yet (a non-empty file is
skipped, so a resubmitted job only redoes what is missing). Each shard is
``harvest.per_shard`` prompts sampled in one vLLM call, under the conditioning
of ``harvest.sampling`` (rotated topics from ``synrl.topics``,
``enable_thinking=False``, top_p 0.95, engine seed 0).

The per-shard seed of the topic rotation, ``seed_for``:

* ``index``           ``i`` -- the math pairs (shards seeded 0..15);
* ``7919*i+104729``   the code pairs.

The seed MUST be a function of the shard -- the math campaign once lost eight
parallel shards to byte-identical output because the salt left the shard id
out -- and must NOT be a timestamp, or a resumed shard replays nothing.

Output cleaning: R1-Distill bases can still emit a thinking block even with the
patched no-think template, so each completion goes through
``strip_thinking(...).strip()`` (an unclosed block counts as no question) and
only non-empty questions are written. Nothing is dropped on finish reason: a
truncated question is caught later by the quality filters (math
``bad_ending``, code ``truncated_tail``). This is deliberately NOT the sweep's
writer, which keeps the raw text.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable, Mapping

from omegaconf import DictConfig

from ask2distill import config as a2d_config
from ask2distill.common.io import count_lines, write_jsonl
from ask2distill.common.text import strip_question
from ask2distill.harvest import sampling

SEED_FORMULAS = {
    "index": lambda i: i,
    "7919*i+104729": lambda i: 7919 * i + 104729,
}


def _harvest_node(cfg: DictConfig | Mapping) -> Mapping:
    return cfg["harvest"] if "harvest" in cfg else cfg


def seed_for(cfg: DictConfig | Mapping, i: int) -> int:
    """Topic-rotation RNG seed of shard ``i`` (``harvest.seed_formula``).

    Accepts the full config or its ``harvest`` section.
    """
    formula = str(_harvest_node(cfg)["seed_formula"])
    if formula not in SEED_FORMULAS:
        raise a2d_config.ConfigError(f"harvest.seed_formula must be one of "
                                     f"{sorted(SEED_FORMULAS)}, got {formula!r}")
    return SEED_FORMULAS[formula](int(i))


def shard_path(cfg: DictConfig, i: int) -> Path:
    """``harvest.out_dir/raw.s<i>.jsonl``."""
    return Path(str(cfg.harvest.out_dir)) / f"raw.s{i}.jsonl"


def harvest_records(texts: Iterable[str], topics: Iterable[str] | None = None) -> list[dict]:
    """``{"problem": q, "topic": topic}`` for every completion that still has a
    question after ``strip_thinking(...).strip()`` (``topic`` omitted when
    ``topics`` is None)."""
    texts = list(texts)
    out = []
    for text, topic in zip(texts, topics if topics is not None else [None] * len(texts)):
        q = strip_question(text)
        if q:
            out.append({"problem": q} if topic is None else {"problem": q, "topic": topic})
    return out


def sample_questions(cfg: DictConfig, model: str | os.PathLike, n: int, seed: int,
                     out: str | os.PathLike, *, topics: list[str] | None = None) -> Path:
    """Sample ``n`` questions from ``model`` with RNG ``seed`` into ``out``.

    Sampling settings come from ``cfg.harvest``; ``topics`` defaults to the
    list in ``synrl.topics``.
    """
    h = cfg.harvest
    if h.strip_thinking is not True:
        raise a2d_config.ConfigError("harvest.strip_thinking must be true: every final bank was "
                                     "harvested from thinking-stripped completions")
    domain = str(cfg.domain)
    if topics is None:
        topics = sampling.load_topics(cfg)
    tok = sampling.load_tokenizer(model)
    prompts, drawn = sampling.build_prompts(tok, n, random.Random(seed), domain, topics)
    texts = sampling.generate(model, prompts, max_new=int(h.max_new), prompt_head=int(h.prompt_head),
                              temperature=float(h.temperature), top_p=float(h.top_p),
                              gpu_memory_utilization=float(h.gpu_memory_utilization),
                              seed=int(h.vllm_seed))
    out = Path(out)
    write_jsonl(out, harvest_records(texts, drawn))
    print(f"HARVEST_SHARD_DONE {out} n={count_lines(out)}", flush=True)
    return out


def check_synthesizer(model: str | os.PathLike) -> Path:
    """Fail early, naming the fix, when the merged synthesizer is not on disk."""
    model = Path(model)
    if not (model / "config.json").is_file():
        raise FileNotFoundError(
            f"synthesizer {model} is not a merged HF directory. Run `a2d select sweep` (it merges "
            "every swept checkpoint), or point harvest.synthesizer at a released synthesizer. The "
            "code synthesizers' weights were not kept: re-harvesting code needs a retrained synthesizer.")
    return model


def run_shard(cfg: DictConfig, i: int, out: Path | None = None) -> Path:
    """Harvest shard ``i`` of ``cfg.harvest.synthesizer`` into ``out`` (default ``raw.s<i>.jsonl``)."""
    a2d_config.require(cfg, "harvest")
    model = check_synthesizer(str(cfg.harvest.synthesizer))
    out = shard_path(cfg, i) if out is None else Path(out)
    seed = seed_for(cfg, i)
    print(f"[harvest] shard {i}: {cfg.harvest.per_shard} from {model} seed={seed} -> {out}",
          flush=True)
    return sample_questions(cfg, model, int(cfg.harvest.per_shard), seed, out)


def main(cfg: DictConfig, args: argparse.Namespace | None = None) -> int:
    """``a2d harvest --pair P [--shard i]``."""
    a2d_config.require(cfg, "harvest")
    n = int(cfg.harvest.shards)
    shard = getattr(args, "shard", None)
    todo = [shard] if shard is not None else list(range(n))
    if any(not 0 <= i < n for i in todo):
        raise SystemExit(f"--shard must be in [0, {n}) for {cfg.pair}")
    pending = [i for i in todo if not shard_path(cfg, i).is_file()
               or os.path.getsize(shard_path(cfg, i)) == 0]
    for i in sorted(set(todo) - set(pending)):
        print(f"[harvest] shard {i} exists, skip", flush=True)
    if getattr(args, "dry_run", False):
        for i in pending:
            print(f"[harvest] would sample shard {i}: n={cfg.harvest.per_shard} seed={seed_for(cfg, i)} "
                  f"model={cfg.harvest.synthesizer} -> {shard_path(cfg, i)}", flush=True)
        return 0
    if not pending:
        return 0
    sampling.export_stage_env("harvest")
    topics = sampling.load_topics(cfg)
    model = check_synthesizer(str(cfg.harvest.synthesizer))
    for i in pending:
        seed = seed_for(cfg, i)
        print(f"[harvest] shard {i}: {cfg.harvest.per_shard} from {model} seed={seed}", flush=True)
        sample_questions(cfg, model, int(cfg.harvest.per_shard), seed, shard_path(cfg, i),
                         topics=topics)
    done = sum(shard_path(cfg, i).is_file() for i in range(n))
    print(f"HARVEST_DONE {done}/{n} shards on disk", flush=True)
    return 0
