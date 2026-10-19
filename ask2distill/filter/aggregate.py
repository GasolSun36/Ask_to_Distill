"""Stage 3c: merge the per-slice filter outputs into one pool with a cross-slice dedup.

Each slice deduped only within itself, so near-duplicates can still sit across
slice boundaries. The cross-slice pass runs on embeddings alone: cosine dedup
at ``filter.agg_thr`` (0.95) is strictly stronger than the Jaccard stage it
replaces -- it catches paraphrase, which token overlap misses -- and it is GPU
work, so it does not reproduce the O(n*kept) CPU cost that made the
single-job version time out.

Slice order is preserved, and the bank takes the FIRST N questions of the pool
in that order, so the order decides which questions an arm trains on. It is
``filter.agg_order``, kept as run:

* Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B, Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B,
  Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507, Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct:
  numeric (for at most 10 slices lexicographic and numeric agree);
* Code_Qwen3-4B_Qwen3-4B-Instruct-2507: lexicographic, s0, s10, s11, s1, ..., s9 -- the final run's
  aggregation listed the slice files with ``ls`` (its log prints that order),
  and its 10,000-row bank came from s0 / s10 / s11 (5,138 / 4,252 / 610 rows);
* Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500 has no cross-slice step (``filter.aggregate: false``): its single
  80k-question saturation pass is the pool.

Output ``filter.pool``. The bank is its own stage (``a2d bank``).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import DictConfig

from ask2distill import config as a2d_config
from ask2distill.common.embed import SupportsEmbed, greedy_dedup
from ask2distill.common.io import iter_jsonl, write_jsonl


def aggregate(slice_files: list[Path], out_pool: Path, thr: float, embedder: SupportsEmbed) -> int:
    """Concatenate ``slice_files`` in the given order, dedup at ``thr``, write ``out_pool``.

    The caller orders the files (``ordered_slice_files``). A missing file is
    skipped with a warning, as the original did; returns the pool size.
    """
    qs: list[str] = []
    per: list[tuple[str, int]] = []
    for f in slice_files:
        if not os.path.exists(f):
            print(f"[agg] WARNING: missing {f}", flush=True)
            continue
        cur = [r["problem"] for r in iter_jsonl(f)]
        per.append((os.path.basename(f), len(cur)))
        qs += cur
    for name, k in per:
        print(f"[agg] {name}: {k}", flush=True)
    print(f"[agg] merged {len(qs)} questions, cross-slice embedding dedup @{thr}", flush=True)
    kept = [qs[i] for i in greedy_dedup(qs, thr, embedder)]
    print(f"[agg] {len(kept)} after cross-slice dedup (removed {len(qs) - len(kept)})", flush=True)
    write_jsonl(out_pool, ({"problem": q} for q in kept))
    print(f"[agg] -> {out_pool}", flush=True)
    return len(kept)


def ordered_slice_files(cfg: DictConfig) -> list[Path]:
    """Slice question files in ``filter.agg_order``."""
    from ask2distill.filter.saturation import questions_path, slice_output
    order = [int(k) for k in (cfg.filter.agg_order or [])]
    if sorted(order) != list(range(len(cfg.filter.slices))):
        raise a2d_config.ConfigError(f"filter.agg_order {order} is not a permutation of the "
                                     f"{len(cfg.filter.slices)} slices")
    return [questions_path(slice_output(cfg, k)) for k in order]


def main(cfg: DictConfig, args: argparse.Namespace | None = None) -> int:
    """``a2d aggregate --pair P`` (a no-op for pairs without a cross-slice step)."""
    a2d_config.require(cfg, "filter")
    f = cfg.filter
    if not f.aggregate:
        print(f"[agg] {cfg.pair} has no cross-slice step (filter.aggregate=false); "
              f"bank.pool reads the slice output directly", flush=True)
        return 0
    files = ordered_slice_files(cfg)
    missing = [str(p) for p in files if not p.is_file()]
    out = Path(str(f.pool))
    if getattr(args, "dry_run", False):
        print(f"[agg] would merge {[p.name for p in files]} @{f.agg_thr} -> {out}", flush=True)
        if missing:
            print(f"[agg] not ready: missing slice output(s) {missing}", flush=True)
        return 0
    if missing:
        raise SystemExit(f"[agg] filter incomplete, missing slice output(s): {missing}")
    from ask2distill.common.embed import Embedder
    from ask2distill.filter.saturation import check_embed_bs
    from ask2distill.harvest.sampling import export_stage_env
    check_embed_bs(cfg)
    export_stage_env("filter")
    embedder = Embedder(str(cfg.models.embedder), device="cuda")
    aggregate(files, out, float(f.agg_thr), embedder)
    return 0
