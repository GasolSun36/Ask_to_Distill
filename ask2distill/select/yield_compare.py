"""Stage 2c (optional): saturation yield of candidate synthesizer steps.

Compares how many semantically independent questions each candidate synthesizer
actually yields: per candidate, harvest
``select.yield.n`` (5000) questions with topic seed ``select.yield.seed``
(100) under the harvest sampling settings, run ONE saturation pass over them
(format + quality -> Jaccard 0.7 -> embedding dedup ``filter.sat_thr``), and
compare the final ``emb`` count. The highest count wins; ``tie: earliest``
takes the smallest step among equals. The candidates are given per pair in
``select.yield.candidates``; ``select.yield: null`` skips this step.

Outputs: ``$A2D_OUT/<pair>/yield/raw_step<N>.jsonl``, ``sat_step<N>.json`` (+
``_questions.jsonl``) and the summary ``$A2D_OUT/<pair>/yield.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import DictConfig

from ask2distill import config as a2d_config
from ask2distill.common.io import atomic_write_text


def work_dir(cfg: DictConfig) -> Path:
    """``$A2D_OUT/<pair>/yield``."""
    return Path(str(cfg.paths.out)) / str(cfg.pair) / "yield"


def summary_path(cfg: DictConfig) -> Path:
    """``$A2D_OUT/<pair>/yield.json``."""
    return Path(str(cfg.paths.out)) / str(cfg.pair) / "yield.json"


def choose(emb: dict[int, int], tie: str) -> int:
    """The candidate with the largest yield; ``tie="earliest"`` breaks ties by smallest step."""
    if tie != "earliest":
        raise a2d_config.ConfigError(f"select.yield.tie must be 'earliest', got {tie!r}")
    best = max(emb.values())
    return min(st for st, n in emb.items() if n == best)


def main(cfg: DictConfig, args: argparse.Namespace | None = None) -> int:
    """``a2d select yield --pair P``."""
    y = cfg.select.get("yield")
    if y is None:
        raise SystemExit(f"[yield] {cfg.pair} has no yield comparison (select.yield is null); "
                         "its synthesizer step is the pick")
    a2d_config.require(cfg, "select.yield")
    a2d_config.require(cfg, "harvest")
    a2d_config.require(cfg, "filter")
    if str(cfg.domain) != "math":
        raise SystemExit("[yield] the yield comparison was only used for math pairs")
    candidates = [int(c) for c in y.candidates]
    if not candidates:
        raise a2d_config.ConfigError("select.yield.candidates is empty; pass the steps to compare")
    wd = work_dir(cfg)
    raw = {c: wd / f"raw_step{c}.jsonl" for c in candidates}
    sat = {c: wd / f"sat_step{c}.json" for c in candidates}
    if getattr(args, "dry_run", False):
        for c in candidates:
            print(f"[yield] step {c}: harvest {y.n} at seed {y.seed} -> {raw[c]}; "
                  f"saturation @{cfg.filter.sat_thr} -> {sat[c]}", flush=True)
        return 0

    from ask2distill.harvest import harvest, sampling
    from ask2distill.select.sweep import merged_dir
    sampling.export_stage_env("select")
    wd.mkdir(parents=True, exist_ok=True)
    topics = sampling.load_topics(cfg)
    for c in candidates:
        if raw[c].is_file() and raw[c].stat().st_size > 0:
            print(f"[yield] step {c}: {raw[c]} exists, skip harvest", flush=True)
            continue
        model = harvest.check_synthesizer(merged_dir(cfg, c))
        harvest.sample_questions(cfg, model, int(y.n), int(y.seed), raw[c],
                                 topics=topics)

    from ask2distill.common.embed import Embedder
    from ask2distill.filter.saturation import questions_path, run_slice
    embedder = None
    emb: dict[int, int] = {}
    for c in candidates:
        if sat[c].is_file() and questions_path(sat[c]).is_file():
            emb[c] = int(json.loads(sat[c].read_text())["emb"])
            print(f"[yield] step {c}: {sat[c]} exists (emb {emb[c]})", flush=True)
            continue
        if embedder is None:
            embedder = Embedder(str(cfg.models.embedder), device="cuda")
        res = run_slice([raw[c]], sat[c], "math", float(cfg.filter.sat_thr), embedder,
                        jaccard_thr=float(cfg.filter.jaccard))
        emb[c] = int(res["emb"])

    step = choose(emb, str(y.tie))
    for c in candidates:
        print(f"[yield] step {c:>4}: {emb[c]:>6} independent questions"
              f"{'  <- chosen' if c == step else ''}", flush=True)
    if step != int(cfg.select.chosen_step):
        print(f"[yield] note: select.chosen_step={cfg.select.chosen_step} (frozen) differs "
              f"from this comparison ({step})", flush=True)
    record = {"candidates": candidates, "n": int(y.n), "seed": int(y.seed),
              "emb": {str(c): emb[c] for c in candidates}, "tie": str(y.tie), "step": step}
    atomic_write_text(summary_path(cfg), json.dumps(record, indent=1) + "\n")
    print(f"[yield] -> {summary_path(cfg)}", flush=True)
    return 0
