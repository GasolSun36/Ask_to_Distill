"""Stage 3b: filter one slice of harvest shards down to semantically independent questions.

Raw count is meaningless here: dedup is against a growing pool, so the
marginal new-question rate falls as the pool grows. What matters is where the
curve bends -- that is the point past which more sampling stops paying -- so
the funnel is also recorded at fixed raw counts (``MARKS``).

Three stages, in the order the pipeline applies them, per question:

``format``   the domain's format gate (``math_filter.comp_filter`` /
             ``code_filter.comp_filter``) ...
``jaccard``  ... which also rejects token-set Jaccard > 0.7 against the slice's
             pool, then the quality filter (``strong_filter``); every question
             passing the gate adds its token set to the pool, and one passing
             both adds the salvaged copy too;
``embed``    greedy cosine dedup at ``thr`` (0.95) over all survivors, the only
             stage that catches paraphrase.

The reported ``fmt`` count is format + quality WITHOUT dedup (each question
gated against an empty pool), ``jac`` the survivors of format -> Jaccard ->
quality, ``emb`` the final count. The math filters reject well-formed
programming statements outright (verb list, ``$``/``{}`` balance, "must end in
?.$)]"), so the code domain uses ``code_filter``, calibrated on real
code_contests statements to the same bar (false positives under 5%).

Slices (``filter.slices``) decide the Jaccard scope, which is in-slice only
(a slice costs ~n^2 in the Jaccard pass -- 24k questions ran 1h26, 8k ~10 min;
the cross-slice embedding step catches what the smaller scope misses):

* Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500: one 80k pass;
* Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B: 4 slices x 4 shards;
* Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B: 2 x 4;
* Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507, Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct: 4 x 3;
* Code_Qwen3-4B_Qwen3-4B-Instruct-2507: 12 x 1.

Shards are read in the listed order.

Outputs ``filter.out_dir/sat.s<k>.json`` (``{"raw", "fmt", "jac", "emb",
"curve"}``) and ``sat.s<k>_questions.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Sequence

from omegaconf import DictConfig

from ask2distill import config as a2d_config
from ask2distill.common.embed import EMBED_BS, SupportsEmbed, greedy_dedup
from ask2distill.common.io import atomic_write_text, iter_jsonl, write_jsonl
from ask2distill.common.text import jaccard, norm_text
from ask2distill.filter import code_filter, math_filter

MARKS = [1000, 2000, 5000, 10000, 20000, 40000, 60000]

Gate = Callable[[str, list], tuple]


def _code_gate(jaccard_thr: float) -> Gate:
    def gate(raw: str, norms: list) -> tuple:
        t, why = code_filter.comp_filter(raw)
        if why:
            return None, why
        nt = norm_text(t)
        if any(jaccard(nt, x) > jaccard_thr for x in norms):
            return None, "dup_self"
        norms.append(nt)
        return t, None
    return gate


def domain_filters(domain: str, jaccard_thr: float = math_filter.JACCARD
                   ) -> tuple[Gate, Callable[[str], tuple]]:
    """``(gate(raw, norms), strong_filter)`` for ``domain``."""
    if domain == "math":
        return (lambda raw, norms: math_filter.comp_filter(raw, norms, jaccard_thr),
                math_filter.strong_filter)
    if domain == "code":
        return _code_gate(jaccard_thr), code_filter.strong_filter
    raise ValueError(f"domain must be 'math' or 'code', got {domain!r}")


def questions_path(out_json: str | os.PathLike) -> Path:
    """``sat.s0.json`` -> ``sat.s0_questions.jsonl`` (same directory)."""
    out_json = Path(out_json)
    return out_json.with_name(out_json.name.replace(".json", "_questions.jsonl"))


def read_raw(files: Sequence[str | os.PathLike]) -> list[str]:
    """Every ``problem`` of ``files`` in order; a missing file is skipped with a warning."""
    raw: list[str] = []
    for f in files:
        if not os.path.exists(f):
            print(f"[sat] WARNING: {f} does not exist, skipped", flush=True)
            continue
        raw += [r["problem"] for r in iter_jsonl(f)]
    return raw


def funnel(raw: Sequence[str], domain: str, jaccard_thr: float = math_filter.JACCARD
           ) -> tuple[list[str], int, list[tuple[int, int, int]]]:
    """Format -> Jaccard -> quality over ``raw``: ``(survivors, fmt count, curve)``."""
    gate, strong = domain_filters(domain, jaccard_thr)
    fmt = 0
    norms: list = []
    jac: list[str] = []
    curve: list[tuple[int, int, int]] = []
    for i, q in enumerate(raw, 1):
        t, why = gate(q, norms)
        if not why:
            c, _ = strong(t)
            if c is not None:
                jac.append(c)
                norms.append(norm_text(c))
        t2, why2 = gate(q, [])
        if not why2 and strong(t2)[0] is not None:
            fmt += 1
        if i in MARKS:
            curve.append((i, fmt, len(jac)))
            print(f"  raw {i:>6}  format {fmt:>6}  jaccard {len(jac):>6}", flush=True)
    return jac, fmt, curve


def run_slice(files: list[Path], out_json: Path, domain: str, thr: float, embedder: SupportsEmbed,
              *, jaccard_thr: float = math_filter.JACCARD) -> dict:
    """Filter the shards ``files`` (in order) and write ``out_json`` + its question file."""
    raw = read_raw(files)
    print(f"[sat] raw {len(raw)}", flush=True)
    jac, fmt, curve = funnel(raw, domain, jaccard_thr)
    print(f"[sat] embedding dedup @{thr} ...", flush=True)
    final = [jac[i] for i in greedy_dedup(jac, thr, embedder)]
    for lab, n in (("raw", len(raw)), ("format", fmt), (f"jaccard>{jaccard_thr}", len(jac)),
                   (f"embed@{thr}", len(final))):
        print(f"{lab:<14}{n:>8}{100 * n / max(len(raw), 1):8.1f}%", flush=True)
    record = {"raw": len(raw), "fmt": fmt, "jac": len(jac), "emb": len(final),
              "curve": [list(c) for c in curve]}
    # questions first: a slice counts as done once its question file exists
    write_jsonl(questions_path(out_json), ({"problem": q} for q in final))
    atomic_write_text(out_json, json.dumps(record, indent=1))
    print("SAT_DONE", flush=True)
    return record


def check_embed_bs(cfg: DictConfig) -> None:
    """bf16 embeddings depend on the batch shapes near the 0.95 threshold; only 16 is faithful."""
    if int(cfg.filter.embed_bs) != EMBED_BS:
        raise a2d_config.ConfigError(f"filter.embed_bs={cfg.filter.embed_bs}: the final pools were "
                                     f"deduped with batches of {EMBED_BS}")


def slice_files(cfg: DictConfig, k: int) -> list[Path]:
    """Raw shard files of slice ``k`` in listed order: ``harvest.out_dir/raw.s<i>.jsonl``."""
    from ask2distill.harvest.harvest import shard_path
    return [shard_path(cfg, int(i)) for i in cfg.filter.slices[k]]


def slice_output(cfg: DictConfig, k: int) -> Path:
    """``filter.out_dir/sat.s<k>.json``."""
    return Path(str(cfg.filter.out_dir)) / f"sat.s{k}.json"


def main(cfg: DictConfig, args: argparse.Namespace | None = None) -> int:
    """``a2d filter --pair P [--slice k]``."""
    a2d_config.require(cfg, "filter")
    f = cfg.filter
    if str(f.domain) != str(cfg.domain):
        raise a2d_config.ConfigError(f"filter.domain={f.domain} but the pair is {cfg.domain}")
    check_embed_bs(cfg)
    n = len(f.slices)
    k_arg = getattr(args, "slice", None)
    todo = [k_arg] if k_arg is not None else list(range(n))
    if any(not 0 <= k < n for k in todo):
        raise SystemExit(f"--slice must be in [0, {n}) for {cfg.pair}")
    pending = []
    missing: dict[int, list[str]] = {}
    for k in todo:
        q = questions_path(slice_output(cfg, k))
        if q.is_file() and q.stat().st_size > 0 and slice_output(cfg, k).is_file():
            print(f"[filter {k}] exists, skip", flush=True)
            continue
        absent = [str(p) for p in slice_files(cfg, k) if not p.is_file()]
        if absent:
            missing[k] = absent
        pending.append(k)
    if getattr(args, "dry_run", False):
        for k in pending:
            print(f"[filter {k}] would filter {[str(p) for p in slice_files(cfg, k)]} "
                  f"({f.domain}, jaccard {f.jaccard}, embed @{f.sat_thr}) -> {slice_output(cfg, k)}",
                  flush=True)
            if k in missing:
                print(f"[filter {k}] not ready: missing harvest shard(s) {missing[k]}", flush=True)
        return 0
    if missing:
        raise SystemExit(f"[filter] missing harvest shard(s): {missing}")
    if not pending:
        return 0
    from ask2distill.common.embed import Embedder
    from ask2distill.harvest.sampling import export_stage_env
    export_stage_env("filter")
    embedder = Embedder(str(cfg.models.embedder), device="cuda")
    for k in pending:
        print(f"[filter {k}] {len(f.slices[k])} shard(s)", flush=True)
        run_slice(slice_files(cfg, k), slice_output(cfg, k), str(f.domain), float(f.sat_thr),
                  embedder, jaccard_thr=float(f.jaccard))
    return 0
