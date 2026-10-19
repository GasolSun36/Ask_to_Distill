"""``reward_config.json``: the only channel through which the reward worker is configured.

``synrl.launch`` writes the resolved ``reward`` config section to
``<run_dir>/reward_config.json`` and passes its absolute path to every Ray
worker as ``$A2D_REWARD_CONFIG`` (through ``ray_kwargs.ray_init.runtime_env``).
The worker reads nothing else: every key is required, so a code default that
differs from the configured one (for example a default length penalty, solve
budget or δ_ref) can never leak in through an unset variable.

This module has no verl or CUDA side effects, so the launcher can validate a
config before anything starts; the worker itself lives in ``reward_worker``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping

from ask2distill.synrl.solve import solve_template

REWARD_CONFIG_ENV = "A2D_REWARD_CONFIG"
# The math/code worker embeds in batches of this size (common.embed.EMBED_BS);
# kept as a literal so this module does not import torch.
EMBED_BS = 16
REQUIRED_KEYS = (
    "domain", "solver", "teacher", "embedder", "qwen3_nothink", "solve_template", "solve_system",
    "delta_ref", "topk", "chunk", "use_omega", "len_alpha", "len_pen", "max_new", "rolls",
    "temperature", "top_p", "sample_top_k", "trunc_zero", "gen_bs", "min_question_chars",
    "min_rollout_tokens", "fix_eos", "step_counter", "diversity", "dump",
)
DIVERSITY_KEYS = ("mode", "thresh", "pen", "group", "max_len", "embed_bs", "per_group_embed")
# bleu: sentence-BLEU clustering over the full scoring batch (paper §2.3);
# emb: embedding-cosine clustering per group of the rank batch.
DIVERSITY_MODES = ("bleu", "emb")


class RewardConfigError(RuntimeError):
    """``reward_config.json`` is missing, incomplete or asks for an unsupported mode."""


def check_reward_config(cfg: Mapping[str, Any]) -> None:
    """Raise ``RewardConfigError`` on a missing key or an unsupported setting."""
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    missing += [f"diversity.{k}" for k in DIVERSITY_KEYS if k not in (cfg.get("diversity") or {})]
    if missing:
        raise RewardConfigError("reward config is missing: " + ", ".join(missing))
    div = cfg["diversity"]
    problems = []
    if cfg["domain"] not in ("math", "code"):
        problems.append(f"domain={cfg['domain']!r} (math or code)")
    if div["mode"] not in DIVERSITY_MODES:
        problems.append(f"diversity.mode={div['mode']!r} (one of {DIVERSITY_MODES})")
    if not 0 < float(div["thresh"]) <= 1:
        problems.append(f"diversity.thresh={div['thresh']} (the clustering distance τ lies in (0, 1])")
    if float(div["pen"]) < 0:
        problems.append(f"diversity.pen={div['pen']} (λ_div must be >= 0)")
    if div["mode"] == "emb":
        if not div["per_group_embed"]:
            problems.append("diversity.per_group_embed=false (each group is embedded in its own call)")
        if int(div["embed_bs"]) != EMBED_BS:
            problems.append(f"diversity.embed_bs={div['embed_bs']} (the embedder runs at {EMBED_BS})")
    if cfg["step_counter"] != "local":
        problems.append(f"step_counter={cfg['step_counter']!r} (math/code dumps use the local counter)")
    if cfg["delta_ref"] is None:
        problems.append("delta_ref is not set (run `a2d calibrate` and set reward.delta_ref to its "
                        "delta_ref_rounded)")
    elif round(float(cfg["delta_ref"]), 4) != float(cfg["delta_ref"]):
        problems.append(f"delta_ref={cfg['delta_ref']} is not rounded to 4 decimals")
    try:
        solve_template(cfg["solve_template"])
    except ValueError as e:
        problems.append(str(e))
    if problems:
        raise RewardConfigError("unsupported reward config: " + "; ".join(problems))


def load_reward_config(path: str | os.PathLike | None = None) -> dict:
    """Read and check the reward config (``$A2D_REWARD_CONFIG`` when ``path`` is None)."""
    path = path or os.environ.get(REWARD_CONFIG_ENV)
    if not path:
        raise RewardConfigError(f"${REWARD_CONFIG_ENV} is not set; the reward worker reads its "
                                "configuration only from that file (written by `a2d synrl`)")
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    check_reward_config(cfg)
    return cfg
