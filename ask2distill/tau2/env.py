"""Render the complete ``TAU2_*`` environment of each tau2 role from the config.

The env-side modules (and the verl-side reward worker and agent loop) read
their knobs from ``TAU2_*`` variables, several of them at import time
(``reward_base.py``, ``reward_v6.py``, ``agent_loop.py``). The originals relied
on a mix of exported values and module defaults; here every variable a role
reads is rendered explicitly from the config, so no module default ever
decides a run.

========  ==================================================================
role      variables
========  ==================================================================
synrl     ``reward_env`` (gates, turn caps, judge, diversity, served names),
          ``TAU2_REWARD_V`` of the version, and the reward worker's
          ``TAU2_DUMP`` / ``TAU2_GEN_TOKENIZER`` / ``TAU2_EMB_MODEL`` /
          ``TAU2_DIV_GROUP``; the service URL is added by the launcher
harvest   the scoring part of ``reward_env`` overlaid with ``harvest.env``
          (grounding, teacher gate, ``TAU2_KT``), as expand_v{6,7}_int.sh set it
opd       ``opd.agent_env`` (turn cap, stop-on-repeat, tool-output cap, user
          retry) and the user-simulator name; URLs are added by the launcher
eval      ``TAU2_MAX_TOKENS``, ``TAU2_ORCHESTRATOR_PROMPT``, ``TAU2_OPENAI_DIRECT``
========  ==================================================================

Every role also gets ``TAU2_CTX`` (``domain_ctx.json``), ``TAU2_USER_GUIDELINES``
and the DB directory under ``tau2_data.db_dir_env`` (``A2D_TAU2_DB_DIR``; the
original name ``TAU2_DATA`` collided with the training job's prompt-parquet
variable and is never set).

Roles whose env-side process needs structured settings (harvest, eval) also
get ``A2D_TAU2_SETTINGS``, the resolved config section as JSON (see
``role_settings``).
"""
from __future__ import annotations

import json
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf

Role = Literal["synrl", "harvest", "opd", "eval"]
ROLES = ("synrl", "harvest", "opd", "eval")

# reward_env keys the card scorers read (reward_base / reward_v5..v7); the rest
# of reward_env belongs to the synthesizer-RL worker and the service endpoints.
SCORING_KEYS = ("TAU2_KT", "TAU2_GROUNDING", "TAU2_TEACHER_GATE", "TAU2_MAX_TURNS",
                "TAU2_GATE_TURNS", "TAU2_PREF_TURNS", "TAU2_V7_TURNS",
                "TAU2_PREF_TEACHER_GATE", "TAU2_PREF_JUDGE")
SETTINGS_ENV = "A2D_TAU2_SETTINGS"


def _s(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _container(cfg: DictConfig, key: str) -> Any:
    node = OmegaConf.select(cfg, key, throw_on_missing=True)
    if node is None:
        raise KeyError(f"config has no {key!r}")
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
    return node


def version_config(cfg: DictConfig, version: str) -> dict:
    """``synrl.versions.<version>``, resolved."""
    versions = _container(cfg, "synrl.versions")
    if version not in versions:
        raise KeyError(f"synrl.versions has no {version!r} (has {sorted(versions)})")
    return versions[version]


def common(cfg: DictConfig) -> dict[str, str]:
    """Paths every role needs."""
    td = cfg.tau2_data
    return {"TAU2_CTX": str(td.domain_ctx),
            "TAU2_USER_GUIDELINES": str(td.user_guidelines),
            str(td.db_dir_env): str(td.db_dir)}


def render(cfg: DictConfig, role: Role, version: str | None = None) -> dict[str, str]:
    """The ``TAU2_*`` environment of ``role`` (``version`` selects the synthesizer-RL version)."""
    if role not in ROLES:
        raise ValueError(f"unknown tau2 role {role!r}; expected one of {ROLES}")
    env = common(cfg)
    reward_env = {k: _s(v) for k, v in _container(cfg, "reward_env").items()}
    if role == "synrl":
        env.update(reward_env)
        g = cfg.synrl
        env.update({"TAU2_GEN_TOKENIZER": str(g.gen_tokenizer),
                    "TAU2_EMB_MODEL": str(cfg.models.embedder),
                    "TAU2_DIV_GROUP": _s(g.diversity.group)})
        if version is not None:
            v = version_config(cfg, version)
            env["TAU2_REWARD_V"] = _s(v["reward_v"])
            env["TAU2_DUMP"] = str(v["dump"])
    elif role == "harvest":
        env.update({k: reward_env[k] for k in SCORING_KEYS})
        env.update({k: _s(v) for k, v in _container(cfg, "harvest.env").items()})
    elif role == "opd":
        env.update({k: _s(v) for k, v in _container(cfg, "opd.agent_env").items()})
        env["TAU2_USER_MODEL"] = reward_env["TAU2_USER_MODEL"]
    else:  # eval
        ev = cfg.eval
        env.update({"TAU2_MAX_TOKENS": _s(ev.TAU2_MAX_TOKENS),
                    "TAU2_ORCHESTRATOR_PROMPT": _s(ev.TAU2_ORCHESTRATOR_PROMPT),
                    "TAU2_OPENAI_DIRECT": _s(ev.TAU2_OPENAI_DIRECT)})
    return env


def role_settings(cfg: DictConfig, role: Role) -> dict[str, Any] | None:
    """Structured settings for the env-side process of ``role`` (``None`` if it needs none)."""
    if role == "harvest":
        return {"harvest": _container(cfg, "harvest")}
    if role == "eval":
        return {"eval": _container(cfg, "eval")}
    return None


def settings_env(cfg: DictConfig, role: Role) -> dict[str, str]:
    """``{A2D_TAU2_SETTINGS: <json>}`` for roles that have settings, else ``{}``."""
    s = role_settings(cfg, role)
    return {} if s is None else {SETTINGS_ENV: json.dumps(s, ensure_ascii=False, sort_keys=True)}
