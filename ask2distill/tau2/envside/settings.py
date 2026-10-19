"""Settings for env-side processes: ``TAU2_*`` variables and the role settings blob.

``ask2distill.tau2.env.render`` emits every ``TAU2_*`` variable a role reads,
so the defaults kept in the ported modules (they equal the final-run values)
are never what decides a run. Paths have no defaults beyond what the fork's
own layout implies.

Roles that need structured settings (harvest, eval) also get
``A2D_TAU2_SETTINGS``: a JSON object with the resolved config section,
produced in the a2d env where the config can be parsed.
"""
from __future__ import annotations

import json
import os
from typing import Any

SETTINGS_ENV = "A2D_TAU2_SETTINGS"
DB_DIR_ENV = "A2D_TAU2_DB_DIR"


class SettingsError(RuntimeError):
    """A required variable or setting is missing."""


def require(name: str) -> str:
    """The value of environment variable ``name``; raise when unset or empty."""
    value = os.environ.get(name, "")
    if not value:
        raise SettingsError(f"{name} is not set (render it with ask2distill.tau2.env, "
                            "or start this module through `a2d tau2 ...`)")
    return value


def flag(name: str, default: str = "0") -> bool:
    """``True`` when ``$name == "1"`` (the convention of every tau2 switch)."""
    return os.environ.get(name, default) == "1"


def db_dir() -> str:
    """Directory holding ``<domain>/db.json`` (``$A2D_TAU2_DB_DIR``, else the fork's)."""
    value = os.environ.get(DB_DIR_ENV, "")
    if value:
        return value
    from ask2distill.tau2.envside import repo
    return os.path.join(repo.data_dir(), "domains")


def role_settings() -> dict[str, Any]:
    """The decoded ``A2D_TAU2_SETTINGS`` object."""
    raw = require(SETTINGS_ENV)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise SettingsError(f"{SETTINGS_ENV} is not valid JSON: {e}") from None
    if not isinstance(value, dict):
        raise SettingsError(f"{SETTINGS_ENV} must be a JSON object")
    return value
