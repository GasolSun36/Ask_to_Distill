"""Locate the ToolOrchestra tau2 fork and put it on ``sys.path``.

The fork is not pip-installed; it is imported from ``$REPO_PATH`` and
``$REPO_PATH/evaluation/tau2-bench``. ``REPO_PATH`` is required, not defaulted:
the fork's own ``tau2/utils/utils.py`` joins it into ``DATA_DIR`` at import
and raises ``TypeError`` when it is unset.
"""
from __future__ import annotations

import os
import sys


class Tau2RepoError(RuntimeError):
    """``REPO_PATH`` is unset or does not point at the vendored fork."""


def repo_path() -> str:
    """``$REPO_PATH``, or a clear error."""
    repo = os.environ.get("REPO_PATH", "").strip()
    if not repo:
        raise Tau2RepoError("REPO_PATH is not set; run envside modules through "
                            "ask2distill.tau2.spawn (it points REPO_PATH at "
                            "third_party/toolorchestra_tau2)")
    return repo


def setup() -> str:
    """Prepend the fork to ``sys.path`` (idempotent) and return ``REPO_PATH``."""
    repo = repo_path()
    for p in (f"{repo}/evaluation/tau2-bench", repo):
        if p not in sys.path:
            sys.path.insert(0, p)
    return repo


def data_dir() -> str:
    """``$REPO_PATH/data/tau2``: domains, user-simulator guidelines."""
    return os.path.join(repo_path(), "data", "tau2")
