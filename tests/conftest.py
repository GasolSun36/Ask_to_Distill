"""Shared pytest fixtures.

=================  ===========================================================
``models_dir``     ``$A2D_MODELS`` (``a2d models download``); tests that read
                   a tokenizer or model config skip when a model is absent
``tau2_python``    the tau2 interpreter, ``$A2D_TAU2_PYTHON``; skipped if unset
``tracked_files``  repo-relative paths git tracks, or would track (see below)
=================  ===========================================================

Also importable (``from conftest import ...``): the pair names.

Markers (declared in ``pyproject.toml``): ``gpu``, ``equiv``, ``smoke``,
``hydra``, ``tau2env``. The CPU unit suite is ``pytest -m "not gpu and not smoke"``.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO = Path(os.path.abspath(__file__)).parents[1]

# ------------------------------------------------------------------------ pair names
# Domain_Student_Teacher; each is configs/pairs/<name>.yaml and the HF repo gasolsun/<name>.
MATH_QWEN3_1P7B = "Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500"
MATH_R1_1P5B = "Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B"
MATH_R1_7B = "Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B"
CODE_QWEN3_30B = "Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507"
CODE_CODER_32B = "Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct"
CODE_QWEN3_4B = "Code_Qwen3-4B_Qwen3-4B-Instruct-2507"
AGENT_QWEN3_4B = "Agent_Qwen3-4B_Qwen3-4B-Thinking-2507"
MATH_PAIRS = [MATH_QWEN3_1P7B, MATH_R1_1P5B, MATH_R1_7B]
CODE_PAIRS = [CODE_QWEN3_30B, CODE_CODER_32B, CODE_QWEN3_4B]
ALL_PAIRS = MATH_PAIRS + CODE_PAIRS + [AGENT_QWEN3_4B]


@pytest.fixture(scope="session")
def models_dir() -> Path:
    """``$A2D_MODELS``: the downloaded models (``a2d models download``)."""
    from ask2distill.paths import get_paths
    root = get_paths().models
    if not root.is_dir():
        pytest.skip(f"{root} absent (a2d models download)")
    return root


@pytest.fixture(scope="session")
def tau2_python() -> Path:
    """``$A2D_TAU2_PYTHON``: the interpreter of the separate tau2 env."""
    value = os.environ.get("A2D_TAU2_PYTHON", "").strip()
    if not value or not Path(value).is_file():
        pytest.skip("A2D_TAU2_PYTHON is not set (env/build_tau2_env.sh)")
    return Path(value)


def _git(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, **kw)


def list_tracked_files() -> tuple[list[str], str]:
    """``(paths, mode)``: what git tracks, or what it WOULD track.

    In a git checkout: ``git ls-files`` plus untracked files that are not
    ignored (so a file about to be committed is covered too). Outside one (a
    fresh copy of the tree), a throwaway bare repository is pointed at the
    work tree and asked the same question, which applies every ``.gitignore``
    exactly as a first ``git add -A`` would; nothing is written to the tree.
    """
    inside = _git(["rev-parse", "--is-inside-work-tree"])
    if inside.returncode == 0 and inside.stdout.strip() == "true" \
            and Path(_git(["rev-parse", "--show-toplevel"]).stdout.strip()) == REPO.resolve():
        out = _git(["ls-files", "--cached", "--others", "--exclude-standard", "-z"], check=True)
        return sorted(p for p in out.stdout.split("\0") if p), "git"
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["git", "init", "-q", "--bare", td], check=True)
        out = subprocess.run(["git", f"--git-dir={td}", f"--work-tree={REPO}", "ls-files", "--others",
                              "--exclude-standard", "-z"], cwd=REPO, capture_output=True, text=True,
                             check=True)
    return sorted(p for p in out.stdout.split("\0") if p), "simulated"


@pytest.fixture(scope="session")
def tracked_files() -> list[str]:
    """Repo-relative paths under version control (see ``list_tracked_files``)."""
    try:
        files, _ = list_tracked_files()
    except (OSError, subprocess.CalledProcessError) as e:
        pytest.skip(f"git unavailable: {e}")
    return files
