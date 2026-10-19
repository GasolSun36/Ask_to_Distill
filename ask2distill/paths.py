"""Filesystem locations, resolved from ``A2D_*`` environment variables.

Every location the pipeline reads or writes is derived here, so no module
carries a cluster path. Defaults live under the repository root, which is
taken from this file's location (the package must be installed editable, see
``env/build_a2d_env.sh``):

==================  ==========================  =====================================
variable            default                     holds
==================  ==========================  =====================================
``A2D_DATA``        ``<repo>/data``             released / built datasets
``A2D_MODELS``      ``<repo>/models``           downloaded base models
``A2D_CKPT``        ``<repo>/checkpoints``      training checkpoints, merged HF dirs
``A2D_OUT``         ``<repo>/outputs``          stage outputs, eval results, run dirs
``A2D_CACHE``       ``<repo>/.cache``           HF / triton / vLLM / evalplus caches
==================  ==========================  =====================================

``A2D_TAU2_PYTHON`` (interpreter of the separate tau2 env), ``A2D_HF_DATA_REPO``
(release dataset repo id) and ``A2D_OPENAI_ENV_FILE`` (the user's own key file,
never copied) are optional and only needed by the stages that use them.

A relative value is resolved against the repository root, not the current
directory: Ray workers, the tau2 env-service and LiveCodeBench all run with
different working directories, so a relative path would silently point at
different places in different processes. Anything still relative after
resolution is an error.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Kept as os.path.abspath rather than Path.resolve(): resolving would rewrite a
# symlinked mount into its target and every recorded path would change spelling.
REPO_ROOT = Path(os.path.abspath(__file__)).parents[1]

_DIR_VARS = {
    "data": ("A2D_DATA", "data"),
    "models": ("A2D_MODELS", "models"),
    "ckpt": ("A2D_CKPT", "checkpoints"),
    "out": ("A2D_OUT", "outputs"),
    "cache": ("A2D_CACHE", ".cache"),
}


class PathError(RuntimeError):
    """A configured location could not be resolved to an absolute path."""


@dataclass(frozen=True)
class Paths:
    """Absolute locations used by every stage (see the module docstring)."""

    repo: Path
    data: Path
    models: Path
    ckpt: Path
    out: Path
    cache: Path
    third_party: Path
    tau2_python: Path | None = None
    hf_data_repo: str | None = None
    openai_env_file: Path | None = None

    def as_dict(self) -> dict[str, str | None]:
        """String form, used to inject the ``paths`` node into a config."""
        return {k: (None if v is None else str(v)) for k, v in self.__dict__.items()}


def resolve(value: str | os.PathLike, root: Path = REPO_ROOT) -> Path:
    """Expand ``~`` and variables, then anchor a relative path at ``root``."""
    p = Path(os.path.expandvars(os.path.expanduser(os.fspath(value))))
    if not p.is_absolute():
        p = root / p
    p = Path(os.path.normpath(p))
    if not p.is_absolute():
        raise PathError(f"{value!r} did not resolve to an absolute path (got {p})")
    return p


def get_paths(env: dict[str, str] | None = None) -> Paths:
    """Resolve every location from ``env`` (default: the process environment)."""
    env = os.environ if env is None else env
    if not (REPO_ROOT / "configs").is_dir():
        raise PathError(
            f"{REPO_ROOT} has no configs/ directory; ask2distill must be installed "
            "editable from the repository (pip install -e . --no-deps)")
    dirs = {name: resolve(env.get(var) or default) for name, (var, default) in _DIR_VARS.items()}
    tau2_py = env.get("A2D_TAU2_PYTHON", "").strip()
    openai_env = env.get("A2D_OPENAI_ENV_FILE", "").strip()
    return Paths(
        repo=REPO_ROOT,
        third_party=REPO_ROOT / "third_party",
        tau2_python=resolve(tau2_py) if tau2_py else None,
        hf_data_repo=env.get("A2D_HF_DATA_REPO", "").strip() or None,
        openai_env_file=resolve(openai_env) if openai_env else None,
        **dirs,
    )


def require_tau2_python(paths: Paths | None = None) -> Path:
    """The tau2 interpreter, or a clear error naming the variable to set."""
    p = (paths or get_paths()).tau2_python
    if p is None:
        raise PathError("A2D_TAU2_PYTHON is not set; build the tau2 env with "
                        "env/build_tau2_env.sh and export the interpreter path it prints")
    return p
