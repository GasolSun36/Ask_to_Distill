"""``--model`` of ``a2d eval math|code``: evaluate a ready HF model instead of an OPD checkpoint.

    a2d eval math --model gasolsun/<pair>                # released student; --pair is inferred
    a2d eval code --pair P --model /path/to/hf_dir       # any local HF directory

``--model`` is a local HF directory or an HF repo id (``owner/name``). A repo id is
downloaded once to ``$A2D_MODELS/<owner>__<name>`` with ``huggingface_hub.snapshot_download``
(``$HF_ENDPOINT`` selects a mirror) and reused while that copy is complete. No merge runs.

Results are named after the model: ``<owner>__<name>`` for a repo id, and
``<dir name>-<8 hex digits of its absolute path>`` for a local directory, so two
directories with the same basename never share LiveCodeBench's cached generations
(``--continue_existing`` keys them on this name).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from omegaconf import DictConfig

_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def check_args(args) -> str | None:
    """``--model`` alone, or ``--arm`` with ``--step``; returns ``--model`` (None for a checkpoint)."""
    model = getattr(args, "model", None)
    arm, step = getattr(args, "arm", None), getattr(args, "step", None)
    if model and (arm or step is not None):
        raise SystemExit("give either --model or --arm with --step, not both")
    if not model and (arm is None or step is None):
        raise SystemExit("give --arm and --step (an OPD checkpoint) or --model (a ready model)")
    return model


def model_name(spec: str) -> str:
    """The name results are stored under (eval directory and LCB label)."""
    p = Path(spec).expanduser()
    if p.is_dir():
        p = p.resolve()
        return f"{p.name}-{hashlib.sha1(str(p).encode()).hexdigest()[:8]}"
    if not _REPO_ID.match(spec):
        raise SystemExit(f"--model {spec!r}: neither a local directory nor an HF repo id (owner/name)")
    return spec.replace("/", "__")


def pair_of(spec: str) -> str | None:
    """The pair a released model belongs to: its repo or directory name, when that names a pair."""
    from ask2distill.config import list_pairs
    name = Path(spec.rstrip("/")).name
    return name if name in list_pairs() else None


def is_complete(model_dir: Path) -> bool:
    """An HF directory with its config, tokenizer and safetensors weights."""
    return ((model_dir / "config.json").is_file() and (model_dir / "tokenizer_config.json").is_file()
            and any(model_dir.glob("*.safetensors")))


def resolve(cfg: DictConfig, spec: str, *, dry_run: bool = False) -> Path:
    """Local HF directory for ``spec``, downloading a repo id into ``$A2D_MODELS`` if needed."""
    p = Path(spec).expanduser()
    if p.is_dir():
        p = p.resolve()
        if not is_complete(p):
            raise SystemExit(f"--model {p}: not an HF directory (config.json, tokenizer_config.json, "
                             "*.safetensors)")
        return p
    local = Path(str(cfg.paths.models)) / model_name(spec)
    if is_complete(local):
        print(f"[model] {spec}: using {local}", flush=True)
        return local
    if dry_run:
        print(f"[dry-run] download {spec} -> {local}", flush=True)
        return local
    from huggingface_hub import snapshot_download
    print(f"[model] downloading {spec} -> {local}", flush=True)
    snapshot_download(repo_id=spec, local_dir=str(local))
    if not is_complete(local):
        raise SystemExit(f"--model {spec}: download finished but {local} is not a complete HF directory")
    return local
