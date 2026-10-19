"""Merge an FSDP actor checkpoint into a Hugging Face model directory.

Wraps ``python -m verl.model_merger merge --backend fsdp`` with the
coordination concurrent jobs need:

* **lock** -- several jobs may need the same checkpoint at once (one eval job
  per math task, a sweep and an eval). One takes ``<target>.lock`` (a
  directory, so ``mkdir`` is the atomic test-and-set) and merges; the others
  wait for it instead of racing into the same directory.
* **skip** -- a target that already holds a finished merge is reused.
* **chat template** (tau2) -- ``--chat-template ROLE`` copies
  ``models.templates[ROLE]`` next to the weights, so a served checkpoint
  carries the template it must be served with. A missing template is an error,
  never a silent fallback: vLLM's fallback is the model's own template, which
  re-enables thinking for Qwen3.
* **login-node guard** -- a 4B fp32 merge needs more memory than the 24 GB a
  login node allows per user; it is refused there unless ``--force``.

Layout: ``$A2D_CKPT/<experiment>/global_step_<N>/actor`` ->
``$A2D_CKPT/<experiment>/hf/step_<N>``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ask2distill import runtime
from ask2distill.paths import get_paths

MERGER_MODULE = "verl.model_merger"
# A login node caps each user's memory (24 GB here); anything this small is
# treated as one when we are not inside a SLURM allocation.
LOGIN_MEM_LIMIT_BYTES = 32 * 2**30
_POLL_S = 10
_HOLDER_GRACE_S = 20


class MergeError(RuntimeError):
    """The merge failed, or the process holding the lock died without a model."""


# --------------------------------------------------------------------------- layout

def actor_dir(cfg: DictConfig, experiment: str, step: int) -> Path:
    """``$A2D_CKPT/<experiment>/global_step_<step>/actor``."""
    return Path(str(cfg.paths.ckpt)) / experiment / f"global_step_{int(step)}" / "actor"


def hf_dir(cfg: DictConfig, experiment: str, step: int) -> Path:
    """``$A2D_CKPT/<experiment>/hf/step_<step>``: where merged weights live."""
    return Path(str(cfg.paths.ckpt)) / experiment / "hf" / f"step_{int(step)}"


def lock_path(target_dir: Path) -> Path:
    """``<target>.lock``, a sibling of the merged model."""
    target_dir = Path(target_dir)
    return target_dir.with_name(target_dir.name + ".lock")


def is_merged(target_dir: Path) -> bool:
    """True when ``target_dir`` holds a finished merge and nobody is writing to it.

    The merger writes config.json early and the tokenizer files last; the lock
    is released only after it returns, so all three are checked.
    """
    t = Path(target_dir)
    return ((t / "config.json").is_file() and (t / "tokenizer_config.json").is_file()
            and any(t.glob("*.safetensors")) and not lock_path(t).is_dir())


# ---------------------------------------------------------------- login-node guard

def _cgroup_memory_limits() -> list[int]:
    """Memory limits (bytes) of this process's cgroup and its ancestors (v2 or v1)."""
    limits: list[int] = []
    root = Path("/sys/fs/cgroup")
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return limits
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hier, controllers, rel = parts
        if hier == "0" and controllers == "":
            d = root / rel.lstrip("/")
            name = "memory.max"
        elif "memory" in controllers.split(","):
            d = root / "memory" / rel.lstrip("/")
            name = "memory.limit_in_bytes"
        else:
            continue
        while True:
            try:
                text = (d / name).read_text().strip()
                if text.isdigit():
                    limits.append(int(text))
            except OSError:
                pass
            if d in (root, root / "memory") or d.parent == d:
                break
            d = d.parent
    return limits


def on_login_node() -> bool:
    """Outside any SLURM allocation and under a small per-user memory limit."""
    if os.environ.get("SLURM_JOB_ID"):
        return False
    return any(limit <= LOGIN_MEM_LIMIT_BYTES for limit in _cgroup_memory_limits())


# --------------------------------------------------------------------------- merging

def merge_command(actor: Path, target: Path) -> list[str]:
    """The verl merger invocation."""
    return [sys.executable, "-m", MERGER_MODULE, "merge", "--backend", "fsdp",
            "--local_dir", str(actor), "--target_dir", str(target)]


def copy_chat_template(template: Path, target_dir: Path) -> Path:
    """Copy ``template`` into ``target_dir`` under its own name (``<name>.nothink.jinja``)."""
    template = Path(template)
    if not template.is_file():
        raise FileNotFoundError(f"chat template {template} does not exist; refusing to serve "
                                "the checkpoint with the model's own (thinking) template")
    dst = Path(target_dir) / template.name
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(template, tmp)
    os.replace(tmp, dst)
    return dst


def _wait_for_holder(target: Path, lock: Path, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if (target / "tokenizer_config.json").is_file() and not lock.is_dir():
            return
        if not lock.is_dir():
            # The holder released the lock; give the filesystem a moment, then a
            # model must be there -- otherwise the holder died mid-merge.
            time.sleep(_HOLDER_GRACE_S)
            if not (target / "config.json").is_file():
                raise MergeError(f"MERGE_HOLDER_GONE: {lock} released without a model in {target}")
            return
        time.sleep(_POLL_S)
    if not (target / "config.json").is_file():
        raise MergeError(f"MERGE_WAIT_TIMEOUT: no model in {target} after {timeout_s}s")


def merge_fsdp(actor_dir: Path, target_dir: Path, *, chat_template: Path | None = None,
               lock: bool = True, timeout_s: int = 3600, allow_login_node: bool = False,
               dry_run: bool = False) -> Path:
    """Merge ``actor_dir`` (an FSDP ``global_step_<N>/actor``) into ``target_dir``.

    Skips the merge when ``target_dir`` already holds one. With ``lock`` the
    first caller merges and concurrent callers wait up to ``timeout_s``.
    Returns ``target_dir``. Raises ``FileNotFoundError`` for a missing
    checkpoint and ``MergeError`` when no model results.
    """
    actor_dir, target_dir = Path(actor_dir), Path(target_dir)
    if not actor_dir.is_dir() and not is_merged(target_dir):
        if not dry_run:
            raise FileNotFoundError(f"NO_SUCH_CHECKPOINT: {actor_dir}")
        print(f"[merge] note: NO_SUCH_CHECKPOINT yet: {actor_dir}", file=sys.stderr, flush=True)
    cmd = merge_command(actor_dir, target_dir)
    env = runtime.base_env(get_paths(), stage="merge")
    if dry_run:
        runtime.run(cmd, env, dry_run=True)
        return target_dir
    if on_login_node() and not allow_login_node and not is_merged(target_dir):
        raise MergeError("refusing to merge on a login node (per-user memory limit; a 4B fp32 "
                         "merge exceeds it) -- run inside a GPU job, or pass --force")

    if is_merged(target_dir):
        print(f"[merge] already merged: {target_dir}", flush=True)
    elif not lock:
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if runtime.run(cmd, env) != 0 or not (target_dir / "config.json").is_file():
            raise MergeError(f"MERGE_FAILED: {actor_dir} -> {target_dir}")
    else:
        # The lock lives next to the merged model, so its parent has to exist
        # first: for a fresh experiment hf/ does not, mkdir fails for EVERY
        # concurrent caller, all of them take the waiter branch, and the
        # checkpoint dies of MERGE_WAIT_TIMEOUT without anyone ever merging.
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        lk = lock_path(target_dir)
        try:
            lk.mkdir()
            holder = True
        except FileExistsError:
            holder = False
        if holder:
            print(f"[merge] {actor_dir} -> {target_dir}", flush=True)
            try:
                rc = runtime.run(cmd, env)
            finally:
                lk.rmdir()
            if rc != 0 or not (target_dir / "config.json").is_file():
                raise MergeError(f"MERGE_FAILED: {actor_dir} -> {target_dir} (exit {rc})")
        else:
            print(f"[merge] waiting for the holder of {lk}", flush=True)
            _wait_for_holder(target_dir, lk, timeout_s)
    if chat_template is not None:
        dst = copy_chat_template(Path(chat_template), target_dir)
        print(f"[merge] chat template -> {dst}", flush=True)
    return target_dir


# ------------------------------------------------------------------------------ CLI

def template_for_role(cfg: DictConfig | None, role: str) -> Path:
    """``models.templates[role]`` of the pair; raises when it is absent or null."""
    templates = None if cfg is None else OmegaConf.select(cfg, "models.templates", default=None)
    if templates is None or role not in templates:
        raise SystemExit(f"--chat-template {role}: the config has no models.templates.{role}")
    value = templates[role]
    if value is None:
        raise SystemExit(f"--chat-template {role}: models.templates.{role} is null (that role "
                         "is served with the model's own template)")
    return Path(str(value))


def main(cfg: DictConfig | None, args: argparse.Namespace) -> int:
    """``a2d merge --pair P --arm A --step N`` or ``a2d merge --actor-dir D --target-dir T``."""
    if getattr(args, "actor_dir", None) or getattr(args, "target_dir", None):
        if not (args.actor_dir and args.target_dir):
            raise SystemExit("a2d merge: --actor-dir and --target-dir go together")
        actor, target = Path(os.path.abspath(args.actor_dir)), Path(os.path.abspath(args.target_dir))
    elif cfg is not None and getattr(args, "arm", None) and getattr(args, "step", None) is not None:
        from ask2distill.opd.launch import arm_config
        exp = str(arm_config(cfg, args.arm).experiment)
        actor, target = actor_dir(cfg, exp, args.step), hf_dir(cfg, exp, args.step)
    else:
        raise SystemExit("a2d merge: give --pair P --arm A --step N, or --actor-dir D --target-dir T")
    template = template_for_role(cfg, args.chat_template) if getattr(args, "chat_template", None) else None
    try:
        merge_fsdp(actor, target, chat_template=template,
                   allow_login_node=bool(getattr(args, "force", False)),
                   dry_run=bool(getattr(args, "dry_run", False)))
    except (FileNotFoundError, MergeError) as e:
        print(str(e), file=sys.stderr, flush=True)
        return 1
    return 0
