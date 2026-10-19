"""Stage 4 launcher: render verl's ``main_ppo`` overrides for one OPD arm and run it.

One launcher serves every domain and both arms. ``build_overrides`` emits the
same override strings, in the same order, as the shell launcher of the final
runs produced under each arm's job environment (every math, code and tau2 arm
of the main table), followed by two non-semantic groups:

* ``hydra.run.dir=<run_dir>/hydra`` -- otherwise hydra writes ``outputs/<date>/``
  into the current directory;
* ``+ray_kwargs.ray_init.runtime_env.env_vars.<K>=<V>`` -- run-specific
  environment. verl merges it into ``ray.init`` (verl's main_ppo.py), so it
  reaches every worker on every node, including agent-loop workers that read
  their settings at import (tau2). The shell launcher got the same effect by
  exporting everything before its own ``ray start``.

Translation from the shell launcher:

* ``MAX_VAL_RESP_LENGTH`` is ``opd.max_val_response_length`` (per pair, never
  the launcher default); it feeds ``max_model_len`` and ``+val_kwargs.max_tokens``.
* ``TEST_FILE`` is ``opd.test_files``: the math validation sets (math and tau2,
  where TEST_FILE was never set), or ``[train]`` for code, which spelled the
  training parquet as TEST_FILE. Both are rendered exactly as the shell did.
* ``PROJECT_PATH`` is ``$A2D_CKPT``; ``custom_reward_function.path`` is absolute
  (the shell launcher relied on its working directory).
* ``trainer.nnodes`` comes from ``opd.nnodes`` instead of ``$SLURM_NNODES``,
  which every final arm pinned to 1.
* ``data.train_max_samples`` / ``data.val_max_samples`` are emitted only when
  capped (smoke); the final runs never set them.

Ray: inside a cluster somebody else built (``$RAY_ADDRESS`` set) this only
attaches. The shell launcher opened with ``ray stop --force``, which tears down
any cluster on the node -- including a synthesizer-RL job's when a real-data arm
ran next to it.

Save-before-kill (ESI): with ``opd.esi_save`` the job's end time goes to verl as
``MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP``. It comes from
``--wall-clock-end`` (launchers/slurm/common.sh) or, inside SLURM, from the
job's remaining time (``squeue %L``; scontrol's EndTime is printed in local
time and is easily misread as UTC).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from ask2distill import config as a2d_config
from ask2distill import runtime
from ask2distill.paths import get_paths

ENTRY_MODULE = "verl.trainer.main_ppo"
RUNTIME_ENV_PREFIX = "+ray_kwargs.ray_init.runtime_env.env_vars."
# verl's checkpoint manager force-saves once the time left before this unix
# timestamp drops below "longest step so far + save time + esi_redundant_time"
# (checkpoint_manager.py:should_save_ckpt_esi; meant for Volcengine elastic
# instances, reused here for the SLURM wall clock).
ESI_ENV = "MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"
# opd.test_files == [TRAIN_SENTINEL]: validate on the arm's own training parquet.
TRAIN_SENTINEL = "train"
ARMS = ("ours", "vanilla")
# verl writes this tracker only after a save completes; find_latest_ckpt_path
# reads it, so it is what training actually resumes from.
TRACKER = "latest_checkpointed_iteration.txt"

_SAFE_UNQUOTED = re.compile(r"[A-Za-z0-9_/.:@+\-]+")


# ------------------------------------------------------------------ value spelling

def fmt(value: Any) -> str:
    """A config value spelled as the shell launcher spelled it.

    Booleans as ``True``/``False``, ``None`` as ``null``, floats in Python's
    shortest form with the exponent shortened (``1e-06`` -> ``1e-6``),
    everything else ``str``.
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "null"
    if isinstance(value, float):
        s = repr(value)
        if "e" in s:
            mant, exp = s.split("e")
            sign = "-" if exp.startswith("-") else ""
            s = f"{mant}e{sign}{exp.lstrip('+-').lstrip('0') or '0'}"
        return s
    return str(value)


def _looks_non_string(value: str) -> bool:
    if value.lower() in ("true", "false", "null", "none", "nan", "inf", "+inf", "-inf"):
        return True
    try:
        float(value)
    except ValueError:
        return False
    return True


def hydra_string(value: Any) -> str:
    """``value`` as a hydra override value that parses back to exactly ``str(value)``.

    Ray requires ``runtime_env.env_vars`` values to be strings, but hydra turns
    an unquoted ``1700000000`` into an int and splits ``a,b`` into a list, so
    anything that is not a plain path-like token is single-quoted.
    """
    s = str(value)
    if _SAFE_UNQUOTED.fullmatch(s) and not _looks_non_string(s):
        return s
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


# --------------------------------------------------------------------- arm view

def arm_config(cfg: DictConfig, arm: str) -> DictConfig:
    """The ``opd`` section with ``opd.arms.<arm>`` laid over it (arm keys win)."""
    if arm not in ARMS:
        raise a2d_config.ConfigError(f"unknown OPD arm {arm!r}; expected one of {ARMS}")
    a2d_config.require(cfg, "opd")
    arms = cfg.opd.arms
    if arm not in arms:
        raise a2d_config.ConfigError(
            f"pair {cfg.pair} has no opd.arms.{arm} (has: {', '.join(arms.keys())})")
    base = OmegaConf.to_container(cfg.opd, resolve=True, throw_on_missing=True)
    base.pop("arms")
    merged = OmegaConf.merge(OmegaConf.create(base),
                             OmegaConf.to_container(arms[arm], resolve=True, throw_on_missing=True))
    for key in ("train", "experiment", "total_epochs", "save_freq",
                "ppo_max_token_len_per_gpu", "reward_micro_bs"):
        if OmegaConf.select(merged, key, default=None) is None:
            raise a2d_config.ConfigError(f"opd.arms.{arm}.{key} is not set for pair {cfg.pair}")
    return merged


def ckpt_dir(cfg: DictConfig, arm: str) -> Path:
    """``$A2D_CKPT/<experiment>`` (verl's ``trainer.default_local_dir``)."""
    return Path(str(cfg.paths.ckpt)) / str(arm_config(cfg, arm).experiment)


def latest_step(cfg: DictConfig, arm: str) -> int | None:
    """Step recorded in the arm's checkpoint tracker, or ``None`` for a fresh run."""
    tracker = ckpt_dir(cfg, arm) / TRACKER
    try:
        text = tracker.read_text().strip()
    except OSError:
        return None
    return int(text) if text else None


def val_files(a: DictConfig) -> list[str]:
    """Validation parquets of the arm (``[train]`` resolves to the training file)."""
    files = [str(f) for f in a.test_files]
    if files == [TRAIN_SENTINEL]:
        return [str(a.train)]
    if TRAIN_SENTINEL in files:
        raise a2d_config.ConfigError(f"opd.test_files mixes {TRAIN_SENTINEL!r} with paths: {files}")
    return files


def render_val_files(a: DictConfig) -> str:
    """``data.val_files`` exactly as the shell launcher rendered it.

    Code arms set ``TEST_FILE='["<train parquet>"]'`` (quotes kept, JSON style);
    math and tau2 left it unset, and the launcher's default
    ``["$D/AIME25/test.parquet", ...]`` lost its quotes to bash's quote removal
    inside ``${TEST_FILE:-...}``, leaving ``[a, b, c]``. Hydra reads both as the
    same list of strings.
    """
    if [str(f) for f in a.test_files] == [TRAIN_SENTINEL]:
        return '["' + str(a.train) + '"]'
    return "[" + ", ".join(val_files(a)) + "]"


def max_model_len(a: DictConfig) -> int:
    """vLLM caps the WHOLE sequence: prompt + the longer of train / val responses."""
    return int(a.max_prompt_length) + max(int(a.max_response_length),
                                          int(a.max_val_response_length))


# ------------------------------------------------------------------- environment

def base_runtime_env(cfg: DictConfig, arm: str) -> dict[str, str]:
    """Run-specific variables every Ray worker needs, whoever started Ray."""
    return {"SWANLAB_LOG_DIR": str(Path(str(cfg.paths.ckpt)) / "swanlab_log")}


def build_env(cfg: DictConfig, arm: str) -> dict[str, str | None]:
    """Process environment of the training driver (also what a private Ray head starts with).

    ``runtime.base_env(stage="opd")`` carries the launcher's process hygiene
    (caches, NCCL / Ray / hydra settings, ``NCCL_DEBUG=WARN``,
    ``SWANLAB_MODE=local``, ``TOKENIZERS_PARALLELISM=true``, and never
    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments``).
    """
    env = runtime.base_env(get_paths(), stage="opd")
    env.update(base_runtime_env(cfg, arm))
    # An end time inherited from a parent job must not leak in: only run() sets it,
    # and only when opd.esi_save is on.
    env[ESI_ENV] = None
    return env


def remaining_seconds(text: str) -> int | None:
    """Seconds in squeue's ``%L`` field (``[D-]HH:MM:SS``, ``MM:SS`` or ``SS``)."""
    text = text.strip()
    if not text or text in ("INVALID", "UNLIMITED", "NOT_SET"):
        return None
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [int(p) for p in text.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + s


def slurm_wall_clock_end(job_id: str | None = None) -> int | None:
    """Unix time this SLURM job is killed, from its remaining time; ``None`` outside SLURM."""
    job_id = job_id or os.environ.get("SLURM_JOB_ID", "").strip()
    if not job_id:
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%L"], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    secs = remaining_seconds(out.splitlines()[0] if out.strip() else "")
    return None if secs is None else int(time.time()) + secs


def esi_env(cfg: DictConfig, arm: str, wall_clock_end: int | None) -> dict[str, str]:
    """``{ESI_ENV: ts}`` when this arm force-saves before the wall clock, else ``{}``.

    Prints the same line the shell launcher printed, so the job log says which way it went.
    """
    if not bool(arm_config(cfg, arm).esi_save):
        print("save-before-kill DISABLED (opd.esi_save=false); resubmit loses up to "
              "save_freq steps", flush=True)
        return {}
    end = wall_clock_end if wall_clock_end is not None else slurm_wall_clock_end()
    if end is None:
        print("WARNING: no wall-clock end time -- no save-before-kill, only every "
              "save_freq steps", flush=True)
        return {}
    print(f"wall clock ends at ts={end}; will force-save before then", flush=True)
    return {ESI_ENV: str(int(end))}


# --------------------------------------------------------------------- overrides

def build_overrides(cfg: DictConfig, arm: str, run_dir: Path,
                    runtime_env: Mapping[str, Any] | None = None) -> list[str]:
    """Hydra overrides for ``python -m verl.trainer.main_ppo``, in the final runs' order.

    ``runtime_env`` holds extra run-specific variables for every Ray worker
    (tau2 agent-loop settings, the ESI end time); they are appended after
    ``SWANLAB_LOG_DIR`` in the ``+ray_kwargs...env_vars`` tail.
    """
    a = arm_config(cfg, arm)
    off = a.offload
    project_path = Path(str(cfg.paths.ckpt))
    exp = str(a.experiment)
    ov = [
        # advantage = reward x mask: no baseline, no GAE, no group normalisation
        f"algorithm.adv_estimator={a.adv_estimator}",
        f"algorithm.grpo_outcome_weight={fmt(float(a.grpo_outcome_weight))}",
        # False for math/code: every epoch replays the same prompts in the same order
        f"data.shuffle={fmt(bool(a.shuffle))}",
        f"data.train_files={a.train}",
        # never scored (val_before_train False, test_freq -1); only has to load
        f"data.val_files={render_val_files(a)}",
        f"data.train_batch_size={int(a.mini_batch_size) * int(a.parallel_size)}",
        # 1024 fits DAPO math. A code_contests statement in the LiveCodeBench
        # template runs p90 938 / p99 1444 / max 1887, and filter_overlong_prompts
        # below DROPS anything over the cap silently -- at 1024 that is 7% of a
        # real-question arm and a different 7% of a synthetic one, which quietly
        # destroys the budget match the arms are compared under.
        f"data.max_prompt_length={fmt(a.max_prompt_length)}",
        f"data.max_response_length={fmt(a.max_response_length)}",
        f"data.filter_overlong_prompts={fmt(bool(a.filter_overlong_prompts))}",
        f"data.truncation={a.truncation}",
        f"data.return_raw_chat={fmt(bool(a.return_raw_chat))}",
    ]
    # Qwen3 thinks by default; a non-thinking pair MUST pass enable_thinking=False
    # or the rollouts silently grow a reasoning chain. Empty for pre-Qwen3 pairs.
    ov += str(a.chat_template_args or "").split()
    ov += [
        f"actor_rollout_ref.model.path={a.student}",
        f"actor_rollout_ref.model.use_remove_padding={fmt(bool(a.use_remove_padding))}",
        f"actor_rollout_ref.model.enable_activation_offload={fmt(bool(off.activation))}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={fmt(bool(off.gradient_checkpointing))}",
        f"actor_rollout_ref.actor.optim.lr={fmt(float(a.lr))}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={fmt(a.mini_batch_size)}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={fmt(a.ppo_micro_batch_size_per_gpu)}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={fmt(a.ppo_max_token_len_per_gpu)}",
        f"actor_rollout_ref.actor.ulysses_sequence_parallel_size={fmt(a.parallel_size)}",
    ]
    # KL to the reference is off by default. The same-family code pair
    # (Qwen2.5-Coder 3B <- 32B) needs it: its token distributions barely differ
    # (δ_ref 1.3629 vs 3.9756 for Qwen3-4B <- Qwen3-30B-A3B), the OPD gradient is weak and length is the only free
    # direction left; without the KL / entropy terms entropy and rollout
    # length collapse.
    if bool(a.use_kl):
        ov += ["actor_rollout_ref.actor.use_kl_loss=True",
               f"actor_rollout_ref.actor.kl_loss_coef={fmt(float(a.kl_coef))}",
               f"actor_rollout_ref.actor.kl_loss_type={a.kl_loss_type}"]
    else:
        ov.append("actor_rollout_ref.actor.use_kl_loss=False")
    # fp32 is the paper default and it matters: in bf16 at lr 1e-6 the relative
    # update is below bf16's resolution and gets rounded away.
    dtype = str(a.model_dtype)
    ov += [
        f"actor_rollout_ref.actor.loss_agg_mode={a.loss_agg_mode}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={fmt(bool(off.actor_param))}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={fmt(bool(off.actor_optimizer))}",
        f"actor_rollout_ref.actor.fsdp_config.forward_prefetch={fmt(bool(off.forward_prefetch))}",
        f"actor_rollout_ref.actor.fsdp_config.model_dtype={dtype}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={fmt(a.ppo_max_token_len_per_gpu)}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={fmt(bool(off.ref_param))}",
        f"actor_rollout_ref.ref.fsdp_config.model_dtype={dtype}",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.temperature={fmt(float(a.temperature))}",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        f"+actor_rollout_ref.rollout.log_prob_top_k={fmt(a.log_prob_top_k)}",
        f"+actor_rollout_ref.rollout.top_k_strategy={a.top_k_strategy}",
        f"+actor_rollout_ref.rollout.reward_weight_mode={a.reward_weight_mode}",
        f"+actor_rollout_ref.rollout.teacher_temperature={fmt(float(a.teacher_temperature))}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={fmt(a.parallel_size)}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={fmt(float(a.rollout_gpu_memory_utilization))}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len(a)}",
        f"actor_rollout_ref.rollout.n={fmt(a.n_responses)}",
        f"actor_rollout_ref.rollout.val_kwargs.do_sample={fmt(bool(a.val_kwargs.do_sample))}",
        f"+actor_rollout_ref.rollout.val_kwargs.max_tokens={fmt(a.max_val_response_length)}",
        f"actor_rollout_ref.rollout.val_kwargs.n={fmt(a.val_kwargs.n)}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={fmt(float(a.val_kwargs.temperature))}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={fmt(float(a.val_kwargs.top_p))}",
        f"actor_rollout_ref.rollout.repetition_penalty={fmt(float(a.repetition_penalty))}",
        f"actor_rollout_ref.rollout.calculate_log_probs={fmt(bool(a.calculate_log_probs))}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={fmt(a.ref_log_prob_micro_batch_size_per_gpu)}",
        # The teacher sits in verl's reward-model slot and scores the student's own
        # token ids, so the pair must share a tokenizer (checked per pair).
        "reward_model.enable=True",
        f"+reward_model.reward_kwargs.enable_format_reward={fmt(bool(a.enable_format_reward))}",
        f"reward_model.model.path={a.teacher}",
        f"reward_model.model.input_tokenizer={fmt(a.reward_model_input_tokenizer)}",
        f"reward_model.model.use_remove_padding={fmt(bool(a.use_remove_padding))}",
        f"reward_model.model.fsdp_config.param_offload={fmt(bool(off.reward_model_param))}",
        f"+reward_model.model.dtype={dtype}",
        f"reward_model.micro_batch_size_per_gpu={fmt(a.reward_micro_bs)}",
        # Logged only: the teacher's token-level reward is the whole signal.
        f"custom_reward_function.path={a.custom_reward.path}",
        f"custom_reward_function.name={a.custom_reward.name}",
        f"trainer.val_before_train={fmt(bool(a.val_before_train))}",
        f"trainer.log_val_generations={fmt(a.log_val_generations)}",
        "trainer.logger=[" + ",".join(str(x) for x in a.logger) + "]",
        f"trainer.project_name={a.project_name}",
        f"trainer.experiment_name={exp}",
        f"trainer.validation_data_dir={project_path / 'validation_log' / exp}",
        f"trainer.n_gpus_per_node={fmt(a.n_gpus_per_node)}",
        f"trainer.nnodes={fmt(a.nnodes)}",
        f"trainer.save_freq={fmt(a.save_freq)}",
        # Keep this at 0. verl re-evaluates the expiration check every step and has
        # no "already saved for expiration" flag, so the trigger window is one save
        # PER STEP until the job dies. A run with 600 force-saved on steps
        # 114-120: 7 checkpoints, 6 of them redundant, 120GB of disk.
        f"trainer.esi_redundant_time={fmt(a.esi_redundant_time)}",
        f"trainer.test_freq={fmt(a.test_freq)}",
        f"trainer.total_epochs={fmt(a.total_epochs)}",
        f"trainer.resume_mode={a.resume_mode}",
        # A stable name per arm, never a timestamp: the 4h wall clock cannot hold a
        # whole arm, so resume_mode=auto continues from the same directory.
        f"trainer.default_local_dir={project_path / exp}",
        f"actor_rollout_ref.actor.entropy_coeff={fmt(a.entropy_coeff)}",
        f"trainer.is_plot={fmt(bool(a.is_plot))}",
    ]
    for key in ("train_max_samples", "val_max_samples"):
        if int(a[key]) != -1:
            ov.append(f"data.{key}={int(a[key])}")
    ov += [str(x) for x in (a.extra_overrides or [])]
    forbidden = set(a.forbidden_overrides or [])
    bad = [o for o in ov if o.split("=", 1)[0].lstrip("+~") in forbidden]
    if bad:
        raise a2d_config.ConfigError(f"OPD overrides set forbidden key(s): {bad}")

    ov.append(f"hydra.run.dir={Path(os.path.abspath(run_dir)) / 'hydra'}")
    env_vars = {**base_runtime_env(cfg, arm), **{k: str(v) for k, v in (runtime_env or {}).items()}}
    ov += [f"{RUNTIME_ENV_PREFIX}{k}={hydra_string(v)}" for k, v in env_vars.items()]
    return ov


# ---------------------------------------------------------------------- running

def missing_inputs(cfg: DictConfig, arm: str) -> list[str]:
    """Training and validation files that do not exist.

    The validation sets are loaded but never scored, so they are only a
    startup gate -- kept because a missing file would otherwise fail deep
    inside the first dataloader instead of here.
    """
    a = arm_config(cfg, arm)
    files = [str(a.train)] + [f for f in val_files(a) if f != str(a.train)]
    return [f for f in files if not Path(f).is_file()]


def num_gpus(cfg: DictConfig, arm: str) -> int:
    """GPUs the arm's Ray cluster must offer."""
    a = arm_config(cfg, arm)
    return int(a.n_gpus_per_node) * int(a.nnodes)


def run(cfg: DictConfig, arm: str, *, runtime_env: Mapping[str, Any] | None = None,
        wall_clock_end: int | None = None, dry_run: bool = False) -> int:
    """Launch one OPD arm of any domain; returns the trainer's exit code.

    ``runtime_env`` is delivered to every Ray worker and also set in the
    driver's environment (tau2 passes its agent-loop settings this way).
    """
    a = arm_config(cfg, arm)
    missing = missing_inputs(cfg, arm)
    if missing:
        msg = "MISSING DATA: " + ", ".join(missing) + \
              " -- `a2d data download` / `a2d data build` (or `a2d bank`) first"
        if not dry_run:
            print(msg, flush=True)
            return 1
        print(f"[dry-run] {msg}", flush=True)
    run_dir = runtime.new_run_dir(cfg, f"opd_{arm}")
    ck = ckpt_dir(cfg, arm)
    step = latest_step(cfg, arm)
    print(f"RESUMING from step {step} (resume_mode=auto)" if step is not None
          else f"FRESH RUN (no {ck / TRACKER})", flush=True)
    print(f"EXPERIMENT_NAME: {a.experiment}", flush=True)
    print(f"MODEL_DTYPE: {a.model_dtype} | ROLLOUT_GPU_MEM_UTIL: "
          f"{a.rollout_gpu_memory_utilization} | PPO_MAX_TOKEN_LEN_PER_GPU: "
          f"{a.ppo_max_token_len_per_gpu}", flush=True)

    worker_env = {**esi_env(cfg, arm, wall_clock_end),
                  **{k: str(v) for k, v in (runtime_env or {}).items()}}
    overrides = build_overrides(cfg, arm, run_dir, worker_env)
    env = build_env(cfg, arm)
    env.update(worker_env)
    cmd = [sys.executable, "-m", ENTRY_MODULE, *overrides]
    runtime.write_run_record(run_dir, env, cmd)
    if not dry_run:
        ck.mkdir(parents=True, exist_ok=True)
        Path(str(env["SWANLAB_LOG_DIR"])).mkdir(parents=True, exist_ok=True)

    gpus = num_gpus(cfg, arm)
    if int(a.nnodes) > 1 and not os.environ.get("RAY_ADDRESS") and not dry_run:
        raise RuntimeError(f"opd.nnodes={a.nnodes}: start the multi-node Ray cluster first "
                           "(launchers/slurm/common.sh exports RAY_ADDRESS)")
    print(f"[opd] run dir {run_dir}", flush=True)
    with runtime.ray_session(gpus, env, dry_run=dry_run):
        rc = runtime.run(cmd, env, log=run_dir / "opd.log", dry_run=dry_run, cwd=run_dir)
    print(f"exit code: {rc}   ckpt: {ck}", flush=True)
    return rc


def main(cfg: DictConfig, args: argparse.Namespace | None = None) -> int:
    """``a2d opd --pair P --arm ours|vanilla`` for math and code pairs."""
    if cfg.domain == "tau2":
        raise a2d_config.ConfigError("tau2 OPD needs its env service and user simulator: "
                                     "use `a2d tau2 opd --arm A --user-endpoints FILE`")
    for section in ("models", "opd"):
        a2d_config.require(cfg, section)
    return run(cfg, args.arm, wall_clock_end=getattr(args, "wall_clock_end", None),
               dry_run=bool(getattr(args, "dry_run", False)))
