"""Launch tau2 synthesizer RL for one version (v4..v7): env service + verl GRPO.

Steps, as the final runs ran them on the training node:

1. read the endpoints file written by ``a2d tau2 serve --plan synrl`` (kinds
   stu / usr, and tea when the teacher gate is on) -- every ready replica is
   used, the service spreads load over all of them;
2. start ``envside.env_service`` under the tau2 interpreter with the version's
   reward (``TAU2_REWARD_V``), ``svc_workers`` scoring threads and the
   endpoints, and wait for ``/health``;
3. run ``python -m ask2distill.tau2.trainside.synrl_main`` with the overrides
   of ``build_overrides`` -- the same strings, in the same order, as the
   final runs' launch -- followed by ``hydra.run.dir`` and the
   ``+ray_kwargs.ray_init.runtime_env.env_vars`` tail that carries the
   worker's ``TAU2_*`` settings to every Ray worker.

``max_prompt_length`` is per version (7168 / 8192 / 8192 / 10240); the actor's
token budget and vLLM's ``max_model_len`` are prompt + response (1536), and
``filter_overlong_prompts`` drops any residual outlier instead of killing the
DataLoader, which is how the third launch died. The synthesizer starts from the
previous version's chosen, merged checkpoint (``synrl.versions.<v>.init``).

Save-before-kill: verl's ESI path force-saves once the time left is under
"longest step so far + save time"; the end time comes from
``--wall-clock-end`` or the job's remaining time (squeue ``%L``, not
scontrol's EndTime, which is printed in local time and was once read as UTC:
the first restart logged a wall clock ending three hours before it began).

Ray: a private head on this node unless ``$RAY_ADDRESS`` is set;
``ray stop --force`` is never run.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig

from ask2distill import config as a2d_config
from ask2distill import runtime
from ask2distill.tau2 import env as tau2_env
from ask2distill.tau2 import spawn
from ask2distill.tau2.envside import endpoints as E

ENTRY_MODULE = "ask2distill.tau2.trainside.synrl_main"


def _fmt(value: Any) -> str:
    from ask2distill.opd.launch import fmt
    return fmt(value)


def max_prompt_length(cfg: DictConfig, version: str) -> int:
    return int(tau2_env.version_config(cfg, version)["max_prompt_length"])


def token_budget(cfg: DictConfig, version: str) -> int:
    """Prompt + response length: actor token budget and vLLM max_model_len."""
    g = cfg.synrl
    if str(g.ppo_max_token_len_per_gpu) != "prompt_plus_resp":
        return int(g.ppo_max_token_len_per_gpu)
    return max_prompt_length(cfg, version) + int(g.max_response_length)


def build_overrides(cfg: DictConfig, version: str, run_dir: Path,
                    runtime_env: Mapping[str, Any] | None = None) -> list[str]:
    """Hydra overrides for ``synrl_main``, in the final runs' order, plus the tail."""
    from ask2distill.opd.launch import RUNTIME_ENV_PREFIX, hydra_string
    g = cfg.synrl
    v = tau2_env.version_config(cfg, version)
    plen, resp, budget = max_prompt_length(cfg, version), int(g.max_response_length), token_budget(cfg, version)
    ckpt = Path(str(cfg.paths.ckpt))
    exp = str(v["experiment"])
    f = _fmt
    ov = [
        "algorithm.adv_estimator=grpo",
        f"custom_reward_function.path={g.custom_reward.path}",
        f"custom_reward_function.name={g.custom_reward.name}",
        f"trainer.val_before_train={f(bool(g.val_before_train))}",
        f"data.shuffle={f(bool(g.shuffle))}",
        f"data.train_files={v['prompts']}",
        f"data.val_files={v['prompts']}",
        f"data.train_batch_size={f(g.train_batch_size)}",
        f"data.max_prompt_length={plen}",
        f"data.max_response_length={resp}",
        f"data.return_raw_chat={f(bool(g.return_raw_chat))}",
        f"data.filter_overlong_prompts={f(bool(g.filter_overlong_prompts))}",
        f"+data.apply_chat_template_kwargs.enable_thinking={f(bool(g.enable_thinking))}",
        f"actor_rollout_ref.model.path={v['init']}",
        f"actor_rollout_ref.model.use_remove_padding={f(bool(g.use_remove_padding))}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={f(bool(g.gradient_checkpointing))}",
        f"actor_rollout_ref.actor.optim.lr={f(float(g.lr))}",
        f"actor_rollout_ref.actor.optim.lr_scheduler_type={g.lr_scheduler_type}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={f(g.ppo_mini_batch_size)}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={f(g.ppo_micro_batch_size_per_gpu)}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={budget}",
        f"actor_rollout_ref.actor.use_kl_loss={f(bool(g.use_kl_loss))}",
        f"actor_rollout_ref.actor.kl_loss_coef={f(float(g.kl_loss_coef))}",
        f"actor_rollout_ref.actor.entropy_coeff={f(float(g.entropy_coeff))}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={f(bool(g.offload))}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={f(bool(g.offload))}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={f(float(g.gpu_memory_utilization))}",
        f"actor_rollout_ref.rollout.max_model_len={budget}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={f(g.max_num_batched_tokens)}",
        f"actor_rollout_ref.rollout.n={f(g.rollout_n)}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={f(bool(g.ref_param_offload))}",
        # The reward-model slot hosts Tau2RewardWorker; it loads no reward model
        # (the path is only a placeholder verl requires). No expandable_segments in
        # this worker, so sharing the actor's GPUs with vLLM is safe.
        "reward_model.enable=True",
        f"reward_model.enable_resource_pool={f(bool(g.enable_resource_pool))}",
        f"reward_model.model.path={g.reward_model_path}",
        f"reward_model.micro_batch_size_per_gpu={f(g.reward_micro_batch_size_per_gpu)}",
        f"trainer.critic_warmup={f(g.critic_warmup)}",
        f"trainer.logger={g.logger}",
        f"trainer.project_name={g.project_name}",
        f"trainer.experiment_name={exp}",
        f"trainer.n_gpus_per_node={f(g.n_gpus_per_node)}",
        f"trainer.nnodes={f(g.nnodes)}",
        f"trainer.save_freq={f(g.save_freq)}",
        f"trainer.esi_redundant_time={f(g.esi_redundant_time)}",
        f"trainer.test_freq={f(g.test_freq)}",
        f"trainer.total_epochs={f(g.total_epochs)}",
        f"trainer.total_training_steps={f(g.total_steps)}",
        # A stable name per version, never a timestamp: resume_mode=auto continues
        # from the same directory across 4h windows.
        f"trainer.default_local_dir={ckpt / exp}",
        f"trainer.resume_mode={g.resume_mode}",
    ]
    # Not set in the final runs (verl defaults); smoke configs use them.
    if g.model_dtype is not None:
        ov += [f"actor_rollout_ref.actor.fsdp_config.model_dtype={g.model_dtype}",
               f"actor_rollout_ref.ref.fsdp_config.model_dtype={g.model_dtype}"]
    if int(g.train_max_samples) != -1:
        ov.append(f"data.train_max_samples={int(g.train_max_samples)}")
    ov += [str(x) for x in (g.extra_overrides or [])]
    forbidden = set(g.forbidden_overrides or [])
    bad = [o for o in ov if o.split("=", 1)[0].lstrip("+~") in forbidden]
    if bad:
        raise a2d_config.ConfigError(f"tau2 synrl overrides set forbidden key(s): {bad}")
    ov.append(f"hydra.run.dir={Path(os.path.abspath(run_dir)) / 'hydra'}")
    ov += [f"{RUNTIME_ENV_PREFIX}{k}={hydra_string(val)}" for k, val in (runtime_env or {}).items()]
    return ov


def worker_env(cfg: DictConfig, version: str, svc_url: str,
               wall_clock_end: int | None = None) -> dict[str, str]:
    """Run-specific variables every Ray worker needs (reward worker settings, ESI)."""
    from ask2distill.opd.launch import ESI_ENV
    env = tau2_env.render(cfg, "synrl", version)
    env["TAU2_SVC"] = svc_url
    if wall_clock_end is not None:
        env[ESI_ENV] = str(int(wall_clock_end))
    return env


def build_env(cfg: DictConfig, worker: Mapping[str, str]) -> dict[str, str | None]:
    """Driver / private-Ray-head environment: process hygiene plus the worker variables."""
    env = runtime.base_env(stage="tau2_synrl")
    env.update(worker)
    return env


def service_env(cfg: DictConfig, eps: list[E.Endpoint]) -> dict[str, str]:
    """``TAU2_{STUDENT,USER,TEACHER}_BASE`` from the endpoints file."""
    re_ = cfg.reward_env
    need = ["stu", "usr"] + (["tea"] if int(re_.TAU2_TEACHER_GATE) == 1 else [])
    names = {"stu": str(re_.TAU2_STUDENT_MODEL), "usr": str(re_.TAU2_USER_MODEL),
             "tea": str(re_.TAU2_TEACHER_MODEL)}
    out = {}
    for kind, var in (("stu", "TAU2_STUDENT_BASE"), ("usr", "TAU2_USER_BASE"), ("tea", "TAU2_TEACHER_BASE")):
        urls = E.urls(eps, kind)
        if kind in need and not urls:
            raise SystemExit(f"endpoints file has no ready {kind!r} server (need {need})")
        if urls and E.served_name(eps, kind) != names[kind]:
            raise SystemExit(f"{kind} is served as {E.served_name(eps, kind)!r}, the reward expects "
                             f"{names[kind]!r} (reward_env)")
        out[var] = ",".join(urls)
    return out


def _service_port(port: int) -> int:
    if runtime.port_free(port):
        return port
    ports = runtime.job_ports()
    for p in range(ports["serve_port_end"], ports["serve_port_start"] - 1, -1):
        if runtime.port_free(p):
            print(f"[tau2-synrl] port {port} is taken; env service on {p}", flush=True)
            return p
    raise RuntimeError("no free port for the env service")


def missing_inputs(cfg: DictConfig, version: str) -> list[str]:
    v = tau2_env.version_config(cfg, version)
    paths = [str(v["prompts"]), str(cfg.tau2_data.domain_ctx)]
    missing = [p for p in paths if not Path(p).is_file()]
    if not Path(str(v["init"])).is_dir():
        missing.append(str(v["init"]))
    return missing


def run(cfg: DictConfig, version: str, endpoints_file: str | None, *,
        wall_clock_end: int | None = None, dry_run: bool = False) -> int:
    """Synthesizer RL for one version; returns the trainer's exit code."""
    from ask2distill.opd.launch import slurm_wall_clock_end
    for section in ("models", "tau2_data", "reward_env", "synrl"):
        a2d_config.require(cfg, section)
    v = tau2_env.version_config(cfg, version)
    missing = missing_inputs(cfg, version)
    if missing:
        msg = "MISSING INPUTS: " + ", ".join(missing)
        if not dry_run:
            print(msg, flush=True)
            return 1
        print(f"[dry-run] {msg}", flush=True)
    run_dir = runtime.new_run_dir(cfg, f"tau2_synrl_{version}")
    end = wall_clock_end if wall_clock_end is not None else slurm_wall_clock_end()
    if end is None:
        print("[tau2-synrl] no wall-clock end time: no save-before-kill; a restart loses up to "
              "save_freq steps", flush=True)
    port = int(cfg.synrl.env_service.port) if dry_run else _service_port(int(cfg.synrl.env_service.port))
    worker = worker_env(cfg, version, f"http://127.0.0.1:{port}", end)
    overrides = build_overrides(cfg, version, run_dir, worker)
    env = build_env(cfg, worker)
    cmd = [sys.executable, "-m", ENTRY_MODULE, *overrides]
    runtime.write_run_record(run_dir, env, cmd)
    print(f"[tau2-synrl] {version}: exp {v['experiment']}, reward v{v['reward_v']}, init {v['init']}, "
          f"prompt len {v['max_prompt_length']}, run dir {run_dir}", flush=True)
    if dry_run:
        print(f"[dry-run] env service: role synrl, reward v{v['reward_v']}, port {port}, "
              f"workers {v['svc_workers']}", flush=True)
        with runtime.ray_session(int(cfg.synrl.n_gpus_per_node) * int(cfg.synrl.nnodes), env, dry_run=True):
            return runtime.run(cmd, env, dry_run=True)
    if not endpoints_file:
        raise SystemExit("a2d tau2 synrl needs --endpoints (written by `a2d tau2 serve --plan synrl`)")
    svc_extra = service_env(cfg, E.read(endpoints_file))
    Path(str(v["dump"])).parent.mkdir(parents=True, exist_ok=True)
    svc = spawn.start_env_service(cfg, "synrl", port=port, workers=int(v["svc_workers"]),
                                  extra=svc_extra, version=version, log=run_dir / "env_service.log")
    print(f"[tau2-synrl] env service ready on :{port}", flush=True)
    try:
        with runtime.ray_session(int(cfg.synrl.n_gpus_per_node) * int(cfg.synrl.nnodes), env):
            rc = runtime.run(cmd, env, log=run_dir / "synrl.log", cwd=run_dir)
    finally:
        spawn.stop(svc)
    print(f"[tau2-synrl] exit code {rc}", flush=True)
    return rc


def main(cfg: DictConfig, args: argparse.Namespace) -> int:
    """``a2d tau2 synrl --version vX --endpoints FILE [--wall-clock-end TS]``."""
    version = getattr(args, "version", None)
    if not version:
        raise SystemExit("a2d tau2 synrl needs --version")
    return run(cfg, version, getattr(args, "endpoints", None),
               wall_clock_end=getattr(args, "wall_clock_end", None),
               dry_run=bool(getattr(args, "dry_run", False)))
