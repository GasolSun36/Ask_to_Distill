"""Stage 1 launcher: render the synthesizer-RL hydra overrides and run ``synrl.main``.

``build_overrides`` emits verl's hydra overrides (pinned by
``tests/golden/overrides.json``), then two non-semantic groups:

* ``hydra.run.dir=<run_dir>/hydra`` -- otherwise hydra writes ``outputs/<date>/``
  into the current directory;
* ``+ray_kwargs.ray_init.runtime_env.env_vars.<K>=<V>`` -- run-specific
  environment. verl merges it into ``ray.init`` (main_ppo.py), so it reaches
  every worker on every node, including the reward workers that read
  ``$A2D_REWARD_CONFIG`` (the resolved ``reward`` section, written here as
  ``<run_dir>/reward_config.json``). sbatch therefore never needs to know the
  timestamped run directory.

Layout. S_θ trains on ``synrl.layout.actor_gpus`` GPUs (the global pool,
node 0); the frozen student and teacher sit in their own reward pool
(``reward_gpus`` x ``reward_nnodes``). ``reward_model.enable_resource_pool`` is
what puts the reward worker on its own GPUs; without it student and teacher
would share memory with the trained model -- and the reward worker's
expandable-segments allocator setting would break the synthesizer's vLLM
rollout (config.validate refuses that layout). The step's questions must
split evenly over the reward ranks; in ``bleu`` diversity mode the ranks
exchange their question texts so the share covers the whole step.

The prompt parquet (``synrl.prompts``) is built from the config when it does
not exist yet (``bank.build_synrl_prompts``).

Ray. Inside an sbatch-built multi-node cluster (``$RAY_ADDRESS`` set by
``launchers/slurm/common.sh``) this only attaches. Opening with
``ray stop --force`` and starting a fresh head would tear down a cluster the
job had just built on that node, so nothing here ever stops a cluster it did
not start (``runtime.ray_session``).

The real-question arm runs as its own job (``a2d opd --arm vanilla``), not
inside the synthesizer-RL job.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ask2distill import config as a2d_config
from ask2distill import runtime
from ask2distill.synrl.reward_config import REWARD_CONFIG_ENV, check_reward_config
from ask2distill.paths import get_paths

ENTRY_MODULE = "ask2distill.synrl.main"
RUNTIME_ENV_PREFIX = "+ray_kwargs.ray_init.runtime_env.env_vars."
# verl's checkpoint manager force-saves ahead of this unix time when it is set
# (checkpoint_manager.py). The final synthesizer runs did not use it.
ESI_ENV = "MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"


def fmt(value: Any) -> str:
    """A config value spelled as the final runs' shell launcher spelled it.

    Booleans as ``True``/``False``, floats in Python's shortest form with a
    two-digit exponent shortened (``1e-06`` -> ``1e-6``), everything else ``str``.
    """
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        s = repr(value)
        if "e" in s:
            mant, exp = s.split("e")
            sign = "-" if exp.startswith("-") else ""
            s = f"{mant}e{sign}{exp.lstrip('+-').lstrip('0') or '0'}"
        return s
    return str(value)


def reward_config_path(run_dir: Path) -> Path:
    """``<run_dir>/reward_config.json`` (absolute)."""
    return Path(os.path.abspath(run_dir)) / "reward_config.json"


def write_reward_config(cfg: DictConfig, run_dir: Path) -> Path:
    """Write the resolved ``reward`` section for the reward workers and check it.

    δ_ref is taken from the config as is; ``config.validate`` already insists it
    is rounded to 4 decimals, and ``check_reward_config`` refuses an unset one.
    """
    path = a2d_config.section_json(cfg, "reward", reward_config_path(run_dir))
    check_reward_config(json.loads(path.read_text()))
    return path


def runtime_env_vars(cfg: DictConfig, run_dir: Path, wall_clock_end: int | None = None
                     ) -> dict[str, str]:
    """Run-specific environment delivered to every Ray worker."""
    env = {REWARD_CONFIG_ENV: str(reward_config_path(run_dir))}
    if wall_clock_end is not None:
        env[ESI_ENV] = str(int(wall_clock_end))
    return env


def build_overrides(cfg: DictConfig, run_dir: Path, wall_clock_end: int | None = None
                    ) -> list[str]:
    """Hydra overrides for ``python -m ask2distill.synrl.main``, in the final runs' order."""
    g, m, r = cfg.synrl, cfg.models, cfg.reward
    lay = g.layout
    ov = [
        "algorithm.adv_estimator=grpo",
        f"custom_reward_function.path={g.custom_reward.path}",
        f"custom_reward_function.name={g.custom_reward.name}",
        f"trainer.val_before_train={fmt(bool(g.val_before_train))}",
        f"data.shuffle={fmt(bool(g.shuffle))}",
        f"data.train_files={g.prompts}",
        # never scored (test_freq -1, val_before_train False); it only has to load
        f"data.val_files={g.prompts}",
        f"data.train_batch_size={fmt(g.train_batch_size)}",
        f"data.max_prompt_length={fmt(g.max_prompt_length)}",
        f"data.max_response_length={fmt(g.max_response_length)}",
        f"data.filter_overlong_prompts={fmt(bool(g.filter_overlong_prompts))}",
        f"data.truncation={g.truncation}",
        f"data.return_raw_chat={fmt(bool(g.return_raw_chat))}",
    ]
    # Qwen3 takes enable_thinking through the template kwargs; DeepSeek has no
    # such switch and passing it raises, so it is emitted for Qwen3 pairs only.
    if m.qwen3_nothink:
        ov.append("+data.apply_chat_template_kwargs.enable_thinking=False")
    offload = fmt(bool(g.offload))
    ov += [
        f"actor_rollout_ref.model.path={m.synthesizer_init}",
        f"actor_rollout_ref.model.use_remove_padding={fmt(bool(g.use_remove_padding))}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={fmt(bool(g.gradient_checkpointing))}",
        f"actor_rollout_ref.actor.optim.lr={fmt(g.lr)}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={fmt(g.ppo_mini_batch_size)}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={fmt(g.ppo_micro_batch_size_per_gpu)}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={fmt(g.ppo_max_token_len_per_gpu)}",
        f"actor_rollout_ref.actor.entropy_coeff={fmt(g.entropy_coeff)}",
        f"actor_rollout_ref.actor.use_kl_loss={fmt(bool(g.use_kl_loss))}",
        f"actor_rollout_ref.actor.kl_loss_coef={fmt(g.kl_loss_coef)}",
        f"actor_rollout_ref.actor.kl_loss_type={g.kl_loss_type}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={offload}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={offload}",
        f"actor_rollout_ref.actor.fsdp_config.model_dtype={g.model_dtype}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.temperature={fmt(g.rollout_temperature)}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={fmt(g.rollout_tp)}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={fmt(g.gpu_memory_utilization)}",
        # vLLM caps the WHOLE sequence, so this has to cover prompt + response.
        f"actor_rollout_ref.rollout.max_model_len={g.max_prompt_length + g.max_response_length}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={fmt(g.max_num_batched_tokens)}",
        f"actor_rollout_ref.rollout.n={fmt(g.rollout_n)}",
        f"actor_rollout_ref.ref.fsdp_config.param_offload={fmt(bool(g.ref_param_offload))}",
        "reward_model.enable=True",
        f"reward_model.enable_resource_pool={fmt(bool(lay.enable_resource_pool))}",
        f"reward_model.n_gpus_per_node={fmt(lay.reward_gpus)}",
        f"reward_model.nnodes={fmt(lay.reward_nnodes)}",
        # The worker loads no reward model of its own; the path only has to name
        # a real model directory, and the teacher is the natural one.
        f"reward_model.model.path={r.teacher}",
        f"reward_model.micro_batch_size_per_gpu={fmt(g.reward_micro_batch_size_per_gpu)}",
        f"trainer.critic_warmup={fmt(g.critic_warmup)}",
        f"trainer.logger={g.logger}",
        f"trainer.project_name={g.project_name}",
        f"trainer.experiment_name={g.experiment}",
        f"trainer.n_gpus_per_node={fmt(lay.actor_gpus)}",
        "trainer.nnodes=1",                   # S_theta always trains on one node
        f"trainer.save_freq={fmt(g.save_freq)}",
        f"trainer.test_freq={fmt(g.test_freq)}",
        f"trainer.total_epochs={fmt(g.total_epochs)}",
        f"trainer.total_training_steps={fmt(g.total_steps)}",
        f"trainer.default_local_dir={g.ckpt_dir}",
        f"trainer.resume_mode={g.resume_mode}",
    ]
    ov += [str(x) for x in (g.extra_overrides or [])]
    forbidden = set(g.forbidden_overrides or [])
    bad = [o for o in ov if o.split("=", 1)[0].lstrip("+~") in forbidden]
    if bad:
        raise a2d_config.ConfigError(f"synrl overrides set forbidden key(s): {bad}")
    ov.append(f"hydra.run.dir={Path(os.path.abspath(run_dir)) / 'hydra'}")
    # Ray wants string env values; hydra would parse the ESI timestamp as an int.
    from ask2distill.opd.launch import hydra_string
    ov += [f"{RUNTIME_ENV_PREFIX}{k}={hydra_string(v)}"
           for k, v in runtime_env_vars(cfg, run_dir, wall_clock_end).items()]
    return ov


def build_env(cfg: DictConfig, run_dir: Path, wall_clock_end: int | None = None
              ) -> dict[str, str | None]:
    """Process environment of the training driver (also exported before ``ray start``)."""
    env = runtime.base_env(get_paths(), stage="synrl")
    env.update(runtime_env_vars(cfg, run_dir, wall_clock_end))
    return env


def num_gpus(cfg: DictConfig) -> int:
    """GPUs a private single-node Ray head must offer for this layout."""
    lay = cfg.synrl.layout
    return int(lay.actor_gpus) + int(lay.reward_gpus) * int(lay.reward_nnodes)


def main(cfg: DictConfig, args: argparse.Namespace | None = None) -> int:
    """``a2d synrl --pair P``: train the question synthesizer for a math/code pair."""
    dry_run = bool(getattr(args, "dry_run", False))
    wall_clock_end = getattr(args, "wall_clock_end", None)
    if cfg.domain not in ("math", "code"):
        raise a2d_config.ConfigError(f"`a2d synrl` is for math/code pairs; {cfg.pair} is "
                                     f"{cfg.domain} (use `a2d tau2 synrl`)")
    for section in ("models", "reward", "synrl"):
        a2d_config.require(cfg, section)
    prompts = Path(str(cfg.synrl.prompts))
    if not prompts.is_file():
        from ask2distill.bank import build_synrl_prompts
        if dry_run:
            build_synrl_prompts.check(cfg)
            print(f"[synrl] {prompts} absent: would build it first (`a2d prompts build`)", flush=True)
        else:
            build_synrl_prompts.build(cfg)
    run_dir = runtime.new_run_dir(cfg, "synrl")
    write_reward_config(cfg, run_dir)
    Path(str(cfg.reward.dump)).parent.mkdir(parents=True, exist_ok=True)
    overrides = build_overrides(cfg, run_dir, wall_clock_end)
    env = build_env(cfg, run_dir, wall_clock_end)
    cmd = [sys.executable, "-m", ENTRY_MODULE, *overrides]
    runtime.write_run_record(run_dir, env, cmd)
    gpus = num_gpus(cfg)
    attached = bool(os.environ.get("RAY_ADDRESS"))
    if not attached and gpus > 8 and not dry_run:
        raise RuntimeError(
            f"this layout needs {gpus} GPUs on more than one node; start the multi-node Ray "
            "cluster first (launchers/slurm/synrl_2node.sbatch exports RAY_ADDRESS)")
    print(f"[synrl] run dir {run_dir}", flush=True)
    with runtime.ray_session(gpus, env, dry_run=dry_run):
        return runtime.run(cmd, env, log=run_dir / "synrl.log", dry_run=dry_run, cwd=run_dir)
