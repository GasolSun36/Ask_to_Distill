"""Launch one tau2 OPD arm: env-only service + verl OPD with the multi-turn agent loop.

Qwen3-4B student <- Qwen3-4B-Thinking-2507 teacher, both non-thinking.
The per-token objective is the math / code OPD objective unchanged (see
``trainside.agent_loop`` and docs/TAU2.md, "Agent-specific OPD"): multi-turn
rollouts come from ``trainside.agent_loop`` through verl's async agent loop,
and the teacher's top-k signal only lands on tokens the student generated
(``response_mask``), never on tool output or user turns. The OPD launcher
itself is the shared ``ask2distill.opd.launch``, configured exactly as for the
other domains; this module adds what the tau2 arm needs around it:

1. the user-simulator endpoints (kind ``usr`` of the file written by
   ``a2d tau2 serve --plan opd``) become ``TAU2_USER_BASE`` and a
   ``TAU2_USER_BASE_FILE`` holding the comma list -- on a failed simulator call
   the agent loop re-reads that file for up to ``TAU2_USER_RETRY_S`` seconds;
2. ``envside.env_service`` starts as an env-only service (no student / user
   endpoints: only ``/env/*`` is used) with ``opd.env_service.workers`` threads;
3. ``opd.launch.run`` runs with the agent-loop settings (``opd.agent_env``,
   ``TAU2_CTX``, the guidelines, the service and simulator URLs) delivered to
   every AgentLoopWorker through ``runtime_env``.

The agent-loop hydra keys (async rollout, ``tau2_agent``, hermes format,
total steps, checkpoints kept) come from ``opd.extra_overrides``;
``trainer.rollout_data_dir`` is forbidden there (verl's ``_dump_generations``
json-dumps a Tensor from the agent-loop batch and crashed the first step).

An arm already at ``opd.arms.<arm>.total_steps`` is skipped.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ask2distill import config as a2d_config
from ask2distill import runtime
from ask2distill.paths import REPO_ROOT
from ask2distill.tau2 import env as tau2_env
from ask2distill.tau2 import spawn
from ask2distill.tau2.envside import endpoints as E


def user_bases(cfg: DictConfig, endpoints_file: str | Path) -> list[str]:
    """Base URLs of the user simulator, checked against the configured served name."""
    eps = E.read(endpoints_file)
    urls = E.urls(eps, "usr")
    if not urls:
        raise SystemExit(f"{endpoints_file}: no ready user-simulator ('usr') endpoint")
    want = str(cfg.reward_env.TAU2_USER_MODEL)
    if E.served_name(eps, "usr") != want:
        raise SystemExit(f"user simulator served as {E.served_name(eps, 'usr')!r}, the agent loop "
                         f"requests {want!r}")
    return urls


def agent_env(cfg: DictConfig, svc_url: str, bases: list[str], base_file: Path) -> dict[str, str]:
    """Every variable the agent loop reads at import, plus PYTHONPATH for the loop module."""
    env = tau2_env.render(cfg, "opd")
    env.update({"TAU2_SVC": svc_url,
                "TAU2_USER_BASE": ",".join(bases),
                "TAU2_USER_BASE_FILE": str(base_file)})
    inherited = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + inherited if inherited else "")
    return env


def _service_port(port: int) -> int:
    if runtime.port_free(port):
        return port
    ports = runtime.job_ports()
    for p in range(ports["serve_port_end"], ports["serve_port_start"] - 1, -1):
        if runtime.port_free(p):
            print(f"[tau2-opd] port {port} is taken; env service on {p}", flush=True)
            return p
    raise RuntimeError("no free port for the env service")


def run(cfg: DictConfig, arm: str, user_endpoints: str | None, *, wall_clock_end: int | None = None,
        dry_run: bool = False) -> int:
    """One OPD arm; returns the trainer's exit code."""
    from ask2distill.opd import launch as opd_launch
    for section in ("models", "tau2_data", "reward_env", "opd"):
        a2d_config.require(cfg, section)
    a = opd_launch.arm_config(cfg, arm)
    total = int(OmegaConf.select(a, "total_steps", default=None) or cfg.opd.total_training_steps)
    step = opd_launch.latest_step(cfg, arm)
    if step is not None and step >= total and not dry_run:
        print(f"[tau2-opd] {a.experiment} already at step {step} >= {total}", flush=True)
        return 0
    port = int(cfg.opd.env_service.port) if dry_run else _service_port(int(cfg.opd.env_service.port))
    svc_url = f"http://127.0.0.1:{port}"
    run_root = Path(str(cfg.paths.out)) / str(cfg.pair) / "opd" / str(a.experiment)
    run_root.mkdir(parents=True, exist_ok=True)
    base_file = run_root / f"user_bases.{runtime.job_key()}.txt"
    if user_endpoints:
        bases = user_bases(cfg, user_endpoints)
    elif dry_run:
        bases = ["http://<user-sim>/v1"]
    else:
        raise SystemExit("a2d tau2 opd needs --user-endpoints (written by `a2d tau2 serve --plan opd`)")
    if not dry_run:
        from ask2distill.common.io import atomic_write_text
        atomic_write_text(base_file, ",".join(bases))
    print(f"[tau2-opd] user simulator: {','.join(bases)}", flush=True)
    rt_env = agent_env(cfg, svc_url, bases, base_file)
    if dry_run:
        print(f"[dry-run] env service: role opd, port {port}, workers {cfg.opd.env_service.workers}",
              flush=True)
        return opd_launch.run(cfg, arm, runtime_env=rt_env, wall_clock_end=wall_clock_end, dry_run=True)
    svc = spawn.start_env_service(cfg, "opd", port=port, workers=int(cfg.opd.env_service.workers),
                                  log=run_root / f"env_service.{runtime.job_key()}.log")
    print(f"[tau2-opd] env service ready on :{port}", flush=True)
    try:
        return opd_launch.run(cfg, arm, runtime_env=rt_env, wall_clock_end=wall_clock_end)
    finally:
        spawn.stop(svc)


def main(cfg: DictConfig, args: argparse.Namespace) -> int:
    """``a2d tau2 opd --arm ours|vanilla --user-endpoints FILE [--wall-clock-end TS]``."""
    return run(cfg, args.arm, getattr(args, "user_endpoints", None),
               wall_clock_end=getattr(args, "wall_clock_end", None),
               dry_run=bool(getattr(args, "dry_run", False)))
