"""Hydra entry point for tau2 synthesizer RL: verl's GRPO trainer with the tau2 reward worker.

Run as ``python -m ask2distill.tau2.trainside.synrl_main <hydra overrides>``;
the overrides are rendered by ``synrl_launch.build_overrides``.

The worker swap has to happen inside the Ray actor: verl runs its TaskRunner in
a separate process that re-imports verl cleanly, so a patch applied in the
driver is invisible there and training silently proceeds against the stock
RewardModelWorker.

The synthesizer writes one JSON card per rollout -- a single turn. The
multi-turn agentic part happens inside the reward (the env service), not in
the actor's rollout, so no agent loop is needed here (that is only for the
downstream OPD run).

The hydra config directory is resolved from the installed verl package.
"""
from __future__ import annotations

import os

import hydra
import ray
import verl
from verl.trainer.main_ppo import TaskRunner, run_ppo

VERL_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl.__file__)), "trainer", "config")


class Tau2TaskRunner(TaskRunner):
    """verl's TaskRunner with ``RewardModelWorker`` replaced by ``Tau2RewardWorker``."""

    def run(self, config):
        import verl.workers.fsdp_workers as fw

        from ask2distill.tau2.trainside.synrl_worker import Tau2RewardWorker
        fw.RewardModelWorker = Tau2RewardWorker
        print(f"[tau2-synrl] in-actor patch: RewardModelWorker -> {Tau2RewardWorker.__name__}",
              flush=True)
        return super().run(config)


@hydra.main(config_path=VERL_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config) -> None:
    """Compose verl's ``ppo_trainer`` config with the CLI overrides and train the synthesizer."""
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(Tau2TaskRunner))


if __name__ == "__main__":
    main()
