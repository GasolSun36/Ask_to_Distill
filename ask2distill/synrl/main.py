"""Hydra entry point for synthesizer RL: verl's GRPO trainer with our reward worker.

Run as ``python -m ask2distill.synrl.main <hydra overrides>``; the overrides are
rendered by ``ask2distill.synrl.launch.build_overrides``.

The substitution has to happen INSIDE the Ray actor, not in the driver. verl
runs its TaskRunner as a Ray actor in a separate process, which re-imports verl
cleanly -- a patch applied in the driver is invisible there, and the first
attempt silently trained against verl's stock RewardModelWorker instead (the
traceback pointed into verl's FSDP worker, at a method our worker does not have).

run_ppo(task_runner_class=...) is the documented hook for recipes to swap the
TaskRunner, so the patch rides inside a subclass that applies it in its own
process before verl resolves any worker class.

The hydra config directory is resolved from the installed verl package rather
than a path relative to this file, so it does not depend on the checkout layout.
"""
from __future__ import annotations

import os

import hydra
import ray
import verl
from verl.trainer.main_ppo import TaskRunner, run_ppo

VERL_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl.__file__)), "trainer", "config")


class SynRLTaskRunner(TaskRunner):
    """verl's TaskRunner with ``RewardModelWorker`` replaced by ``SynRewardWorker``."""

    def run(self, config):
        import verl.workers.fsdp_workers as fw

        from ask2distill.synrl.reward_worker import SynRewardWorker
        fw.RewardModelWorker = SynRewardWorker
        print(f"[synrl] in-actor patch: RewardModelWorker -> {SynRewardWorker.__name__}",
              flush=True)
        return super().run(config)


@hydra.main(config_path=VERL_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config) -> None:
    """Compose verl's ``ppo_trainer`` config with the CLI overrides and train the synthesizer."""
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(SynRLTaskRunner))


if __name__ == "__main__":
    main()
