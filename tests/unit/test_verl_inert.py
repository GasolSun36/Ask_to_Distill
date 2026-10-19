"""The later verl change (token ledger; THIRD_PARTY.md change 4) is inert by default.

It must stay off unless a config opts in: the rollout config defaults keep the original top-k
strategy, no shipped trainer YAML sets an opt-in key, and the code reads every opt-in key with
an off default. Static checks only (no verl import, CPU).
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

REPO = Path(os.path.abspath(__file__)).parents[2]
VERL = REPO / "third_party" / "verl" / "verl"
OPT_IN_KEYS = ("token_budget", "log_token_counts")


def _dataclass_defaults(path: Path, cls: str) -> dict[str, object]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            out: dict[str, object] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                    try:
                        out[stmt.target.id] = ast.literal_eval(stmt.value)
                    except ValueError:
                        pass  # field(default_factory=...) and similar
            return out
    raise AssertionError(f"class {cls} not found in {path}")


def test_rollout_config_defaults_keep_new_branches_off() -> None:
    d = _dataclass_defaults(VERL / "workers" / "config" / "rollout.py", "RolloutConfig")
    assert d["top_k_strategy"] == "only_stu"
    assert d["reward_weight_mode"] == "student_p"
    assert d["log_prob_top_k"] == 256
    assert d["teacher_temperature"] == 1.0


@pytest.mark.parametrize("yaml_name", ["ppo_trainer.yaml", "_generated_ppo_trainer.yaml"])
def test_trainer_yaml_sets_no_opt_in_key(yaml_name: str) -> None:
    text = (VERL / "trainer" / "config" / yaml_name).read_text()
    for key in OPT_IN_KEYS:
        assert key not in text, f"{yaml_name} sets {key}"


def test_opt_in_keys_are_read_with_off_defaults() -> None:
    trainer = (VERL / "trainer" / "ppo" / "ray_trainer.py").read_text()
    assert 'self.config.trainer.get("token_budget", None)' in trainer
    assert 'self.config.trainer.get("log_token_counts", False)' in trainer


def test_vendored_verl_is_clean() -> None:
    root = VERL.parent
    assert (VERL / "models" / "transformers" / "monkey_patch.py").is_file()
    assert (root / "setup.py").is_file() and (VERL / "version" / "version").read_text().strip() == "0.7.0.dev"
    # No *.bak* backup may ship next to an edited file.
    leftovers = [p for p in root.rglob("*.bak*")]
    assert not leftovers, leftovers[:5]
