"""``synrl.launch.build_overrides``: the synthesizer-RL command line and its run record.

Paths are normalised to <MODELS>/, <PROMPTS>/, <CKPT>/ and <RULE_REWARD>; the
non-semantic ``hydra.*`` / ``+ray_kwargs.*`` tail (run directory and the
reward-config handle passed through Ray's runtime env) is checked separately.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from conftest import CODE_PAIRS, CODE_QWEN3_4B, MATH_PAIRS, MATH_QWEN3_1P7B

REPO = Path(os.path.abspath(__file__)).parents[2]
PAIRS = MATH_PAIRS + CODE_PAIRS
TAIL_PREFIXES = ("hydra.", "+ray_kwargs.")


@pytest.fixture
def a2d_dirs(tmp_path, monkeypatch):
    dirs = {k: tmp_path / k for k in ("data", "models", "ckpt", "out", "cache")}
    for name, var in (("data", "A2D_DATA"), ("models", "A2D_MODELS"), ("ckpt", "A2D_CKPT"),
                      ("out", "A2D_OUT"), ("cache", "A2D_CACHE")):
        monkeypatch.setenv(var, str(dirs[name]))
    return dirs


def new_overrides(pair: str, dirs: dict, run_dir: Path) -> tuple[list[str], list[str]]:
    """(normalised semantic overrides, tail) for ``pair``."""
    from ask2distill import config
    from ask2distill.synrl.launch import build_overrides
    cfg = config.load(pair)
    ov = build_overrides(cfg, run_dir)
    body = [o for o in ov if not o.startswith(TAIL_PREFIXES)]
    tail = [o for o in ov if o.startswith(TAIL_PREFIXES)]
    rule = str(REPO / "ask2distill" / "synrl" / "rule_reward.py")
    out = []
    for s in body:
        s = s.replace(rule, "<RULE_REWARD>")
        s = s.replace(f"{dirs['models']}/", "<MODELS>/")
        s = s.replace(f"{dirs['data']}/prompts/", "<PROMPTS>/")
        s = s.replace(f"{dirs['ckpt']}/", "<CKPT>/")
        out.append(s)
    return out, tail


@pytest.mark.parametrize("pair", PAIRS)
def test_tail_and_reward_config(pair, a2d_dirs, tmp_path):
    from ask2distill import config
    from ask2distill.synrl.launch import write_reward_config
    from ask2distill.synrl.reward_config import load_reward_config
    run_dir = tmp_path / "run"
    _, tail = new_overrides(pair, a2d_dirs, run_dir)
    rc = run_dir / "reward_config.json"
    assert tail == [f"hydra.run.dir={run_dir / 'hydra'}",
                    f"+ray_kwargs.ray_init.runtime_env.env_vars.A2D_REWARD_CONFIG={rc}"]
    write_reward_config(config.load(pair), run_dir)
    loaded = load_reward_config(rc)
    assert loaded["delta_ref"] == round(loaded["delta_ref"], 4)


def test_fmt_spells_values_like_the_shell_script():
    from ask2distill.synrl.launch import fmt
    assert [fmt(x) for x in (1e-6, 0.005, 0.01, 1.0, 0.6, True, False, 16384, "error")] == \
           ["1e-6", "0.005", "0.01", "1.0", "0.6", "True", "False", "16384", "error"]


def test_forbidden_extra_override_is_refused(a2d_dirs, tmp_path):
    from omegaconf import OmegaConf

    from ask2distill import config
    from ask2distill.synrl.launch import build_overrides
    cfg = config.load(MATH_QWEN3_1P7B)
    OmegaConf.update(cfg, "synrl.extra_overrides", ["trainer.rollout_data_dir=/x"], merge=False)
    OmegaConf.update(cfg, "synrl.forbidden_overrides", ["trainer.rollout_data_dir"], merge=False)
    with pytest.raises(config.ConfigError, match="forbidden"):
        build_overrides(cfg, tmp_path)


def test_wall_clock_end_goes_through_runtime_env(a2d_dirs, tmp_path):
    from ask2distill import config
    from ask2distill.synrl.launch import ESI_ENV, build_overrides
    ov = build_overrides(config.load(MATH_QWEN3_1P7B), tmp_path, wall_clock_end=1234567890)
    # quoted, so hydra keeps it a string (Ray rejects non-string env values)
    assert f"+ray_kwargs.ray_init.runtime_env.env_vars.{ESI_ENV}='1234567890'" in ov
    assert not any(ESI_ENV in o for o in build_overrides(config.load(MATH_QWEN3_1P7B), tmp_path))


def test_dry_run_writes_run_record(a2d_dirs, monkeypatch, capsys):
    import argparse

    from ask2distill import config
    from ask2distill.synrl import launch
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    rc = launch.main(config.load(CODE_QWEN3_4B), argparse.Namespace(dry_run=True, wall_clock_end=None))
    assert rc == 0
    runs = list((a2d_dirs["out"] / CODE_QWEN3_4B / "runs").iterdir())
    assert len(runs) == 1
    cmd = (runs[0] / "command.txt").read_text()
    assert "-m ask2distill.synrl.main" in cmd and "trainer.save_freq=10" in cmd
    rc_json = json.loads((runs[0] / "reward_config.json").read_text())
    assert rc_json["gen_bs"] == 8 and rc_json["delta_ref"] == 3.8221
    env = json.loads((runs[0] / "env.json").read_text())
    assert env["A2D_REWARD_CONFIG"] == str(runs[0] / "reward_config.json")
    assert env["VLLM_ATTENTION_BACKEND"] == "FLASH_ATTN"
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env or env["PYTORCH_CUDA_ALLOC_CONF"] is None
