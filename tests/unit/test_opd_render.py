"""``opd.launch.build_overrides``: the OPD command line of every arm (every pair x ours / vanilla).

Paths are normalised (<MODELS>/, <DATA>/<file>, <EVAL>/<task>, <CKPT>/,
<TTRL>, <RULE_REWARD>, <AGENT_LOOPS>); the non-semantic ``hydra.*`` /
``+ray_kwargs.*`` tail is checked separately. The full override lists are
pinned in ``tests/unit/test_golden_overrides.py``.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

import pytest

from ask2distill.paths import REPO_ROOT

from conftest import (
    AGENT_QWEN3_4B, CODE_CODER_32B, CODE_QWEN3_30B, CODE_QWEN3_4B, MATH_QWEN3_1P7B, MATH_R1_1P5B,
    MATH_R1_7B,
)

ARMS = [(MATH_QWEN3_1P7B, "ours"), (MATH_QWEN3_1P7B, "vanilla"), (MATH_R1_1P5B, "ours"), (MATH_R1_1P5B, "vanilla"),
        (MATH_R1_7B, "ours"), (MATH_R1_7B, "vanilla"), (CODE_QWEN3_30B, "ours"), (CODE_QWEN3_30B, "vanilla"),
        (CODE_CODER_32B, "ours"), (CODE_CODER_32B, "vanilla"), (CODE_QWEN3_4B, "ours"), (CODE_QWEN3_4B, "vanilla"),
        (AGENT_QWEN3_4B, "ours"), (AGENT_QWEN3_4B, "vanilla")]
IDS = [f"{p}-{a}" for p, a in ARMS]
TAIL_PREFIXES = ("hydra.", "+ray_kwargs.")


@pytest.fixture
def a2d_dirs(tmp_path, monkeypatch):
    dirs = {k: tmp_path / k for k in ("data", "models", "ckpt", "out", "cache")}
    for name, var in (("data", "A2D_DATA"), ("models", "A2D_MODELS"), ("ckpt", "A2D_CKPT"),
                      ("out", "A2D_OUT"), ("cache", "A2D_CACHE")):
        monkeypatch.setenv(var, str(dirs[name]))
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    return dirs


_PARQUET = re.compile(r"(?:(?<=[=\[\"])|(?<=, ))(/[^\s,\"\]]*/)([^/\s,\"\]]+\.parquet)")


def _norm_parquets(s: str, eval_prefixes: tuple[str, ...]) -> str:
    for pre in eval_prefixes:
        s = re.sub(re.escape(pre) + r"/([^/]+)/test\.parquet", r"<EVAL>/\1", s)
    return _PARQUET.sub(lambda m: f"<DATA>/{m.group(2)}", s)


def norm_new(items: list[str], dirs: dict) -> list[str]:
    out = []
    for s in items:
        s = s.replace(str(REPO_ROOT / "ask2distill" / "synrl" / "rule_reward.py"), "<RULE_REWARD>")
        s = s.replace(str(REPO_ROOT / "third_party" / "verl" / "verl" / "utils" / "reward_score"
                          / "ttrl_math" / "__init__.py"), "<TTRL>")
        s = s.replace(str(REPO_ROOT / "ask2distill" / "tau2" / "trainside" / "agent_loops.yaml"),
                      "<AGENT_LOOPS>")
        s = s.replace(f"{dirs['models']}/", "<MODELS>/")
        s = s.replace(f"{dirs['ckpt']}/", "<CKPT>/")
        out.append(_norm_parquets(s, (f"{dirs['data']}/eval/math",)))
    return out


def new_overrides(pair: str, arm: str, dirs: dict, run_dir: Path,
                  runtime_env: dict | None = None) -> tuple[list[str], list[str]]:
    """(normalised semantic overrides, raw tail) for one arm."""
    from ask2distill import config
    from ask2distill.opd.launch import build_overrides
    ov = build_overrides(config.load(pair), arm, run_dir, runtime_env)
    body = [o for o in ov if not o.startswith(TAIL_PREFIXES)]
    tail = [o for o in ov if o.startswith(TAIL_PREFIXES)]
    return norm_new(body, dirs), tail


@pytest.mark.parametrize("pair,arm", ARMS, ids=IDS)
def test_tail(pair, arm, a2d_dirs, tmp_path):
    run_dir = tmp_path / "run"
    _, tail = new_overrides(pair, arm, a2d_dirs, run_dir, {"TAU2_USER_BASE": "http://a:1/v1,http://b:2/v1"})
    assert tail == [f"hydra.run.dir={run_dir / 'hydra'}",
                    f"+ray_kwargs.ray_init.runtime_env.env_vars.SWANLAB_LOG_DIR={a2d_dirs['ckpt'] / 'swanlab_log'}",
                    "+ray_kwargs.ray_init.runtime_env.env_vars.TAU2_USER_BASE='http://a:1/v1,http://b:2/v1'"]


@pytest.mark.parametrize("pair,expected", [(MATH_QWEN3_1P7B, 8192), (MATH_R1_1P5B, 8192), (MATH_R1_7B, 8192), (CODE_QWEN3_30B, 6144),
                                           (CODE_CODER_32B, 6144), (CODE_QWEN3_4B, 6144), (AGENT_QWEN3_4B, 14336)])
def test_max_model_len(pair, expected, a2d_dirs, tmp_path):
    from ask2distill import config
    from ask2distill.opd.launch import build_overrides
    for arm in ("ours", "vanilla"):
        ov = build_overrides(config.load(pair), arm, tmp_path)
        assert f"actor_rollout_ref.rollout.max_model_len={expected}" in ov


def test_val_files_rendering(a2d_dirs, tmp_path):
    """Code validates on its own training parquet (quoted, as TEST_FILE spelled it);
    math and tau2 on the math sets (unquoted, as the launcher default rendered)."""
    from ask2distill import config
    from ask2distill.opd.launch import build_overrides
    val = lambda pair, arm: next(o for o in build_overrides(config.load(pair), arm, tmp_path)  # noqa: E731
                                 if o.startswith("data.val_files="))
    data = a2d_dirs["data"]
    assert val(CODE_QWEN3_4B, "vanilla") == f'data.val_files=["{data}/real/OpenCodeReasoning-10k.parquet"]'
    math = ", ".join(f"{data}/eval/math/{t}/test.parquet" for t in ("AIME25", "AMC23", "AIME24"))
    assert val(MATH_QWEN3_1P7B, "ours") == f"data.val_files=[{math}]"
    assert val(AGENT_QWEN3_4B, "ours") == f"data.val_files=[{math}]"


def test_fmt_and_hydra_string():
    from ask2distill.opd.launch import fmt, hydra_string
    assert [fmt(x) for x in (1e-6, 0.005, 0, 1.0, 0.8, True, False, None, 16384, "token-mean")] == \
           ["1e-6", "0.005", "0", "1.0", "0.8", "True", "False", "null", "16384", "token-mean"]
    hydra = pytest.importorskip("hydra.core.override_parser.overrides_parser")
    parser = hydra.OverridesParser.create()
    for value in ("/a/b/c.json", "http://127.0.0.1:8711", "1700000000", "0.9", "True", "null",
                  "http://a:1/v1,http://b:2/v1", "/x:/y", "it's", "a b", ""):
        got = parser.parse_override(f"k={hydra_string(value)}").value()
        assert got == value and isinstance(got, str), (value, got)
    assert hydra_string("/a/b/c.json") == "/a/b/c.json"


def test_smoke_caps_and_extra_overrides(a2d_dirs, tmp_path):
    from ask2distill import config
    from ask2distill.opd.launch import build_overrides
    ov = build_overrides(config.load(None, str(REPO_ROOT / "configs/smoke/math_smoke.yaml")), "ours", tmp_path)
    body = [o for o in ov if not o.startswith(TAIL_PREFIXES)]
    assert body[-4:] == ["data.train_max_samples=64", "data.val_max_samples=8",
                         "trainer.total_training_steps=2", "+trainer.max_actor_ckpt_to_keep=1"]
    assert "trainer.n_gpus_per_node=2" in body and "data.train_batch_size=8" in body
    assert not any("max_samples" in o for o in build_overrides(config.load(MATH_QWEN3_1P7B), "ours", tmp_path))


def test_forbidden_override_is_refused(a2d_dirs, tmp_path):
    from omegaconf import OmegaConf

    from ask2distill import config
    from ask2distill.opd.launch import build_overrides
    cfg = config.load(AGENT_QWEN3_4B)
    OmegaConf.update(cfg, "opd.extra_overrides", ["trainer.rollout_data_dir=/x"], merge=False)
    with pytest.raises(config.ConfigError, match="forbidden"):
        build_overrides(cfg, "ours", tmp_path)


def test_unknown_arm(a2d_dirs, tmp_path):
    from ask2distill import config
    from ask2distill.opd.launch import build_overrides
    with pytest.raises(config.ConfigError, match="unknown OPD arm"):
        build_overrides(config.load(MATH_QWEN3_1P7B), "sg", tmp_path)


def test_esi(a2d_dirs, monkeypatch, capsys):
    from ask2distill import config
    from ask2distill.opd import launch
    assert launch.esi_env(config.load(MATH_QWEN3_1P7B), "ours", 1234567890) == {launch.ESI_ENV: "1234567890"}
    assert launch.esi_env(config.load(CODE_QWEN3_30B), "ours", 1234567890) == {}
    assert "DISABLED" in capsys.readouterr().out
    monkeypatch.setattr(launch, "slurm_wall_clock_end", lambda job_id=None: None)
    assert launch.esi_env(config.load(MATH_QWEN3_1P7B), "vanilla", None) == {}
    assert launch.remaining_seconds("1-02:03:04") == 93784
    assert launch.remaining_seconds("03:04") == 184
    assert launch.remaining_seconds("") is None


def test_missing_data_gate(a2d_dirs, capsys):
    from ask2distill import config
    from ask2distill.opd import launch
    cfg = config.load(MATH_QWEN3_1P7B)
    assert launch.run(cfg, "ours") == 1
    assert "MISSING DATA" in capsys.readouterr().out
    missing = launch.missing_inputs(cfg, "ours")
    assert len(missing) == 4 and missing[0].endswith(f"banks/{MATH_QWEN3_1P7B}.parquet")
    # code validates on its training file, so only that one is required
    assert len(launch.missing_inputs(config.load(CODE_QWEN3_30B), "vanilla")) == 1


def test_dry_run_writes_run_record(a2d_dirs, capsys):
    from ask2distill import config
    from ask2distill.opd import launch
    cfg = config.load(MATH_R1_1P5B)
    tracker = a2d_dirs["ckpt"] / f"opd_{MATH_R1_1P5B}_ours" / launch.TRACKER
    tracker.parent.mkdir(parents=True)
    tracker.write_text("120\n")
    rc = launch.main(cfg, argparse.Namespace(arm="ours", dry_run=True, wall_clock_end=1700000000))
    assert rc == 0
    out = capsys.readouterr().out
    assert "RESUMING from step 120" in out and "MISSING DATA" in out
    runs = list((a2d_dirs["out"] / MATH_R1_1P5B / "runs").iterdir())
    assert len(runs) == 1 and runs[0].name.startswith("opd_ours_")
    cmd = shlex.split((runs[0] / "command.txt").read_text())
    assert cmd[1:3] == ["-m", "verl.trainer.main_ppo"]
    assert "trainer.save_freq=10" in cmd and "reward_model.micro_batch_size_per_gpu=4" in cmd
    assert f"+ray_kwargs.ray_init.runtime_env.env_vars.{launch.ESI_ENV}='1700000000'" in cmd
    env = json.loads((runs[0] / "env.json").read_text())
    assert env[launch.ESI_ENV] == "1700000000"
    assert env["SWANLAB_MODE"] == "local" and env["NCCL_DEBUG"] == "WARN"
    assert env.get("PYTORCH_CUDA_ALLOC_CONF") is None
    assert (runs[0] / "resolved_config.yaml").is_file()


def test_tau2_main_points_to_tau2_launcher(a2d_dirs):
    from ask2distill import config
    from ask2distill.opd import launch
    with pytest.raises(config.ConfigError, match="a2d tau2 opd"):
        launch.main(config.load(AGENT_QWEN3_4B), argparse.Namespace(arm="ours", dry_run=True))
