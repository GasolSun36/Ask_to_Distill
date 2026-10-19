"""tau2 launch rendering: synthesizer-RL overrides, TAU2_* environments, spawn env, OPD agent env.

Values are checked against the settings of the final runs; the full override
lists are pinned in ``tests/unit/test_golden_overrides.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex

import pytest

from ask2distill.paths import REPO_ROOT

from conftest import AGENT_QWEN3_4B

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


@pytest.fixture
def cfg(a2d_dirs):
    from ask2distill import config
    return config.load(AGENT_QWEN3_4B)


# ------------------------------------------------------------------ pure rendering

def test_synrl_v7_values(cfg, a2d_dirs, tmp_path):
    from ask2distill.tau2.trainside.synrl_launch import build_overrides
    ov = build_overrides(cfg, "v7", tmp_path / "run", {"TAU2_SVC": "http://127.0.0.1:8710"})
    assert "data.max_prompt_length=10240" in ov
    assert "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=11776" in ov
    assert "actor_rollout_ref.rollout.max_model_len=11776" in ov
    assert "actor_rollout_ref.actor.optim.lr=1e-6" in ov and "trainer.total_training_steps=60" in ov
    assert f"actor_rollout_ref.model.path={a2d_dirs['ckpt']}/synthesizer_{AGENT_QWEN3_4B}_v6/hf/step_50" in ov
    assert ov[-2:] == [f"hydra.run.dir={tmp_path / 'run' / 'hydra'}",
                       "+ray_kwargs.ray_init.runtime_env.env_vars.TAU2_SVC=http://127.0.0.1:8710"]
    assert not any("model_dtype" in o or "max_samples" in o for o in ov)


def test_synrl_smoke_extras(a2d_dirs, tmp_path):
    from ask2distill import config
    from ask2distill.tau2.trainside.synrl_launch import build_overrides
    cfg = config.load(None, str(REPO_ROOT / "configs" / "smoke" / "tau2_smoke.yaml"))
    body = [o for o in build_overrides(cfg, "v7", tmp_path) if not o.startswith(TAIL_PREFIXES)]
    assert body[-5:] == ["actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
                         "actor_rollout_ref.ref.fsdp_config.model_dtype=bf16",
                         "data.train_max_samples=4", "+trainer.max_actor_ckpt_to_keep=1",
                         "actor_rollout_ref.rollout.tensor_model_parallel_size=1"]
    assert "trainer.n_gpus_per_node=1" in body and "actor_rollout_ref.rollout.n=4" in body


def test_render_roles(cfg, a2d_dirs):
    from ask2distill.tau2 import env as tau2_env
    common = {"TAU2_CTX": str(a2d_dirs["data"] / "tau2" / "domain_ctx.json"),
              "TAU2_USER_GUIDELINES": str(REPO_ROOT / "third_party" / "toolorchestra_tau2" / "data" / "tau2"
                                          / "user_simulator" / "simulation_guidelines.md"),
              "A2D_TAU2_DB_DIR": str(REPO_ROOT / "third_party" / "toolorchestra_tau2" / "data" / "tau2" / "domains")}
    g = tau2_env.render(cfg, "synrl", "v6")
    assert {k: g[k] for k in common} == common and "TAU2_DATA" not in g
    assert (g["TAU2_REWARD_V"], g["TAU2_GATE_TURNS"], g["TAU2_PREF_TURNS"], g["TAU2_V7_TURNS"],
            g["TAU2_PREF_TEACHER_GATE"], g["TAU2_PREF_JUDGE"], g["TAU2_SVC_TIMEOUT"]) == \
        ("6", "10", "25", "30", "0", "1", "3600")
    assert g["TAU2_DUMP"] == str(a2d_dirs["out"] / AGENT_QWEN3_4B / "synrl" / f"dump_synthesizer_{AGENT_QWEN3_4B}_v6.jsonl")
    h = tau2_env.render(cfg, "harvest")
    assert (h["TAU2_GROUNDING"], h["TAU2_TEACHER_GATE"], h["TAU2_KT"], h["TAU2_MAX_TURNS"]) == ("1", "1", "3", "12")
    assert "TAU2_K" not in h and "TAU2_REWARD_V" not in h
    o = tau2_env.render(cfg, "opd")
    assert {k: o[k] for k in ("TAU2_MAX_TURNS", "TAU2_STOP_ON_REPEAT", "TAU2_REPEAT_SIM", "TAU2_MAX_TOOL_CHARS",
                              "TAU2_USER_RETRY_S", "TAU2_USER_MODEL")} == \
        {"TAU2_MAX_TURNS": "20", "TAU2_STOP_ON_REPEAT": "1", "TAU2_REPEAT_SIM": "0.9",
         "TAU2_MAX_TOOL_CHARS": "2000", "TAU2_USER_RETRY_S": "1800", "TAU2_USER_MODEL": "Qwen3-4B-Instruct-2507"}
    e = tau2_env.render(cfg, "eval")
    assert (e["TAU2_MAX_TOKENS"], e["TAU2_ORCHESTRATOR_PROMPT"], e["TAU2_OPENAI_DIRECT"]) == ("8000", "0", "1")
    with pytest.raises(ValueError):
        tau2_env.render(cfg, "serve")


def test_spawn_env(cfg, a2d_dirs, monkeypatch, tmp_path):
    from ask2distill.tau2 import spawn
    base = {"PATH": "/usr/bin", "TAU2_DATA": "GENRL_TAU2_V7", "TAU2_STOP_ON_REPEAT": "0",
            "PYTHONPATH": "/somewhere/else", "HOME": "/home/x"}
    env = spawn.env(cfg, "harvest", {"X_EXTRA": "1"}, base=base)
    repo_path = str(REPO_ROOT / "third_party" / "toolorchestra_tau2")
    assert env["PYTHONPATH"] == os.pathsep.join([str(REPO_ROOT), repo_path, f"{repo_path}/evaluation/tau2-bench"])
    assert env["REPO_PATH"] == repo_path and env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_HOME"] == str(a2d_dirs["cache"] / "hf")
    assert "TAU2_DATA" not in env and "TAU2_STOP_ON_REPEAT" not in env
    assert env["PATH"] == "/usr/bin" and env["X_EXTRA"] == "1"
    settings = json.loads(env["A2D_TAU2_SETTINGS"])
    assert settings["harvest"]["v7"]["seed0"] == 2500000 and settings["harvest"]["v6"]["per_target"] == 222
    assert settings["harvest"]["gen_sampling"] == {"temperature": 1.0, "max_tokens": 2048}
    ev = json.loads(spawn.env(cfg, "eval", base=base)["A2D_TAU2_SETTINGS"])["eval"]
    assert (ev["trials"], ev["concurrency"], ev["max_steps"], ev["user_llm"]) == (4, 12, 200, "gpt-4o-mini")
    assert "A2D_TAU2_SETTINGS" not in spawn.env(cfg, "synrl", base=base)
    monkeypatch.setenv("A2D_TAU2_PYTHON", "/opt/tau2/bin/python")
    assert spawn.command("harvest", ["--version", "v7"]) == \
        ["/opt/tau2/bin/python", "-m", "ask2distill.tau2.envside.harvest", "--version", "v7"]


def test_synrl_dry_run_records(cfg, a2d_dirs, capsys):
    from ask2distill.tau2.trainside import synrl_launch
    rc = synrl_launch.main(cfg, argparse.Namespace(version="v7", endpoints=None, wall_clock_end=1700000000,
                                                   dry_run=True))
    assert rc == 0
    out = capsys.readouterr().out
    assert "MISSING INPUTS" in out and "reward v7" in out and "workers 64" in out
    runs = list((a2d_dirs["out"] / AGENT_QWEN3_4B / "runs").iterdir())
    cmd = shlex.split((runs[0] / "command.txt").read_text())
    assert cmd[1:3] == ["-m", "ask2distill.tau2.trainside.synrl_main"]
    assert "+ray_kwargs.ray_init.runtime_env.env_vars.TAU2_SVC=http://127.0.0.1:8710" in cmd
    assert "+ray_kwargs.ray_init.runtime_env.env_vars.MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP='1700000000'" in cmd
    env = json.loads((runs[0] / "env.json").read_text())
    assert env["TOKENIZERS_PARALLELISM"] == "false" and env["TAU2_REWARD_V"] == "7"
    assert "VLLM_ATTENTION_BACKEND" not in env


def test_opd_dry_run_runtime_env(cfg, a2d_dirs, capsys):
    from ask2distill.tau2.trainside import opd_launch
    rc = opd_launch.main(cfg, argparse.Namespace(arm="ours", user_endpoints=None, wall_clock_end=1700000000,
                                                 dry_run=True))
    assert rc == 0
    runs = list((a2d_dirs["out"] / AGENT_QWEN3_4B / "runs").iterdir())
    cmd = shlex.split((runs[0] / "command.txt").read_text())
    # values are hydra-quoted when they would not parse back as strings ('20')
    tail = {o.split("=", 1)[0].rsplit(".", 1)[1]: o.split("=", 1)[1].strip("'")
            for o in cmd if o.startswith("+ray_kwargs.ray_init.runtime_env.env_vars.")}
    assert tail["TAU2_SVC"] == "http://127.0.0.1:8711" and tail["TAU2_MAX_TURNS"] == "20"
    assert tail["TAU2_STOP_ON_REPEAT"] == "1" and tail["TAU2_CTX"].endswith("tau2/domain_ctx.json")
    assert "actor_rollout_ref.rollout.agent.default_agent_loop=tau2_agent" in cmd
    assert f"actor_rollout_ref.rollout.agent.agent_loop_config_path={REPO_ROOT}/ask2distill/tau2/trainside/agent_loops.yaml" in cmd


def test_agent_loops_yaml_targets_the_ported_loop():
    import yaml
    (entry,) = yaml.safe_load((REPO_ROOT / "ask2distill" / "tau2" / "trainside" / "agent_loops.yaml").read_text())
    assert entry == {"name": "tau2_agent", "_target_": "ask2distill.tau2.trainside.agent_loop.Tau2AgentLoop"}
