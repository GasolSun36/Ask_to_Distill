"""Per-job Ray, port blocks, the process environment and the subprocess runner.

No Ray is started: subprocess calls are mocked and the tests check what would
be run (or that nothing is run when attaching to an existing cluster).
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from ask2distill import runtime as R
from ask2distill.paths import get_paths

from conftest import MATH_QWEN3_1P7B, MATH_R1_7B


def test_job_ports_deterministic_and_disjoint():
    a = R.job_ports("22334455", check_free=False)
    assert a == R.job_ports("22334455", check_free=False)
    assert a["base"] == 20000 + (22334455 % 40) * 1000
    assert a["ray_port"] == a["base"] and a["max_worker_port"] == a["base"] + 999
    blocks = [R.job_ports(str(j), check_free=False) for j in range(100, 140)]
    ranges = [set(range(b["base"], b["base"] + R.PORT_BLOCK)) for b in blocks]
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            assert not ranges[i] & ranges[j]
    for b in blocks:
        vals = [v for k, v in b.items() if k != "base"]
        assert all(b["base"] <= v < b["base"] + R.PORT_BLOCK for v in vals)
        assert len({b["VLLM_PORT"], b["MASTER_PORT"], b["ray_port"], b["node_manager_port"],
                    b["object_manager_port"], b["dashboard_agent_listen_port"],
                    b["ray_client_server_port"]}) == 7
        assert max(vals) < 60000


def test_job_ports_non_numeric_key_is_stable():
    assert R.job_ports("abc123ef", check_free=False) == R.job_ports("abc123ef", check_free=False)


def test_job_ports_skips_busy_block(monkeypatch):
    first = R.job_ports("5", check_free=False)
    monkeypatch.setattr(R, "port_free", lambda p, host="0.0.0.0": p != first["ray_port"])
    got = R.job_ports("5")
    assert got["base"] == first["base"] + R.PORT_BLOCK


@pytest.mark.parametrize("key", ["22334455", "1234567890123", "deadbeef"])
def test_ray_temp_dir_short(monkeypatch, key):
    monkeypatch.setenv("USER", "a_rather_long_user_name_here")
    d = R.ray_temp_dir(key)
    assert d.startswith("/tmp/ray_") and len(d) < 40


def test_job_key_prefers_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    assert R.job_key() == "777"
    monkeypatch.delenv("SLURM_JOB_ID")
    k = R.job_key()
    assert k == R.job_key() and len(k) == 8


def _forbid_subprocess(monkeypatch):
    def boom(*a, **k):
        raise AssertionError(f"unexpected subprocess call: {a}")
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(R, "stop_own_head", boom)


def test_ray_session_attach_only(monkeypatch):
    _forbid_subprocess(monkeypatch)
    monkeypatch.setenv("RAY_ADDRESS", "10.0.0.1:20000")
    with R.ray_session(8, {}) as addr:
        assert addr == "10.0.0.1:20000"
    monkeypatch.delenv("RAY_ADDRESS")
    with R.ray_session(8, {"RAY_ADDRESS": "10.0.0.2:21000"}) as addr:
        assert addr == "10.0.0.2:21000"


def test_ray_session_dry_run_starts_nothing(monkeypatch, capsys):
    _forbid_subprocess(monkeypatch)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "22334455")
    with R.ray_session(2, {}, dry_run=True) as addr:
        assert addr.endswith(f":{20000 + (22334455 % 40) * 1000}")
    out = capsys.readouterr().out
    assert "ray start --head" in out and "--temp-dir=/tmp/ray_" in out and "--force" not in out


def test_ray_session_private_head(monkeypatch):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "22334456")
    monkeypatch.setattr(R, "port_free", lambda p, host="0.0.0.0": True)
    calls, stopped = [], []

    def fake_run(cmd, env=None, check=False, **kw):
        calls.append((cmd, env))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(R, "stop_own_head", lambda d, grace_s=15.0: stopped.append(d))
    with R.ray_session(4, {"FOO": "1"}, cuda_visible="1") as addr:
        import os
        assert os.environ["RAY_ADDRESS"] == addr
    assert "RAY_ADDRESS" not in __import__("os").environ
    start = calls[0][0]
    assert start[:3] == ["ray", "start", "--head"]
    assert "--num-gpus=4" in start and f"--temp-dir={R.ray_temp_dir('22334456')}" in start
    assert calls[0][1]["FOO"] == "1" and calls[0][1]["CUDA_VISIBLE_DEVICES"] == "1"
    assert all("stop" not in c[0] for c in calls)              # never `ray stop`
    assert stopped == [R.ray_temp_dir("22334456")]


def test_stop_own_head_only_matches_temp_dir(monkeypatch):
    killed = []
    monkeypatch.setattr(R, "_own_ray_pids", lambda d: [11, 12] if d == "/tmp/ray_u_1" else [99])
    monkeypatch.setattr(R.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(R.Path, "exists", lambda self: False)
    R.stop_own_head("/tmp/ray_u_1", grace_s=0.1)
    assert killed == [11, 12]


def test_base_env(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    paths = get_paths()
    g = R.base_env(paths, stage="synrl")
    o = R.base_env(paths, stage="opd")
    assert g["VLLM_ATTENTION_BACKEND"] == "FLASH_ATTN" and "VLLM_ATTENTION_BACKEND" not in o
    for env in (g, o):
        assert env["PYTORCH_CUDA_ALLOC_CONF"] is None          # inherited expandable_segments is unset
        assert env["NCCL_TIMEOUT"] == "7200" and env["TORCH_NCCL_BLOCKING_WAIT"] == "1"
        assert env["RAY_memory_usage_threshold"] == "0.99" and env["HYDRA_FULL_ERROR"] == "1"
        assert env["HF_HOME"] == str(paths.cache / "hf")
        assert env["OUTLINES_CACHE_DIR"].startswith(str(paths.cache / "outlines"))
    assert R.base_env(paths, stage="tau2_synrl")["TOKENIZERS_PARALLELISM"] == "false"
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF")
    assert "PYTORCH_CUDA_ALLOC_CONF" not in R.base_env(paths, stage="opd")
    assert R.merged_env({"PATH": None, "X": 1})["X"] == "1"
    assert "PATH" not in R.merged_env({"PATH": None})


def test_run_dry_run_and_log(tmp_path, capsys):
    assert R.run(["echo", "hi there"], {"A": "1", "B": None}, dry_run=True) == 0
    out = capsys.readouterr().out
    assert "echo 'hi there'" in out and "A=1" in out and "B=<unset>" in out
    log = tmp_path / "logs" / "x.log"
    rc = R.run([sys.executable, "-c", "import os; print('v=' + os.environ['A2D_T'])"],
               {"A2D_T": "7"}, log=log)
    assert rc == 0 and "v=7" in log.read_text()
    assert R.run([sys.executable, "-c", "raise SystemExit(3)"]) == 3


def test_new_run_dir_and_record(tmp_path, monkeypatch):
    monkeypatch.setenv("A2D_OUT", str(tmp_path))
    from ask2distill.config import load
    cfg = load(MATH_QWEN3_1P7B)
    d = R.new_run_dir(cfg, "synrl")
    assert d.parent == tmp_path / MATH_QWEN3_1P7B / "runs" and (d / "resolved_config.yaml").is_file()
    R.write_run_record(d, {"B": "2", "A": "1"}, ["python", "-m", "x", "a b"])
    assert (d / "command.txt").read_text() == "python -m x 'a b'\n"
    assert list(__import__("json").loads((d / "env.json").read_text())) == ["A", "B"]


def test_mem_preflight(tmp_path, monkeypatch):
    import json
    model = tmp_path / "models" / "Qwen3-1.7B"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"hidden_size": 2048, "num_hidden_layers": 28,
                                                    "intermediate_size": 6144, "vocab_size": 151936}))
    monkeypatch.setenv("A2D_MODELS", str(tmp_path / "models"))
    from ask2distill.config import load
    est = R.mem_preflight(load(MATH_QWEN3_1P7B), "opd")
    assert est["gpus"] == 8 and est["dtype"] == "fp32" and est["logits_gb"] > 0
    est = R.mem_preflight(load(MATH_QWEN3_1P7B), "synrl")
    assert est["sharded_state_gb"] > 0 and est["total_gb"] >= est["logits_gb"]
    assert "note" in R.mem_preflight(load(MATH_R1_7B), "opd")             # model not downloaded


def test_shell_exports_agree_with_python(monkeypatch):
    monkeypatch.setattr(R, "port_free", lambda p, host="0.0.0.0": True)
    out = R.shell_exports("synrl", "22334455")
    ports = R.job_ports("22334455")
    assert f"export A2D_PORT_RAY_PORT={ports['ray_port']}" in out
    assert f"export VLLM_PORT={ports['VLLM_PORT']}" in out
    assert f"export A2D_RAY_TMPDIR={R.ray_temp_dir('22334455')}" in out
    assert "export VLLM_ATTENTION_BACKEND=FLASH_ATTN" in out
    assert "expandable_segments" not in out


def test_eval_math_slot_env_distinct_vllm_ports():
    """Side-by-side math eval engines must not share VLLM_PORT (a smoke run:
    MATH500 died with EADDRINUSE on the port AIME24 had just bound)."""
    from ask2distill.eval.math.run import slot_env
    block = R.job_ports("22334457", check_free=False)
    envs = [slot_env(str(g), g, str(block["VLLM_PORT"])) for g in range(8)]
    ports = [int(e["VLLM_PORT"]) for e in envs]
    assert len(set(ports)) == 8 and ports[0] == block["VLLM_PORT"]
    assert all(block["VLLM_PORT"] <= p < block["MASTER_PORT"] for p in ports)
    assert [e["CUDA_VISIBLE_DEVICES"] for e in envs] == [str(g) for g in range(8)]
    assert envs[0]["TOKENIZERS_PARALLELISM"] == "false"
    assert "VLLM_PORT" not in slot_env("0", 0, None)        # no block: vLLM picks a free port
