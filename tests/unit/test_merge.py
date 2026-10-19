"""``ask2distill.merge``: layout, lock / wait protocol, chat-template copy, login-node guard.

The verl merger itself is replaced by a fake that writes the files a finished
merge leaves behind; no model is loaded.
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import pytest

from ask2distill import merge
from ask2distill.paths import REPO_ROOT

from conftest import AGENT_QWEN3_4B, CODE_QWEN3_4B, MATH_QWEN3_1P7B


@pytest.fixture
def a2d_dirs(tmp_path, monkeypatch):
    dirs = {k: tmp_path / k for k in ("data", "models", "ckpt", "out", "cache")}
    for name, var in (("data", "A2D_DATA"), ("models", "A2D_MODELS"), ("ckpt", "A2D_CKPT"),
                      ("out", "A2D_OUT"), ("cache", "A2D_CACHE")):
        monkeypatch.setenv(var, str(dirs[name]))
    monkeypatch.setattr(merge, "on_login_node", lambda: False)
    return dirs


def _write_model(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (target / name).write_text("{}")


@pytest.fixture
def fake_merger(monkeypatch):
    calls = []

    def fake_run(cmd, env=None, log=None, dry_run=False, cwd=None):
        calls.append(list(cmd))
        if dry_run:
            return 0
        target = Path(cmd[cmd.index("--target_dir") + 1])
        assert merge.lock_path(target).is_dir(), "merger ran without holding the lock"
        _write_model(target)
        return 0

    monkeypatch.setattr(merge.runtime, "run", fake_run)
    return calls


def _actor(tmp_path: Path) -> Path:
    d = tmp_path / "ckpt" / "exp" / "global_step_5" / "actor"
    d.mkdir(parents=True)
    return d


def test_layout(a2d_dirs):
    from ask2distill import config
    cfg = config.load(MATH_QWEN3_1P7B)
    exp = f"opd_{MATH_QWEN3_1P7B}_ours"
    assert merge.hf_dir(cfg, exp, 150) == a2d_dirs["ckpt"] / exp / "hf" / "step_150"
    assert merge.actor_dir(cfg, exp, 150) == a2d_dirs["ckpt"] / exp / "global_step_150" / "actor"
    assert merge.lock_path(Path("/x/hf/step_150")) == Path("/x/hf/step_150.lock")


def test_holder_merges_and_releases_lock(a2d_dirs, tmp_path, fake_merger):
    actor = _actor(tmp_path)
    target = tmp_path / "ckpt" / "exp" / "hf" / "step_5"        # parent hf/ does not exist yet
    assert merge.merge_fsdp(actor, target) == target
    assert merge.is_merged(target) and not merge.lock_path(target).exists()
    cmd = fake_merger[0]
    assert cmd[1:6] == ["-m", "verl.model_merger", "merge", "--backend", "fsdp"]
    assert cmd[cmd.index("--local_dir") + 1] == str(actor)
    merge.merge_fsdp(actor, target)                               # second call: already merged
    assert len(fake_merger) == 1


def test_failed_merge_raises_and_releases_lock(a2d_dirs, tmp_path, monkeypatch):
    monkeypatch.setattr(merge.runtime, "run", lambda *a, **k: 3)
    target = tmp_path / "hf" / "step_5"
    with pytest.raises(merge.MergeError, match="MERGE_FAILED"):
        merge.merge_fsdp(_actor(tmp_path), target)
    assert not merge.lock_path(target).exists()


def test_waiter_returns_when_holder_finishes(a2d_dirs, tmp_path, monkeypatch, fake_merger):
    monkeypatch.setattr(merge, "_POLL_S", 0.05)
    target = tmp_path / "hf" / "step_5"
    lock = merge.lock_path(target)
    lock.mkdir(parents=True)

    def holder():
        time.sleep(0.3)
        _write_model(target)
        lock.rmdir()

    t = threading.Thread(target=holder)
    t.start()
    assert merge.merge_fsdp(_actor(tmp_path), target, timeout_s=10) == target
    t.join()
    assert fake_merger == []                                      # the waiter never merged


def test_waiter_detects_dead_holder_and_timeout(a2d_dirs, tmp_path, monkeypatch, fake_merger):
    monkeypatch.setattr(merge, "_POLL_S", 0.05)
    monkeypatch.setattr(merge, "_HOLDER_GRACE_S", 0)
    actor = _actor(tmp_path)
    target = tmp_path / "hf" / "step_5"
    lock = merge.lock_path(target)
    lock.mkdir(parents=True)
    with pytest.raises(merge.MergeError, match="MERGE_WAIT_TIMEOUT"):
        merge.merge_fsdp(actor, target, timeout_s=0.3)
    threading.Timer(0.2, lock.rmdir).start()
    with pytest.raises(merge.MergeError, match="MERGE_HOLDER_GONE"):
        merge.merge_fsdp(actor, target, timeout_s=10)


def test_missing_checkpoint(a2d_dirs, tmp_path, fake_merger):
    with pytest.raises(FileNotFoundError, match="NO_SUCH_CHECKPOINT"):
        merge.merge_fsdp(tmp_path / "nope" / "actor", tmp_path / "hf" / "step_1")


def test_login_node_guard(a2d_dirs, tmp_path, monkeypatch, fake_merger):
    monkeypatch.setattr(merge, "on_login_node", lambda: True)
    actor, target = _actor(tmp_path), tmp_path / "hf" / "step_5"
    with pytest.raises(merge.MergeError, match="login node"):
        merge.merge_fsdp(actor, target)
    assert merge.merge_fsdp(actor, target, allow_login_node=True) == target
    merge.merge_fsdp(actor, target)               # an existing merge is reused even there


def test_on_login_node_respects_slurm(monkeypatch):
    monkeypatch.setattr(merge, "_cgroup_memory_limits", lambda: [24 * 2**30])
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert merge.on_login_node()
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    assert not merge.on_login_node()
    monkeypatch.delenv("SLURM_JOB_ID")
    monkeypatch.setattr(merge, "_cgroup_memory_limits", lambda: [512 * 2**30])
    assert not merge.on_login_node()


def test_chat_template_by_role(a2d_dirs, tmp_path, fake_merger):
    from ask2distill import config
    cfg = config.load(AGENT_QWEN3_4B)
    template = merge.template_for_role(cfg, "synthesizer")
    assert template == REPO_ROOT / "ask2distill" / "tau2" / "templates" / "Qwen3-4B.nothink.jinja"
    with pytest.raises(SystemExit, match="null"):
        merge.template_for_role(cfg, "user")
    with pytest.raises(SystemExit, match="no models.templates"):
        merge.template_for_role(config.load(MATH_QWEN3_1P7B), "student")

    src = tmp_path / "Qwen3-4B.nothink.jinja"
    src.write_text("{{ template }}")
    target = merge.merge_fsdp(_actor(tmp_path), tmp_path / "hf" / "step_5", chat_template=src)
    assert (target / "Qwen3-4B.nothink.jinja").read_text() == "{{ template }}"
    with pytest.raises(FileNotFoundError, match="thinking"):
        merge.merge_fsdp(_actor(tmp_path / "b"), tmp_path / "hf" / "step_6",
                         chat_template=tmp_path / "missing.jinja")


def test_main_resolves_pair_arm_step(a2d_dirs, fake_merger, capsys):
    from ask2distill import config
    cfg = config.load(CODE_QWEN3_4B)
    actor = a2d_dirs["ckpt"] / f"opd_{CODE_QWEN3_4B}_ours" / "global_step_150" / "actor"
    actor.mkdir(parents=True)
    args = argparse.Namespace(arm="ours", step=150, actor_dir=None, target_dir=None,
                              chat_template=None, force=False, dry_run=False)
    assert merge.main(cfg, args) == 0
    assert merge.is_merged(a2d_dirs["ckpt"] / f"opd_{CODE_QWEN3_4B}_ours" / "hf" / "step_150")
    args.step = 95
    assert merge.main(cfg, args) == 1
    assert "NO_SUCH_CHECKPOINT" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        merge.main(None, argparse.Namespace(arm=None, step=None, actor_dir="/a", target_dir=None,
                                            chat_template=None, force=False, dry_run=False))
