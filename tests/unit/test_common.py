"""paths, common.io and the a2d command registry / argument parsing."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from ask2distill import cli
from ask2distill import paths as P
from ask2distill.common import io

from conftest import AGENT_QWEN3_4B, CODE_CODER_32B, CODE_QWEN3_4B, MATH_QWEN3_1P7B, MATH_R1_7B

# ------------------------------------------------------------------------------ paths

def test_paths_defaults_under_repo():
    p = P.get_paths({})
    assert p.repo == P.REPO_ROOT and (p.repo / "configs").is_dir()
    assert p.data == p.repo / "data" and p.models == p.repo / "models"
    assert p.ckpt == p.repo / "checkpoints" and p.out == p.repo / "outputs"
    assert p.cache == p.repo / ".cache" and p.third_party == p.repo / "third_party"
    assert p.tau2_python is None and p.hf_data_repo is None and p.openai_env_file is None
    assert all(Path(v).is_absolute() for k, v in p.as_dict().items() if v and k != "hf_data_repo")


def test_paths_env_relative_and_user(monkeypatch):
    monkeypatch.setenv("HOME", "/home/someone")
    p = P.get_paths({"A2D_DATA": "rel/d", "A2D_OUT": "~/o", "A2D_TAU2_PYTHON": "/envs/tau2/bin/python",
                     "A2D_HF_DATA_REPO": "org/release", "A2D_CACHE": ""})
    assert p.data == P.REPO_ROOT / "rel" / "d"
    assert p.out == Path("/home/someone/o")
    assert p.cache == P.REPO_ROOT / ".cache"                   # empty value = default
    assert p.tau2_python == Path("/envs/tau2/bin/python") and p.hf_data_repo == "org/release"


def test_require_tau2_python():
    with pytest.raises(P.PathError, match="A2D_TAU2_PYTHON"):
        P.require_tau2_python(P.get_paths({}))


# --------------------------------------------------------------------------------- io

def test_jsonl_roundtrip_and_atomic(tmp_path):
    rows = [{"problem": "x²", "i": 1}, {"problem": "日本", "i": 2}]
    f = io.write_jsonl(tmp_path / "a" / "b.jsonl", rows)
    assert io.read_jsonl(f) == rows and io.count_lines(f) == 2
    assert "日本" in f.read_text(encoding="utf-8")             # ensure_ascii=False, like the originals
    assert not [x for x in os.listdir(f.parent) if x.endswith(".tmp")]
    assert io.count_lines(tmp_path / "missing") == 0


def test_atomic_write_failure_keeps_old_file(tmp_path, monkeypatch):
    f = tmp_path / "x.txt"
    io.atomic_write_text(f, "old")

    def bad_replace(a, b):
        raise OSError("disk full")
    monkeypatch.setattr(io.os, "replace", bad_replace)
    with pytest.raises(OSError):
        io.atomic_write_text(f, "new")
    assert f.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["x.txt"]


def test_json_md5_parquet(tmp_path):
    f = io.atomic_write_json(tmp_path / "j.json", {"a": [1, 2]})
    assert json.loads(f.read_text()) == {"a": [1, 2]}
    assert io.md5_file(f) == io.md5_text(f.read_text())
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"prompt": [[{"role": "user", "content": "q"}]], "n": [1]})
    out = io.write_parquet(tmp_path / "t.parquet", df)
    back = io.read_parquet(out)
    assert back["n"].tolist() == [1] and back["prompt"][0][0]["content"] == "q"


# -------------------------------------------------------------------------------- cli

def test_registry_targets_are_well_formed():
    for path, cmd in cli.REGISTRY.items():
        module, _, func = cmd.target.partition(":")
        assert module.startswith("ask2distill.") and func, path
    # every command family is present
    names = {" ".join(k) for k in cli.REGISTRY}
    for must in ("models download", "models syn-nothink", "data download", "data eval", "data build math",
                 "data build code-eval", "data build ocr", "data build code-real", "data build tau2",
                 "calibrate", "synrl", "select sweep", "select pick", "select yield",
                 "harvest", "filter", "aggregate", "bank", "prompts build", "opd", "merge",
                 "eval math", "eval code", "collect math", "collect code", "collect tau2",
                 "tau2 build-prompts", "tau2 rows2parquet", "tau2 serve", "tau2 env-service",
                 "tau2 synrl", "tau2 select", "tau2 harvest", "tau2 bank", "tau2 official-split",
                 "tau2 opd", "tau2 eval", "tau2 summarize", "tau2 domain-ctx"):
        assert must in names, must


def test_own_targets_import():
    assert cli.resolve_target("ask2distill.cli:run_envside") is cli.run_envside
    for cmd in cli.REGISTRY.values():
        module = cmd.target.split(":")[0]
        if module.startswith(("ask2distill.cli", "ask2distill.config", "ask2distill.runtime")):
            assert importlib.util.find_spec(module) is not None


def test_parse_overrides_and_flags():
    path, cmd, args = cli.parse(["harvest", "--pair", MATH_QWEN3_1P7B, "--shard", "3", "harvest.per_shard=10",
                                 "+reward.x=1", "--dry-run"])
    assert path == ("harvest",) and args.shard == 3 and args.pair == MATH_QWEN3_1P7B and args.dry_run
    assert args.overrides == ["harvest.per_shard=10", "+reward.x=1"]
    path, cmd, args = cli.parse(["data", "build", "ocr", "--from-dir", "/x"])
    assert path == ("data", "build", "ocr") and args.target == "ocr"
    assert args.from_dir == "/x" and not cmd.needs_config
    path, cmd, args = cli.parse(["data", "download", "--from-dir", "/r", "--banks-dir", "/b"])
    assert (args.from_dir, args.banks_dir) == ("/r", "/b")
    path, cmd, args = cli.parse(["eval", "math", "--pair", MATH_R1_7B, "--arm", "ours", "--step", "150",
                                 "--tasks", "AIME24", "MATH500"])
    assert args.tasks == ["AIME24", "MATH500"] and args.step == 150


def test_parse_rejects_unknown_but_envside_passes_through():
    with pytest.raises(SystemExit):
        cli.parse(["synrl", "--pair", MATH_QWEN3_1P7B, "--no-such-flag"])
    _, cmd, args = cli.parse(["tau2", "harvest", "--pair", AGENT_QWEN3_4B, "--version", "v7",
                              "--endpoints", "/e.txt", "opd.ckpt_keep=2"])
    assert cmd.passthrough and args.passthrough == ["--version", "v7", "--endpoints", "/e.txt"]
    assert args.overrides == ["opd.ckpt_keep=2"] and args.envside_module == "harvest"
    assert args.role == "harvest"
    with pytest.raises(SystemExit):
        cli.parse(["frobnicate"])


def test_load_config_sources(monkeypatch):
    _, cmd, args = cli.parse(["synrl", "--pair", CODE_CODER_32B, "synrl.total_steps=3"])
    cfg = cli.load_config(cmd, args)
    assert cfg.pair == CODE_CODER_32B and cfg.synrl.total_steps == 3
    monkeypatch.setenv("A2D_CONFIG", "configs/smoke/math_smoke.yaml")
    _, cmd, args = cli.parse(["synrl"])
    assert cli.load_config(cmd, args).smoke is True             # sbatch path: A2D_CONFIG only
    monkeypatch.delenv("A2D_CONFIG")
    _, cmd, args = cli.parse(["synrl"])
    with pytest.raises(SystemExit, match="needs --pair"):
        cli.load_config(cmd, args)
    _, cmd, args = cli.parse(["data", "download"])
    assert cli.load_config(cmd, args) is None
    _, cmd, args = cli.parse(["synrl", "--pair", MATH_QWEN3_1P7B, "reward.delta_ref=0.12345"])
    with pytest.raises(SystemExit, match="config error"):
        cli.load_config(cmd, args)


def test_main_dispatch_with_stub_target(monkeypatch, capsys):
    seen = {}

    def stub(cfg, args):
        seen["cfg"], seen["args"] = cfg, args
        return 0

    monkeypatch.setattr(cli, "resolve_target", lambda target: (seen.setdefault("target", target), stub)[1])
    assert cli.main(["calibrate", "--pair", CODE_QWEN3_4B]) == 0
    assert seen["target"] == "ask2distill.calib.calibrate:main"
    with pytest.raises(SystemExit):
        cli.parse(["calibrate", "--pair", CODE_QWEN3_4B, "--build-set"])
    seen.clear()
    assert cli.main(["opd", "--pair", AGENT_QWEN3_4B, "--arm", "ours", "--dry-run"]) == 0
    assert seen["target"] == "ask2distill.tau2.trainside.opd_launch:main"
    out = capsys.readouterr().out
    assert "# ---- resolved config ----" in out and "memory pre-flight" in out
    assert seen["args"].dry_run and seen["cfg"].pair == AGENT_QWEN3_4B


def test_usage_lists_commands():
    with pytest.raises(SystemExit) as e:
        cli.parse([])
    assert "calibrate" in str(e.value) and "tau2 env-service" in str(e.value)
