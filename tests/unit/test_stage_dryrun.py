"""Every selection / harvest / filter / bank command resolves its config and dry-runs.

``--dry-run`` reads every config key the stage uses, so a missing or renamed
key surfaces here, on CPU, for all six math/code pairs and both smoke configs.
"""
from __future__ import annotations

import pytest

from ask2distill import cli

from conftest import (
    AGENT_QWEN3_4B, CODE_PAIRS, CODE_QWEN3_30B, MATH_PAIRS, MATH_QWEN3_1P7B, MATH_R1_1P5B,
    MATH_R1_7B,
)

PAIRS = MATH_PAIRS + CODE_PAIRS
COMMANDS = [["select", "sweep"], ["select", "pick"], ["harvest"], ["harvest", "--shard", "1"],
            ["filter"], ["filter", "--slice", "0"], ["aggregate"], ["bank"], ["prompts", "build"]]


@pytest.fixture(autouse=True)
def _isolated_out(tmp_path, monkeypatch):
    monkeypatch.setenv("A2D_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("A2D_CKPT", str(tmp_path / "ckpt"))
    monkeypatch.delenv("A2D_CONFIG", raising=False)


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("command", COMMANDS, ids=" ".join)
def test_dry_run(pair, command, capsys):
    assert cli.main([*command, "--pair", pair, "--dry-run"]) == 0
    assert "# ---- command ----" in capsys.readouterr().out


@pytest.mark.parametrize("pair", [MATH_QWEN3_1P7B, MATH_R1_1P5B])
def test_yield_dry_run(pair, capsys):
    assert cli.main(["select", "yield", "--pair", pair, "--dry-run"]) == 0
    assert "seed 100" in capsys.readouterr().out


@pytest.mark.parametrize("pair", [MATH_R1_7B, CODE_QWEN3_30B])
def test_yieldelta_refused_without_candidates(pair):
    with pytest.raises(SystemExit):
        cli.main(["select", "yield", "--pair", pair, "--dry-run"])


@pytest.mark.parametrize("smoke", ["configs/smoke/math_smoke.yaml", "configs/smoke/code_smoke.yaml"])
@pytest.mark.parametrize("command", [["select", "sweep"], ["harvest"], ["filter"], ["aggregate"],
                                     ["bank"]], ids=" ".join)
def test_smoke_dry_run(smoke, command):
    assert cli.main([*command, "--config", smoke, "--dry-run"]) == 0


def test_shard_out_of_range():
    with pytest.raises(SystemExit):
        cli.main(["harvest", "--pair", MATH_R1_7B, "--shard", "8", "--dry-run"])


def test_no_dry_run_targets_are_registered():
    targets = {c.target for c in cli.REGISTRY.values()}
    assert cli._NO_DRY_RUN_TARGETS <= targets


@pytest.mark.parametrize("command", [["data", "download", "--from-dir", "/nonexistent"], ["data", "eval"],
                                     ["data", "build", "math"], ["data", "build", "code-eval"],
                                     ["data", "build", "ocr"], ["data", "build", "code-real"],
                                     ["data", "build", "tau2"],
                                     ["collect", "math", "--runs", "/x", "--out", "/y.tsv"],
                                     ["tau2", "summarize", "--pair", AGENT_QWEN3_4B, "--models", "M"],
                                     ["tau2", "select", "--pair", AGENT_QWEN3_4B, "--version", "v7"]],
                         ids=" ".join)
def test_dry_run_never_calls_builders(command, monkeypatch, capsys):
    """Downloads, data builders and table writers do not read args.dry_run themselves."""
    def _called(target):
        raise AssertionError(f"--dry-run resolved {target}")
    monkeypatch.setattr(cli, "resolve_target", _called)
    assert cli.main([*command, "--dry-run"]) == 0
    assert "# ---- command ----" in capsys.readouterr().out
