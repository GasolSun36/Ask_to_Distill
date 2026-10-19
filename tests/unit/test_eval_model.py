"""``a2d eval math|code --model``: evaluate a ready HF model (a released student) out of the box.

* ``--model`` alone, or ``--arm`` with ``--step``; never both, never neither;
* results are named after the repo id, or after a local directory plus a hash of its path;
* ``gasolsun/<pair>`` names its own pair, so ``--pair`` can be left out;
* a complete local copy is reused, nothing is downloaded in a dry run, and no merge runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from ask2distill import cli
from ask2distill.eval import model_source as M

from conftest import CODE_QWEN3_30B, MATH_QWEN3_1P7B


@pytest.fixture
def a2d_dirs(tmp_path, monkeypatch):
    dirs = {k: tmp_path / k for k in ("data", "models", "ckpt", "out", "cache")}
    for name, var in (("data", "A2D_DATA"), ("models", "A2D_MODELS"), ("ckpt", "A2D_CKPT"),
                      ("out", "A2D_OUT"), ("cache", "A2D_CACHE")):
        monkeypatch.setenv(var, str(dirs[name]))
    monkeypatch.delenv("A2D_CONFIG", raising=False)
    return dirs


@pytest.fixture
def no_download(monkeypatch):
    import huggingface_hub

    def fail(**kw):
        raise AssertionError(f"unexpected download {kw}")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)


def _write_model(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (target / name).write_text("{}")
    return target


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**{"model": None, "arm": None, "step": None, **kw})


def test_model_or_checkpoint_not_both():
    assert M.check_args(_ns(model="gasolsun/x")) == "gasolsun/x"
    assert M.check_args(_ns(arm="ours", step=10)) is None
    for bad in (_ns(), _ns(arm="ours"), _ns(model="gasolsun/x", arm="ours"), _ns(model="gasolsun/x", step=10)):
        with pytest.raises(SystemExit):
            M.check_args(bad)


def test_model_names(tmp_path):
    assert M.model_name(f"gasolsun/{MATH_QWEN3_1P7B}") == f"gasolsun__{MATH_QWEN3_1P7B}"
    a, b = _write_model(tmp_path / "a" / "hf"), _write_model(tmp_path / "b" / "hf")
    assert M.model_name(str(a)).startswith("hf-") and M.model_name(str(a)) != M.model_name(str(b))
    with pytest.raises(SystemExit):
        M.model_name("not a repo id")


def test_pair_of_released_model(tmp_path):
    assert M.pair_of(f"gasolsun/{CODE_QWEN3_30B}") == CODE_QWEN3_30B
    assert M.pair_of(str(tmp_path / MATH_QWEN3_1P7B) + "/") == MATH_QWEN3_1P7B
    assert M.pair_of("someone/other-model") is None


def test_resolve_reuses_complete_copies(a2d_dirs, no_download):
    from ask2distill import config
    cfg = config.load(MATH_QWEN3_1P7B)
    local = _write_model(a2d_dirs["models"] / "elsewhere")
    assert M.resolve(cfg, str(local)) == local.resolve()
    repo = f"gasolsun/{MATH_QWEN3_1P7B}"
    copy = a2d_dirs["models"] / f"gasolsun__{MATH_QWEN3_1P7B}"
    assert M.resolve(cfg, repo, dry_run=True) == copy          # dry run: nothing fetched
    _write_model(copy)
    assert M.resolve(cfg, repo) == copy                        # complete copy: reused
    (local / "model.safetensors").unlink()
    with pytest.raises(SystemExit):
        M.resolve(cfg, str(local))                             # incomplete local dir


def test_eval_math_dry_run_infers_pair(a2d_dirs, no_download, capsys):
    assert cli.main(["eval", "math", "--model", f"gasolsun/{MATH_QWEN3_1P7B}", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"eval/math/gasolsun__{MATH_QWEN3_1P7B}" in out and "global_step" not in out


def test_eval_code_dry_run_names_lcb_run_after_model(a2d_dirs, no_download, capsys):
    assert cli.main(["eval", "code", "--model", f"gasolsun/{CODE_QWEN3_30B}", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert f"tag=gasolsun__{CODE_QWEN3_30B} " in out
    assert f"--model gasolsun__{CODE_QWEN3_30B} " in out


def test_eval_needs_model_or_checkpoint(a2d_dirs):
    with pytest.raises(SystemExit):
        cli.main(["eval", "math", "--pair", MATH_QWEN3_1P7B, "--dry-run"])
