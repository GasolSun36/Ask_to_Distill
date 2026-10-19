"""Golden trainer command lines: every hydra override the launchers render, pinned.

``tests/golden/overrides.json`` holds, for every OPD arm (7 pairs x ours /
vanilla), every math / code synthesizer-RL run and every tau2 synthesizer version,
the exact override list -- order included -- with the run-specific locations
replaced by placeholders (<DATA>, <MODELS>, <CKPT>, <OUT>, <CACHE>, <REPO>,
<RUN>). The OPD lines are those the released OPD runs were launched with (up to
the renamed synthesizer-RL module path of the null rule reward), so any change to
a config value or to a launcher's rendering shows up here.

After an intended change, regenerate the file and review the diff::

    A2D_UPDATE_GOLDEN=1 pytest tests/unit/test_golden_overrides.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ask2distill.paths import REPO_ROOT

from conftest import AGENT_QWEN3_4B, ALL_PAIRS, CODE_PAIRS, MATH_PAIRS

GOLDEN = REPO_ROOT / "tests" / "golden" / "overrides.json"
TAU2_VERSIONS = ("v4", "v5", "v6", "v7")


def _render_all(tmp: Path) -> dict[str, list[str]]:
    from ask2distill import config
    from ask2distill.synrl.launch import build_overrides as synrl_overrides
    from ask2distill.opd.launch import build_overrides as opd_overrides
    from ask2distill.tau2.trainside.synrl_launch import build_overrides as tau2_synrl_overrides

    run = tmp / "run"
    subs = [(str(run), "<RUN>"), (str(tmp / "data"), "<DATA>"), (str(tmp / "models"), "<MODELS>"),
            (str(tmp / "ckpt"), "<CKPT>"), (str(tmp / "out"), "<OUT>"), (str(tmp / "cache"), "<CACHE>"),
            (str(REPO_ROOT), "<REPO>")]

    def norm(items: list[str]) -> list[str]:
        out = []
        for s in items:
            for a, b in subs:
                s = s.replace(a, b)
            out.append(s)
        return out

    got: dict[str, list[str]] = {}
    for pair in ALL_PAIRS:
        cfg = config.load(pair)
        for arm in ("ours", "vanilla"):
            got[f"opd/{pair}/{arm}"] = norm(opd_overrides(cfg, arm, run))
    for pair in MATH_PAIRS + CODE_PAIRS:
        got[f"synrl/{pair}"] = norm(synrl_overrides(config.load(pair), run))
    cfg = config.load(AGENT_QWEN3_4B)
    for v in TAU2_VERSIONS:
        got[f"tau2_synrl/{AGENT_QWEN3_4B}/{v}"] = norm(tau2_synrl_overrides(cfg, v, run))
    return got


@pytest.fixture
def rendered(tmp_path, monkeypatch):
    for name, var in (("data", "A2D_DATA"), ("models", "A2D_MODELS"), ("ckpt", "A2D_CKPT"),
                      ("out", "A2D_OUT"), ("cache", "A2D_CACHE")):
        monkeypatch.setenv(var, str(tmp_path / name))
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    return _render_all(tmp_path)


def test_overrides_match_golden(rendered):
    if os.environ.get("A2D_UPDATE_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(rendered, indent=1, ensure_ascii=False) + "\n")
        pytest.skip(f"rewrote {GOLDEN}")
    want = json.loads(GOLDEN.read_text())
    assert sorted(rendered) == sorted(want), "rendered run set differs from the golden file"
    for key in want:
        assert rendered[key] == want[key], f"{key}: overrides differ from tests/golden/overrides.json"


def test_golden_is_placeholder_only():
    """No machine-specific location may end up in the golden file."""
    text = GOLDEN.read_text()
    assert "/tmp/" not in text and str(Path.home()) not in text
