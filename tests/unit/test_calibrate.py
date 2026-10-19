"""Calibration on CPU: question records, the δ_ref record, the dry run.

The model-dependent part (vLLM sampling, real student / teacher forwards) is in
tests/equiv/test_calib_gpu.py.
"""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from conftest import CODE_QWEN3_4B, MATH_R1_1P5B


def test_record_keys_and_percentile():
    from ask2distill.calib.calibrate import summarize
    from ask2distill.config import ConfigError
    v = np.arange(1, 101, dtype=np.float32)
    out = summarize([v[:40], v[40:]], 95, 4)
    assert out["percentile"] == 95.0 and out["n_rollouts"] == 2 and out["n_tokens"] == 100
    assert out["delta_ref"] == pytest.approx(np.percentile(v, 95))
    assert out["delta_ref_rounded"] == round(out["delta_ref"], 4)
    assert summarize([v], 90, 4)["delta_ref"] == pytest.approx(np.percentile(v, 90))
    with pytest.raises(ConfigError):
        summarize([v], 100, 4)
    with pytest.raises(RuntimeError):
        summarize([], 95, 4)


def test_question_records_clean_like_the_reward_worker():
    from ask2distill.calib.calibrate import question_records
    texts = ["<think>plan</think>\n\nFind all n with n^2+1 | 2^n+1.", "  short  ",
             "<think>never closed", "Count the lattice paths that avoid the diagonal."]
    rows = question_records(texts, ["a", "b", "c", "d"], 20)
    assert rows == [{"problem": "Find all n with n^2+1 | 2^n+1.", "topic": "a"},
                    {"problem": "Count the lattice paths that avoid the diagonal.", "topic": "d"}]


@pytest.mark.parametrize("pair", [CODE_QWEN3_4B, MATH_R1_1P5B])
def test_dry_run_runs_nothing(pair, capsys):
    from ask2distill import config
    from ask2distill.calib.calibrate import main
    cfg = config.load(pair)
    assert main(cfg, argparse.Namespace(dry_run=True)) == {}
    printed = capsys.readouterr().out
    assert f"would sample 200 questions from {cfg.models.synthesizer_init}" in printed
    assert str(cfg.synrl.topics) in printed and "percentile 95" in printed
    if pair == CODE_QWEN3_4B:
        assert "cuda:1" in printed and "max_new 4096" in printed
    else:
        assert "SynNoThink" in printed                   # the DeepSeek pairs sample their no-think base


def test_existing_questions_are_reused(tmp_path, monkeypatch):
    from ask2distill import config
    from ask2distill.calib import calibrate as C
    from ask2distill.common.io import write_jsonl
    q = tmp_path / "calib_questions.jsonl"
    write_jsonl(q, [{"problem": f"question number {i} of the calibration set", "topic": "t"}
                    for i in range(5)])
    cfg = config.load(CODE_QWEN3_4B, overrides=[f"calib.questions={q}", "calib.n_questions=4"])
    monkeypatch.setattr(C, "sample_questions", lambda cfg: pytest.fail("must not sample"))
    assert C.ensure_questions(cfg) == [f"question number {i} of the calibration set" for i in range(4)]
