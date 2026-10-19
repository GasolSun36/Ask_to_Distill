"""Calibration reproduces a code pair's δ_ref within 5% (initial-synthesizer questions).

Code_Qwen3-4B_Qwen3-4B-Instruct-2507 on 2 GPUs (teacher on ``cuda:1``): the initial
synthesizer writes N_cal = 200 questions, the student answers each once under the
4096-token budget, δ_ref is the 95th percentile of δ_t. vLLM sampling is not
bit-reproducible across versions and batch shapes, so the criterion is
|δ_ref - expected| / expected < 5%, not equality. About 20 minutes; run inside a
GPU job, never on a login node.
"""
from __future__ import annotations

import os

import pytest

from conftest import CODE_QWEN3_4B


@pytest.mark.gpu
@pytest.mark.equiv
def test_calibration_reproduces_delta_ref(tmp_path):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    from ask2distill import config
    from ask2distill.calib.calibrate import run
    cfg = config.load(CODE_QWEN3_4B, overrides=[f"calib.questions={tmp_path / 'q.jsonl'}"])
    for m in (cfg.models.synthesizer_init, cfg.models.student, cfg.models.teacher):
        if not os.path.isdir(str(m)):
            pytest.skip(f"model not downloaded: {m}")
    out = run(cfg, tmp_path / "calib_delta_ref.json")
    expected = float(cfg.calib.expected_delta_ref)
    assert out["n_questions"] == 200
    assert abs(out["delta_ref_rounded"] - expected) / expected < 0.05
    assert (tmp_path / "calib_delta_ref.json").is_file()
