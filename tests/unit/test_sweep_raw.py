"""The sweep writer keeps raw text; the harvest writer strips thinking blocks.

"""
from __future__ import annotations

import json

import pytest

from ask2distill.harvest import harvest as H
from ask2distill.select import sweep as S


RAW = ["<think>plan the problem</think>\n\nFind the sum of all primes below 30.",
       "  plain question without tags?  ",
       "<think>never closed, the budget ran out",
       "",
       "   ",
       "Q with a stray </think> close in the middle",
       "<think>a</think>"]


def test_sweep_keeps_raw_text():
    rows = S.sweep_records(RAW)
    assert [r["problem"] for r in rows] == [
        "<think>plan the problem</think>\n\nFind the sum of all primes below 30.",
        "plain question without tags?",
        "<think>never closed, the budget ran out",
        "Q with a stray </think> close in the middle",
        "<think>a</think>"]


def test_harvest_strips_thinking():
    rows = H.harvest_records(RAW)
    assert [r["problem"] for r in rows] == [
        "Find the sum of all primes below 30.",
        "plain question without tags?",
        "close in the middle"]


def test_writers_are_distinct():
    # The sweep must never share the harvest's cleaning helper.
    assert S.sweep_records is not H.harvest_records
    assert "strip_question" not in S.__dict__ and "strip_thinking" not in S.__dict__


def test_seed_formulas():
    math_cfg = {"harvest": {"seed_formula": "index"}}
    code_cfg = {"harvest": {"seed_formula": "7919*i+104729"}}
    assert [H.seed_for(math_cfg, i) for i in range(4)] == [0, 1, 2, 3]
    assert [H.seed_for(code_cfg, i) for i in (0, 1, 11)] == [104729, 112648, 191838]
    assert H.seed_for({"seed_formula": "index"}, 7) == 7
    with pytest.raises(Exception):
        H.seed_for({"seed_formula": "time"}, 0)


# ------------------------------------------------------------------ fakes for vLLM / HF


def test_sweep_done_rule(tmp_path):
    p = tmp_path / "sweep_step5.jsonl"
    p.write_text("".join(json.dumps({"problem": str(i)}) + "\n" for i in range(270)))
    assert S.is_done(p, 300)
    p.write_text("".join(json.dumps({"problem": str(i)}) + "\n" for i in range(269)))
    assert not S.is_done(p, 300)
    assert not S.is_done(tmp_path / "missing.jsonl", 300)
