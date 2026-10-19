"""Slice and aggregation order resolve exactly as the final runs ran them.

Code_Qwen3-4B_Qwen3-4B-Instruct-2507 aggregated its 12 slice files in ``ls`` (lexicographic)
order, s0, s10, s11, s1, ..., s9; every other pair is numeric;
Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500 has no cross-slice step.
"""
from __future__ import annotations


import pytest

from ask2distill import config
from ask2distill.filter.aggregate import ordered_slice_files
from ask2distill.filter.saturation import questions_path, slice_files

from conftest import (
    CODE_CODER_32B, CODE_QWEN3_30B, CODE_QWEN3_4B, MATH_QWEN3_1P7B, MATH_R1_1P5B, MATH_R1_7B,
)


def _names(paths):
    return [p.name for p in paths]


def test_qwen3_4b_code_pair_is_lexicographic():
    cfg = config.load(CODE_QWEN3_4B)
    names = _names(ordered_slice_files(cfg))
    assert names == [f"sat.s{k}_questions.jsonl" for k in (0, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9)]
    # ... which is exactly what `ls` printed for the twelve files
    assert names == sorted(f"sat.s{k}_questions.jsonl" for k in range(12))
    assert [_names(slice_files(cfg, k)) for k in range(12)] == [[f"raw.s{k}.jsonl"] for k in range(12)]


@pytest.mark.parametrize("pair,n", [(MATH_R1_1P5B, 4), (MATH_R1_7B, 2), (CODE_QWEN3_30B, 4), (CODE_CODER_32B, 4)])
def test_other_pairs_are_numeric(pair, n):
    cfg = config.load(pair)
    assert cfg.filter.aggregate is True
    assert _names(ordered_slice_files(cfg)) == [f"sat.s{k}_questions.jsonl" for k in range(n)]


def test_qwen3_math_pair_single_pass_numeric_shards():
    cfg = config.load(MATH_QWEN3_1P7B)
    assert cfg.filter.aggregate is False and len(cfg.filter.slices) == 1
    # numeric shard order: s10 must not come before s2
    assert _names(slice_files(cfg, 0)) == [f"raw.s{i}.jsonl" for i in range(16)]
    assert str(cfg.bank.pool) == str(questions_path(cfg.filter.out_dir + "/sat.s0.json"))


def test_slice_shard_layout():
    expect = {MATH_R1_1P5B: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
              MATH_R1_7B: [[0, 1, 2, 3], [4, 5, 6, 7]],
              CODE_QWEN3_30B: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
              CODE_CODER_32B: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]}
    for pair, slices in expect.items():
        cfg = config.load(pair)
        assert [_names(slice_files(cfg, k)) for k in range(len(slices))] == \
            [[f"raw.s{i}.jsonl" for i in s] for s in slices], pair


def test_bad_order_rejected():
    cfg = config.load(CODE_QWEN3_4B)
    cfg.filter.agg_order[0] = 1
    with pytest.raises(config.ConfigError):
        ordered_slice_files(cfg)
