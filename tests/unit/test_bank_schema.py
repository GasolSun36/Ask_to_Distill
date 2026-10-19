"""Bank rows, row ids, the code-pool re-filter and the synthesizer prompt parquet.

Row-id prefixes (``q3armg9``, ``code_d_syn10k``, ...) are the ones of the
released banks, so a rebuilt bank is byte-identical to the released one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ask2distill.bank import build_bank as B
from ask2distill.bank import build_synrl_prompts as GP
from ask2distill.bank.refilter import refilter, refilter_questions
from ask2distill.common.prompts_code import solve_prompt
from ask2distill.common.prompts_math import DAPO_PREFIX

from conftest import CODE_QWEN3_4B, MATH_QWEN3_1P7B

QS = [f"  Question {i}: find the value of x_{i} given that x_{i} + {i} = {2 * i}?" for i in range(10)]


def _pool(tmp_path: Path, qs=QS) -> Path:
    p = tmp_path / "pool.jsonl"
    p.write_text("".join(json.dumps({"problem": q}, ensure_ascii=False) + "\n" for q in qs))
    return p


def _records(path: Path) -> list[dict]:
    df = pd.read_parquet(path)
    return json.loads(df.to_json(orient="records"))


def test_math_bank_rows(tmp_path):
    out = tmp_path / f"{MATH_QWEN3_1P7B}.parquet"
    assert B.build(_pool(tmp_path), out, 8, "math", "q3armg9") == 8
    rows = _records(out)
    assert list(pd.read_parquet(out).columns) == ["data_source", "prompt", "ability",
                                                  "reward_model", "extra_info"]
    assert rows[0] == {"data_source": "math_dapo",
                       "prompt": [{"role": "user", "content": DAPO_PREFIX + QS[0]}],
                       "ability": "MATH",
                       "reward_model": {"ground_truth": "N/A", "style": "rule-lighteval/MATH_v2"},
                       "extra_info": {"index": "q3armg9_0"}}
    assert [r["extra_info"]["index"] for r in rows] == [f"q3armg9_{i}" for i in range(8)]
    assert "Remember" not in rows[3]["prompt"][0]["content"]


def test_code_bank_rows(tmp_path):
    out = tmp_path / f"{CODE_QWEN3_4B}.parquet"
    B.build(_pool(tmp_path), out, 10, "code", "code_d_syn10k")
    rows = _records(out)
    assert rows[9]["data_source"] == "code_stdio" and rows[9]["ability"] == "CODE"
    assert rows[9]["prompt"] == [{"role": "user", "content": solve_prompt(QS[9])}]
    assert rows[9]["extra_info"] == {"index": "code_d_syn10k_9"}


def test_short_pool(tmp_path):
    with pytest.raises(ValueError):
        B.build(_pool(tmp_path), tmp_path / "x.parquet", 11, "math", "x")
    assert B.build(_pool(tmp_path), tmp_path / "x.parquet", 11, "math", "x", allow_short=True) == 10


# ------------------------------------------------------------------- code-pool re-filter

STMT = ("You are given an array of n integers a_1, a_2, ..., a_n. In one operation you may "
        "choose two adjacent elements and replace them with their sum. Find the minimum number "
        "of operations needed to make all elements of the array equal.\n\n"
        "Input\nThe first line contains a single integer n (1 <= n <= 2*10^5). The second line "
        "contains n integers a_i (1 <= a_i <= 10^9).\n\n"
        "Output\nPrint one integer: the minimum number of operations.")
REFILTER_POOL = [
    STMT,
    STMT + "\n\nWait, maybe the constraints should be smaller.",     # salvages onto STMT: dup
    "<think>x</think>" + STMT,                                       # format reject
    "too short",                                                     # format reject
    STMT.replace("integers", "numbers"),
    "You are a competitive programmer who loves arrays.\n" + STMT.replace("sum", "product"),
    STMT.replace("integers", "numbers"),                              # exact dup
    "A kingdom has many cities connected by roads and the king wants a count of them all. " * 3,
]


def test_refilter_semantics(tmp_path):
    out = refilter_questions(REFILTER_POOL)
    assert out == [STMT, STMT.replace("integers", "numbers"), STMT.replace("sum", "product")]
    assert refilter(_pool(tmp_path, REFILTER_POOL), tmp_path / "clean.jsonl") == 3


# ---------------------------------------------------------------- synthesizer prompt parquet

TOPICS = ["algebra", "geometry", "number theory"]


@pytest.mark.parametrize("domain,ability", [("math", "MATH"), ("code", "CODE")])
def test_prompt_rows(domain, ability):
    rows = GP.prompt_rows(domain, TOPICS, 7, 0)
    assert len(rows) == 7
    assert rows[0]["data_source"] == "synrl" and rows[0]["ability"] == ability
    assert [r["extra_info"]["index"] for r in rows] == [f"synrl_{domain}_{i}" for i in range(7)]
    assert rows[0]["reward_model"] == {"ground_truth": "N/A", "style": "rule-lighteval/MATH_v2"}
    topics = [r["extra_info"]["topic"] for r in rows]
    assert sorted(topics[:3]) == sorted(TOPICS) and sorted(topics[3:6]) == sorted(TOPICS)   # rotation
    for r in rows:
        system, user = r["prompt"]
        assert system["role"] == "system" and user["role"] == "user"
        assert system["content"].startswith(f"Topic: {r['extra_info']['topic']}.")
    assert GP.prompt_rows(domain, TOPICS, 7, 0) == rows                                      # seeded
    assert GP.prompt_rows(domain, TOPICS, 7, 1) != rows


def test_prompt_build_rejects_too_few_rows():
    from ask2distill import config
    from ask2distill.config import ConfigError
    cfg = config.load(MATH_QWEN3_1P7B, overrides=["synrl.prompt_rows=100"])
    with pytest.raises(ConfigError, match="prompt_rows"):
        GP.check(cfg)
