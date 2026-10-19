"""``a2d data eval``: the evaluation release converted into the files the evaluators read.

* math rows get the verl schema of ``build_math_data`` (AIME24 keeps DAPO's own row layout);
* code rows get LiveCodeBench's format, starter code joined to the question as the builder did;
* a full offline build (``--from-dir``) writes every math / code file and the EvalPlus cache.

Byte equality with the files the final runs used is checked by the command itself against
``data/MANIFEST.json`` (it needs the locked pandas / pyarrow for the parquet files).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ask2distill.common.prompts_math import DAPO_PREFIX, DAPO_TAIL
from ask2distill.data import eval_data as E

MATH_ROW = {"id": "aime24-000", "problem": "Find x.", "answer": "540", "source_index": "2"}
CODE_ROW = {"id": "apps_1", "question": "Q", "starter_code": "class S:\n    def f(self): ", "difficulty": "interview",
            "tests": '{"inputs": [[1]], "outputs": [[2]], "fn_name": "f"}'}


def test_math_rows():
    aime24 = E.math_frame([MATH_ROW], "aime24").iloc[0]
    assert aime24["prompt"] == [{"content": DAPO_PREFIX + "Find x." + DAPO_TAIL, "role": "user"}]
    assert aime24["reward_model"] == {"ground_truth": "540", "style": "rule-lighteval/MATH_v2"}
    assert aime24["extra_info"] == {"index": 2, "raw_problem": "Find x.", "split": None}
    other = E.math_frame([MATH_ROW], "math500").iloc[0]
    assert other["prompt"][0]["content"] == DAPO_PREFIX + "Find x." + DAPO_TAIL
    assert other["extra_info"] == {"index": 0, "raw_problem": "Find x.", "split": "math-500"}


def test_extra_bench_line():
    r = json.loads(E.extra_bench_line(CODE_ROW))
    assert r == {"id": "apps_1", "difficulty": "interview", "input_output": CODE_ROW["tests"],
                 "question": "Q\n\nUse this starter code:\n```python\nclass S:\n    def f(self):\n```"}
    plain = json.loads(E.extra_bench_line({**CODE_ROW, "starter_code": ""}))
    assert plain["question"] == "Q"


def _snapshot(root: Path) -> Path:
    rel = root / "Ask_to_Distill-eval"
    for name in E.MATH:
        (rel / "math").mkdir(parents=True, exist_ok=True)
        (rel / "math" / f"{name}.jsonl").write_text(json.dumps(MATH_ROW) + "\n")
    for bench in E.EXTRA_BENCH:
        (rel / "code").mkdir(parents=True, exist_ok=True)
        (rel / "code" / f"{bench}.jsonl").write_text(json.dumps(CODE_ROW) + "\n")
    from ask2distill.data.build_code_eval_data import EVALPLUS_FILES
    for f in EVALPLUS_FILES:
        (rel / "code" / "evalplus_native").mkdir(parents=True, exist_ok=True)
        (rel / "code" / "evalplus_native" / f).write_text(f"{f}\n")
    import pandas as pd
    (root / "AMC23").mkdir()
    pd.DataFrame({"question": ["What is 3^3?"], "answer": [27.0]}).to_parquet(root / "AMC23" / E.AMC23[2])
    return root


def test_offline_build_writes_every_eval_file(tmp_path, monkeypatch):
    import ask2distill.data.build_code_eval_data as B
    import pandas as pd
    monkeypatch.setattr(B, "hub_file", lambda repo, f, cache, rev=None: str(cache / f))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    data, cache = tmp_path / "data", tmp_path / "cache"
    written = E.build(data, cache, from_dir=_snapshot(tmp_path / "src"))
    assert sorted(written) == sorted([f"eval/math/{d}/test.parquet" for d in [*E.MATH.values(), "AMC23"]]
                                     + [f"eval/code/extra_bench/{b}.jsonl" for b in E.EXTRA_BENCH])
    assert all((data / w).is_file() for w in written)
    amc = pd.read_parquet(data / "eval/math/AMC23/test.parquet").iloc[0]
    assert amc["reward_model"]["ground_truth"] == "27"
    for f in B.EVALPLUS_FILES:
        assert (cache / "evalplus" / f).read_text() == f"{f}\n"


def test_missing_snapshot_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        E.fetch_release(tmp_path, E.EVAL_REVISION, tmp_path / "nothing")
