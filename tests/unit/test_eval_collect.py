"""Collection: math avg6 and code Avg from synthetic result files.

The formulas are those the main table was computed with (avg6; code Avg); every
score below is synthetic. Collecting the released result directories of the
main points is checked in ``tests/equiv/test_eval_regrade.py``.
"""
from __future__ import annotations

import argparse
import json

import pytest

from ask2distill.eval import collect as C

from conftest import MATH_QWEN3_1P7B


def _grading(path, scores: dict[str, float]):
    path.mkdir(parents=True, exist_ok=True)
    rows = [{"hyperparameters": {"temperature": "0.7", "top_p": "0.95", "n": "16",
                                 "max_tokens": "31744", "task_name": k}, "mean_score": v}
            for k, v in scores.items()]
    (path / "grading_results.json").write_text(json.dumps(rows, indent=4))
    return path / "grading_results.json"


SCORES = {"aime24": 0.4167, "aime25": 0.3, "aime26": 0.2833, "amc23": 0.6, "brumo25": 0.35,
          "math500": 0.9, "hmmt_feb25": 0.1}


def test_math_avg6(tmp_path):
    s = C.math_avg6(_grading(tmp_path / "r", SCORES))
    expected = sum(SCORES[k] * 100 for k in C.AVG6_KEYS) / 6
    assert s["avg6"] == pytest.approx(expected) and s["complete"]
    assert list(s["tasks"]) == list(C.AVG6_KEYS)
    row = C.format_math_row("r", s)
    assert row.split("\t") == ["r", "41.7", "30.0", "28.3", "60.0", "35.0", "90.0", f"{expected:.2f}"]


def test_math_avg6_incomplete(tmp_path):
    s = C.math_avg6(_grading(tmp_path / "r", {"aime24": 0.5}))
    assert s["avg6"] is None and not s["complete"] and "math500" in s["missing"]
    assert C.format_math_row("r", s).endswith("\t-")


def _code_dirs(root, lcb=0.4236, mbpp=(0.85, 0.7834), xb=(0.6, 0.35, 0.15, 0.2)):
    tag = root / "tag_step150"
    xbd = root / "xb_tag"
    tag.mkdir(parents=True)
    xbd.mkdir()
    (tag / "livecodebench_v5_v6_eval.json").write_text(json.dumps([{"pass@1": lcb, "detail": {}}]))
    (tag / "evalplus_mbpp_lenient.log").write_text(
        f"mbpp (base tests)\npass@1:\t{mbpp[0]}\nmbpp+ (base + extra tests)\npass@1:\t{mbpp[1]}\n")
    for b, v in zip(C.XB_BENCHES, xb):
        (xbd / f"{b}.json").write_text(json.dumps({"overall": v, "n_problems": 10, "by_difficulty": {}}))
    return tag, xbd


def test_code_avg(tmp_path):
    tag, xbd = _code_dirs(tmp_path)
    s = C.code_avg(tag, xbd)
    vals = [42.36, 78.34, 60.0, 35.0, 15.0, 20.0]
    assert [round(v, 2) for v in s["items"].values()] == vals          # MBPP+ = the LAST pass@1
    # Paper convention: cells 42.4 78.3 60.0 35.0 15.0 20.0 -> mean 41.7833.. -> Avg 41.8;
    # the mean of the unrounded items is kept as avg_exact.
    assert s["avg_cells"] == pytest.approx((42.4 + 78.3 + 60 + 35 + 15 + 20) / 6)
    assert s["avg"] == 41.8
    assert s["avg_exact"] == pytest.approx(sum(vals) / 6)
    assert C.format_code_row("t", s).split("\t")[-3:] == ["41.8", "41.78", "41.78"]
    missing = C.code_avg(tag)                                           # xb defaults to <tag>/xb
    assert missing["avg"] is None and missing["missing"] == list(C.XB_BENCHES)
    assert missing["avg_cells"] is None and missing["avg_exact"] is None
    assert C.format_code_row("t", missing).split("\t")[-3:] == ["-", "-", "-"]


@pytest.mark.parametrize("values, cells, table", [
    ([10.04, 20.0, 30.0, 40.0, 50.0, 0.31], 25.05, 25.1),             # cell mean ends in 5: half-up
    ([12.34, 23.45, 34.56, 45.67, 56.78, 67.89], 40.12, 40.1),
    ([1.04, 2.0, 3.0, 4.0, 5.0, 6.0], 3.5, 3.5),
    ([30.0, 30.0], 30.0, 30.0),
    ([0.05, 0.0], 0.05, 0.1),                                          # cells 0.1 / 0.0 (f-string rounding)
])
def test_paper_avg_rounds_cells_then_half_up(values, cells, table):
    got_cells, got_table = C.paper_avg(values)
    assert round(got_cells, 2) == pytest.approx(cells)
    assert got_table == table


def test_collect_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("A2D_OUT", str(tmp_path))
    _grading(tmp_path / "eval" / "math" / "opd_x_step130", SCORES)
    _grading(tmp_path / "eval" / "math" / "opd_x_step140", {"aime24": 0.1})
    out = tmp_path / "math.json"
    rc = C.main(None, argparse.Namespace(domain="math", runs=None, out=str(out)))
    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split("\t") == ["run", *C.AVG6_KEYS, "avg6"]
    assert [ln.split("\t")[0] for ln in lines[1:3]] == ["opd_x_step130", "opd_x_step140"]
    recs = json.loads(out.read_text())
    assert recs[0]["complete"] and not recs[1]["complete"]

    code_root = tmp_path / "eval" / "code"
    _code_dirs(code_root)
    rc = C.main(None, argparse.Namespace(domain="code", runs=["tag_step150:xb_tag"],
                                         out=str(tmp_path / "code.tsv")))
    assert rc == 0
    tsv = (tmp_path / "code.tsv").read_text().splitlines()
    assert tsv[0].split("\t") == ["run", *C.CODE_ITEMS, "avg", "avg_cells", "avg_exact"]
    assert tsv[1].split("\t")[-3:] == ["41.8", "41.78", f"{(42.36 + 78.34 + 60 + 35 + 15 + 20) / 6:.2f}"]


def test_math_keys_from_config():
    from ask2distill.config import load
    assert C.math_keys(load(MATH_QWEN3_1P7B)) == list(C.AVG6_KEYS)
    assert C.math_keys(None) == list(C.AVG6_KEYS)
