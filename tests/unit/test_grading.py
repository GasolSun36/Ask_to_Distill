"""Math grading: answer checking and the grading_results.json schema.

The pinned cases document the grader's behaviour, including the ``Answer:``
fallback the final runs graded with.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ask2distill.eval.math import grade as G
from ask2distill.eval.math import grader_utils as GU


_FORMS = [
    ("\\boxed{{{a}}}", "boxed"),
    ("The answer is $\\boxed{{{a}}}$.", "boxed in text"),
    ("Answer: {a}", "Answer line"),
    ("**Answer**: {a}", "bold Answer"),
    ("**Answer:** {a}", "bold Answer, colon inside"),
    ("Answer：{a}", "full-width colon"),
    ("Some work.\nAnswer: {a}.", "trailing period"),
    ("{a}", "bare"),
    ("\\fbox{{{a}}}", "fbox"),
]
_ANSWERS = [
    ("42", "42"), ("42", "42.0"), ("42", "42 inches"), ("1/2", "\\frac{1}{2}"),
    ("\\frac{1}{2}", "0.5"), ("\\dfrac{3}{4}", "\\frac34"), ("2\\sqrt{3}", "2\\sqrt3"),
    ("(1, 2)", "(1,2)"), ("[0, 1)", "[0,1)"), ("1,000", "1000"), ("-7", "- 7"),
    ("\\pi", "\\pi"), ("3", "4"), ("x^2+1", "1+x^2"), ("10^{\\circ}", "10"),
]
CASES: list[tuple[str, str]] = []
for gt, given in _ANSWERS:
    for form, _ in _FORMS:
        CASES.append((form.format(a=given), gt))
CASES += [
    ("no final answer at all", "5"),
    ("", "5"),
    ("\\boxed{5", "5"),
    ("first \\boxed{3} then \\boxed{5}", "5"),
    ("Answer: 12 apples", "12"),
    ("Answer: twelve", "12"),
    ("answer: 12", "12"),
    ("The final answer: 12", "12"),
    ("\\boxed{\\text{(B)}}", "B"),
    ("\\boxed{5}", "\\boxed{5}"),
    ("\\boxed{5}", ""),
    ("Answer: 3/4 cups", "\\frac{3}{4}"),
]

PINNED = [
    ("\\boxed{42}", "42", True),
    ("Answer: 42 inches", "42", True),          # numeric answer, unit word dropped
    ("**Answer**: 42", "42", True),
    ("**Answer:** 42", "42", False),            # the bold must close before the colon
    ("Some work.\nAnswer: 42.", "42", True),
    ("42", "42", False),                        # no \boxed, no Answer: line
    ("\\boxed{\\dfrac{1}{2}}", "0.5", True),
    ("\\boxed{3}", "4", False),
    ("first \\boxed{3} then \\boxed{5}", "5", True),   # the LAST box counts
    ("\\boxed{5}", "", False),
    ("\\boxed{5", "5", False),
]


@pytest.mark.parametrize("response,gt,expected", PINNED)
def test_pinned_cases(response, gt, expected):
    assert GU.grade_answer_verl(response, gt) == expected


def test_strict_boxed_disables_answer_fallback():
    assert GU.grade_answer_verl("Answer: 42", "42") is True
    assert GU.grade_answer_verl("Answer: 42", "42", strict_boxed=True) is False
    assert GU.grade_answer_verl("\\boxed{42}", "42", strict_boxed=True) is True


def test_case_count():
    assert len(CASES) >= 100


def _write_gen(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return path


def _synthetic_rows(n_problems: int = 5, n: int = 4) -> list[dict]:
    rows = []
    for rollout in range(n):
        for i in range(n_problems):
            resp = [f"so \\boxed{{{i}}}", f"Answer: {i}", "no idea", f"\\boxed{{{i + 1}}}"][(i + rollout) % 4]
            rows.append({"example_id": i, "prompt": f"q{i}", "answer": str(i), "seed": rollout,
                         "response": resp + " the the the the end"})
    return rows


def test_grade_dir_schema(tmp_path):
    _write_gen(tmp_path / "hmmt_feb25_t0.7_p0.95_n4-MNT31744.jsonl", _synthetic_rows())
    _write_gen(tmp_path / "aime24_t0.7_p0.95_n4-MNT31744.jsonl", _synthetic_rows(3))
    (tmp_path / "notes.jsonl").write_text("")                 # unrecognised name: skipped
    out = G.grade_dir(tmp_path)
    res = json.loads(out.read_text())
    assert [r["hyperparameters"]["task_name"] for r in res] == ["aime24", "hmmt_feb25"]
    r = res[1]
    assert r["hyperparameters"] == {"temperature": "0.7", "top_p": "0.95", "n": "4",
                                    "max_tokens": "31744", "task_name": "hmmt_feb25"}
    assert set(r) == {"hyperparameters", "mean_score", "distinct_4gram", "best_score", "solve_none",
                      "solve_all", "avg_output_length", "format_error_rollouts"}
    # each problem has 2 of 4 correct responses (boxed i, Answer: i)
    assert r["mean_score"] == pytest.approx(0.5)
    assert r["best_score"] == 1.0
    assert r["format_error_rollouts"] == 5 * 2           # "Answer:" and "no idea" have no "boxed"


# ------------------------------------------------------------------------ equivalence
