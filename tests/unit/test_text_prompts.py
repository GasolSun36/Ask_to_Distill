"""Text helpers (thinking-block stripping, normalisation) and the synthesizer prompts.
"""
from __future__ import annotations


import pytest

from ask2distill.common import prompts_code as PC
from ask2distill.common import text as T


STRINGS = [
    "", " ", "\n", "plain question", "  padded question \n", "<think>", "</think>", "<think></think>",
    "<think>reasoning</think>Q?", "<think>a</think>\n\n  What is 2+2?  ", "no close <think> here",
    "<think>x</think>A</think>B", "</think>leading close", "text <think>open only",
    "Find all $x$ such that $x^2=4$.", "UPPER Case Mixed 123 abc", "a-b_c.d,e;f", "ÄÖÜ unicode ü 5",
    "1 1 1 1 1", "The the THE tHe", "x" * 500, "line1\nline2\n\n\nline3", "tab\tsep\tvalues",
    "<THINK>upper</THINK>q", "</think></think>", "<think>\n</think>\n\nProblem: count the subsets.",
    "Let f(x)=x^3; find f'(2).", "  \t ", "emoji 🙂 question?", "answer: 42", "\\boxed{7}",
    "Problem 1: a\n\nProblem 2: b", "Q with <think> in the middle</think> tail", "a</think>",
    "x</think>  ", "multi\n</think>\nlast part", "numbers 12 345 6789", "a b c d e f g h",
    "same words same words", "Same Words", "ab", "a", "0", "-1", "3.14159", "x_1 + x_2 = y",
    "Compute $\\sum_{k=1}^{n} k$.", "def f(): pass", "```python\nprint(1)\n```", "end."]
assert len(STRINGS) == 50


def test_text_helpers_semantics():
    assert T.strip_thinking("<think>still thinking") == ""
    assert T.strip_thinking("  no tags ") == "  no tags "
    assert T.strip_question("  no tags ") == "no tags"
    assert T.strip_thinking("<think>a</think> q </think> r ") == "r"
    assert T.norm_text("A-b a") == {"a", "b"}
    assert T.jaccard(set(), set()) == 0.0


@pytest.mark.parametrize("module", ["prompts_math", "prompts_code"])
def test_synthesizer_prompt_is_topic_only(module):
    """One topic, the instruction text, no example problems."""
    import importlib
    mod = importlib.import_module(f"ask2distill.common.{module}")
    system, user = mod.synthesizer_messages("graph theory")
    assert system["role"] == "system" and user["role"] == "user"
    assert system["content"].startswith("Topic: graph theory.\n")
    text = system["content"].lower()
    for must in ("self-contained", "output only the problem", "do not include a solution"):
        assert must in text
    assert "example" not in text and "problem 1" not in text and "you are" not in text
    if module == "prompts_code":
        assert all(k in system["content"] for k in ("Input format", "Output format", "constraints",
                                                    "standard input", "standard output"))
        assert not [n for n in dir(PC) if "seed" in n.lower() or "shot" in n.lower()]
