"""Code prompt strings: the single solve template and the synthesizer prompt.

Solve template. Every code stage -- δ_ref calibration, the synthesizer reward,
the OPD banks (synthetic and OpenCodeReasoning-10k) and the LiveCodeBench eval -- goes
through ``solve_prompt``. The math pipeline grew two different solve prompts,
so divergence was measured under one conditioning and trained under another;
that is not repeated here. The text is LiveCodeBench's own stdin/stdout
template verbatim, so a question is scored, trained on, and finally evaluated
under byte-identical instructions.

Synthesizer prompt. The synthesizer-RL prompt parquet, calibration, the
checkpoint sweep and the harvest all build their messages with
``synthesizer_messages`` from one topic of ``configs/topics/code.yaml`` (the
technique the intended solution should centre on). Duplicating these strings is
how a synthesizer ends up sampled under a distribution it was never optimised
on, so there is exactly one copy. The prompt carries no example problems: the
instruction alone asks for the Input format, the Output format, the constraints
and stdin/stdout I/O, and for the statement only (no solution, code or answer).
"""
from __future__ import annotations

LCB_HEADER = ("You will be given a question (problem specification) and will generate a "
              "correct Python program that matches the specification and passes all "
              "tests. You will NOT return anything except for the program.\n\n")

LCB_FORMAT = ("Read the inputs from stdin solve the problem and write the answer to "
              "stdout (do not directly test on the sample inputs). Enclose your code "
              "within delimiters as follows. Ensure that when the python program runs, "
              "it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
              "```python\n# YOUR CODE HERE\n```\n\n")

SYNTHESIZER_SYSTEM = (
    "Topic: {topic}.\n"
    "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, "
    "non-trivial competitive programming problem whose intended solution centres on "
    "this topic. Aim for a difficulty such that a strong contestant would need several "
    "non-obvious algorithmic insights, not a single library call. Avoid re-using "
    "classic textbook tasks or well-known contest problems.\n"
    "The problem must be self-contained and must state its Input format, its Output "
    "format, and the constraints on the input. The program that solves it reads from "
    "standard input and writes to standard output.\n"
    "THEN, without revealing any of your private thoughts, output ONLY the problem "
    "statement. Do not include a solution, sample code, an answer, or any commentary.")

SYNTHESIZER_USER = "Generate one new, challenging competitive programming problem now."


def solve_prompt(question: str) -> str:
    """User-turn content for solving ``question``. Chat template applied by caller."""
    return f"{LCB_HEADER}Question: {question.strip()}\n\n{LCB_FORMAT}"


# The reward worker and calibration keep a template and fill it with
# str.replace("{q}", q.strip()), which yields exactly solve_prompt(q).
SOLVE_TEMPLATE = solve_prompt("{q}")


def synthesizer_messages(topic: str) -> list[dict]:
    """The exact message list used at train time, sweep time and harvest time."""
    return [{"role": "system", "content": SYNTHESIZER_SYSTEM.format(topic=topic)},
            {"role": "user", "content": SYNTHESIZER_USER}]
