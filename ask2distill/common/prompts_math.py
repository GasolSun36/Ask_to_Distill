"""Math prompt strings: the solve prompts of the final runs and the synthesizer prompt.

The solve prompts are byte-identical to the ones the final runs used. The math
pipeline uses FOUR different solve prompts, and the difference is
preserved on purpose (docs/DEVIATIONS.md): δ_ref, the synthesizer reward and the
results were all produced under exactly these strings.

=====================  ===============================================  =======================
where                  user-turn content                                function
=====================  ===============================================  =======================
synthesizer reward,    ``q.strip() + REWARD_SOLVE_SUFFIX``              ``reward_solve_prompt``
δ_ref calibration      (``\\boxed{}`` hint, no DAPO prefix)
synthetic OPD banks    ``DAPO_PREFIX + q`` (no "Remember..." tail)      ``bank_prompt``
Vanilla OPD (DAPO-17k) ``DAPO_PREFIX + q + DAPO_TAIL``                  ``real_prompt``
evaluation             the parquet prompt (``real_prompt``) +           ``eval_prompt``
                       ``EVAL_BOXED_SUFFIX``
=====================  ===============================================  =======================

The synthesizer itself is conditioned on one topic from ``configs/topics/math.yaml``
through ``synthesizer_messages``; the same system/user pair is used for the
synthesizer-RL prompt parquet, calibration, the checkpoint sweep and the
harvest, so S_θ is always sampled under the conditioning it was trained under.
"""
from __future__ import annotations

DAPO_PREFIX = ("Solve the following math problem step by step. The last line of your response "
               "should be of the form Answer: $Answer (without quotes) where $Answer is the "
               "answer to the problem.\n\n")

DAPO_TAIL = '\n\nRemember to put your answer on its own line after "Answer:".'

REWARD_SOLVE_SUFFIX = ("\n\nPlease reason step by step, and put your final answer within "
                       "\\boxed{}.")
# The worker keeps the template as a string and fills it with str.replace, not
# str.format: a code template carries braces of its own and so may the question.
REWARD_SOLVE_TEMPLATE = "{q}" + REWARD_SOLVE_SUFFIX

EVAL_BOXED_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."

SYNTHESIZER_SYSTEM = (
    "Topic: {topic}.\n"
    "FIRST, in your private scratch-pad, think step-by-step to design a brand-new, "
    "non-trivial mathematics problem on this topic. Aim for a difficulty such "
    "that fewer than 30% of advanced high-school students could solve it. Avoid "
    "re-using textbook cliches or famous contest problems.\n"
    "The problem must be self-contained, have a well-defined answer, and require "
    "several non-obvious reasoning steps that build on each other, not a single "
    "formula lookup.\n"
    "THEN, without revealing any of your private thoughts, output ONLY the problem "
    "statement. Do not include a solution, an answer, or any commentary.")

SYNTHESIZER_USER = "Generate one new, challenging mathematical reasoning problem now."


def reward_solve_prompt(q: str) -> str:
    """User content the frozen student solves in the reward worker and in calibration."""
    return REWARD_SOLVE_TEMPLATE.replace("{q}", q.strip())


def bank_prompt(q: str) -> str:
    """User content of a synthetic-bank row (no "Remember..." tail, q not stripped)."""
    return DAPO_PREFIX + q


def real_prompt(q: str) -> str:
    """User content of a DAPO-17k / math test-set row (as ``a2d data build math`` writes it)."""
    return DAPO_PREFIX + q + DAPO_TAIL


def eval_prompt(problem: str) -> str:
    """Eval-time prompt: the parquet prompt plus the boxed hint."""
    return problem + EVAL_BOXED_SUFFIX


def synthesizer_messages(topic: str) -> list[dict]:
    """Chat messages that ask the math synthesizer for one question on ``topic``."""
    return [{"role": "system", "content": SYNTHESIZER_SYSTEM.format(topic=topic)},
            {"role": "user", "content": SYNTHESIZER_USER}]
