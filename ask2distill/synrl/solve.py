"""The prompt under which the frozen student solves a question, for calibration and reward.

δ_ref is the p95 of δ_t, and δ_t depends on the prompt. Calibrating under one
instruction and scoring under another silently rescales every reward, so
calibration (``calib.calibrate``) and the reward worker (``synrl.reward_worker``)
build the solve prompt from the same template table and the same chat-template
call. This module stays free of verl and of the reward worker's import-time
allocator setting, so calibration can use it.

Templates (configs: ``calib.solve_template`` / ``reward.solve_template``):

=============  ==============================================================
``math_boxed``  ``"{q}\\n\\nPlease reason step by step, and put your final
                answer within \\boxed{}."`` (``prompts_math.REWARD_SOLVE_TEMPLATE``)
``lcb``         LiveCodeBench's stdin/stdout template (``prompts_code.SOLVE_TEMPLATE``),
                so δ_ref, the reward, the OPD banks and the LCB eval share one prompt
=============  ==============================================================

The two original call sites differ in one detail that is kept: calibration
strips the tokenizer's BOS and then the literal DeepSeek BOS string from the
rendered chat template, the reward worker strips only the tokenizer's BOS.
For every tokenizer used here the two agree.
"""
from __future__ import annotations

from typing import Any, Sequence

from ask2distill.common import prompts_code, prompts_math

SOLVE_TEMPLATES = {
    "math_boxed": prompts_math.REWARD_SOLVE_TEMPLATE,
    "lcb": prompts_code.SOLVE_TEMPLATE,
}
DEEPSEEK_BOS = "<｜begin▁of▁sentence｜>"


def solve_template(name: str) -> str:
    """The solve template registered under ``name`` (a ``{q}`` placeholder string)."""
    try:
        return SOLVE_TEMPLATES[name]
    except KeyError:
        raise ValueError(f"unknown solve_template {name!r}; one of {sorted(SOLVE_TEMPLATES)}") from None


def solve_messages(template: str, question: str, system: str) -> list[dict]:
    """Chat messages for solving ``question``.

    A non-empty ``system`` goes to BOTH models. It is shared, not teacher-only,
    because the token reward scores a single token sequence that the solver and
    the teacher both consume -- giving them different prompts would leave δ_t
    at position t comparing two different contexts.
    """
    # str.replace, not str.format: a code template carries braces of its own
    # and the question text certainly does.
    msgs = [{"role": "user", "content": template.replace("{q}", question.strip())}]
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    return msgs


def strip_prefixes(s: str, prefixes: Sequence[str]) -> str:
    """Remove each non-empty prefix in turn when ``s`` starts with it.

    The tokenizer adds BOS again when the rendered prompt is encoded with
    ``add_special_tokens=True``, so a template that already emits it would
    double it.
    """
    for b in prefixes:
        if b and s.startswith(b):
            s = s[len(b):]
    return s


def render_solve_prompt(tok: Any, question: str, *, template: str, system: str,
                        qwen3_nothink: bool, strip_deepseek_bos: bool) -> str:
    """The chat-templated solve prompt (generation prompt appended, BOS stripped).

    ``qwen3_nothink`` passes ``enable_thinking=False`` to the template (hybrid
    Qwen3 students); other templates have no such switch. ``strip_deepseek_bos``
    selects the calibration variant of the BOS strip (see the module docstring).
    """
    kw = {"enable_thinking": False} if qwen3_nothink else {}
    s = tok.apply_chat_template(solve_messages(template, question, system), tokenize=False,
                                add_generation_prompt=True, **kw)
    prefixes = [tok.bos_token or ""]
    if strip_deepseek_bos:
        prefixes.append(DEEPSEEK_BOS)
    return strip_prefixes(s, prefixes)
