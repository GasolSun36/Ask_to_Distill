"""Text normalisation shared by the reward worker, harvest and the filters.
"""
from __future__ import annotations

import re


def strip_thinking(text: str) -> str:
    """Return the answer part of a response that may contain a thinking block.

    The R1-Distill chat templates end with "<|Assistant|><think>\\n", so those
    models always start inside a thinking block; the synthesizer variants use a
    patched template that closes it immediately. This is the safety net for when
    one slips through: text after the last </think> is the question, and a block
    that never closed means the budget ran out mid-thought, which is not a
    question at all.

    Text without any tag is returned unchanged (not stripped); callers that
    need whitespace trimmed use ``strip_question``.
    """
    if "</think>" not in text:
        return "" if "<think>" in text else text
    return text.rsplit("</think>", 1)[1].strip()


def strip_question(text: str) -> str:
    """``strip_thinking(text).strip()``: the question text as the reward worker
    and the harvest see it.

    The checkpoint sweep deliberately does NOT use this; it writes the raw
    ``.strip()``-ed text, which is what the pick thresholds were set on.
    """
    return strip_thinking(text).strip()


def norm_text(text: str) -> set[str]:
    """Lower-cased alphanumeric token set used for Jaccard near-duplicate checks."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: set, b: set) -> float:
    """|a & b| / |a | b|, with an empty union counted as 1 in the denominator."""
    return len(a & b) / max(len(a | b), 1)
