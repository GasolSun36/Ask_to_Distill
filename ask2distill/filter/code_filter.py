"""Question filter for generated COMPETITIVE-PROGRAMMING problems.

The math filter (``math_filter``) cannot be reused: its ``ASK`` verb list, the
``$``/``{}`` balance checks and the "must end in ?.$)]" rule all reject
well-formed programming problems, while its 80-2000 char window sits below
their median (real code_contests problems: p50 1537, p90 2770).

What replaces them is a structural test the math domain has no analogue for: a
competitive problem states its INPUT and its OUTPUT. That is a far stronger
signal than any verb list, and it is what separates a real problem statement
from a solution sketch or a fragment of reasoning.

Calibrated against OpenCodeReasoning ``input`` fields (real
codeforces/atcoder/codechef statements); the bar is the same one the math
filter used for DAPO: false-positive rate under 5%.

This is the version the pairs Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct
and Code_Qwen3-4B_Qwen3-4B-Instruct-2507 were filtered with, and the one that re-filtered the
pool of Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507 (``bank.refilter``). Its known misfires
are kept on purpose (docs/DEVIATIONS.md):

* ``ships_solution`` does not require the keyword to sit inside the code
  block: any line three or more lines after an opening fence that contains
  ``for ``/``while ``/... trips it ("For the sample, ..." after a fenced
  sample input), while a short fenced solution of under four lines slips
  through;
* ``no_io_spec`` only fires when neither the section headers nor the very
  loose inline pattern match, so "Print the minimum cost ..." alone passes.
"""
from __future__ import annotations

import re
from collections import Counter

# Model meta-commentary. Anchored at a line start or after sentence-ending
# punctuation -- the math version matched mid-sentence and "okay" or "what"
# inside a problem narrative tripped it constantly.
META = re.compile(
    r"(?:^|\n)\s*(?:Wait\b|Hmm\b|Alright\b|Okay\b|OK,\s|Let me\b|Let'?s (?:see|consider|design|create|think)\b|"
    r"I think\b|I need to\b|I'?ll\b|I will (?:design|create)\b|I can'?t\b|Actually,|But wait\b|"
    r"So,? (?:the problem|this problem|I |we)|"
    r"Here(?:'s| is) (?:the|a) (?:solution|code|problem|new)\b|"
    r"This (?:problem )?(?:can be|is) solved\b|"
    r"The (?:answer|solution) is\b)", re.IGNORECASE)

# The synthesizer narrates its design before writing the statement ~15% of the time:
# "Designing a new, non-trivial problem based on ... Let's consider ..." then a rule
# and the real problem. salvage() cuts AT a marker and keeps what precedes it, which
# is exactly backwards when the narration comes first -- hence this separate pass.
LEAD_META = re.compile(
    r"^(?:[^\n]{0,200}?)\b(?:Designing|Let'?s (?:consider|design|create|think)|"
    r"Here(?:'s| is) a (?:new|problem)|I(?:'ll| will) (?:design|create)|"
    r"This problem (?:is designed|involves creating))", re.I)
# Where the statement actually starts, when the model marked the boundary itself.
SEP = re.compile(r"\n\s*(?:---+|\*\*\s*Problem Statement\s*\*\*|##+\s*Problem[^\n]*)\s*\n", re.I)

# A role sentence at the start of the statement:
# "You are a researcher studying societal phenomena. ..." -- not a problem opening.
# "You are given ..." is, so it is excluded.
ROLE_PREAMBLE = re.compile(r"^You are (?:an?|the)\s+(?!given\b)[^.\n]{5,90}(?:\.|,)", re.I)

# The structural core: a statement declares what it reads and what it prints.
INPUT_SEC = re.compile(r"(?:^|\n)\s*(?:#+\s*|\*+\s*)?input\b", re.IGNORECASE)
OUTPUT_SEC = re.compile(r"(?:^|\n)\s*(?:#+\s*|\*+\s*)?output\b", re.IGNORECASE)
# Fallback for statements that phrase it inline rather than as a section.
IO_INLINE = re.compile(
    r"\b(?:print|output|return|prints?|outputs?)\b.{0,80}\b(?:the|a|an|each|every)\b|"
    r"\bthe (?:first|next|last) line (?:of the input )?contains\b|"
    r"\bread\b.{0,40}\bfrom (?:standard input|stdin)\b", re.IGNORECASE)

# A statement that ships its own solution is not a question.
SOLUTION_LEAK = re.compile(
    r"```[a-z+]*\s*\n(?:[^\n]*\n){3,}?[^\n]*\b(?:def |class |for |while |import )",
    re.IGNORECASE)


def salvage(q: str) -> str:
    """Strip design narration before the statement, then trailing commentary after it."""
    # Leading narration first: if the model drew its own boundary, take what follows.
    if LEAD_META.match(q):
        m = SEP.search(q)
        q = q[m.end():] if m else q
    # A role sentence sits on its own line or ends the first sentence; drop just it.
    m = ROLE_PREAMBLE.match(q)
    if m:
        q = q[m.end():].lstrip()
    m = META.search(q)
    if m:
        q = q[:m.start()]
    return q.strip()


def reject_reason(q: str) -> str | None:
    """Why a (salvaged) statement is rejected, or ``None`` if it is kept."""
    w = q.split()
    if not (200 <= len(q) <= 8000 and len(w) >= 40):
        return "length"
    # Code statements repeat their own identifiers far more than prose, so the
    # math threshold of 0.35 would reject genuine problems.
    if len(w) >= 60 and len(set(w)) / len(w) < 0.22:
        return "low_diversity"
    # These three detectors exist to catch MODEL degeneration, and all of them
    # misfire on the structured repetition every competitive statement carries:
    # "a_1, a_2, ..., a_n" is not ellipsis spam, "0 0 0 0 0 0" in a sample input
    # is not a token loop, and a repeated constraint line is not an n-gram loop.
    # Each one is therefore restricted to alphabetic content.
    if re.search(r"\b([A-Za-z]{2,}\s+)\1{6,}", q):
        return "token_loop"
    alpha_grams = [g for g in (" ".join(w[i:i + 8]) for i in range(len(w) - 7))
                   if sum(bool(re.fullmatch(r"[A-Za-z']+", t)) for t in g.split()) >= 5]
    if alpha_grams and Counter(alpha_grams).most_common(1)[0][1] >= 5:
        return "ngram_loop"
    # Only ellipses that are NOT part of a sequence like "a_1, ..., a_n".
    loose = len(re.findall(r"(?<![,\s])\.\.\.(?![,\s])", q))
    if loose > 8 or q.count("…") > 12:
        return "ellipsis_spam"
    if SOLUTION_LEAK.search(q):
        return "ships_solution"
    if META.search(q) or LEAD_META.match(q):
        return "residual_meta"
    if ROLE_PREAMBLE.match(q):
        return "role_preamble"
    if not ((INPUT_SEC.search(q) and OUTPUT_SEC.search(q)) or IO_INLINE.search(q)):
        return "no_io_spec"
    # Hard truncation: the synthesizer ran out of budget mid-sentence. Example
    # blocks and constraint tables legitimately end without punctuation, so this
    # only fires when the last line is prose that simply stops.
    last = [line for line in q.strip().split("\n") if line.strip()]
    if last:
        tail = last[-1].strip()
        prose = sum(bool(re.fullmatch(r"[A-Za-z',.;:()-]+", t)) for t in tail.split())
        if (len(tail.split()) >= 6 and prose >= len(tail.split()) * 0.7
                and tail[-1] not in ".!?:;\"')]}`"):
            return "truncated_tail"

    # A statement that opens mid-thought is a fragment of something else.
    if re.match(r"\s*(?:Now\b|Then\b|Next\b|Also\b|And\b|But\b|Thus\b|Therefore\b|"
                r"Similarly\b|Again\b|Finally\b|So\b)", q):
        return "dangling_opener"
    return None


def strong_filter(q: str) -> tuple[str | None, str | None]:
    """``(clean, None)`` or ``(None, reason)``. The clean text may be salvaged."""
    q = salvage(q)
    r = reject_reason(q)
    return (None, r) if r else (q, None)


def comp_filter(raw: str) -> tuple[str | None, str | None]:
    """Cheap format gate, the code analogue of ``math_filter.comp_filter``.

    Dedup is NOT done here: the callers (saturation) own that, and keeping it
    out makes this function pure and unit-testable.
    """
    text = raw.strip()
    if "<think>" in text or "</think>" in text:
        return None, "think_tag"
    if not (200 <= len(text) <= 8000):
        return None, "length"
    if re.search(r"(?im)^\s*problem\s*\d+\s*:", text):
        return None, "multi_problem"
    return text, None
