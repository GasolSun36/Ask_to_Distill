"""Math question filters: a cheap format gate (with in-slice Jaccard dedup) and
the salvage-then-reject quality filter.

Format gate (``comp_filter``), in this order:

* ``think_tag``     the output still contains ``<think>`` / ``</think>``;
* the text is cut at the first triple newline;
* ``length``        outside 80-2000 characters;
* ``leaks_answer``  contains ``answer:`` or ``\\boxed`` (anti-leak: a question
  that ships its answer teaches nothing);
* ``multi_problem`` a line starting with ``Problem <digit>``;
* ``dup_self``      token-set Jaccard above 0.7 against the slice's dedup pool.

Every text that passes the gate adds its token set to the pool, even if the
quality filter then rejects it; ``saturation`` additionally adds the salvaged
copy of every question that passes both. That pool semantics is what the final
banks were filtered under and is kept as is.

Quality filter (``strong_filter``): ``salvage`` cuts trailing model
meta-commentary ("Wait", "Let me", ...) and ``reject_reason`` rejects what is
left if it is degenerate, structurally broken, or not actually a question.
Calibrated on DAPO real questions (target false-positive rate under 5%).

The original gate also compared against "seed" and "source" pools for the
completion-style question writers of an early pipeline; those were always empty in
the saturation filter and are dropped here.
"""
from __future__ import annotations

import re
from collections import Counter

from ask2distill.common.text import jaccard, norm_text

# Fixed in the format gate the final runs used; filter.jaccard records it.
JACCARD = 0.7

META = re.compile(
    r"(?:^|\n|\s)(?:[Ww]ait\b|[Hh]mm\b|[Aa]lright\b|[Oo]kay\b|OK,\s|Oh\b|sorry\b|Sorry\b|"
    r"the user\b|I think\b|I need to\b|I can't\b|Let me\b|Let's see\b|"
    r"So,? (?:the problem|this problem|I |we have solved)|"
    r"But wait\b|Actually\b|it seems\b|This seems\b|"
    r"Problem \d|problem statement|duplicate of)"
)
ASK = re.compile(
    r"\?|\b(?:find|compute|determine|evaluate|calculate|prove|show that|"
    r"how many|how much|what|which|solve|express|state|give|denote|"
    r"in how many|for what|at what|the (?:value|number|sum|product|area|remainder) of)\b",
    re.IGNORECASE)


def comp_filter(raw: str, norms: list[set[str]], jaccard_thr: float = JACCARD
                ) -> tuple[str | None, str | None]:
    """Format gate plus Jaccard dedup against ``norms``; ``(text, None)`` or ``(None, reason)``.

    On success the text's token set is appended to ``norms``.
    """
    text = raw.strip()
    if "<think>" in text or "</think>" in text:
        return None, "think_tag"
    text = text.split("\n\n\n")[0].strip()
    if not (80 <= len(text) <= 2000):
        return None, "length"
    if "answer:" in text.lower() or "\\boxed" in text:
        return None, "leaks_answer"
    if re.search(r"(?im)^\s*problem\s*\d", text):
        return None, "multi_problem"
    nt = norm_text(text)
    if any(jaccard(nt, s) > jaccard_thr for s in norms):
        return None, "dup_self"
    norms.append(nt)
    return text, None


def salvage(q: str) -> str:
    """Cut the question off at the first meta-commentary marker."""
    m = META.search(q)
    if m:
        q = q[:m.start()]
    return q.strip()


def reject_reason(q: str) -> str | None:
    """Why a (salvaged) question is rejected, or ``None`` if it is kept."""
    w = q.split()
    if not (30 <= len(q) <= 2500 and len(w) >= 8):
        return "length"
    if len(w) >= 30 and len(set(w)) / len(w) < 0.35:
        return "low_diversity"
    if re.search(r"(\S+ )\1{5,}", q):
        return "token_loop"
    if q.count("...") > 3 or q.count("…") > 3:
        return "ellipsis_spam"
    # repeated 5-gram >= 4 times
    grams = [" ".join(w[i:i + 5]) for i in range(len(w) - 4)]
    if grams and Counter(grams).most_common(1)[0][1] >= 4:
        return "ngram_loop"
    if q.count("$") % 2 == 1:
        return "unbalanced_dollar"
    if abs(q.count("{") - q.count("}")) > 2:
        return "unbalanced_brace"
    if re.search(r"\\end\{(\w+)\}", q) and not re.search(r"\\begin\{", q):
        return "orphan_end"
    if not ASK.search(q):
        return "no_ask"
    if META.search(q):
        return "residual_meta"
    tail = q.rstrip()
    if tail and tail[-1] not in "?.$)]":
        return "bad_ending"
    # unanswerable without an attached figure
    if re.search(r"\b(?:figure|diagram|graph below|shown below|as shown|indicated)\b",
                 q, re.IGNORECASE):
        return "figure_ref"
    # dangling opener: refers to context that does not exist
    if re.match(r"\s*(?:Now\b|Then\b|Next\b|Also\b|And\b|But\b|So\b|Thus\b|"
                r"Similarly\b|Again\b|Finally\b|[a-z])", q):
        return "dangling_opener"
    return None


def strong_filter(q: str) -> tuple[str | None, str]:
    """``(clean, "salvaged" | "intact")`` or ``(None, reason)``.

    "salvaged" means more than five characters were cut off the stripped input.
    """
    s = salvage(q)
    r = reject_reason(s)
    if r:
        return None, r
    return s, ("salvaged" if len(s) < len(q.strip()) - 5 else "intact")
