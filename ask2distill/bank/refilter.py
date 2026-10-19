"""Re-filter an aggregated pool with the current ``code_filter`` (``bank.refilter``).

Used by Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507 only. Its slice funnel ran under an earlier version of
``code_filter`` that no longer exists, before the other two code pairs
(Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct and Code_Qwen3-4B_Qwen3-4B-Instruct-2507) ran. Before its bank was rebuilt, the
already embedding-deduped pool was passed once more through the current
filter: ``comp_filter`` (format gate), then ``strong_filter`` -- whose salvage
may rewrite the text -- and finally an exact-string dedup (salvage can map two
pool entries onto the same text). Nothing is re-embedded; the filter only
removes. On the final run: 29,675 -> 26,275 rows.
"""
from __future__ import annotations

import os
from pathlib import Path

from ask2distill.common.io import iter_jsonl, write_jsonl
from ask2distill.filter.code_filter import comp_filter, strong_filter


def refilter_questions(qs: list[str]) -> list[str]:
    """``comp_filter`` -> ``strong_filter`` (salvaged text kept) -> exact dedup, in order."""
    kept = []
    for q in qs:
        t, w = comp_filter(q)
        if w:
            continue
        c, w2 = strong_filter(t)
        if not w2:
            kept.append(c)
    # The pool was already cross-shard embedding-deduped; the filter only removes.
    seen: set[str] = set()
    out = []
    for q in kept:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def refilter(pool: str | os.PathLike, out: str | os.PathLike) -> int:
    """Re-filter ``pool`` into ``out``; returns the number of rows written."""
    rows = refilter_questions([r["problem"] for r in iter_jsonl(pool)])
    write_jsonl(Path(out), ({"problem": q} for q in rows))
    print(f"[rebuild] {pool}: kept {len(rows)}", flush=True)
    return len(rows)
