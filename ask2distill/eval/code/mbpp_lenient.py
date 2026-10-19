"""Re-score MBPP samples after repairing entry-point names.

MBPP hides the required function name inside the assert in the prompt, so a
model that solves the task but calls its function something else scores zero.
Training on competitive-programming problems, which never impose an external
function name, drifts exactly that habit: tasks lose their entry point during
training while the logic stays correct (``to_uppercase`` for ``is_upper``).

This separates "cannot solve it" from "did not follow the interface". The
rename is deliberately conservative -- only when the sample defines exactly ONE
top-level function -- so it repairs naming and nothing else.

Usage::

    python -m ask2distill.eval.code.mbpp_lenient --samples S.jsonl --out lenient.jsonl
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from typing import Mapping, Sequence


def top_level_funcs(src: str) -> list[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    return [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]


def repair(src: str, entry: str | None) -> tuple[str, bool]:
    """Rename the single top-level function of ``src`` to ``entry``; ``(source, renamed?)``."""
    if not entry or re.search(rf"\bdef\s+{re.escape(entry)}\s*\(", src):
        return src, False
    names = top_level_funcs(src)
    if len(names) != 1:
        return src, False          # ambiguous: leave it alone
    # \b on both sides so a name that is a prefix of another identifier is safe;
    # recursive calls get renamed with the definition, which is what we want.
    return re.sub(rf"\b{re.escape(names[0])}\b", entry, src), True


def rescore(samples: str, out: str, entry_points: Mapping[str, str] | None = None) -> tuple[int, int]:
    """Write the repaired samples to ``out``; returns ``(n_samples, n_renamed)``.

    ``entry_points`` maps MBPP+ task ids to their function names (default: from
    ``evalplus.data.get_mbpp_plus``).
    """
    if entry_points is None:
        from evalplus.data import get_mbpp_plus
        entry_points = {k: v["entry_point"] for k, v in get_mbpp_plus().items()}
    n = fixed = 0
    with open(out, "w") as f, open(samples) as src:
        for line in src:
            r = json.loads(line)
            n += 1
            r["solution"], did = repair(r["solution"], entry_points.get(r["task_id"]))
            fixed += did
            f.write(json.dumps(r) + "\n")
    print(f"[lenient] {os.path.basename(samples)}: {n} samples, renamed {fixed}", flush=True)
    return n, fixed


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repair MBPP entry-point names before re-scoring.")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rescore(a.samples, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
