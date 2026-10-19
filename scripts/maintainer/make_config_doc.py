"""Regenerate the key reference in docs/CONFIG.md from the YAML files.

    python scripts/maintainer/make_config_doc.py [--check]

Every key of every file under ``configs/`` becomes one table row: its dotted
path, its value as written, and the comment on its line plus any comment
lines that continue it -- the reason for a value where it is not obvious.
The hand-written part of docs/CONFIG.md, above
the ``<!-- BEGIN GENERATED -->`` marker, is kept. ``--check`` exits 1 if the
file is out of date (``tests/unit/test_docs.py`` runs it).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "CONFIG.md"
MARKER = "<!-- BEGIN GENERATED: scripts/maintainer/make_config_doc.py -->"
ORDER = ["base/math.yaml", "base/code.yaml", "base/tau2.yaml", "base/opd.yaml", "base/eval_math.yaml",
         "base/eval_code.yaml",
         "pairs/Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500.yaml",
         "pairs/Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B.yaml",
         "pairs/Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B.yaml",
         "pairs/Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507.yaml",
         "pairs/Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct.yaml",
         "pairs/Code_Qwen3-4B_Qwen3-4B-Instruct-2507.yaml",
         "pairs/Agent_Qwen3-4B_Qwen3-4B-Thinking-2507.yaml", "pairs/TEMPLATE_custom.yaml",
         "smoke/math_smoke.yaml", "smoke/code_smoke.yaml", "smoke/tau2_smoke.yaml"]
KEY_LINE = re.compile(r"^(?P<indent> *)(?P<dash>- )?(?P<key>[A-Za-z0-9_+.\-]+):(?:\s+(?P<rest>.*))?$")


def _split_comment(text: str) -> tuple[str, str]:
    """``value  # comment`` -> (value, comment), ignoring ``#`` inside quotes."""
    quote = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or text[i - 1].isspace()):
            return text[:i].rstrip(), text[i + 1:].strip()
    return text.rstrip(), ""


def _balance(s: str) -> int:
    return s.count("{") + s.count("[") - s.count("}") - s.count("]")


def parse(path: Path) -> list[tuple[str, str, str]]:
    """``[(dotted key, value, note)]`` in file order.

    A comment line aligned with the previous line's inline comment continues
    it; any other comment block (indented like a key) is the explanation of the
    key that follows and is put in front of that key's own comment.
    """
    rows: list[list[str]] = []
    stack: list[tuple[int, str]] = []
    pending: list[str] = []
    last_col = -1                                   # column of the previous inline comment
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        stripped = raw.strip()
        if not stripped:
            last_col = -1
            continue
        if stripped.startswith("#"):
            text = stripped.lstrip("#").strip()
            col = len(raw) - len(raw.lstrip())
            if rows and last_col >= 0 and col == last_col and not pending:
                rows[-1][2] += " " + text
            elif raw.startswith(" "):
                pending.append(text)
            continue
        m = KEY_LINE.match(raw)
        if not m:
            if stripped.startswith("- ") and rows:          # list item of the previous key
                value, comment = _split_comment(stripped[2:])
                rows[-1][1] += (", " if rows[-1][1] else "") + value
                if comment:
                    rows[-1][2] += (" " if rows[-1][2] else "") + comment
            continue
        indent = len(m.group("indent")) + (2 if m.group("dash") else 0)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        key = ".".join([k for _, k in stack] + [m.group("key")])
        rest = m.group("rest") or ""
        value, comment = _split_comment(rest)
        last_col = raw.find("#", len(raw) - len(rest)) if comment else -1
        while _balance(value) > 0 and i < len(lines):     # multi-line flow mapping / list
            more, c2 = _split_comment(lines[i].strip())
            i += 1
            value += " " + more
            if c2:
                comment += (" " if comment else "") + c2
        note = " ".join(pending + ([comment] if comment else []))
        pending = []
        if value:
            rows.append([key, value, note])
        else:
            stack.append((indent, m.group("key")))
            if note:
                rows.append([key, "", note])
    return [(k, v, c) for k, v, c in rows]


def _cell(s: str, limit: int = 120) -> str:
    s = s.replace("|", "\\|")
    return s if len(s) <= limit else s[:limit - 1] + "…"


def render() -> str:
    out = []
    for rel in ORDER:
        path = REPO / "configs" / rel
        out += [f"### `configs/{rel}`", "", "| Key | Value | Note |", "|---|---|---|"]
        for key, value, comment in parse(path):
            out.append(f"| `{key}` | {('`' + _cell(value, 80) + '`') if value else ''} | {_cell(comment, 400)} |")
        out.append("")
    # configs/topics/ holds plain topic lists, not configuration keys.
    missing = sorted(p.relative_to(REPO / "configs").as_posix() for p in (REPO / "configs").rglob("*.yaml")
                     if p.relative_to(REPO / "configs").as_posix() not in ORDER
                     and not p.relative_to(REPO / "configs").as_posix().startswith("topics/"))
    if missing:
        raise SystemExit(f"config files not in ORDER: {missing}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if docs/CONFIG.md is out of date")
    a = ap.parse_args(argv)
    text = DOC.read_text()
    if MARKER not in text:
        raise SystemExit(f"{DOC} has no generation marker")
    new = text.split(MARKER)[0] + MARKER + "\n\n" + render()
    if a.check:
        if new != text:
            print(f"{DOC} is out of date: run python {Path(__file__).relative_to(REPO)}", file=sys.stderr)
            return 1
        return 0
    DOC.write_text(new)
    print(f"wrote {DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
