"""The documentation stays true to the code.

* docs/CONFIG.md's generated key reference matches the YAML files;
* every ``a2d ...`` command the docs show is a registered CLI command, and every
  ``launchers/slurm/submit.sh <stage>`` names a stage submit.sh knows;
* every repository path the docs name exists.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCS = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md"))]
TOP_DIRS = ("ask2distill", "configs", "launchers", "env", "tests", "scripts", "docs", "release",
            "third_party", "data")


def _text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_config_reference_is_current():
    res = subprocess.run([sys.executable, str(REPO / "scripts/maintainer/make_config_doc.py"), "--check"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_a2d_commands_exist(doc):
    from ask2distill.cli import REGISTRY
    bad = []
    for m in re.finditer(r"(?<![\w/.-])a2d ((?:[a-z][a-z0-9-]*)(?: [a-z][a-z0-9-]*){0,2})", _text(doc)):
        words = m.group(1).split()
        if words[0] == "env":                      # prose: "the a2d env" (the conda environment)
            continue
        known = any(tuple(words[:n]) in REGISTRY for n in range(len(words), 0, -1))
        prefix = any(k[:len(words)] == tuple(words) for k in REGISTRY)   # e.g. `a2d data build {...}`
        if not (known or prefix):
            bad.append(m.group(0))
    assert not bad, f"{doc.name}: unknown a2d commands {sorted(set(bad))}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_submit_stages_exist(doc):
    script = _text(REPO / "launchers/slurm/submit.sh")
    stages = set(re.findall(r"^\s+([a-z0-9-]+)\)\s+SCRIPT=", script, flags=re.M))
    used = set(re.findall(r"(?<![\w-])submit\.sh ([a-z0-9-]+) ", _text(doc))) - {"math", "code", "tau2"}
    assert used <= stages, f"{doc.name}: unknown submit.sh stages {sorted(used - stages)}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_named_paths_exist(doc):
    missing = []
    for tok in re.findall(r"`([^`\s]+)`", _text(doc)) + re.findall(r"\]\(([^)]+)\)", _text(doc)):
        tok = tok.rstrip(".,:;")
        root_file = tok in ("README.md", "LICENSE", "THIRD_PARTY.md", "pyproject.toml")
        doc_name = re.fullmatch(r"[A-Z_]+\.md", tok) is not None    # e.g. DATA.md (a sibling doc)
        if not root_file and not doc_name and not ("/" in tok and tok.startswith(TOP_DIRS)):
            continue
        if any(c in tok for c in "<>{}*$") or tok.endswith("/"):
            continue
        rel = tok if not doc.parent.name == "docs" or tok.startswith(TOP_DIRS) else f"docs/{tok}"
        if not (REPO / rel).exists() and not (REPO / tok).exists():
            missing.append(tok)
    # Run-time locations (gitignored or produced by a stage) are not part of the tree.
    runtime = re.compile(r"^(data/(?!MANIFEST\.json)|third_party/toolorchestra_tau2/data|configs/paths\.env$"
                         r"|launchers/slurm/cluster\.env$)")
    missing = [m for m in missing if not runtime.match(m)]
    assert not missing, f"{doc.name}: paths that do not exist: {sorted(set(missing))}"
