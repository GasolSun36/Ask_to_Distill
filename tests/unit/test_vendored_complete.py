"""Version control keeps every vendored file (the .gitignore cannot drop one).

An unanchored ignore pattern -- ``models/`` instead of ``/models/`` -- would
silently drop ``third_party/verl/verl/models/``, and the vendored trees bring
their own ``.gitignore`` files that apply inside them. This checks the set git
tracks (or would track on a first ``git add -A``; ``conftest.list_tracked_files``)
under ``third_party/``:

* every vendored file on disk is tracked, apart from run-time
  artefacts, and ``verl/models/transformers/monkey_patch.py`` is among them;
* the root ``.gitignore`` uses anchored patterns only.

A vendored tree's own ``.gitignore`` can ignore a file its upstream tracks
(evalplus ignores ``*.png`` yet ships ``gallary/overview.png``). Such files are
listed in ``FORCE_ADD`` and must be committed with ``git add -f``; in a real
checkout the test then sees them tracked.
"""
from __future__ import annotations

import re

import pytest

from conftest import REPO, list_tracked_files

TP = REPO / "third_party"
# Produced at run time inside the vendored trees; never part of the source.
RUNTIME_ONLY = re.compile(
    r"(^|/)(__pycache__|[^/]+\.egg-info)(/|$)|\.pyc$"
    r"|^third_party/evalplus/evalplus/_version\.py$"
    r"|^third_party/toolorchestra_tau2/data/"
    r"|^third_party/toolorchestra_tau2/evaluation/tau2-bench/cut_error_message\.json$"
    r"|^third_party/LiveCodeBench/output/")
# Ignored by the vendored tree's own .gitignore but part of its upstream: `git add -f`.
FORCE_ADD = {"third_party/evalplus/gallary/overview.png"}
MUST_TRACK = "third_party/verl/verl/models/transformers/monkey_patch.py"


@pytest.fixture(scope="module")
def tracked():
    try:
        files, mode = list_tracked_files()
    except Exception as e:  # git missing or failing: nothing to check against
        pytest.skip(f"git unavailable: {e}")
    return {f for f in files if f.startswith("third_party/")}, mode


def _on_disk() -> set[str]:
    out = set()
    for p in TP.rglob("*"):
        if p.is_file():
            rel = p.relative_to(REPO).as_posix()
            if not RUNTIME_ONLY.search(rel):
                out.add(rel)
    return out


def test_monkey_patch_is_tracked(tracked):
    files, _ = tracked
    assert (REPO / MUST_TRACK).is_file()
    assert MUST_TRACK in files


def test_every_vendored_file_is_tracked(tracked):
    files, mode = tracked
    missing = _on_disk() - files
    if mode == "simulated":
        missing -= FORCE_ADD
    assert not missing, (f"{len(missing)} vendored file(s) would not be committed "
                         f"({mode}):\n" + "\n".join(sorted(missing)[:50]))


def test_force_add_list_is_current(tracked):
    files, mode = tracked
    for f in FORCE_ADD:
        assert (REPO / f).is_file(), f"{f} no longer vendored; drop it from FORCE_ADD"
        if mode == "simulated":
            assert f not in files, f"{f} is no longer ignored; drop it from FORCE_ADD"


def test_root_gitignore_is_anchored():
    bad = []
    for line in (REPO / ".gitignore").read_text().splitlines():
        pat = line.strip()
        if not pat or pat.startswith("#"):
            continue
        body = pat[1:] if pat.startswith("!") else pat
        # A pattern with a slash before its last character is matched relative
        # to the .gitignore only if it starts with "/"; without one it can hit
        # any directory level (e.g. `models/`).
        if "/" in body.rstrip("/") and not body.startswith("/"):
            bad.append(pat)
        if body.endswith("/") and "/" not in body.rstrip("/") and body.strip("/") not in (
                "__pycache__",) and not body.endswith(".egg-info/"):
            bad.append(pat)
    assert not bad, f"unanchored .gitignore patterns: {bad}"
