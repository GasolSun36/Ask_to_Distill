"""Vendored third-party trees and their patch records.

* every recorded patch matches the vendored tree (it reverse-applies cleanly);
* the a2d LiveCodeBench patch does what THIRD_PARTY.md says: few-shot files load from any
  working directory, and the dataset revision is pinned (and overridable).

The git-level completeness check (``git ls-files third_party``) lives in
``test_vendored_complete.py``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(os.path.abspath(__file__)).parents[2]
TP = REPO / "third_party"
PATCHES = TP / "patches"
LCB = TP / "LiveCodeBench"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
A2D_PATCHED = {
    "lcb_runner/benchmarks/code_generation.py",
    "lcb_runner/prompts/code_generation.py",
}


def _patch_bin() -> str:
    exe = shutil.which("patch")
    if exe is None:
        pytest.skip("`patch` not available")
    return exe


def _apply(patch_file: Path, root: Path, *, reverse: bool, dry_run: bool) -> None:
    cmd = [_patch_bin(), "-p1", "-d", str(root), "--batch", "--forward" if not reverse else "-R"]
    if dry_run:
        cmd.append("--dry-run")
    res = subprocess.run(cmd, stdin=patch_file.open("rb"), capture_output=True, text=True)
    assert res.returncode == 0, f"{patch_file.name} does not match {root}:\n{res.stdout}{res.stderr}"
    assert "offset" not in res.stdout and "fuzz" not in res.stdout, res.stdout


# --------------------------------------------------------------------------- patches


@pytest.mark.parametrize(
    "patch_name, tree",
    [
        ("verl_vs_rethinking_opd_ac26e38.patch", "verl"),
        ("evalplus_26d6d00.patch", "evalplus"),
        ("toolorchestra_c1c0a40.patch", "toolorchestra_tau2"),
    ],
)
def test_patch_matches_vendored_tree(patch_name: str, tree: str) -> None:
    _apply(PATCHES / patch_name, TP / tree, reverse=True, dry_run=True)


def test_livecodebench_patch_stack(tmp_path: Path) -> None:
    """a2d patch on top of our 28fef95 patch: undo both in order on a scratch copy."""
    work = tmp_path / "lcb"
    shutil.copytree(LCB, work, ignore=shutil.ignore_patterns("__pycache__", "output"))
    _apply(PATCHES / "livecodebench_a2d.patch", work, reverse=True, dry_run=False)
    for rel in A2D_PATCHED:
        assert (work / rel).read_text() != (LCB / rel).read_text()
    _apply(PATCHES / "livecodebench_28fef95.patch", work, reverse=True, dry_run=True)


def test_patch_set_is_complete() -> None:
    assert sorted(p.name for p in PATCHES.iterdir()) == [
        "evalplus_26d6d00.patch",
        "livecodebench_28fef95.patch",
        "livecodebench_a2d.patch",
        "toolorchestra_c1c0a40.patch",
        "verl_vs_rethinking_opd_ac26e38.patch",
    ]


# --------------------------------------------------------------------------- a2d LCB patch


def test_lcb_few_shot_loads_from_any_cwd(tmp_path: Path) -> None:
    code = (
        "import json; from lcb_runner.prompts import code_generation as cg; "
        "print(json.dumps([len(cg.func), len(cg.stdin)]))"
    )
    env = dict(os.environ, PYTHONPATH=str(LCB), PYTHONDONTWRITEBYTECODE="1")
    res = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    n_func, n_stdin = json.loads(res.stdout.strip().splitlines()[-1])
    assert n_func > 0 and n_stdin > 0


_REVISION_PROBE = """
import json, sys, huggingface_hub
calls = []
def fake(repo_id, filename, **kw):
    calls.append({"repo_id": repo_id, "filename": filename, **kw})
    return sys.argv[1]
huggingface_hub.hf_hub_download = fake
from lcb_runner.benchmarks import code_generation as b
rows = b._load_lcb_jsonl("v5_v6")
print(json.dumps({"calls": calls, "rows": len(rows)}))
"""


@pytest.mark.parametrize("override", [None, "main"])
def test_lcb_dataset_revision_is_pinned(tmp_path: Path, override: str | None) -> None:
    shard = tmp_path / "shard.jsonl"
    shard.write_text('{"a": 1}\n{"a": 2}\n')
    env = dict(os.environ, PYTHONPATH=str(LCB), PYTHONDONTWRITEBYTECODE="1")
    env.pop("LCB_DATASET_REVISION", None)
    if override:
        env["LCB_DATASET_REVISION"] = override
    res = subprocess.run([sys.executable, "-c", _REVISION_PROBE, str(shard)], cwd=tmp_path, env=env,
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert [c["filename"] for c in out["calls"]] == ["test5.jsonl", "test6.jsonl"]
    for c in out["calls"]:
        assert c["repo_id"] == "livecodebench/code_generation_lite"
        assert c["repo_type"] == "dataset"
        assert c["revision"] == (override or LCB_REVISION)
    assert out["rows"] == 4
