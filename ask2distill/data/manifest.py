"""The data-release manifest: schema, file metadata and verification.

``data/MANIFEST.json`` is the only part of the data release that is committed.
It lists every released file with its md5, size, row count, license and
provenance, grouped so that ``a2d data download --only <group>`` can fetch a
subset. ``a2d data download`` checks every file it lands against it.

Where a file comes from is given by its optional ``source``:

* no ``source``: the release's own dataset repo (``release.repo_id``);
* ``{"kind": "hf", "repo_id", "path", "revision"}``: one file of another HF
  dataset repo (the seven synthetic banks are published as ``gasolsun/<pair>``);
* ``{"kind": "tau2_bank", "pools", "domains"}``: built locally from released
  tau2 card pools with ``ask2distill.tau2.bank.build_synth`` (the two parts of
  the agent bank).

``reference`` lists files the repository builds from public upstream data but
does not distribute -- the real-question training sets of the Vanilla arms,
the math and code evaluation sets, and the tau2 test tasks and domain context;
the builders (``a2d data build ...``) compare their output with the md5
recorded there.

Schema (``schema: 1``)::

    {
      "schema": 1,
      "release": {"name": ..., "repo_id": ..., "revision": ..., "created": ...},
      "groups": {"<group>": {"description": ..., "optional": bool}},
      "files": [
        {"path": "banks/<pair>.parquet",        # relative to $A2D_DATA
         "group": "banks",
         "md5": "fd38b03f30...",                # null while the file is pending
         "md5_prefix": "fd38b03f30",            # first 10 hex digits recorded in the inventory
         "size": 461234, "rows": 2889,
         "license": "...", "provenance": "...",
         "regenerable": "no" | "exact" | "equivalent",
         "builder": "a2d data build math" | null,
         "note": "...",
         "source": {...}}                       # optional, see above
      ],
      "reference": [
        {"path": "real/DAPO-Math-17k.parquet", "md5": ..., "size": ..., "rows": ...,
         "license": ..., "upstream": "<URL>", "builder": "a2d data build math",
         "provenance": ..., "note": ...}
      ]
    }

``md5`` may be ``null`` for an entry whose source was not available when the
manifest was written; such a file is then checked against ``md5_prefix`` and
``rows`` where those are known, and reported as only partially verified.
``release.pending`` lists release paths (directories and globs included) that
are not in the release yet.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ask2distill.common.io import atomic_write_text, md5_file
from ask2distill.paths import REPO_ROOT

SCHEMA = 1
REPO_MANIFEST = REPO_ROOT / "data" / "MANIFEST.json"
REGENERABLE = ("no", "exact", "equivalent")
ENTRY_KEYS = ("path", "group", "md5", "md5_prefix", "size", "rows", "license", "provenance",
              "regenerable", "builder", "note")
OPTIONAL_ENTRY_KEYS = ("source",)
REFERENCE_KEYS = ("path", "md5", "size", "rows", "license", "upstream", "builder", "provenance",
                  "note")
SOURCE_KINDS = {"hf": ("repo_id", "path", "revision"), "tau2_bank": ("pools", "domains")}


class ManifestError(RuntimeError):
    """The manifest is malformed, or a file does not match it."""


def load(path: str | os.PathLike = REPO_MANIFEST) -> dict:
    """Read and validate a manifest."""
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"{path} not found")
    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    validate(man)
    return man


def validate(man: Mapping[str, Any]) -> None:
    """Raise ``ManifestError`` listing every schema problem."""
    errs: list[str] = []
    if man.get("schema") != SCHEMA:
        errs.append(f"schema must be {SCHEMA}, got {man.get('schema')!r}")
    groups = man.get("groups") or {}
    seen: set[str] = set()
    all_paths = {str(f.get("path")) for f in man.get("files") or []}
    for i, e in enumerate(man.get("files") or []):
        where = e.get("path", f"files[{i}]")
        missing = [k for k in ENTRY_KEYS if k not in e]
        if missing:
            errs.append(f"{where}: missing keys {missing}")
        if e.get("group") not in groups:
            errs.append(f"{where}: unknown group {e.get('group')!r}")
        if e.get("regenerable") not in REGENERABLE:
            errs.append(f"{where}: regenerable must be one of {REGENERABLE}")
        p = str(e.get("path", ""))
        if p.startswith("/") or ".." in Path(p).parts:
            errs.append(f"{where}: path must be relative and inside the release")
        if p in seen:
            errs.append(f"{where}: listed twice")
        seen.add(p)
        src = e.get("source")
        if src is not None:
            kind = src.get("kind") if isinstance(src, Mapping) else None
            if kind not in SOURCE_KINDS:
                errs.append(f"{where}: source.kind must be one of {sorted(SOURCE_KINDS)}")
            else:
                miss = [k for k in SOURCE_KINDS[kind] if not src.get(k)]
                if miss:
                    errs.append(f"{where}: source of kind {kind!r} lacks {miss}")
            if kind == "tau2_bank":
                for pool in src.get("pools") or []:
                    if pool not in all_paths:
                        errs.append(f"{where}: source pool {pool!r} is not a release file")
    for i, e in enumerate(man.get("reference") or []):
        where = e.get("path", f"reference[{i}]")
        missing = [k for k in REFERENCE_KEYS if k not in e]
        if missing:
            errs.append(f"reference {where}: missing keys {missing}")
        if e.get("path") in seen:
            errs.append(f"reference {where}: also listed as a release file")
    if errs:
        raise ManifestError("invalid manifest:\n  " + "\n  ".join(errs))


def select(man: Mapping[str, Any], groups: Iterable[str] | None = None) -> list[dict]:
    """Entries of ``groups``; by default every non-optional group."""
    known = man["groups"]
    if groups:
        groups = list(groups)
        bad = [g for g in groups if g not in known]
        if bad:
            raise ManifestError(f"unknown group(s) {bad}; known: {sorted(known)}")
        wanted = set(groups)
    else:
        wanted = {g for g, spec in known.items() if not spec.get("optional", False)}
    return [e for e in man["files"] if e["group"] in wanted]


def count_rows(path: str | os.PathLike) -> int | None:
    """Rows of a jsonl / parquet file, elements of a JSON list; ``None`` otherwise."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, "rb") as fh:
            return sum(1 for line in fh if line.strip())
    if suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    if suffix == ".json":
        try:
            with open(path, encoding="utf-8") as fh:
                obj = json.load(fh)
        except (ValueError, UnicodeDecodeError):
            return None
        return len(obj) if isinstance(obj, list) else None
    return None


def describe(path: str | os.PathLike) -> dict[str, Any]:
    """``{md5, size, rows}`` of one file."""
    path = Path(path)
    return {"md5": md5_file(path), "size": path.stat().st_size, "rows": count_rows(path)}


@dataclass
class Report:
    """Outcome of ``verify``; ``ok`` only if nothing is missing or wrong."""

    checked: int = 0
    missing: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)   # no full md5 known: prefix / rows only

    @property
    def ok(self) -> bool:
        return not self.missing and not self.mismatched

    def summary(self) -> str:
        lines = [f"verified {self.checked} file(s): {len(self.missing)} missing, "
                 f"{len(self.mismatched)} mismatched, {len(self.partial)} without a full md5 (checked by prefix / rows only)"]
        lines += [f"  MISSING   {p}" for p in self.missing]
        lines += [f"  MISMATCH  {p}" for p in self.mismatched]
        lines += [f"  PARTIAL   {p}" for p in self.partial]
        return "\n".join(lines)


def source_kind(entry: Mapping[str, Any]) -> str:
    """``"release"`` (the release repo), ``"hf"`` (another HF dataset repo) or ``"tau2_bank"``."""
    src = entry.get("source")
    return "release" if not src else str(src["kind"])


def compare_entries(man: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Release files and reference entries: everything a builder can compare its output with."""
    return list(man.get("files") or []) + list(man.get("reference") or [])


def report_reference(data_dir: str | os.PathLike, paths: Iterable[str],
                     manifest_path: str | os.PathLike | None = None, log=print) -> None:
    """Print whether each built file of ``paths`` equals the md5 recorded in the manifest."""
    try:
        man = load(manifest_path or REPO_MANIFEST)
    except ManifestError:
        return
    wanted = set(paths)
    for e in compare_entries(man):
        if e["path"] not in wanted:
            continue
        status, detail = check_entry(data_dir, e)
        if status == "ok":
            log(f"  {e['path']}: identical to the file the final runs used (md5 {e['md5']})")
        elif status == "mismatch":
            log(f"  note: {detail} (the final runs used md5 {e.get('md5')}; {e.get('note') or 'see docs/DATA.md'})")


def check_entry(root: str | os.PathLike, entry: Mapping[str, Any]) -> tuple[str, str]:
    """Check one file; returns ``(status, detail)`` with status ok / partial / missing / mismatch."""
    path = Path(root) / entry["path"]
    if not path.is_file():
        return "missing", str(path)
    got = md5_file(path)
    if entry.get("md5"):
        if got != entry["md5"]:
            return "mismatch", f"{entry['path']}: md5 {got} != {entry['md5']}"
        return "ok", ""
    prefix = entry.get("md5_prefix")
    if prefix and not got.startswith(prefix):
        return "mismatch", f"{entry['path']}: md5 {got} does not start with {prefix}"
    rows = entry.get("rows")
    if rows is not None and count_rows(path) != rows:
        return "mismatch", f"{entry['path']}: {count_rows(path)} rows != {rows}"
    return "partial", entry["path"]


def verify(root: str | os.PathLike, entries: Iterable[Mapping[str, Any]]) -> Report:
    """Check every entry's file under ``root``."""
    rep = Report()
    for e in entries:
        status, detail = check_entry(root, e)
        rep.checked += 1
        if status == "missing":
            rep.missing.append(e["path"])
        elif status == "mismatch":
            rep.mismatched.append(detail)
        elif status == "partial":
            rep.partial.append(detail)
    return rep


def dumps(man: Mapping[str, Any]) -> str:
    """Canonical text form: files sorted by path, stable key order, trailing newline."""
    out = {k: man[k] for k in ("schema", "release", "groups") if k in man}
    out["files"] = [{**{k: e.get(k) for k in ENTRY_KEYS},
                     **{k: e[k] for k in OPTIONAL_ENTRY_KEYS if e.get(k) is not None}}
                    for e in sorted(man["files"], key=lambda e: e["path"])]
    if man.get("reference"):
        out["reference"] = [{k: e.get(k) for k in REFERENCE_KEYS}
                            for e in sorted(man["reference"], key=lambda e: e["path"])]
    return json.dumps(out, indent=1, ensure_ascii=False) + "\n"


def write(path: str | os.PathLike, man: Mapping[str, Any]) -> Path:
    """Validate and write a manifest atomically."""
    validate(man)
    return atomic_write_text(path, dumps(man))
