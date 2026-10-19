"""Fetch the data release and the synthetic banks into ``$A2D_DATA`` and md5-verify them.

    a2d data download [--repo-id ID] [--revision R] [--endpoint URL] [--only GROUP ...]
    a2d data download --from-dir DIR [--banks-dir DIR] [--only GROUP ...]

Each manifest entry names where its file comes from (``manifest.source_kind``):

* **the release repo** (no ``source``): ``huggingface_hub.snapshot_download``
  of the files the manifest lists for the selected groups (δ_ref records, the
  tau2 synthesizer-RL prompts and the tau2 card pools). The repo id comes from
  ``--repo-id``, else ``$A2D_HF_DATA_REPO``, else the manifest's
  ``release.repo_id``, else ``DEFAULT_REPO_ID`` (``gasolsun/Ask_to_Distill-data``);
  ``--endpoint`` (or ``$HF_ENDPOINT``) selects a mirror.
* **another HF dataset repo** (``source.kind "hf"``): the seven synthetic
  banks are the published datasets ``gasolsun/<pair>``; their
  ``data/train-00000-of-00001.parquet`` at the pinned ``source.revision`` is
  copied to ``banks/<pair>.parquet``.
* **built locally** (``source.kind "tau2_bank"``): the two parts of the agent
  bank are rebuilt from the released tau2 card pools with
  ``ask2distill.tau2.bank.build_synth`` once the pools are on disk.

Offline (clusters whose HF file CDN is blocked, or tests): ``--from-dir DIR``
reads the release-repo files from ``DIR``, a snapshot of the release repo;
``--banks-dir DIR`` reads the banks from a local directory holding, per pair,
``<pair>.parquet``, ``banks/<pair>.parquet`` or ``<pair>/data/train-00000-of-00001.parquet``
(a snapshot of the bank's dataset repo). Without ``--banks-dir`` the banks are
downloaded from their dataset repos.

Every file is checked against the committed ``data/MANIFEST.json`` (md5). The
release's own ``MANIFEST.json`` and ``README.md`` are never copied over the
committed manifest. Files listed under ``reference`` -- the real-question
training sets of the Vanilla arms, the evaluation sets and the tau2 test tasks
and domain context -- are not distributed: ``a2d data build ...`` makes them
from their public sources and compares them with the md5 recorded there
(docs/DATA.md).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Mapping

from ask2distill.data import manifest as M
from ask2distill.paths import get_paths


# The public dataset repo of the release; --repo-id or $A2D_HF_DATA_REPO override it.
DEFAULT_REPO_ID = "gasolsun/Ask_to_Distill-data"


def resolve_repo_id(repo_id: str | None, man: Mapping) -> str:
    """``repo_id``, else ``$A2D_HF_DATA_REPO``, else the manifest's ``release.repo_id``, else the default."""
    return (repo_id or os.environ.get("A2D_HF_DATA_REPO", "").strip()
            or man["release"].get("repo_id") or DEFAULT_REPO_ID)


class DownloadError(RuntimeError):
    """The release could not be fetched or does not match the manifest."""


def copy_from_dir(src: Path, dst: Path, entries: Iterable[Mapping], *, log=print) -> list[str]:
    """Copy each entry's file from ``src`` to ``dst``; returns the paths not found in ``src``.

    A destination that already matches the manifest md5 is left alone, so a
    re-run only copies what changed.
    """
    absent = []
    for e in entries:
        s, d = src / e["path"], dst / e["path"]
        if not s.is_file():
            absent.append(e["path"])
            continue
        if d.is_file() and e.get("md5") and M.check_entry(dst, e)[0] == "ok":
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_name(f".{d.name}.part")
        shutil.copyfile(s, tmp)
        os.replace(tmp, d)
        log(f"copied {e['path']}")
    return absent


def fetch_hub(repo_id: str, dst: Path, entries: Iterable[Mapping], *, revision: str | None = None,
              endpoint: str | None = None) -> None:
    """``snapshot_download`` restricted to the manifest paths of the selected groups."""
    from huggingface_hub import snapshot_download
    patterns = sorted({e["path"] for e in entries})
    if not patterns:
        return
    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision, endpoint=endpoint,
                      local_dir=str(dst), allow_patterns=patterns)


def local_bank(banks_dir: Path, entry: Mapping) -> Path | None:
    """The local copy of a ``source.kind "hf"`` bank in ``banks_dir``, or None."""
    name = Path(entry["path"]).name
    repo = str(entry["source"]["repo_id"]).split("/")[-1]
    for cand in (banks_dir / name, banks_dir / entry["path"], banks_dir / repo / entry["source"]["path"]):
        if cand.is_file():
            return cand
    return None


def copy_banks(banks_dir: Path, dst: Path, entries: Iterable[Mapping], *, log=print) -> list[str]:
    """Copy each ``source.kind "hf"`` bank from ``banks_dir``; returns the paths not found."""
    absent = []
    for e in entries:
        d = dst / e["path"]
        if d.is_file() and e.get("md5") and M.check_entry(dst, e)[0] == "ok":
            continue
        s = local_bank(banks_dir, e)
        if s is None:
            absent.append(e["path"])
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_name(f".{d.name}.part")
        shutil.copyfile(s, tmp)
        os.replace(tmp, d)
        log(f"copied {e['path']} from {s}")
    return absent


def fetch_sourced(dst: Path, entries: Iterable[Mapping], *, endpoint: str | None = None,
                  log=print) -> None:
    """Copy each ``source.kind "hf"`` file from its own dataset repo to ``dst/<path>``."""
    from huggingface_hub import hf_hub_download
    for e in entries:
        d = dst / e["path"]
        if d.is_file() and e.get("md5") and M.check_entry(dst, e)[0] == "ok":
            continue
        src = e["source"]
        log(f"downloading {src['repo_id']}@{str(src['revision'])[:8]}:{src['path']} -> {e['path']}")
        got = Path(hf_hub_download(src["repo_id"], src["path"], repo_type="dataset",
                                   revision=src["revision"], endpoint=endpoint))
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = d.with_name(f".{d.name}.part")
        shutil.copyfile(got, tmp)
        os.replace(tmp, d)


def build_tau2_parts(dst: Path, entries: Iterable[Mapping], man: Mapping, *, log=print) -> list[str]:
    """Rebuild every ``source.kind "tau2_bank"`` file from its released pools.

    Returns the paths that could not be built because a pool is missing or does
    not match the manifest.
    """
    by_path = {f["path"]: f for f in man["files"]}
    blocked = []
    for e in entries:
        if (dst / e["path"]).is_file() and M.check_entry(dst, e)[0] == "ok":
            continue
        src = e["source"]
        pools = [str(p) for p in src["pools"]]
        bad = [p for p in pools if M.check_entry(dst, by_path[p])[0] not in ("ok", "partial")]
        if bad:
            log(f"cannot build {e['path']}: pool(s) {bad} missing or wrong "
                "(fetch them with `a2d data download --only tau2`)")
            blocked.append(e["path"])
            continue
        from ask2distill.tau2.bank import build_synth
        build_synth([dst / p for p in pools], dst / e["path"], list(src["domains"]))
        log(f"built {e['path']} from {', '.join(pools)}")
    return blocked


def run(*, data_dir: Path, groups: Iterable[str] | None = None, from_dir: str | None = None,
        banks_dir: str | None = None, repo_id: str | None = None, revision: str | None = None,
        endpoint: str | None = None, manifest_path: str | os.PathLike = M.REPO_MANIFEST,
        log=print) -> M.Report:
    """Fetch the selected groups into ``data_dir`` and verify them."""
    man = M.load(manifest_path)
    entries = M.select(man, groups)
    kinds: dict[str, list[dict]] = {"release": [], "hf": [], "tau2_bank": []}
    for e in entries:
        kinds[M.source_kind(e)].append(e)
    pending = man["release"].get("pending", [])
    data_dir.mkdir(parents=True, exist_ok=True)
    if from_dir:
        src = Path(from_dir)
        if not src.is_dir():
            raise DownloadError(f"--from-dir {src} is not a directory")
        absent = copy_from_dir(src, data_dir, kinds["release"], log=log)
        if absent:
            log(f"{len(absent)} file(s) not in {src}: " + ", ".join(absent[:5])
                + (" ..." if len(absent) > 5 else ""))
    elif kinds["release"]:
        repo = resolve_repo_id(repo_id, man)
        rev = revision or man["release"].get("revision")
        log(f"downloading {len(kinds['release'])} file(s) from {repo}@{rev or 'main'}")
        fetch_hub(repo, data_dir, kinds["release"], revision=rev, endpoint=endpoint)
    if kinds["hf"]:
        if banks_dir:
            bdir = Path(banks_dir)
            if not bdir.is_dir():
                raise DownloadError(f"--banks-dir {bdir} is not a directory")
            absent = copy_banks(bdir, data_dir, kinds["hf"], log=log)
            if absent:
                log(f"{len(absent)} bank(s) not in {bdir}: " + ", ".join(absent))
        else:
            log(f"downloading {len(kinds['hf'])} bank(s) from their own dataset repos")
            fetch_sourced(data_dir, kinds["hf"], endpoint=endpoint, log=log)
    if kinds["tau2_bank"]:
        build_tau2_parts(data_dir, kinds["tau2_bank"], man, log=log)
    rep = M.verify(data_dir, entries)
    log(rep.summary())
    if pending:
        log(f"note: the committed manifest lists {len(pending)} release path(s) as pending "
            "(not in the release when it was written); those files are checked by md5 prefix and "
            "row count only, and pending directories not at all")
    if man.get("reference"):
        builders = sorted({str(e["builder"]) for e in man["reference"]})
        log(f"not distributed: {len(man['reference'])} file(s) built from their public sources by "
            + ", ".join(f"`{b}`" for b in builders) + " (docs/DATA.md)")
    return rep


def main(cfg=None, args: argparse.Namespace | None = None) -> int:
    """``a2d data download`` entry point (``cfg`` is unused)."""
    if args is None:
        args = parse_args(sys.argv[1:])
    try:
        rep = run(data_dir=get_paths().data, groups=getattr(args, "only", None),
                  from_dir=getattr(args, "from_dir", None), banks_dir=getattr(args, "banks_dir", None),
                  repo_id=getattr(args, "repo_id", None),
                  revision=getattr(args, "revision", None), endpoint=getattr(args, "endpoint", None))
    except (DownloadError, M.ManifestError) as e:
        raise SystemExit(f"a2d data download: {e}") from None
    return 0 if rep.ok else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m ask2distill.data.download",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo-id", help=f"HF dataset repo (default: $A2D_HF_DATA_REPO, else {DEFAULT_REPO_ID})")
    ap.add_argument("--revision")
    ap.add_argument("--endpoint")
    ap.add_argument("--from-dir", help="a local snapshot of the release repo")
    ap.add_argument("--banks-dir", help="local copies of the bank datasets (no download)")
    ap.add_argument("--only", nargs="+", help="release groups")
    return ap.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(None, parse_args(sys.argv[1:])))
