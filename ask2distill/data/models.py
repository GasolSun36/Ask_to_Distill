"""Base models at pinned revisions, and the ``*-SynNoThink`` synthesizer bases.

    a2d models download --pair P | --all [--modelscope] [--endpoint URL]
    a2d models syn-nothink --pair Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B
    a2d models syn-nothink --pair Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B

**Download.** Every pair config lists its models under ``models.hf`` as
``{role: {repo, revision}}``; the local directory is ``models.<role>``
(``$A2D_MODELS/<name>``). Revisions are the full commit SHAs of the model
downloads the final runs used (their ``.cache/huggingface/download/*.metadata``).

* Default: ``huggingface_hub.snapshot_download(repo, revision=<sha>,
  local_dir=...)``; ``--endpoint`` (or ``$HF_ENDPOINT``) selects a mirror.
* ``--modelscope``: on clusters where the HF file CDN is firewalled but the HF
  API is reachable (where huggingface_hub, hf_xet and hf-mirror all fail),
  files come from the ModelScope REST API instead and
  are verified against the HF API -- LFS files by the sha256 the API reports
  for the pinned revision, small files by their git blob id -- so a ModelScope
  copy that differs from the pinned HF revision is rejected, never used. The
  HF-style ``.metadata`` files are then written, so the directory is
  indistinguishable from an HF download.

**SynNoThink.** The DeepSeek-R1-Distill synthesizers (the two Math_DeepSeek-* pairs) start from the
student with a chat template whose generation prefix closes ``<think>``
immediately -- the DeepSeek analogue of Qwen3's ``enable_thinking=False``.
``make_syn_nothink`` rebuilds them: weights and other large files are
symlinked, small config files copied, and the prefix
``'<｜Assistant｜><think>\\n'`` -> ``'<｜Assistant｜><think>\\n</think>\\n\\n'`` is
patched in ``tokenizer_config.json``'s ``chat_template`` and in any
``chat_template.jinja``; only that key may differ afterwards (checked).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

GEN_PREFIX = "{{'<｜Assistant｜><think>\\n'}}"
GEN_PREFIX_NOTHINK = "{{'<｜Assistant｜><think>\\n</think>\\n\\n'}}"
TEMPLATE_KEY = "chat_template"
# The shipped DeepSeek-R1-Distill SynNoThink tokenizer_config.json files were re-serialised
# with json.dumps(indent=1, ensure_ascii=False) and no trailing newline; keeping
# that makes the rebuilt file byte-identical to the one the synthesizers used.
TOKENIZER_CONFIG_INDENT = 1
# Small text files are copied into a SynNoThink directory; everything else
# (weights, directories, licenses) is symlinked to the source model.
COPY_SUFFIXES = (".json", ".txt", ".jinja", ".model", ".tiktoken", ".py")
COPY_NAMES = ("README.md",)

MODELSCOPE_API = "https://www.modelscope.cn/api/v1/models"
CHUNK = 1 << 20
RETRIES = 8
# Documentation files may legitimately be missing on, or differ in, a ModelScope
# mirror; they are skipped with a warning instead of failing the download.
DOC_FILES = ("README.md", "LICENSE", "LICENSE.txt", ".gitattributes")
DOC_DIRS = ("figures/", "assets/")


class ModelError(RuntimeError):
    """A model could not be fetched, or does not match its pinned revision."""


# --------------------------------------------------------------------------- specs

@dataclass(frozen=True)
class ModelSpec:
    """One base model: HF repo, pinned commit and local directory."""

    repo: str
    revision: str
    local_dir: Path

    @property
    def name(self) -> str:
        return self.local_dir.name


def pair_specs(cfg) -> list[ModelSpec]:
    """The ``models.hf`` entries of one pair config."""
    out = []
    for role, spec in cfg.models.hf.items():
        if role not in cfg.models or cfg.models[role] is None:
            raise ModelError(f"models.hf.{role} has no models.{role} path in pair {cfg.pair}")
        out.append(ModelSpec(str(spec.repo), str(spec.revision), Path(str(cfg.models[role]))))
    return out


def collect_specs(cfgs: Iterable) -> list[ModelSpec]:
    """Union over pairs; a model named twice must carry the same repo and revision."""
    by_dir: dict[Path, ModelSpec] = {}
    for cfg in cfgs:
        for s in pair_specs(cfg):
            prev = by_dir.get(s.local_dir)
            if prev is not None and (prev.repo, prev.revision) != (s.repo, s.revision):
                raise ModelError(f"{s.local_dir} is pinned twice: {prev.repo}@{prev.revision} "
                                 f"vs {s.repo}@{s.revision} (pair {cfg.pair})")
            by_dir[s.local_dir] = s
    return sorted(by_dir.values(), key=lambda s: s.name)


# --------------------------------------------------------------------------- hashing

def git_blob_sha1(path: str | os.PathLike) -> str:
    """Git's blob id of a file (what the HF API reports for a non-LFS file)."""
    h = hashlib.sha1()
    size = os.path.getsize(path)
    h.update(f"blob {size}\0".encode())
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata_dir(local_dir: Path) -> Path:
    return local_dir / ".cache" / "huggingface" / "download"


def read_metadata(local_dir: Path) -> dict[str, tuple[str, str]]:
    """``{relative path: (commit, etag)}`` from HF ``local_dir`` download metadata."""
    root = metadata_dir(local_dir)
    out = {}
    for f in sorted(root.rglob("*.metadata")) if root.is_dir() else []:
        lines = f.read_text().splitlines()
        if len(lines) >= 2:
            out[f.relative_to(root).as_posix()[:-len(".metadata")]] = (lines[0].strip(), lines[1].strip())
    return out


def write_metadata(local_dir: Path, rel: str, commit: str, etag: str) -> None:
    """HF-style ``<rel>.metadata`` (commit, etag, timestamp), as snapshot_download writes it."""
    p = metadata_dir(local_dir) / f"{rel}.metadata"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{commit}\n{etag}\n{time.time()}\n")


def is_complete(spec: ModelSpec) -> bool:
    """Every recorded file is present and every record names the pinned commit."""
    meta = read_metadata(spec.local_dir)
    return bool(meta) and all(c == spec.revision and (spec.local_dir / rel).is_file()
                              for rel, (c, _) in meta.items()) \
        and (spec.local_dir / "config.json").is_file()


def verify_against_metadata(local_dir: Path, *, hash_large: bool = False) -> list[str]:
    """Files whose content does not match their recorded etag.

    Small files are checked by git blob id; LFS files (64-hex etag) by sha256
    only when ``hash_large`` (tens of GB otherwise).
    """
    bad = []
    for rel, (_, etag) in read_metadata(local_dir).items():
        p = local_dir / rel
        if not p.is_file():
            bad.append(f"{rel}: missing")
        elif len(etag) == 64:
            if hash_large and sha256_file(p) != etag:
                bad.append(f"{rel}: sha256 mismatch")
        elif git_blob_sha1(p) != etag:
            bad.append(f"{rel}: blob id mismatch")
    return bad


# --------------------------------------------------------------------------- HF

def hf_tree(spec: ModelSpec, endpoint: str | None = None) -> list[dict]:
    """``[{path, size, blob_id, sha256|None}]`` of the pinned revision, from the HF API."""
    from huggingface_hub import HfApi
    api = HfApi(endpoint=endpoint)
    out = []
    for f in api.list_repo_tree(spec.repo, revision=spec.revision, recursive=True):
        if not hasattr(f, "blob_id"):
            continue                                    # a folder
        lfs = getattr(f, "lfs", None)
        out.append({"path": f.path, "size": f.size, "blob_id": f.blob_id,
                    "sha256": getattr(lfs, "sha256", None) if lfs else None})
    return out


def download_hf(spec: ModelSpec, endpoint: str | None = None) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=spec.repo, revision=spec.revision, local_dir=str(spec.local_dir),
                      endpoint=endpoint)


# --------------------------------------------------------------------------- ModelScope

def _ms_list(repo: str, revision: str = "master") -> dict[str, int]:
    import requests
    r = requests.get(f"{MODELSCOPE_API}/{repo}/repo/files",
                     params={"Revision": revision, "Recursive": "True"}, timeout=60)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code == 404 or (r.ok and not body.get("Data")):
        raise ModelError(f"{repo} is not on ModelScope ({body.get('Message', 'no data')}); "
                         "use --endpoint with an HF mirror, or copy the model in (docs/DATA.md)")
    r.raise_for_status()
    return {f["Path"]: f.get("Size", 0) for f in body["Data"]["Files"] if f.get("Type") != "tree"}


def _ms_fetch(repo: str, path: str, size: int, out: Path, revision: str = "master") -> None:
    """Ranged GET with resume (the server sometimes ignores Range: restart then)."""
    import requests
    url = f"{MODELSCOPE_API}/{repo}/repo?" + urllib.parse.urlencode({"Revision": revision, "FilePath": path})
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, RETRIES + 1):
        have = out.stat().st_size if out.exists() else 0
        if have == size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
                if have and r.status_code == 200:
                    have = 0
                r.raise_for_status()
                with open(out, "ab" if have else "wb") as fh:
                    for chunk in r.iter_content(CHUNK):
                        fh.write(chunk)
        except Exception as e:  # network errors of every kind: retry with backoff
            print(f"  [retry {attempt}/{RETRIES}] {path}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            time.sleep(min(2 ** attempt, 30))
    got = out.stat().st_size if out.exists() else 0
    if got != size:
        raise ModelError(f"{repo}/{path}: expected {size} bytes, got {got}")


def _is_doc(path: str) -> bool:
    return path in DOC_FILES or path.startswith(DOC_DIRS)


def download_modelscope(spec: ModelSpec, endpoint: str | None = None,
                        ms_repo: str | None = None) -> None:
    """Fetch from ModelScope, verifying each file against the pinned HF revision."""
    tree = hf_tree(spec, endpoint)
    ms_repo = ms_repo or spec.repo
    available = _ms_list(ms_repo)
    for f in sorted(tree, key=lambda f: f["size"] or 0):
        rel, out = f["path"], spec.local_dir / f["path"]
        if rel not in available:
            if _is_doc(rel):
                print(f"  [skip] {rel}: not on ModelScope (documentation only)")
                continue
            raise ModelError(f"{ms_repo}: {rel} is not on ModelScope")
        etag = f["sha256"] or f["blob_id"]
        if out.is_file() and out.stat().st_size == f["size"] and _matches(out, f):
            write_metadata(spec.local_dir, rel, spec.revision, etag)
            continue
        if out.exists():
            out.unlink()
        _ms_fetch(ms_repo, rel, available[rel], out)
        if not _matches(out, f):
            if _is_doc(rel):
                print(f"  [skip] {rel}: ModelScope copy differs from HF@{spec.revision[:8]}")
                out.unlink()
                continue
            out.unlink()
            raise ModelError(f"{ms_repo}/{rel} differs from {spec.repo}@{spec.revision}")
        write_metadata(spec.local_dir, rel, spec.revision, etag)
        print(f"  [ok] {rel}", flush=True)


def _matches(path: Path, f: dict) -> bool:
    if f["sha256"]:
        return sha256_file(path) == f["sha256"]
    return git_blob_sha1(path) == f["blob_id"]


# --------------------------------------------------------------------------- SynNoThink

def _patch_template(text: str, where: str) -> str:
    n = text.count(GEN_PREFIX)
    if n != 1:
        raise ModelError(f"{where}: expected the generation prefix {GEN_PREFIX!r} once, found {n}")
    return text.replace(GEN_PREFIX, GEN_PREFIX_NOTHINK)


def make_syn_nothink(src: str | os.PathLike, dst: str | os.PathLike, *,
                     indent: int = TOKENIZER_CONFIG_INDENT) -> Path:
    """Build ``dst`` from model ``src`` with a thinking-closed generation prefix."""
    src, dst = Path(os.path.abspath(src)), Path(os.path.abspath(dst))
    if not (src / "config.json").is_file():
        raise ModelError(f"{src} is not a model directory (no config.json)")
    if dst.exists():
        check_syn_nothink(src, dst)
        print(f"{dst} exists and checks out; left as is")
        return dst
    tmp = dst.with_name(f".{dst.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    patched = 0
    for p in sorted(src.iterdir()):
        out = tmp / p.name
        if p.name == "tokenizer_config.json":
            cfg = json.loads(p.read_text(encoding="utf-8"))
            if TEMPLATE_KEY in cfg:
                cfg[TEMPLATE_KEY] = _patch_template(cfg[TEMPLATE_KEY], str(p))
                patched += 1
                out.write_text(json.dumps(cfg, indent=indent, ensure_ascii=False), encoding="utf-8")
            else:
                shutil.copy2(p, out)
        elif p.name == "chat_template.jinja":
            out.write_text(_patch_template(p.read_text(encoding="utf-8"), str(p)), encoding="utf-8")
            patched += 1
        elif p.is_file() and not p.is_symlink() and (p.suffix in COPY_SUFFIXES or p.name in COPY_NAMES):
            shutil.copy2(p, out)
        else:
            os.symlink(p, out)                         # absolute, like the original recipe
    if not patched:
        shutil.rmtree(tmp)
        raise ModelError(f"{src} has no chat template to patch")
    os.replace(tmp, dst)
    check_syn_nothink(src, dst)
    return dst


def check_syn_nothink(src: str | os.PathLike, dst: str | os.PathLike) -> None:
    """``dst`` differs from ``src`` only in the patched generation prefix."""
    src, dst = Path(src), Path(dst)
    for name in ("tokenizer_config.json", "chat_template.jinja"):
        a, b = src / name, dst / name
        if not a.exists():
            continue
        if not b.exists():
            raise ModelError(f"{b} missing")
        if name.endswith(".json"):
            ja, jb = json.loads(a.read_text(encoding="utf-8")), json.loads(b.read_text(encoding="utf-8"))
            if set(ja) != set(jb) or any(ja[k] != jb[k] for k in ja if k != TEMPLATE_KEY):
                raise ModelError(f"{b}: keys other than {TEMPLATE_KEY} differ from {a}")
            if TEMPLATE_KEY in ja and jb[TEMPLATE_KEY] != ja[TEMPLATE_KEY].replace(GEN_PREFIX, GEN_PREFIX_NOTHINK):
                raise ModelError(f"{b}: {TEMPLATE_KEY} is not the patched source template")
        elif b.read_text(encoding="utf-8") != _patch_template(a.read_text(encoding="utf-8"), str(a)):
            raise ModelError(f"{b} is not the patched source template")
    for p in src.iterdir():
        if p.name not in ("tokenizer_config.json", "chat_template.jinja") and not (dst / p.name).exists():
            raise ModelError(f"{dst / p.name} missing")


# --------------------------------------------------------------------------- CLI

def download(specs: Iterable[ModelSpec], *, modelscope: bool = False, endpoint: str | None = None) -> int:
    """Fetch every spec not already complete; returns the number fetched."""
    n = 0
    for s in specs:
        if is_complete(s):
            print(f"[models] {s.name}: present at {s.revision[:8]}")
            continue
        print(f"[models] {s.name} <- {s.repo}@{s.revision[:8]} ({'ModelScope' if modelscope else 'HF'})",
              flush=True)
        s.local_dir.mkdir(parents=True, exist_ok=True)
        if modelscope:
            download_modelscope(s, endpoint)
        else:
            download_hf(s, endpoint)
        if not is_complete(s):
            raise ModelError(f"{s.local_dir}: incomplete after download")
        n += 1
    return n


def download_main(cfg, args: argparse.Namespace) -> int:
    """``a2d models download``."""
    from ask2distill import config
    if getattr(args, "all", False):
        cfgs = [config.load(p) for p in config.list_pairs()]
    elif cfg is not None:
        cfgs = [cfg]
    else:
        raise SystemExit("a2d models download: give --pair P or --all")
    specs = collect_specs(cfgs)
    if getattr(args, "dry_run", False):
        for s in specs:
            print(f"{s.name:40s} {s.repo}@{s.revision}  -> {s.local_dir}")
        return 0
    try:
        download(specs, modelscope=getattr(args, "modelscope", False), endpoint=getattr(args, "endpoint", None))
    except ModelError as e:
        raise SystemExit(f"a2d models download: {e}") from None
    if not getattr(args, "all", False) and cfg is not None and cfg.models.get("syn_nothink", False):
        print(f"[models] next: a2d models syn-nothink --pair {cfg.pair}")
    return 0


def syn_nothink_main(cfg, args: argparse.Namespace) -> int:
    """``a2d models syn-nothink --pair P`` for a pair with ``models.syn_nothink``."""
    if not cfg.models.get("syn_nothink", False):
        raise SystemExit(f"pair {cfg.pair} does not use a SynNoThink synthesizer base")
    src, dst = Path(str(cfg.models.student)), Path(str(cfg.models.synthesizer_init))
    if getattr(args, "dry_run", False):
        print(f"make_syn_nothink {src} -> {dst}")
        return 0
    try:
        make_syn_nothink(src, dst)
    except ModelError as e:
        raise SystemExit(f"a2d models syn-nothink: {e}") from None
    print(f"[models] {dst} ready")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m ask2distill.data.models")
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    make_syn_nothink(a.src, a.dst)
    sys.exit(0)
