"""tau2 data: domain databases, the official split, the domain context, a tokenizer cache.

    a2d data build tau2 [--from-dir DIR]

1. **Domain data** -- ``data/tau2`` of NVlabs/ToolOrchestra at ``c1c0a40``
   (35 MB, plain git, no LFS) into ``third_party/toolorchestra_tau2/data/tau2``
   (``tau2_data.data_dir``), by a sparse checkout. Every stage uses the fork's
   ``user_simulator/simulation_guidelines.md``, which differs from upstream; its
   md5 is checked.
2. **Official split** -- ``tasks.json`` + ``split_tasks.json`` of
   sierra-research/tau2-bench at ``b7ea907`` (whose gold labels fix 11 airline /
   7 retail tasks relative to the fork), md5-checked, then
   ``envside.official_split`` writes the test task files
   (``$A2D_DATA/tau2/tasks_test/{airline,retail,telecom}.json``, 20 / 40 / 40)
   and the 104 real training rows, made into
   ``$A2D_DATA/real/tau2-official-train.parquet`` here (the tau2 env has no
   pyarrow) -- the training set of the Vanilla agent arm, which is not part of
   the data release. Both are byte-identical to the files the final runs used
   (md5 in ``data/MANIFEST.json``).
3. **Tokenizer cache** -- the fork's ``llm_utils.py`` loads
   ``AutoTokenizer("Qwen/Qwen3-8B")`` at import, and env-side processes run with
   ``HF_HUB_OFFLINE=1``; its tokenizer files are cached at the pinned revision
   in ``tau2_data.tokenizer_cache`` with ``refs/main`` pointing at it, so the
   offline lookup of the bare repo id resolves.
4. **Domain context** -- ``envside.dump_domain_ctx`` under ``$A2D_TAU2_PYTHON``
   writes ``tau2_data.domain_ctx`` (system prompt + tool schemas per domain).
   Skipped with a note when the tau2 interpreter is not configured; the
   released copy is identical.

``--from-dir DIR`` takes local copies instead of the network:
``DIR/ToolOrchestra`` (a checkout holding ``data/tau2``), ``DIR/tau2-bench`` (a
checkout holding ``data/tau2/domains``) and ``DIR/hf/hub`` (an HF cache holding
the Qwen3-8B tokenizer). A part that is absent falls back to the network.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from ask2distill.paths import get_paths

DOMAINS = ("airline", "retail", "telecom")
GUIDELINES = "user_simulator/simulation_guidelines.md"
GUIDELINES_MD5 = "c7ace24fa9fed8f6522ad22545a8ad3c"            # fork @c1c0a40
# md5 of tasks.json / split_tasks.json at sierra-research/tau2-bench@b7ea907.
UPSTREAM_MD5 = {
    "airline/tasks.json": "4588a63c1fa9844914c058b1d3dc107e",
    "airline/split_tasks.json": "251197b7f25416e4981b7975448ea3c0",
    "retail/tasks.json": "c07e89e811d345bc062461aeff7de7d7",
    "retail/split_tasks.json": "0e42ed2ce03b444e252b999f755f1556",
    "telecom/tasks.json": "a610efdf3385d253f0601f2a4e3fc0ab",
    "telecom/split_tasks.json": "4a547c49d64d98ae3fe165acbfca1162",
}
RAW_URL = "https://raw.githubusercontent.com/{repo}/{rev}/data/tau2/domains/{rel}"
TOKENIZER_PATTERNS = ["config.json", "generation_config.json", "tokenizer.json",
                      "tokenizer_config.json", "vocab.json", "merges.txt"]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def fetch_domain_data(repo: str, rev: str, data_dir: Path, from_dir: Path | None) -> Path:
    """``data/tau2`` of the ToolOrchestra fork into ``data_dir`` (sparse checkout)."""
    if (data_dir / GUIDELINES).is_file() and _md5(data_dir / GUIDELINES) == GUIDELINES_MD5:
        print(f"[tau2] domain data present at {data_dir}")
        return data_dir
    src = from_dir / "ToolOrchestra" / "data" / "tau2" if from_dir else None
    if src is None or not src.is_dir():
        tmp = Path(tempfile.mkdtemp(prefix="a2d_toolorchestra_"))
        try:
            url = f"https://github.com/{repo}.git"
            git = ["git", "-C", str(tmp)]
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--sparse", url, str(tmp)],
                           check=True)
            subprocess.run(git + ["sparse-checkout", "set", "data/tau2"], check=True)
            subprocess.run(git + ["checkout", "--quiet", rev], check=True)
            _replace_tree(tmp / "data" / "tau2", data_dir)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        _replace_tree(src, data_dir)
    got = _md5(data_dir / GUIDELINES)
    if got != GUIDELINES_MD5:
        raise RuntimeError(f"{data_dir / GUIDELINES}: md5 {got} != {GUIDELINES_MD5} "
                           f"(not the ToolOrchestra fork @{rev[:7]})")
    print(f"[tau2] {repo}@{rev[:7]} data/tau2 -> {data_dir}")
    return data_dir


def _replace_tree(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".{dst.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp.replace(dst)


def fetch_split_files(repo: str, rev: str, out: Path, from_dir: Path | None) -> Path:
    """The six upstream split inputs into ``out/<domain>/``, md5-checked."""
    local = from_dir / "tau2-bench" / "data" / "tau2" / "domains" if from_dir else None
    for rel, md5 in UPSTREAM_MD5.items():
        dst = out / rel
        if dst.is_file() and _md5(dst) == md5:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if local is not None and (local / rel).is_file():
            shutil.copyfile(local / rel, dst)
        else:
            with urllib.request.urlopen(RAW_URL.format(repo=repo, rev=rev, rel=rel), timeout=120) as r:
                dst.write_bytes(r.read())
        if _md5(dst) != md5:
            raise RuntimeError(f"{dst}: md5 {_md5(dst)} != {md5} (not tau2-bench@{rev[:7]})")
    print(f"[tau2] {repo}@{rev[:7]} split files -> {out}")
    return out


def cache_tokenizer(repo: str, rev: str, hf_home: Path, from_dir: Path | None) -> Path:
    """Tokenizer files of ``repo@rev`` in ``hf_home/hub``, with ``refs/main`` = ``rev``."""
    hub = hf_home / "hub"
    if from_dir is not None and (from_dir / "hf" / "hub").is_dir():
        from ask2distill.data.build_code_eval_data import merge_tree
        merge_tree(from_dir / "hf" / "hub", hub)
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo, revision=rev, cache_dir=str(hub),
                          allow_patterns=TOKENIZER_PATTERNS)
    repo_dir = hub / f"models--{repo.replace('/', '--')}"
    if not (repo_dir / "snapshots" / rev / "tokenizer.json").is_file():
        raise RuntimeError(f"{repo}@{rev[:7]} tokenizer not in {hub}")
    # A pinned-sha download writes no ref; the fork asks for the bare repo id
    # (revision main) offline, which resolves through refs/main.
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(rev)
    print(f"[tau2] tokenizer {repo}@{rev[:7]} cached in {hub}")
    return repo_dir


def official_split(domains_dir: Path, tasks_out: Path, real_parquet: Path) -> None:
    """Test task files + real/tau2-official-train.parquet (stdlib-only module, run in-process)."""
    from ask2distill.tau2 import bank
    from ask2distill.tau2.envside import official_split as osplit
    with tempfile.TemporaryDirectory() as td:
        rows = Path(td) / "rows.json"
        osplit.main(["--upstream", str(domains_dir), "--rows-out", str(rows), "--tasks-out", str(tasks_out)])
        n = bank.rows2parquet(rows, real_parquet)
    print(f"[tau2] {n} real training rows -> {real_parquet}")


def domain_ctx(cfg, out: Path) -> bool:
    """Spawn ``envside.dump_domain_ctx`` under the tau2 interpreter; False if not configured."""
    if get_paths().tau2_python is None:
        print("[tau2] A2D_TAU2_PYTHON not set: domain_ctx.json not rebuilt "
              "(the released copy from `a2d data download` is identical)")
        return False
    from ask2distill.tau2 import spawn
    rc = spawn.run("dump_domain_ctx", ["--out", str(out)], spawn.env(cfg, "synrl")).wait()
    if rc != 0:
        raise RuntimeError(f"dump_domain_ctx exited with {rc}")
    return True


def build(cfg, from_dir: Path | None = None) -> None:
    paths = get_paths()
    td = cfg.tau2_data
    fetch_domain_data(str(td.toolorchestra.repo), str(td.toolorchestra.revision),
                      Path(str(td.data_dir)), from_dir)
    up = td.upstream_split
    domains = fetch_split_files(str(up.repo), str(up.revision),
                                paths.cache / f"tau2-bench@{str(up.revision)[:7]}" / "domains", from_dir)
    cache_tokenizer(str(td.import_tokenizer.repo), str(td.import_tokenizer.revision),
                    Path(str(td.tokenizer_cache)), from_dir)
    official_split(domains, Path(str(td.tasks_test)), paths.data / "real" / "tau2-official-train.parquet")
    domain_ctx(cfg, Path(str(td.domain_ctx)))
    _compare_with_manifest(paths.data)


def _compare_with_manifest(data_dir: Path) -> None:
    from ask2distill.data import manifest as M
    try:
        man = M.load()
    except M.ManifestError:
        return
    for e in M.compare_entries(man):
        if e["path"].startswith(("tau2/tasks_test/", "tau2/domain_ctx", "real/tau2-official-train")):
            status, detail = M.check_entry(data_dir, e)
            if status == "mismatch":
                print(f"  WARNING: {detail}")
            elif status == "ok":
                print(f"  {e['path']}: identical to the file the final runs used")


def main(cfg=None, args: argparse.Namespace | None = None) -> int:
    """``a2d data build tau2`` (uses the shipped tau2-domain pair config)."""
    if cfg is None or cfg.get("domain") != "tau2":
        from ask2distill import config
        cfg = config.load(config.domain_pair("tau2"))
    from_dir = getattr(args, "from_dir", None) if args is not None else None
    build(cfg, Path(from_dir) if from_dir else None)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m ask2distill.data.tau2_data")
    ap.add_argument("--from-dir")
    sys.exit(main(None, ap.parse_args()))

