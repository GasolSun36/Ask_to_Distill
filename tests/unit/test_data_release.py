"""Data release, builders and model helpers (``ask2distill.data``), CPU only, no network.

* the manifest schema, selection and verification (ok / partial / missing / mismatch);
* ``download --from-dir`` on a synthetic release: md5 checks, optional
  groups, partially verified entries; ``--banks-dir`` for banks that live in
  their own dataset repos;
* the committed ``data/MANIFEST.json``: valid, and it covers every released
  file the configs read;
* model specs over all pairs are consistent; ``make_syn_nothink`` patches only
  the generation prefix;
* pins shared with other modules agree (LCB revision, tau2 sources).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from ask2distill.data import manifest as M

from conftest import AGENT_QWEN3_4B, ALL_PAIRS

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- manifest

def _entry(path: str, data: bytes, group: str = "g", **kw) -> dict:
    import hashlib
    e = {"path": path, "group": group, "md5": hashlib.md5(data).hexdigest(), "md5_prefix": None,
         "size": len(data), "rows": None, "license": "x", "provenance": "", "regenerable": "no",
         "builder": None, "note": ""}
    e.update(kw)
    return e


def _manifest(entries, optional=()) -> dict:
    groups = {e["group"] for e in entries} | set(optional)
    return {"schema": 1, "release": {"name": "t"},
            "groups": {g: {"description": g, "optional": g in optional} for g in groups},
            "files": entries}


def test_manifest_verify_statuses(tmp_path):
    (tmp_path / "a.jsonl").write_bytes(b'{"x":1}\n{"x":2}\n')
    (tmp_path / "b.txt").write_bytes(b"other")
    (tmp_path / "c.jsonl").write_bytes(b'{"x":1}\n')
    entries = [
        _entry("a.jsonl", b'{"x":1}\n{"x":2}\n'),
        _entry("b.txt", b"expected"),
        _entry("gone.txt", b""),
        _entry("c.jsonl", b"", md5=None, md5_prefix=M.describe(tmp_path / "c.jsonl")["md5"][:10], rows=1),
    ]
    man = _manifest(entries)
    M.validate(man)
    rep = M.verify(tmp_path, M.select(man))
    assert rep.checked == 4
    assert rep.missing == ["gone.txt"]
    assert len(rep.mismatched) == 1 and rep.mismatched[0].startswith("b.txt")
    assert rep.partial == ["c.jsonl"]
    assert not rep.ok


def test_manifest_rejects_bad_entries():
    bad = _manifest([_entry("/abs/path", b"x"), _entry("../up", b"x"), _entry("dup", b"x"), _entry("dup", b"x")])
    bad["files"][0]["regenerable"] = "maybe"
    with pytest.raises(M.ManifestError) as ei:
        M.validate(bad)
    msg = str(ei.value)
    assert "relative" in msg and "listed twice" in msg and "regenerable" in msg


def test_manifest_select_skips_optional_groups():
    man = _manifest([_entry("a", b"", group="core"), _entry("f", b"", group="fixtures")], optional=("fixtures",))
    assert [e["path"] for e in M.select(man)] == ["a"]
    assert [e["path"] for e in M.select(man, ["fixtures"])] == ["f"]
    with pytest.raises(M.ManifestError):
        M.select(man, ["nope"])


def test_count_rows(tmp_path):
    import pandas as pd
    (tmp_path / "x.jsonl").write_text('{"a":1}\n\n{"a":2}\n')
    (tmp_path / "x.json").write_text("[1, 2, 3]")
    (tmp_path / "y.json").write_text('{"a": 1}')
    pd.DataFrame({"a": [1, 2, 3, 4]}).to_parquet(tmp_path / "x.parquet")
    assert M.count_rows(tmp_path / "x.jsonl") == 2
    assert M.count_rows(tmp_path / "x.json") == 3
    assert M.count_rows(tmp_path / "y.json") is None
    assert M.count_rows(tmp_path / "x.parquet") == 4


# --------------------------------------------------------------------------- download

def test_download_from_dir_round_trip(tmp_path):
    """``a2d data download --from-dir``: md5 checks, optional groups, partial entries."""
    from ask2distill.data import download as D
    rel = tmp_path / "release"
    (rel / "core" / "dir").mkdir(parents=True)
    (rel / "fixtures").mkdir()
    files = {"core/a.jsonl": b'{"q":1}\n{"q":2}\n{"q":3}\n', "core/dir/one.txt": b"1",
             "fixtures/x.txt": b"x"}
    for path, data in files.items():
        (rel / path).write_bytes(data)
    entries = [_entry(p, d, group=p.split("/")[0]) for p, d in files.items()]
    entries[0]["rows"] = 3
    entries.append(_entry("core/absent.parquet", b"", group="core", md5=None, size=None, rows=7,
                          md5_prefix="abcdef0123"))
    man = _manifest(entries, optional=("fixtures",))
    M.write(tmp_path / "MANIFEST.json", man)

    data = tmp_path / "data"
    rep = D.run(data_dir=data, from_dir=str(rel), manifest_path=tmp_path / "MANIFEST.json", log=lambda *_: None)
    assert rep.missing == ["core/absent.parquet"] and not rep.mismatched
    assert not (data / "fixtures").exists()                               # optional group not fetched
    assert not (data / "MANIFEST.json").exists()                          # never copied over
    rep = D.run(data_dir=data, from_dir=str(rel), groups=["fixtures"],
                manifest_path=tmp_path / "MANIFEST.json", log=lambda *_: None)
    assert rep.ok and rep.checked == 1
    (data / "core" / "a.jsonl").write_text("tampered\n")
    rep = M.verify(data, M.select(M.load(tmp_path / "MANIFEST.json"), ["core"]))
    assert any(m.startswith("core/a.jsonl") for m in rep.mismatched)


def test_download_repo_id_precedence(monkeypatch):
    """--repo-id, then $A2D_HF_DATA_REPO, then the manifest, then the public default."""
    from ask2distill.data import download as D
    assert D.DEFAULT_REPO_ID == "gasolsun/Ask_to_Distill-data"
    man = {"release": {"repo_id": None}}
    monkeypatch.delenv("A2D_HF_DATA_REPO", raising=False)
    assert D.resolve_repo_id(None, man) == D.DEFAULT_REPO_ID
    assert D.resolve_repo_id(None, {"release": {"repo_id": "m/r"}}) == "m/r"
    monkeypatch.setenv("A2D_HF_DATA_REPO", "env/r")
    assert D.resolve_repo_id(None, {"release": {"repo_id": "m/r"}}) == "env/r"
    assert D.resolve_repo_id("cli/r", man) == "cli/r"
    assert M.load()["release"]["repo_id"] == D.DEFAULT_REPO_ID


def test_download_banks_dir(tmp_path):
    """Banks of ``source.kind "hf"`` come from --banks-dir in any of the three layouts."""
    from ask2distill.data import download as D
    rel, banks = tmp_path / "release", tmp_path / "banks"
    (rel / "core").mkdir(parents=True)
    (rel / "core" / "a.txt").write_bytes(b"a")
    payload = {"P1": b"bank one", "P2": b"bank two", "P3": b"bank three"}
    entries = [_entry("core/a.txt", b"a", group="core")]
    for name, data in payload.items():
        entries.append(_entry(f"banks/{name}.parquet", data, group="banks",
                              source={"kind": "hf", "repo_id": f"org/{name}",
                                      "path": "data/train-00000-of-00001.parquet", "revision": "r" * 40}))
    (banks / "banks").mkdir(parents=True)
    (banks / "P1.parquet").write_bytes(payload["P1"])                               # flat
    (banks / "banks" / "P2.parquet").write_bytes(payload["P2"])                     # $A2D_DATA layout
    snap = banks / "P3" / "data"
    snap.mkdir(parents=True)
    (snap / "train-00000-of-00001.parquet").write_bytes(payload["P3"])              # dataset snapshot
    M.write(tmp_path / "MANIFEST.json", _manifest(entries))
    data = tmp_path / "data"
    rep = D.run(data_dir=data, from_dir=str(rel), banks_dir=str(banks),
                manifest_path=tmp_path / "MANIFEST.json", log=lambda *_: None)
    assert rep.ok and rep.checked == 4
    assert (data / "banks" / "P3.parquet").read_bytes() == payload["P3"]
    (banks / "P1.parquet").unlink()
    (data / "banks" / "P1.parquet").unlink()
    rep = D.run(data_dir=data, from_dir=str(rel), banks_dir=str(banks),
                manifest_path=tmp_path / "MANIFEST.json", log=lambda *_: None)
    assert rep.missing == ["banks/P1.parquet"]


# --------------------------------------------------------------------------- committed manifest

def test_committed_manifest_is_valid():
    man = M.load()
    assert set(man["groups"]) == {"calib", "prompts", "tau2", "banks"}
    assert not any(g.get("optional") for g in man["groups"].values())
    assert all(e["md5"] for e in man["files"]) and not man["release"]["pending"]
    banks = {e["path"]: e for e in man["files"] if e["group"] == "banks"}
    hf = [e for e in banks.values() if M.source_kind(e) == "hf"]
    assert len(hf) == 7 and all(e["source"]["repo_id"] == "gasolsun/" + e["path"][6:-8] for e in hf)
    assert all(e["source"]["path"] == "data/train-00000-of-00001.parquet" for e in hf)
    assert sorted(M.source_kind(e) for e in banks.values()).count("tau2_bank") == 2
    release = {e["path"] for e in man["files"] if M.source_kind(e) == "release"}
    assert not any(p.startswith(("real/", "eval/", "fixtures/", "banks/")) for p in release)
    ref = {e["path"] for e in man["reference"]}
    assert {"real/DAPO-Math-17k.parquet", "real/OpenCodeReasoning-10k.parquet",
            "real/tau2-official-train.parquet"} <= ref
    assert all(e["md5"] and e["upstream"].startswith("https://") for e in man["reference"])


def _data_paths(node, prefix: str) -> set[str]:
    out = set()
    if isinstance(node, dict):
        for v in node.values():
            out |= _data_paths(v, prefix)
    elif isinstance(node, list):
        for v in node:
            out |= _data_paths(v, prefix)
    elif isinstance(node, str) and node.startswith(prefix):
        out.add(node[len(prefix):])
    return out


@pytest.mark.parametrize("pair", ALL_PAIRS)
def test_manifest_covers_every_released_input_a_config_reads(pair, tmp_path, monkeypatch):
    from ask2distill import config
    from omegaconf import OmegaConf
    data = tmp_path / "data"
    for var, d in (("A2D_DATA", data), ("A2D_MODELS", tmp_path / "m"), ("A2D_CKPT", tmp_path / "c"),
                   ("A2D_OUT", tmp_path / "o"), ("A2D_CACHE", tmp_path / "x")):
        monkeypatch.setenv(var, str(d))
    cfg = OmegaConf.to_container(config.load(pair), resolve=True)
    used = _data_paths(cfg, str(data) + "/")
    man = M.load()
    released = {e["path"] for e in man["files"]} | {e["path"] for e in man["reference"]}
    built = re.compile(r"^(OpenCodeReasoning$|eval/math$|eval/code/extra_bench$|tau2/tasks_test$)")
    missing = sorted(p for p in used if p not in released and not built.match(p))
    assert not missing, f"{pair} reads files the release does not list: {missing}"


# --------------------------------------------------------------------------- models

def test_model_specs_are_consistent_across_pairs(tmp_path, monkeypatch):
    from ask2distill import config
    from ask2distill.data.models import collect_specs
    monkeypatch.setenv("A2D_MODELS", str(tmp_path / "models"))
    cfgs = [config.load(p) for p in config.list_pairs()]
    specs = collect_specs(cfgs)
    names = [s.name for s in specs]
    assert len(names) == len(set(names))
    assert {"Qwen3-1.7B", "Qwen3-4B", "Qwen3-Embedding-4B", "Qwen3-4B-Non-Thinking-RL-Math-Step500",
            "Qwen2.5-Coder-32B-Instruct", "Qwen3-30B-A3B-Instruct-2507"} <= set(names)
    for s in specs:
        assert re.fullmatch(r"[0-9a-f]{40}", s.revision), s
        assert s.local_dir.parent == tmp_path / "models"
    teacher = next(s for s in specs if s.name == "Qwen3-4B-Non-Thinking-RL-Math-Step500")
    assert teacher.repo == "Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500"


def test_git_blob_sha1(tmp_path):
    from ask2distill.data.models import git_blob_sha1
    p = tmp_path / "f"
    p.write_bytes(b"hello\n")
    assert git_blob_sha1(p) == "ce013625030ba8dba906f756967f9e9ca394464a"   # `git hash-object`


def test_metadata_round_trip(tmp_path):
    from ask2distill.data.models import ModelSpec, is_complete, read_metadata, verify_against_metadata, \
        write_metadata, git_blob_sha1
    d = tmp_path / "M"
    d.mkdir()
    (d / "config.json").write_text("{}")
    rev = "a" * 40
    write_metadata(d, "config.json", rev, git_blob_sha1(d / "config.json"))
    assert read_metadata(d) == {"config.json": (rev, git_blob_sha1(d / "config.json"))}
    assert is_complete(ModelSpec("o/M", rev, d))
    assert not is_complete(ModelSpec("o/M", "b" * 40, d))
    assert verify_against_metadata(d) == []
    (d / "config.json").write_text('{"x": 1}')
    assert verify_against_metadata(d) == ["config.json: blob id mismatch"]


def _toy_model(root: Path, template: str, jinja: str | None = None) -> Path:
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"a": 1}')
    (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (root / "LICENSE").write_text("license")
    (root / "tokenizer_config.json").write_text(json.dumps({"x": 1, "chat_template": template}, indent=2))
    if jinja is not None:
        (root / "chat_template.jinja").write_text(jinja)
    return root


def test_make_syn_nothink(tmp_path):
    from ask2distill.data.models import GEN_PREFIX, GEN_PREFIX_NOTHINK, check_syn_nothink, make_syn_nothink
    tpl = "{% if add_generation_prompt %}" + GEN_PREFIX + "{% endif %}"
    src = _toy_model(tmp_path / "M", tpl, jinja=tpl)
    dst = make_syn_nothink(src, tmp_path / "M-SynNoThink")
    cfg = json.loads((dst / "tokenizer_config.json").read_text())
    assert cfg == {"x": 1, "chat_template": tpl.replace(GEN_PREFIX, GEN_PREFIX_NOTHINK)}
    assert (dst / "chat_template.jinja").read_text() == tpl.replace(GEN_PREFIX, GEN_PREFIX_NOTHINK)
    weights = dst / "model-00001-of-00001.safetensors"
    assert weights.is_symlink() and os.path.isabs(os.readlink(weights))
    assert not (dst / "config.json").is_symlink()
    assert (dst / "LICENSE").is_symlink()
    make_syn_nothink(src, dst)                                            # idempotent
    (dst / "tokenizer_config.json").write_text(json.dumps({"x": 2, "chat_template": cfg["chat_template"]}))
    with pytest.raises(Exception, match="keys other than"):
        check_syn_nothink(src, dst)


def test_make_syn_nothink_needs_exactly_one_prefix(tmp_path):
    from ask2distill.data.models import GEN_PREFIX, ModelError, make_syn_nothink
    with pytest.raises(ModelError, match="found 0"):
        make_syn_nothink(_toy_model(tmp_path / "A", "no prefix here"), tmp_path / "A2")
    with pytest.raises(ModelError, match="found 2"):
        make_syn_nothink(_toy_model(tmp_path / "B", GEN_PREFIX * 2), tmp_path / "B2")
    assert not (tmp_path / "A2").exists() and not (tmp_path / "B2").exists()


# --------------------------------------------------------------------------- converters

def test_interview_1k_recipe(tmp_path, monkeypatch):
    import random
    from ask2distill.data import build_code_eval_data as B
    ids = list(range(3000))
    perm = ids[:]
    random.Random(0).shuffle(perm)
    first = f"apps_{perm[0]}"
    full = tmp_path / "full.jsonl"
    full.write_text("".join(json.dumps({"id": f"apps_{i}", "question": "q"}) + "\n" for i in ids))
    monkeypatch.setattr(B, "INTERVIEW_1K_FIRST_ID", first)
    assert B.interview_1k(full, tmp_path / "1k.jsonl") == 1000
    got = [json.loads(line)["id"] for line in (tmp_path / "1k.jsonl").read_text().splitlines()]
    assert got == [f"apps_{i}" for i in perm[:1000]]
    monkeypatch.setattr(B, "INTERVIEW_1K_FIRST_ID", "apps_2639")
    if first != "apps_2639":
        with pytest.raises(RuntimeError, match="apps_2639"):
            B.interview_1k(full, tmp_path / "x.jsonl")


# --------------------------------------------------------------------------- pins

def test_lcb_revision_pins_agree():
    from ask2distill.data.build_code_eval_data import LCB_FILES, LCB_REPO, LCB_REVISION
    text = (REPO / "configs/base/eval_code.yaml").read_text()
    assert f"dataset_revision: {LCB_REVISION}" in text and f"dataset: {LCB_REPO}" in text
    loader = (REPO / "third_party/LiveCodeBench/lcb_runner/benchmarks/code_generation.py").read_text()
    assert f'"{LCB_REVISION}"' in loader
    assert LCB_FILES == ("test5.jsonl", "test6.jsonl")


def test_tau2_pins_agree_with_config(tmp_path, monkeypatch):
    from ask2distill import config
    from ask2distill.data import tau2_data as T
    monkeypatch.setenv("A2D_DATA", str(tmp_path / "d"))
    cfg = config.load(AGENT_QWEN3_4B)
    assert str(cfg.tau2_data.toolorchestra.revision).startswith("c1c0a40")
    assert str(cfg.tau2_data.upstream_split.revision).startswith("b7ea907")
    assert set(T.UPSTREAM_MD5) == {f"{d}/{f}" for d in T.DOMAINS for f in ("tasks.json", "split_tasks.json")}
    assert Path(str(cfg.tau2_data.tasks_test)) == tmp_path / "d" / "tau2" / "tasks_test"


def test_tau2_split_files_are_md5_checked(tmp_path):
    from ask2distill.data import tau2_data as T
    local = tmp_path / "in" / "tau2-bench" / "data" / "tau2" / "domains" / "airline"
    local.mkdir(parents=True)
    for f in ("tasks.json", "split_tasks.json"):
        (local / f).write_text("[]")
    with pytest.raises(RuntimeError, match="not tau2-bench"):
        T.fetch_split_files("sierra-research/tau2-bench", "b7ea907", tmp_path / "out", tmp_path / "in")


def test_tokenizer_cache_gets_a_main_ref(tmp_path):
    from ask2distill.data import tau2_data as T
    rev = "b968826d9c46dd6066d109eabc6255188de91218"
    snap_dir = tmp_path / "in" / "hf" / "hub" / "models--Qwen--Qwen3-8B" / "snapshots" / rev
    snap_dir.mkdir(parents=True)
    (snap_dir / "tokenizer.json").write_text("{}")
    repo_dir = T.cache_tokenizer("Qwen/Qwen3-8B", rev, tmp_path / "hf", tmp_path / "in")
    assert (repo_dir / "refs" / "main").read_text() == rev
    assert (tmp_path / "hf" / "hub" / "models--Qwen--Qwen3-8B" / "snapshots" / rev / "tokenizer.json").is_file()


def test_ocr_pins_and_code_real_recipe():
    from ask2distill.data import build_public as P
    assert P.OCR_SHARDS == {"split_0": 30, "split_1": 10}
    assert P.OCR_REVISION.startswith("20a1ca1")
    assert P.CODE_REAL == {"path": "real/OpenCodeReasoning-10k.parquet", "n": 10000, "seed": 0,
                           "row_id_prefix": "code_ocr10k"}
