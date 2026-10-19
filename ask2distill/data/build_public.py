"""OpenCodeReasoning at a pinned revision, and the OpenCodeReasoning-10k Vanilla code arm.

    a2d data build ocr        [--from-dir DIR]
    a2d data build code-real

**OpenCodeReasoning** (``nvidia/OpenCodeReasoning@20a1ca1``, CC-BY-4.0, about
9 GB, https://huggingface.co/datasets/nvidia/OpenCodeReasoning): needed for the
real-question code arm (``code-real`` below) and to rebuild code calibration
set 2. Both ``split_0`` and ``split_1`` are fetched into
``$A2D_DATA/OpenCodeReasoning``: ``split_1`` contributes no question (its
``input`` is "-" on every row) but its 10 file names are part of
``sorted(glob(**/*.parquet))``, which the selectors shuffle, so leaving them out
changes which questions are drawn. ``--from-dir DIR`` copies the same files
from a local download instead (``DIR/split_0``, ``DIR/split_1``).

**code-real** writes ``$A2D_DATA/real/OpenCodeReasoning-10k.parquet``, the
training set of the Vanilla code arms (``opd.arms.vanilla.train``): the seeded
selection of ``ask2distill.bank.build_code_real`` (10,000 questions, seed 0,
row ids ``code_ocr10k_<i>``) in the bank format of ``build_bank`` (LCB solve
template, ``ground_truth "N/A"``). The selection runs the current
``code_filter``; the file the final Vanilla runs trained on was selected under
an earlier version of that filter, so the two share most but not all questions
(docs/DATA.md).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ask2distill.paths import get_paths

OCR_REPO = "nvidia/OpenCodeReasoning"
OCR_REVISION = "20a1ca19c0d050fe9057fc08339d6b370ec1c67a"
OCR_PATTERNS = ["split_0/*.parquet", "split_1/*.parquet"]
OCR_SHARDS = {"split_0": 30, "split_1": 10}


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def build_ocr(data_dir: Path, from_dir: Path | None = None) -> Path:
    dst = data_dir / "OpenCodeReasoning"
    if from_dir is not None:
        for split in OCR_SHARDS:
            for p in sorted((from_dir / split).glob("*.parquet")):
                _copy(p, dst / split / p.name)
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(OCR_REPO, repo_type="dataset", revision=OCR_REVISION, local_dir=str(dst),
                          allow_patterns=OCR_PATTERNS)
    for split, n in OCR_SHARDS.items():
        got = len(list((dst / split).glob("*.parquet")))
        if got != n:
            raise RuntimeError(f"{dst / split}: {got} parquet files, expected {n}")
    print(f"[ocr] {OCR_REPO}@{OCR_REVISION[:7]} -> {dst}")
    return dst


CODE_REAL = {"path": "real/OpenCodeReasoning-10k.parquet", "n": 10000, "seed": 0,
             "row_id_prefix": "code_ocr10k"}


def build_code_real(data_dir: Path) -> Path:
    """``$A2D_DATA/real/OpenCodeReasoning-10k.parquet`` from ``$A2D_DATA/OpenCodeReasoning``."""
    from ask2distill.bank.build_code_real import build
    ocr = data_dir / "OpenCodeReasoning"
    if not any(ocr.glob("split_0/*.parquet")):
        raise SystemExit(f"{ocr} has no split_0 parquet files; run `a2d data build ocr` first")
    out = data_dir / CODE_REAL["path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    build(ocr, out, n=CODE_REAL["n"], seed=CODE_REAL["seed"],
          row_id_prefix=CODE_REAL["row_id_prefix"])
    from ask2distill.data import manifest as M
    M.report_reference(data_dir, [CODE_REAL["path"]])
    return out


def main(cfg=None, args: argparse.Namespace | None = None) -> int:
    """``a2d data build ocr`` / ``a2d data build code-real``."""
    paths = get_paths()
    from_dir = getattr(args, "from_dir", None)
    from_dir = Path(from_dir) if from_dir else None
    target = getattr(args, "target", "ocr")
    if target == "ocr":
        build_ocr(paths.data, from_dir)
    elif target == "code-real":
        build_code_real(paths.data)
    else:
        raise SystemExit(f"unknown target {target!r}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="python -m ask2distill.data.build_public")
    ap.add_argument("target", choices=["ocr", "code-real"])
    ap.add_argument("--from-dir")
    sys.exit(main(None, ap.parse_args()))
