"""Build a real-question code arm (the OpenCodeReasoning-10k recipe) from OpenCodeReasoning.

Real-question arms sample OpenCodeReasoning, the only reference frame a
synthetic code arm can be read against. Sampling is over the whole corpus, not
the first shards: shards are grouped by source (codeforces/aizu/...), so taking
a prefix would silently pick one judge's house style.

Recipe: ``sorted(glob(<ocr_dir>/**/*.parquet))`` shuffled with
``random.Random(seed)``; every ``input`` passing ``comp_filter`` and
``strong_filter`` (salvaged text, exact-dedup) is collected file by file until
the pool reaches ``2 * n``; the pool is shuffled with the same RNG and
the first ``n`` become rows in the bank format of ``build_bank`` (code,
``extra_info.index "<row_id_prefix>_<i>"``, default ``code_ocr10k`` as in the
final runs' file).

``a2d data build code-real`` runs this recipe with the settings of the final runs (n
10,000, seed 0, prefix ``code_ocr10k``) into ``$A2D_DATA/real/OpenCodeReasoning-10k.parquet``.
It is NOT a reproduction of the file the final Vanilla code arms trained on
(md5 under ``reference`` in ``data/MANIFEST.json``; not distributed): that file was
selected under an earlier version of ``code_filter`` that no longer exists.
Re-running this recipe with the current filter scans the whole corpus (pool
14,889 < 2N), puts 0 rows in the same position and shares 6,765 / 10,000
questions with it.

    python -m ask2distill.bank.build_code_real --ocr-dir DIR --out FILE [--n 10000] [--seed 0]
        [--row-id-prefix code_ocr10k]
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path


def load_ocr(ocr_dir: str | os.PathLike, n: int, seed: int) -> list[str]:
    """``n`` filtered real statements sampled across the whole corpus."""
    import pandas as pd

    from ask2distill.filter.code_filter import comp_filter, strong_filter
    files = sorted(glob.glob(os.path.join(os.fspath(ocr_dir), "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files under {ocr_dir} (`a2d data build ocr`)")
    rng = random.Random(seed)
    rng.shuffle(files)
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        for q in pd.read_parquet(f, columns=["input"])["input"]:
            t, why = comp_filter(q)
            if why:
                continue
            c, w = strong_filter(t)
            if w or c in seen:
                continue
            seen.add(c)
            out.append(c)
        print(f"[ocr] {os.path.basename(f)} -> pool {len(out)}", flush=True)
        if len(out) >= n * 2:
            break
    rng.shuffle(out)
    return out[:n]


def build(ocr_dir: str | os.PathLike, out: str | os.PathLike, n: int = 10000, seed: int = 0,
          row_id_prefix: str = "code_ocr10k") -> int:
    """Write the real-question arm; returns the row count."""
    import pandas as pd

    from ask2distill.bank.build_bank import bank_rows
    from ask2distill.common.io import write_parquet
    qs = load_ocr(ocr_dir, n, seed)
    if len(qs) < n:
        print(f"[build] WARNING only {len(qs)} questions, wanted {n}", flush=True)
    rows = bank_rows(qs, "code", row_id_prefix)
    write_parquet(Path(out), pd.DataFrame(rows), index=None)  # type: ignore[arg-type]
    print(f"BUILD_DONE {out} rows={len(rows)} steps={len(rows) // 64}", flush=True)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point (see the module docstring)."""
    p = argparse.ArgumentParser(prog="python -m ask2distill.bank.build_code_real",
                                description="Rebuild an OpenCodeReasoning-10k-style real-question arm "
                                            "(not identical to the final runs' file).")
    p.add_argument("--ocr-dir", required=True, help="local nvidia/OpenCodeReasoning download")
    p.add_argument("--out", required=True, help="output parquet")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--row-id-prefix", default="code_ocr10k",
                   help="extra_info.index prefix (the final runs' file used code_ocr10k)")
    a = p.parse_args(argv)
    build(a.ocr_dir, a.out, a.n, a.seed, a.row_id_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
