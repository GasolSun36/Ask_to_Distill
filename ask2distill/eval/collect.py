"""``a2d collect {math,code,tau2}``: aggregate metrics from finished eval directories.

* **math avg6** -- mean of AIME24, AIME25, AIME26, AMC23, BRUMO25 and MATH500
  ``mean_score`` x 100 from ``grading_results.json`` (keys are the lower-cased
  task names, ``eval.collect_keys``). Per-task values print with one decimal,
  the average with two, computed from the unrounded scores.
* **code Avg** -- mean of six items: LiveCodeBench v5_v6 pass@1, MBPP+ pass@1
  after the lenient entry-point repair, and the extra benches APPS
  introductory / interview1k / competition and CodeContests. HumanEval+ is
  reported elsewhere and is not in the Avg. The Avg follows the paper table:
  each item is rounded to one decimal (as its cell prints), the Avg is the
  mean of those cells (``avg_cells``) and the table shows it rounded half-up
  to one decimal (``avg``). The mean of the unrounded scores (``avg_exact``)
  is printed next to it.
* **tau2** -- pass^1 per domain and their unweighted mean, delegated to
  ``ask2distill.tau2.summarize``.

Runs are given with ``--runs``:

* math: ``<exp>_step<N>`` under ``eval.out_dir`` (or any directory path);
* code: ``<TAG>`` under ``$A2D_OUT/eval/code`` with the extra benches in
  ``<TAG>/xb``, or ``<TAG>:<XB_DIR>`` when they live elsewhere (the original
  results keep them in separate ``xb_*`` directories);
* tau2: ``<outputs_dir>/<served_name>`` (relative to ``$A2D_OUT/tau2/outputs``).

Without ``--runs`` every finished math/code directory under the root is listed.
``--out FILE`` also writes the table (``.json``: full records; otherwise TSV).
"""
from __future__ import annotations

import argparse
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Iterable, Sequence

from omegaconf import DictConfig

from ask2distill.common.io import atomic_write_text

AVG6_KEYS = ("aime24", "aime25", "aime26", "amc23", "brumo25", "math500")
XB_BENCHES = ("apps_introductory", "apps_interview1k", "apps_competition", "codecontests")
CODE_ITEMS = ("lcb", "mbpp_plus_lenient") + XB_BENCHES
TAU2_COLUMNS = ("airline", "retail", "telecom", "mean")
LCB_VERSION = "v5_v6"
MATH_RESULTS = "grading_results.json"
MBPP_LENIENT_LOG = "evalplus_mbpp_lenient.log"
_PASS1 = re.compile(r"pass@1:\s*([\d.]+)")


# ---------------------------------------------------------------------------- math

def math_keys(cfg: DictConfig | None) -> list[str]:
    """Collected task keys: ``eval.collect_keys`` of ``eval.tasks``, else the avg6 set."""
    if cfg is None or cfg.get("domain") != "math":
        return list(AVG6_KEYS)
    return [str(cfg.eval.collect_keys[t]) for t in cfg.eval.tasks]


def math_avg6(grading_results: Path, keys: Sequence[str] = AVG6_KEYS) -> dict:
    """Per-task ``mean_score`` x 100 and their mean over ``keys``.

    Returns ``{"tasks": {key: score}, "avg6": float | None, "complete": bool,
    "missing": [...]}``; ``avg6`` is ``None`` unless every key is present. A
    task graded twice (two settings in one directory) keeps the later entry.
    """
    scores = {x["hyperparameters"]["task_name"]: x["mean_score"] * 100
              for x in json.loads(Path(grading_results).read_text())}
    missing = [k for k in keys if k not in scores]
    avg = None if missing else sum(scores[k] for k in keys) / len(keys)
    return {"tasks": {k: scores.get(k) for k in keys}, "all_tasks": scores, "avg6": avg,
            "complete": not missing, "missing": missing}


def format_math_row(name: str, summary: dict, keys: Sequence[str] = AVG6_KEYS) -> str:
    cells = [_fmt(summary["tasks"].get(k), 1) for k in keys]
    return "\t".join([name, *cells, _fmt(summary["avg6"], 2)])


# ---------------------------------------------------------------------------- code

def lcb_pass1(tag_dir: Path, version: str = LCB_VERSION) -> float | None:
    """LiveCodeBench pass@1 x 100 from ``livecodebench_<version>_eval.json``."""
    p = Path(tag_dir) / f"livecodebench_{version}_eval.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())[0]["pass@1"] * 100


def mbpp_plus_lenient(tag_dir: Path) -> float | None:
    """The LAST ``pass@1`` of the lenient log (MBPP+, base + extra tests) x 100."""
    p = Path(tag_dir) / MBPP_LENIENT_LOG
    if not p.is_file():
        return None
    found = _PASS1.findall(p.read_text(errors="ignore"))
    return float(found[-1]) * 100 if found else None


def xb_overall(xb_dir: Path, bench: str) -> float | None:
    """Extra-bench ``overall`` pass@1 x 100 from ``<xb_dir>/<bench>.json``."""
    p = Path(xb_dir) / f"{bench}.json"
    if not p.is_file():
        return None
    v = json.loads(p.read_text()).get("overall")
    return None if v is None else v * 100


def paper_avg(values: Sequence[float], cell_digits: int = 1) -> tuple[float, float]:
    """The paper's Avg cell: ``(mean of the cells, that mean rounded half-up)``.

    Each value is first rounded the way its table cell prints (``f"{v:.1f}"``);
    the mean of those cells is exact in decimal (12.35, not 12.349999...), so the
    half-up rounding to ``cell_digits`` gives the printed Avg (12.4).
    """
    q = Decimal(1).scaleb(-cell_digits)
    cells = [Decimal(f"{v:.{cell_digits}f}") for v in values]
    mean = sum(cells) / len(cells)
    return float(mean), float(mean.quantize(q, rounding=ROUND_HALF_UP))


def code_avg(tag_dir: Path, xb_dir: Path | None = None, *, items: Sequence[str] = CODE_ITEMS,
             lcb_version: str = LCB_VERSION) -> dict:
    """The six Avg items of one checkpoint and their Avg.

    Returns ``{"items": {name: score}, "avg": float | None, "avg_cells": ...,
    "avg_exact": ..., "complete": bool, "missing": [...]}``. ``avg`` is the
    paper's Avg (one decimal), ``avg_cells`` the mean of the one-decimal item
    cells it is rounded from, ``avg_exact`` the mean of the unrounded items
    (see the module docstring); all three are ``None`` unless every item exists.
    """
    tag_dir = Path(tag_dir)
    xb_dir = tag_dir / "xb" if xb_dir is None else Path(xb_dir)
    unknown = [i for i in items if i not in CODE_ITEMS]
    if unknown:
        raise ValueError(f"unknown code Avg items {unknown}; known: {CODE_ITEMS}")
    values: dict[str, float | None] = {}
    for item in items:
        if item == "lcb":
            values[item] = lcb_pass1(tag_dir, lcb_version)
        elif item == "mbpp_plus_lenient":
            values[item] = mbpp_plus_lenient(tag_dir)
        else:
            values[item] = xb_overall(xb_dir, item)
    missing = [k for k, v in values.items() if v is None]
    avg = cells = exact = None
    if not missing:
        cells, avg = paper_avg(list(values.values()))
        exact = sum(values.values()) / len(values)
    return {"items": values, "avg": avg, "avg_cells": cells, "avg_exact": exact,
            "complete": not missing, "missing": missing}


CODE_AVG_COLUMNS = ("avg", "avg_cells", "avg_exact")


def format_code_row(name: str, summary: dict, items: Sequence[str] = CODE_ITEMS) -> str:
    """Items with one decimal, then ``avg`` (one decimal), ``avg_cells`` and ``avg_exact`` (two)."""
    cells = [_fmt(summary["items"].get(k), 1) for k in items]
    return "\t".join([name, *cells, _fmt(summary["avg"], 1), _fmt(summary.get("avg_cells"), 2),
                      _fmt(summary.get("avg_exact"), 2)])


# ---------------------------------------------------------------------------- tau2

def tau2_row(outputs_dir: Path, served_name: str) -> dict:
    """pass^1 per domain and mean for one served model (``tau2.summarize``)."""
    from ask2distill.tau2.summarize import summarize
    return summarize(Path(outputs_dir), served_name)


# ----------------------------------------------------------------------------- CLI

def _fmt(v: float | None, digits: int) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _resolve(run: str, root: Path) -> Path:
    p = Path(run)
    return p if p.is_absolute() or p.exists() else root / p


def _discover(root: Path, marker: str) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and any(d.glob(marker)))


def collect(domain: str, runs: Iterable[str] | None, cfg: DictConfig | None = None) -> tuple[list[str], list[dict]]:
    """Table header and one record per run (``record["row"]`` is the TSV line)."""
    from ask2distill.paths import get_paths
    out_root = get_paths().out
    records: list[dict] = []
    if domain == "math":
        keys = math_keys(cfg)
        root = (Path(str(cfg.eval.out_dir)) if cfg is not None and cfg.get("domain") == "math"
                else out_root / "eval" / "math")
        header = ["run", *keys, "avg6"]
        for run in (list(runs) if runs else _discover(root, MATH_RESULTS)):
            d = _resolve(run, root)
            if not (d / MATH_RESULTS).is_file():
                records.append({"run": run, "error": f"no {d / MATH_RESULTS}",
                                "row": "\t".join([run, *["-"] * (len(keys) + 1)])})
                continue
            s = math_avg6(d / MATH_RESULTS, keys)
            records.append({"run": run, "dir": str(d), **s, "row": format_math_row(run, s, keys)})
    elif domain == "code":
        items = (tuple(str(i) for i in cfg.eval.avg_items)
                 if cfg is not None and cfg.get("domain") == "code" else CODE_ITEMS)
        version = (str(cfg.eval.lcb.version)
                   if cfg is not None and cfg.get("domain") == "code" else LCB_VERSION)
        root = out_root / "eval" / "code"
        header = ["run", *items, *CODE_AVG_COLUMNS]
        for run in (list(runs) if runs else _discover(root, "livecodebench_*_eval.json")):
            tag, _, xb = run.partition(":")
            d = _resolve(tag, root)
            x = _resolve(xb, root) if xb else None
            s = code_avg(d, x, items=items, lcb_version=version)
            records.append({"run": run, "dir": str(d), "xb_dir": str(x or d / "xb"), **s,
                            "row": format_code_row(run, s, items)})
    elif domain == "tau2":
        if not runs:
            raise SystemExit("a2d collect tau2: give --runs <outputs_dir>/<served_name> ...")
        root = out_root / "tau2" / "outputs"
        header = ["run", *TAU2_COLUMNS]
        for run in runs:
            p = _resolve(run, root)
            s = tau2_row(p.parent, p.name)
            cells = [_fmt(s.get(c) if isinstance(s, dict) else None, 1) for c in TAU2_COLUMNS]
            records.append({"run": run, "summary": s, "row": "\t".join([run, *cells])})
    else:
        raise SystemExit(f"a2d collect: unknown domain {domain!r}")
    return header, records


def main(cfg: DictConfig | None, args: argparse.Namespace) -> int:
    """``a2d collect {math,code,tau2} [--runs ...] [--out FILE]``."""
    header, records = collect(args.domain, getattr(args, "runs", None), cfg)
    table = "\n".join(["\t".join(header), *(r["row"] for r in records)]) + "\n"
    print(table, end="", flush=True)
    out = getattr(args, "out", None)
    if out:
        out = Path(out)
        if out.suffix == ".json":
            atomic_write_text(out, json.dumps([{k: v for k, v in r.items() if k != "row"}
                                               for r in records], indent=1) + "\n")
        else:
            atomic_write_text(out, table)
        print(f"[collect] wrote {out}", flush=True)
    return 0
