"""tau2 parquets: card pools -> OPD banks, rows.json -> parquet, synthesizer prompt sets.

Bank (``build_synth``). From one or more harvest pools: keep the last record
per ``(pool, idx)`` (a redone row supersedes the failed one), drop rejected
cards, then dedup on task identity -- two cards that perform the same write on
the same order / reservation are the same task however the scenario is
phrased:

* v7 cards: the plan's parts (gold is rebuilt from the plan; the card's action
  list is empty);
* v6 preference cards: the preference spec (gold is rebuilt by pref_solver);
* refusals: the target and the forbidden call (their gold has no write);
* everything else: the sorted non-read gold actions with their arguments.

Refusal cards get the fixed customer persona the v5 reward scored them with,
and v7 cards get their plan's persona appended to ``task_instructions``
(= ``retail_v7.with_persona``). Rows match ``real/tau2-official-train.parquet``:
``extra_info {index, domain, task_id: "syn<idx>", target, scenario}`` with the
first user turn as the prompt. Every card that passed the gates is kept,
including ones the student solved k/k.

Final bank (``bank.parts`` then ``bank.mix``): ``<pair>_part-airline`` (976; v6
pool, airline only -- the command is in no script and was reconstructed, its
output is identical) + ``<pair>_part-retail`` (1,326; v7 pool) = ``<pair>``
(2,302), concatenated with ``extra_info.index`` renumbered. The released pools rebuild
the released banks exactly.
"""
from __future__ import annotations

import argparse
import collections
import json
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

from omegaconf import DictConfig

REFUSE_PERSONA = ("You really want this done and give your reason. If the agent says it cannot be done, "
                  "push back once and ask them to make an exception; if they still decline, accept it and end the call.")
READ = ("get_", "search_", "list_", "calculate", "find_", "think")


# ------------------------------------------------------------------------- banks

def synth_rows(pools: Sequence[str | Path], domains: Iterable[str] | None = None) -> tuple[list[dict], dict]:
    """OPD rows from harvest pools, plus counts (``pool_idx``, ``valid``, ``rows``)."""
    doms = set(domains) if domains else None
    last = {}
    for pi, pool in enumerate(pools):
        with open(pool) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if doms and r.get("domain") not in doms:
                    continue
                last[(pi, r["idx"])] = r
    valid = [r for _, r in sorted(last.items()) if not r.get("rejected")]
    seen, rows = set(), []
    for r in valid:
        acts = r["card"]["actions"]
        target = str(r.get("target") or "")
        refuse, pref, v7 = target.startswith("refuse:"), target.startswith("pref:"), target.startswith("v7:")
        if v7:
            # v7: gold is rebuilt from the plan (the card's action list is empty)
            sig = json.dumps(["v7", r["card"].get("plan", {}).get("parts")], sort_keys=True)
        elif pref:
            # v6: gold is rebuilt from the preference by pref_solver, the card's own
            # action list is not trusted -- identity is the spec itself
            sig = json.dumps(["pref", r["card"].get("preference")], sort_keys=True)
        else:
            # a refusal's gold has no write -- its identity is the forbidden call
            key = ([r["card"].get("forbidden_action") or {}] if refuse else
                   [a for a in acts if not a["name"].startswith(READ)])
            sig = json.dumps([r.get("target") if refuse else None] +
                             sorted(json.dumps([a.get("name"), a.get("arguments")], sort_keys=True) for a in key))
        if sig in seen:
            continue
        seen.add(sig)
        sc = {k: v for k, v in dict(r["card"]["scenario"]).items() if v is not None}
        if refuse:          # same persona the v5 reward scored the card with
            sc["task_instructions"] = REFUSE_PERSONA
        if v7 and (r["card"].get("plan") or {}).get("persona"):   # = retail_v7.with_persona
            ti = str(sc.get("task_instructions") or "").strip()
            sc["task_instructions"] = (ti + " " if ti else "") + r["card"]["plan"]["persona"]
        sc["domain"] = r["domain"]
        first = sc.get("reason_for_call") or sc.get("task_instructions", "")
        rows.append({"data_source": "tau2_opd",
                     "prompt": [{"role": "user", "content": first}],
                     "ability": "agentic",
                     "reward_model": {"style": "model", "ground_truth": ""},
                     "extra_info": {"index": len(rows), "domain": r["domain"], "task_id": f"syn{r['idx']}",
                                    "target": r.get("target"), "scenario": json.dumps(sc, ensure_ascii=False)}})
    return rows, {"pool_idx": len(last), "valid": len(valid), "rows": len(rows)}


def write_rows(rows: list[dict], out: str | Path) -> Path:
    """``pd.DataFrame(rows).to_parquet(out)``, written atomically."""
    import pandas as pd

    from ask2distill.common.io import write_parquet
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    return write_parquet(out, pd.DataFrame(rows), index=None)


def build_synth(pools: Sequence[str | Path], out: str | Path, domains: Iterable[str] | None = None) -> int:
    """One synthetic bank parquet; returns its row count."""
    rows, n = synth_rows(pools, domains)
    write_rows(rows, out)
    print(f"pool idx {n['pool_idx']} valid {n['valid']} after dedup {n['rows']} -> {out}", flush=True)
    print(dict(collections.Counter(r["extra_info"]["domain"] for r in rows)),
          collections.Counter(r["extra_info"]["target"] for r in rows).most_common(), flush=True)
    return n["rows"]


def mix(parts: Sequence[str | Path], out: str | Path) -> int:
    """Concatenate bank parquets and renumber ``extra_info.index`` (as the released mix bank was built)."""
    import pandas as pd

    from ask2distill.common.io import write_parquet
    frames = [pd.read_parquet(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    df["extra_info"] = [{**e, "index": i} for i, e in enumerate(df["extra_info"])]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_parquet(out, df, index=None)
    print("MIX", len(df), " + ".join(str(len(f)) for f in frames), "->", out, flush=True)
    return len(df)


def build_bank(cfg: DictConfig, out_dir: str | Path | None = None) -> Path:
    """Every ``bank.parts`` parquet, then ``bank.mix``; returns the mix path."""
    out_dir = Path(out_dir or str(cfg.bank.out_dir))
    parts = []
    for part in cfg.bank.parts:
        p = out_dir / f"{part.out}.parquet"
        build_synth([str(part.pool)], p, list(part.domains))
        parts.append(p)
    mixed = out_dir / f"{cfg.bank.mix}.parquet"
    mix(parts, mixed)
    return mixed


# ------------------------------------------------------------------ rows.json bridge

def rows2parquet(rows_json: str | Path, out: str | Path) -> int:
    """rows.json written by an env-side builder -> parquet (the tau2 env has no pyarrow)."""
    with open(rows_json) as fh:
        rows = json.load(fh)
    write_rows(rows, out)
    print(f"{len(rows)} rows -> {out}", flush=True)
    return len(rows)


# ----------------------------------------------------------------------------- CLI

def main(cfg: DictConfig, args: argparse.Namespace | None = None) -> int:
    """``a2d tau2 bank``: pools -> ``bank.parts`` parquets -> ``bank.mix`` (the OPD bank)."""
    from ask2distill import config
    config.require(cfg, "bank")
    if getattr(args, "dry_run", False):
        out_dir = Path(str(cfg.bank.out_dir))
        for part in cfg.bank.parts:
            state = "" if Path(str(part.pool)).is_file() else "  (pool missing)"
            print(f"build_synth {part.pool} domains={list(part.domains)} -> "
                  f"{out_dir / f'{part.out}.parquet'}{state}")
        print(f"mix -> {out_dir / f'{cfg.bank.mix}.parquet'}", flush=True)
        return 0
    build_bank(cfg)
    return 0


def rows2parquet_main(cfg: DictConfig | None, args: argparse.Namespace) -> int:
    """``a2d tau2 rows2parquet --rows FILE --out FILE``."""
    rows2parquet(args.rows, args.out)
    return 0


def build_prompts_main(cfg: DictConfig, args: argparse.Namespace) -> int:
    """``a2d tau2 build-prompts --version vX``: env-side rows.json, then parquet.

    Equivalent to the released GENRL_TAU2_<V> parquet, not byte-identical
    (see ``envside.build_prompts``); the released file stays canonical.
    """
    from ask2distill.tau2 import spawn
    version = getattr(args, "version", None)
    if not version:
        raise SystemExit("a2d tau2 build-prompts needs --version")
    out = Path(str(cfg.paths.out)) / "tau2" / "prompts" / f"GENRL_TAU2_{version.upper()}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        rows_json = Path(tmp) / "rows.json"
        argv = ["--version", version, "--out", str(rows_json), *getattr(args, "passthrough", [])]
        env = spawn.env(cfg, "synrl")
        if getattr(args, "dry_run", False):
            print(" ".join(spawn.command("build_prompts", argv)), flush=True)
            return 0
        rc = spawn.run("build_prompts", argv, env).wait()
        if rc != 0:
            return rc
        rows2parquet(rows_json, out)
    return 0
