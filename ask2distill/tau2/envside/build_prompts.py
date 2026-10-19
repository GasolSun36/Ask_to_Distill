"""Build a synthesizer prompt set (GENRL_TAU2_V4..V7): one row per (domain, seed), as rows.json.

The tau2 counterpart of the math / code synthesizer-RL prompt parquet. Each row's prompt already
contains the sampled database records and the domain's real tool signatures, so
the synthesizer never has to guess an id or a tool name -- the first yield test
scored 0% precisely because the tool list was missing. ``extra_info`` carries
the domain and the target write, so the reward worker knows which environment
to execute a candidate's actions against and which write it must contain.

Rows cycle through the domains (``i % len(domains)``) and, within a domain,
through its targets (``(i // len(domains)) % len(targets)``) so every type gets
an equal share; the seed of row ``i`` is ``i``.

========  =======  ==================  ========================================
version   rows     domains             builder
========  =======  ==================  ========================================
v4        2048     retail, airline     ``scenario_prompt.build`` (targeted)
v5        1152     retail, airline     ``scenario_prompt_v5.build``
v6        1152     retail, airline     ``scenario_prompt_v6.build``
v7        1152     retail              ``scenario_prompt_v7.build`` (``TARGETS[i % 4]``)
========  =======  ==================  ========================================

The original wrote parquet directly; the tau2 env has no pyarrow, so this
writes ``rows.json`` and ``a2d tau2 build-prompts`` converts it in the a2d env.
v7 had no standalone builder (its prompts came from an inline call); its branch
is added here.

Equivalent, not byte-identical: ``_retail`` iterates a hash-ordered set, so
the released parquets cannot be regenerated row for row and stay canonical.
``--deterministic`` (default on) sorts that set and pins ``PYTHONHASHSEED=0``
(re-executing the interpreter if needed), so a rebuild is at least repeatable.

    python -m ask2distill.tau2.envside.build_prompts --version v7 --out rows.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

# version -> (rows, domains)
PLAN = {"v4": (2048, ("retail", "airline")),
        "v5": (1152, ("retail", "airline")),
        "v6": (1152, ("retail", "airline")),
        "v7": (1152, ("retail",))}


def _builder(version: str):
    if version == "v4":
        from ask2distill.tau2.envside import scenario_prompt as M
        return M.TARGETS, lambda dom, i, t: M.build(dom, seed=i, target=t)
    if version == "v5":
        from ask2distill.tau2.envside import scenario_prompt_v5 as M
    elif version == "v6":
        from ask2distill.tau2.envside import scenario_prompt_v6 as M
    elif version == "v7":
        from ask2distill.tau2.envside import scenario_prompt_v7 as M
    else:
        raise ValueError(f"unknown version {version!r}")
    return M.TARGETS, lambda dom, i, t: M.build(dom, i, t)


def build_rows(version: str, n: int | None = None) -> list[dict]:
    """The prompt rows of ``version`` (``n`` overrides the row count, for smoke)."""
    total, domains = PLAN[version]
    targets, build = _builder(version)
    rows = []
    for i in range(total if n is None else n):
        dom = domains[i % len(domains)]
        tgts = targets[dom]
        # cycle through the targets rather than sample: every type gets an equal share
        target = tgts[(i // len(domains)) % len(tgts)]
        prompt, _ = build(dom, i, target)
        rows.append({
            "data_source": "tau2_genrl",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "agentic",
            "reward_model": {"style": "model", "ground_truth": ""},
            "extra_info": {"index": i, "domain": dom, "target": target},
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ask2distill.tau2.envside.build_prompts")
    ap.add_argument("--version", required=True, choices=sorted(PLAN))
    ap.add_argument("--out", required=True, help="rows.json to write")
    ap.add_argument("--n", type=int, default=None, help="number of rows (default: the released count)")
    ap.add_argument("--no-deterministic", dest="deterministic", action="store_false",
                    help="keep hash-ordered set iteration, as the original builder ran")
    a = ap.parse_args(argv)
    if a.deterministic and os.environ.get("PYTHONHASHSEED") != "0":
        # String hashing is fixed at interpreter start; restart with a pinned seed.
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", __spec__.name, *(argv or sys.argv[1:])])
    from ask2distill.tau2.envside import scenario_prompt
    scenario_prompt.DETERMINISTIC = a.deterministic
    rows = build_rows(a.version, a.n)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    os.replace(tmp, a.out)
    lens = sorted(len(r["prompt"][0]["content"]) for r in rows)
    print(f"{len(rows)} rows -> {a.out}")
    print("prompt chars: median", lens[len(lens) // 2] if lens else 0)
    print("domains:", dict(Counter(r["extra_info"]["domain"] for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
