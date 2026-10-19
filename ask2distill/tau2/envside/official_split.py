"""Official tau2-bench train/test split -> real-question OPD rows + test task files.

train: airline + retail train ids -> ``rows.json`` for ``real/tau2-official-train.parquet``
       (104 rows = airline 30 + retail 74; ``extra_info {index, domain, task_id,
       scenario}``). telecom is left out: the agent loop / env service only
       carry airline and retail, and telecom is dual-control (the user also
       runs tools).
test:  airline / retail / telecom test ids -> ``<tasks-out>/<domain>.json``
       (20 / 40 / 40), the evaluation task files. Task objects come from the
       upstream sierra-research/tau2-bench repo (pinned b7ea907), whose gold
       labels fix 11 airline / 7 retail tasks relative to the ToolOrchestra fork
       (db.json and policy.md are byte-identical).

The original wrote the parquet itself; the tau2 env has no pyarrow, so this
writes ``rows.json`` and ``a2d tau2 rows2parquet`` makes the parquet in the a2d
env. Only ``json`` is needed here, no tau2 import.

    python -m ask2distill.tau2.envside.official_split \\
        --upstream <tau2-bench>/data/tau2/domains --rows-out rows.json --tasks-out data/tau2/tasks_test
"""
from __future__ import annotations

import argparse
import json
import os

TRAIN_DOMAINS = ("airline", "retail")
TEST_DOMAINS = ("airline", "retail", "telecom")


def load(upstream: str, domain: str) -> tuple[dict, dict]:
    """(tasks by id, split) of one upstream domain."""
    with open(f"{upstream}/{domain}/tasks.json") as fh:
        tasks = {t["id"]: t for t in json.load(fh)}
    with open(f"{upstream}/{domain}/split_tasks.json") as fh:
        split = json.load(fh)
    return tasks, split


def train_rows(upstream: str) -> list[dict]:
    """``tau2-official-train`` rows, in the original order."""
    rows = []
    for d in TRAIN_DOMAINS:
        tasks, sp = load(upstream, d)
        for i in sp["train"]:
            ins = tasks[i]["user_scenario"]["instructions"]
            sc = dict(ins) if isinstance(ins, dict) else {"task_instructions": str(ins)}
            sc.setdefault("domain", d)
            first = sc.get("reason_for_call") or sc.get("task_instructions", "")
            rows.append({"data_source": "tau2_opd",
                         "prompt": [{"role": "user", "content": first}],
                         "ability": "agentic",
                         "reward_model": {"style": "model", "ground_truth": ""},
                         "extra_info": {"index": len(rows), "domain": d, "task_id": i,
                                        "scenario": json.dumps(sc, ensure_ascii=False)}})
    return rows


def write_test(upstream: str, out_dir: str) -> dict[str, int]:
    """Write ``<out_dir>/<domain>.json`` for every test domain; returns the counts."""
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for d in TEST_DOMAINS:
        tasks, sp = load(upstream, d)
        test = [tasks[i] for i in sp["test"]]
        with open(f"{out_dir}/{d}.json", "w") as fh:
            json.dump(test, fh, indent=1)
        counts[d] = len(test)
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ask2distill.tau2.envside.official_split")
    ap.add_argument("--upstream", required=True,
                    help="sierra-research/tau2-bench data/tau2/domains (tasks.json + split_tasks.json)")
    ap.add_argument("--rows-out", required=True, help="rows.json for real/tau2-official-train.parquet")
    ap.add_argument("--tasks-out", required=True, help="directory for the test task files")
    a = ap.parse_args(argv)
    rows = train_rows(a.upstream)
    os.makedirs(os.path.dirname(os.path.abspath(a.rows_out)), exist_ok=True)
    with open(a.rows_out, "w") as fh:
        json.dump(rows, fh, ensure_ascii=False)
    print("train rows", len(rows), "->", a.rows_out)
    for d, n in write_test(a.upstream, a.tasks_out).items():
        print(d, "test", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
