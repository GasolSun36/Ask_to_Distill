"""Card harvest: sample the chosen synthesizer per (domain, target) slot and score every card.

For each job index ``i`` over ``TARGETS x per_target`` (domains in the order
retail, airline), the prompt is ``V.build(domain, seed0 + i, target)``, the
synthesizer answers once (``temperature 1.0, max_tokens 2048``; top_p/top_k come
from the server's generation_config, no sampling flags are sent), the card is
parsed and scored by the version's reward with ``k`` student rollouts and the
teacher gate. One JSON line per index is appended to the pool file.

Resumable, exactly as the original: an index already in the pool is skipped
unless its row failed on infrastructure (``rejected`` starting with
``rollout_error`` or ``gen_error``), in which case it is redone and the new
row appended (the bank keeps the last row per index). The pool is complete
when every index has a non-infrastructure row (``expand_v{6,7}_int.sh``
``complete()``); the process repeats passes until then, up to
``--max-passes``, and exits 3 if the pool is still incomplete -- resubmit.

Endpoints come from an endpoints file (kinds gen / stu / usr / tea) instead of
parsing ``squeue`` and serve logs. The served names in that file are the model
names sent with each request; the synthesizer's must equal the configured
``harvest.<version>.served_name``.

Settings: ``A2D_TAU2_SETTINGS`` holds the resolved ``harvest`` section. The
harvest env (``TAU2_GROUNDING=1 TAU2_TEACHER_GATE=1 TAU2_KT=3`` and no
``TAU2_DATA``) is rendered by ``ask2distill.tau2.env`` for the role
``harvest``.

    python -m ask2distill.tau2.envside.harvest --version v7 --endpoints endpoints.txt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

from ask2distill.tau2.envside import endpoints as E
from ask2distill.tau2.envside import settings

DOMAINS = ("retail", "airline")
INFRA_ERRORS = ("rollout_error", "gen_error")
EXIT_INCOMPLETE = 3


def modules(version: str) -> tuple[ModuleType, ModuleType]:
    """(prompt module, reward module) of a harvest version."""
    if version == "v7":                        # v7: compound retail calls (retail_v7)
        from ask2distill.tau2.envside import reward_v7 as R, scenario_prompt_v7 as V
    elif version == "v6":                      # v6: preference cards (pref_solver)
        from ask2distill.tau2.envside import reward_v6 as R, scenario_prompt_v6 as V
    elif version == "v5":
        from ask2distill.tau2.envside import reward_v5 as R, scenario_prompt_v5 as V
    else:
        raise ValueError(f"no harvest for version {version!r} (v5, v6, v7)")
    return V, R


def jobs(targets: dict[str, list[str]], per_target: int) -> list[tuple[str, str]]:
    """``(domain, target)`` per job index, in the original order."""
    return [(d, t) for d in DOMAINS for t in targets[d] for _ in range(per_target)]


def done_indices(pool: str) -> set[int]:
    """Indices with a row that did not fail on infrastructure."""
    done: set[int] = set()
    if os.path.exists(pool):
        with open(pool) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not str(r.get("rejected") or "").startswith(INFRA_ERRORS):
                    done.add(r["idx"])
    return done


def progress(pool: str, n_jobs: int) -> tuple[int, int]:
    """(indices done, indices wanted): the ``complete()`` rule of expand_v*_int.sh."""
    return len(done_indices(pool)), n_jobs


class Pool:
    """Random choice over the replicas of one served model."""

    def __init__(self, bases: list[str]) -> None:
        from openai import OpenAI
        self.c = [OpenAI(api_key="EMPTY", base_url=b, timeout=900) for b in bases]
        assert self.c, "no endpoints"

    @property
    def chat(self):
        return random.choice(self.c).chat


def parse_card(text: str) -> dict | None:
    """Pull ``{scenario, actions, ...}`` out of a generation (same rule as the reward worker)."""
    t = text.replace("```json", "").replace("```", "")
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(j.get("scenario"), dict) or not isinstance(j.get("actions"), list):
        return None
    out = {"scenario": j["scenario"], "actions": j["actions"]}
    if isinstance(j.get("forbidden_action"), dict):
        out["forbidden_action"] = j["forbidden_action"]
    if isinstance(j.get("preference"), dict):
        out["preference"] = j["preference"]
    if isinstance(j.get("plan"), dict):
        out["plan"] = j["plan"]
    return out


def run_pass(V: ModuleType, R: ModuleType, job_list: list[tuple[str, str]], eps: list[E.Endpoint],
             model: dict[str, str], *, seed0: int, k: int, workers: int, out: str,
             sampling: dict) -> None:
    """One pass over every index not yet done (``gen_v5.main``)."""
    GEN, STU, USR, TEA = (Pool(E.urls(eps, kind)) for kind in ("gen", "stu", "usr", "tea"))
    done = done_indices(out)
    lock = threading.Lock()

    def job(i):
        if i in done:
            return None
        dom, tgt = job_list[i]
        rec = {"idx": i, "domain": dom, "target": tgt}
        try:
            prompt, _ = V.build(dom, seed0 + i, tgt)
            r = GEN.chat.completions.create(model=model["gen"], max_tokens=int(sampling["max_tokens"]),
                                            temperature=float(sampling["temperature"]),
                                            messages=[{"role": "user", "content": prompt}])
            card = parse_card(r.choices[0].message.content or "")
        except Exception as e:
            rec["rejected"] = f"gen_error:{str(e)[:60]}"
            card = None
        if card:
            res = R.score({**card, "domain": dom, "target": tgt}, STU, model["stu"], USR, model["usr"],
                          k=k, teacher_client=TEA, teacher_model=model["tea"])
            rec.update(card=card, **{key: res.get(key) for key in
                                     ("R", "rejected", "teacher_ok", "mean_turns", "student_fail")})
        elif "rejected" not in rec:
            rec["rejected"] = "wrong_shape"
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return rec

    with ThreadPoolExecutor(workers) as ex:
        for j, _ in enumerate(ex.map(job, range(len(job_list)))):
            if j % 50 == 0:
                print(f"[harvest] {j}/{len(job_list)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ask2distill.tau2.envside.harvest")
    ap.add_argument("--version", required=True, choices=["v5", "v6", "v7"])
    ap.add_argument("--endpoints", required=True, help="endpoints file (kinds gen / stu / usr / tea)")
    ap.add_argument("--out", help="pool file (default: harvest.<version>.pool)")
    ap.add_argument("--max-passes", type=int, default=3,
                    help="passes before giving up on infrastructure errors (resubmit to continue)")
    ap.add_argument("--retry-wait-s", type=int, default=60)
    a = ap.parse_args(argv)

    h = settings.role_settings()["harvest"]
    hv = h[a.version]
    out = a.out or hv["pool"]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    eps = E.read(a.endpoints)
    model = {kind: E.served_name(eps, kind) for kind in ("gen", "stu", "usr", "tea")}
    if model["gen"] != hv["served_name"]:
        raise SystemExit(f"endpoints serve the synthesizer as {model['gen']!r}, but "
                         f"harvest.{a.version}.served_name is {hv['served_name']!r}")
    V, R = modules(a.version)
    n_targets = sum(len(V.TARGETS[d]) for d in DOMAINS)
    if n_targets != int(hv["targets"]):
        raise SystemExit(f"{a.version} prompts define {n_targets} target slots, "
                         f"config says harvest.{a.version}.targets={hv['targets']}")
    job_list = jobs(V.TARGETS, int(hv["per_target"]))
    print(f"[harvest] {a.version}: {n_targets} target slots x {hv['per_target']} = {len(job_list)} cards, "
          f"seed0 {hv['seed0']}, k {hv['k']}, workers {hv['workers']} -> {out}; models {model}", flush=True)

    for p in range(1, a.max_passes + 1):
        run_pass(V, R, job_list, eps, model, seed0=int(hv["seed0"]), k=int(hv["k"]),
                 workers=int(hv["workers"]), out=out, sampling=h["gen_sampling"])
        n_done, n = progress(out, len(job_list))
        print(f"[harvest] pass {p}: {n_done}/{n}", flush=True)
        if n_done >= n:
            print(f"HARVEST_DONE {a.version} {n_done}/{n} {out}", flush=True)
            return 0
        if p < a.max_passes:
            time.sleep(a.retry_wait_s)
    print(f"HARVEST_INCOMPLETE {a.version} {progress(out, len(job_list))[0]}/{len(job_list)}: "
          "resubmit to redo the rows that failed on infrastructure", flush=True)
    return EXIT_INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main())
