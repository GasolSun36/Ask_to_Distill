"""verl reward worker for tau2 synthesizer RL.

Nothing here imports tau2 -- the environment lives behind
``envside.env_service`` over HTTP, because tau2 needs loguru/litellm that the
a2d env does not have.

The reward is not divergence: on this pair token KL, path surprisal and
behavioural distance did not rank learnable tasks above the rest better than
trajectory length alone. So the signal is measured directly
by the service (``POST /score``, reward version ``TAU2_REWARD_V``)::

    R_raw = [card passes every gate] x (student failure rate over TAU2_K rollouts)

The gates: gold actions execute, change the DB and are grounded in the card;
the student-first teacher gate (some teacher episode reaches gold when the
student fails k/k; off for v6 preference and v7 plan cards, whose
well-posedness the solvers prove); v5 refusal checks; v6/v7 preference / plan
checks, the Instruct-2507 judge and graded partial credit (<= 0.1) for cards
that dump spec keys or leave a preference unstated. The worker then applies
the diversity term::

    R = R_raw x max(0, 1 - TAU2_DIVW x share)

``share`` is the size of the card's scenario cluster (Qwen3-Embedding mean
pool, average linkage, cosine distance cut at ``TAU2_DIVTHRESH``) over its
group. The group is a slice of ``TAU2_DIV_GROUP`` rows of this rank's batch
(16 = the whole rank batch with 8 actor ranks, so a fully unique card still
gets 1/16 of share), and only parse-valid cards are clustered, on
``json.dumps(card["scenario"], ensure_ascii=False)``: the scenario text is what
collapses -- two cards can share a scenario while differing in an argument.
The term is on from step one: a hash comparison has no interior gradient once
the reward saturates (many valid cards already score 1.0), so collapse
would be unrecoverable without it. The scalar is placed on the last response
token.

When the service call fails the whole step scores 0 (``svc_down`` rows, kept
as run: v6 rank 0 logged 48 of them). Dumps (``<TAU2_DUMP>.r<rank>`` and
``.metrics.r<rank>``) carry verl's global step, so a run spanning several 4h
windows never restarts its step keys.

Settings come from the environment (rendered by ``ask2distill.tau2.env`` and
delivered to every worker through ``runtime_env``); all are required.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Sequence

import requests
import torch
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register

_JSON = re.compile(r"\{.*\}", re.S)


@dataclass(frozen=True)
class WorkerSettings:
    """The worker's knobs, read from the environment."""

    svc: str
    k: int
    divthresh: float
    divw: float
    div_group: int
    dump: str
    timeout: int
    gen_tokenizer: str
    emb_model: str

    @classmethod
    def from_env(cls, env: dict | None = None) -> "WorkerSettings":
        env = os.environ if env is None else env
        names = ("TAU2_SVC", "TAU2_K", "TAU2_DIVTHRESH", "TAU2_DIVW", "TAU2_DIV_GROUP", "TAU2_DUMP",
                 "TAU2_SVC_TIMEOUT", "TAU2_GEN_TOKENIZER", "TAU2_EMB_MODEL")
        # TAU2_DUMP may be empty (no dump); everything else must have a value.
        missing = [n for n in names if n not in env or (not env[n] and n != "TAU2_DUMP")]
        if missing:
            raise RuntimeError(f"tau2 reward worker: {', '.join(missing)} not set (rendered by "
                               "ask2distill.tau2.env and passed through runtime_env)")
        return cls(svc=env["TAU2_SVC"], k=int(env["TAU2_K"]), divthresh=float(env["TAU2_DIVTHRESH"]),
                   divw=float(env["TAU2_DIVW"]), div_group=int(env["TAU2_DIV_GROUP"]),
                   dump=env["TAU2_DUMP"], timeout=int(env["TAU2_SVC_TIMEOUT"]),
                   gen_tokenizer=env["TAU2_GEN_TOKENIZER"], emb_model=env["TAU2_EMB_MODEL"])


def parse_card(text: str) -> dict | None:
    """Pull {scenario, actions} out of a generation, tolerating fences and prose."""
    t = text.replace("```json", "").replace("```", "")
    m = _JSON.search(t)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(j.get("scenario"), dict) or not isinstance(j.get("actions"), list):
        return None
    out = {"scenario": j["scenario"], "actions": j["actions"]}
    if isinstance(j.get("forbidden_action"), dict):      # v5 refusal cards
        out["forbidden_action"] = j["forbidden_action"]
    if isinstance(j.get("preference"), dict):            # v6 preference cards
        out["preference"] = j["preference"]
    if isinstance(j.get("plan"), dict):                  # v7 compound retail calls
        out["plan"] = j["plan"]
    return out


def diversity_share(sc_texts: Sequence[str], thresh: float, group: int, embedder) -> list[float]:
    """Cluster share per row; 0 for unparsable rows and for groups with < 2 valid cards.

    Rows are sliced into consecutive groups of ``group`` (0, or a group larger
    than the batch, means the whole batch); within a group only the parse-valid
    cards (non-empty scenario text) are clustered, in one embedding call.
    """
    from ask2distill.common.embed import cluster_share
    n = len(sc_texts)
    share = [0.0] * n
    g = group if 0 < group < n else n
    for start in range(0, n, max(g, 1)):
        idx = [i for i in range(start, min(start + g, n)) if sc_texts[i]]
        if len(idx) < 2:
            continue
        sh = cluster_share([sc_texts[i] for i in idx], thresh, None, embedder)
        for j, i in enumerate(idx):
            share[i] = float(sh[j])
    return share


class Tau2RewardWorker(Worker):
    """Scores generated cards through the env service; replaces verl's RewardModelWorker."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.s = WorkerSettings.from_env()
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.s.gen_tokenizer or config.model.path,
                                                 trust_remote_code=True)
        self.step = 0
        self._emb = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # No teacher here: this reward never consults it directly (the service runs the
        # teacher gate). Only the embedding model for the diversity term, colocated
        # with the actor (~8GB).
        from ask2distill.common.embed import Embedder
        self._emb = Embedder(self.s.emb_model, device="cuda")
        r = requests.get(f"{self.s.svc}/health", timeout=60).json()
        print(f"[tau2-rm] rank{self.rank} svc={self.s.svc} {r} k={self.s.k}", flush=True)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto, **kwargs):
        s = self.s
        # The true global step, so a run spanning several 4h windows does not
        # restart its step counter -- that reset silently collapsed three windows
        # of the 7B math run onto the same keys and hid steps 30-60 entirely.
        self.step = int(data.meta_info.get("global_steps", self.step + 1))

        resp = data.batch["responses"]
        attn = data.batch["attention_mask"]
        rlen = resp.shape[1]
        resp_mask = attn[:, -rlen:]
        scores = torch.zeros_like(resp, dtype=torch.float32)
        n = resp.shape[0]

        # non_tensor_batch values are numpy object arrays; `arr or default` calls
        # bool() on the whole array and raises for n > 1 -- which is exactly how
        # the fourth launch died. Test for None explicitly.
        ei = data.non_tensor_batch.get("extra_info")
        if ei is None:
            ei = [{}] * n
        doms = [((e if isinstance(e, dict) else {}) or {}).get("domain", "retail") for e in ei]
        tgts = [((e if isinstance(e, dict) else {}) or {}).get("target") for e in ei]
        texts = [self.tok.decode(resp[i][:int(resp_mask[i].sum())], skip_special_tokens=True)
                 for i in range(n)]
        cards = [parse_card(t) for t in texts]
        payload = [({**c, "domain": doms[i], "target": tgts[i]} if c else {"domain": doms[i]})
                   for i, c in enumerate(cards)]

        try:
            res = requests.post(f"{s.svc}/score", json={"cands": payload, "k": s.k},
                                timeout=s.timeout).json()["results"]
        except Exception as e:
            print(f"[tau2-rm] service call failed, the whole step scores 0: {e}", flush=True)
            res = [{"R": 0.0, "rejected": f"svc_down:{str(e)[:40]}"}] * n

        # Diversity is charged on the scenario text, which is what collapses --
        # two cards can share a scenario while differing in an argument.
        share = [0.0] * n
        sc_texts = [json.dumps(c["scenario"], ensure_ascii=False) if c else "" for c in cards]
        if self._emb is not None:
            try:
                share = diversity_share(sc_texts, s.divthresh, s.div_group, self._emb)
            except Exception as e:
                share = [0.0] * n
                print(f"[tau2-rm] diversity failed, no penalty this step: {e}", flush=True)

        finals, diag = [], []
        for i in range(n):
            R = float(res[i].get("R", 0.0)) * max(0.0, 1.0 - s.divw * share[i])
            finals.append(R)
            L = int(resp_mask[i].sum())
            if L > 0:
                scores[i, L - 1] = R
            diag.append({"R": R, "R_raw": float(res[i].get("R", 0.0)),
                         "share": share[i], "rejected": res[i].get("rejected"),
                         "n_actions": len(cards[i]["actions"]) if cards[i] else 0,
                         "turns": res[i].get("mean_turns", 0.0),
                         "teacher_ok": res[i].get("teacher_ok"),
                         "target": tgts[i]})

        nrej = sum(1 for d in diag if d["rejected"])
        metrics = {
            "tau2/R": sum(finals) / n,
            "tau2/R_raw": sum(d["R_raw"] for d in diag) / n,
            "tau2/valid_frac": 1.0 - nrej / n,
            "tau2/parse_frac": sum(1 for c in cards if c) / n,
            "tau2/share": sum(share) / n,
            "tau2/n_actions": sum(d["n_actions"] for d in diag) / n,
            "tau2/turns": sum(d["turns"] for d in diag) / max(n - nrej, 1),
            "tau2/frac_R_eq_1": sum(1 for d in diag if d["R_raw"] >= 1.0) / n,
            "tau2/teacher_unsolvable_frac": sum(1 for d in diag if d["rejected"] == "teacher_unsolvable") / n,
        }
        if s.dump:
            try:
                with open(f"{s.dump}.metrics.r{self.rank}", "a") as fh:
                    fh.write(json.dumps({"step": self.step, **metrics}) + "\n")
                with open(f"{s.dump}.r{self.rank}", "a") as fh:
                    for i in range(n):
                        fh.write(json.dumps({"step": self.step, "domain": doms[i],
                                             **diag[i],
                                             "card": cards[i]}, ensure_ascii=False,
                                            default=str) + "\n")
            except Exception as e:
                print(f"[tau2-rm] dump failed: {e}", flush=True)
        return DataProto.from_dict(tensors={"rm_scores": scores},
                                   meta_info={"metrics": metrics})
