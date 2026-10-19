"""Offline tests of the tau2 pipeline: no GPU, no vLLM, no env service, no simulator.

* the agent loop: the response mask over a scripted multi-turn episode
  (policy tokens 1, tool / user-simulator tokens 0) and stop-on-repeat;
* the synthesizer reward worker on stubbed services;
* the diversity term: one embedding call per group of the rank batch, over
  parse-valid cards only, nothing below two cards;
* the synthesizer pick (window rule) on synthetic dumps;
* the harvest job order and completion rule;
* the synthetic bank (dedup identity, customer personas, row schema) and, when
  the released pools and bank are present, an exact rebuild of the released bank;
* summarize on synthetic outputs.

The tokenizer comes from ``$A2D_MODELS/Qwen3-4B``; tests that need it skip
otherwise.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import types
from pathlib import Path

import pytest

from conftest import AGENT_QWEN3_4B

DOMAINS = ("airline", "retail", "telecom")


# ------------------------------------------------------------------------ fixtures

def _tokenizer_dir() -> Path | None:
    root = os.environ.get("A2D_MODELS", "")
    if root and (Path(root) / "Qwen3-4B" / "tokenizer_config.json").is_file():
        return Path(root) / "Qwen3-4B"
    return None


@pytest.fixture(scope="module")
def tok():
    d = _tokenizer_dir()
    if d is None:
        pytest.skip("Qwen3-4B tokenizer not found (set A2D_MODELS; a2d models download)")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)


@pytest.fixture
def agent_loop_module(monkeypatch, tmp_path):
    """Fresh import of the agent loop with its settings in the environment."""
    pytest.importorskip("verl.experimental.agent_loop.agent_loop")
    ctx = {"retail": {"system": "You are a retail agent.", "tools": []},
           "airline": {"system": "You are an airline agent.", "tools": []}}
    (tmp_path / "ctx.json").write_text(json.dumps(ctx))
    (tmp_path / "guide.md").write_text("Be a customer.")
    monkeypatch.setenv("TAU2_CTX", str(tmp_path / "ctx.json"))
    monkeypatch.setenv("TAU2_USER_GUIDELINES", str(tmp_path / "guide.md"))
    monkeypatch.setenv("TAU2_USER_BASE", "http://unused/v1")
    monkeypatch.setenv("TAU2_MAX_TURNS", "20")

    def load(stop_on_repeat: bool):
        monkeypatch.setenv("TAU2_STOP_ON_REPEAT", "1" if stop_on_repeat else "0")
        import ask2distill.tau2.trainside.agent_loop as M
        M = importlib.reload(M)
        M.Tau2AgentLoop._class_initialized = False
        return M
    return load


def _loop_cfg():
    from omegaconf import OmegaConf
    return OmegaConf.create({"actor_rollout_ref": {"rollout": {"response_length": 4096,
                                                               "multi_turn": {"format": "hermes"}}},
                             "data": {"apply_chat_template_kwargs": {"enable_thinking": False}}})


def _scripted_loop(M, tok, policy, user, tool_out="aarav_patel_1234"):
    M.Tau2AgentLoop.init_class(_loop_cfg(), tok, None)

    class FakeSM:
        def __init__(self):
            self.i = 0

        async def generate(self, request_id, prompt_ids, sampling_params):
            text = policy[self.i] + "<|im_end|>"
            self.i += 1
            return types.SimpleNamespace(token_ids=tok.encode(text, add_special_tokens=False), log_probs=None)

    loop = M.Tau2AgentLoop.__new__(M.Tau2AgentLoop)
    loop.server_manager = FakeSM()
    ui = iter(user)
    posts = []

    async def fake_user(self, scenario, dialogue):
        return next(ui)

    async def fake_post(self, path, body):
        posts.append((path, body))
        return {"sid": "s1"} if path == "/env/new" else ({"out": tool_out} if path == "/env/call" else {"ok": True})

    M.Tau2AgentLoop._user_turn = fake_user
    M.Tau2AgentLoop._post = fake_post
    return loop, posts


# ---------------------------------------------------------------------- agent loop

def test_agent_loop_mask(agent_loop_module, tok):
    """mask=1 tokens are exactly what the policy generated; mask=0 exactly the observations."""
    M = agent_loop_module(stop_on_repeat=False)
    policy = ['<tool_call>\n{"name": "find_user_id_by_email", "arguments": {"email": "a@example.com"}}\n</tool_call>',
              "I found your account. Your order #W1 is pending; shall I cancel it?",
              "Done, it is cancelled."]
    user = ["I want to cancel my order. My email is a@example.com.", "Yes please.", "###STOP###"]
    loop, posts = _scripted_loop(M, tok, policy, user)

    async def go():
        loop.loop = asyncio.get_running_loop()
        return await loop.run({}, extra_info={"domain": "retail", "scenario": {"reason_for_call": "cancel"}})

    out = asyncio.run(go())
    r, m = out.response_ids, out.response_mask
    assert len(r) == len(m)
    pol = tok.decode([t for t, k in zip(r, m) if k == 1])
    obs = tok.decode([t for t, k in zip(r, m) if k == 0])
    for p in policy:
        assert p.strip() in pol
    assert "aarav_patel_1234" in obs and "Yes please." in obs
    assert "aarav_patel_1234" not in pol and "Yes please." not in pol
    assert out.num_turns == 3
    # the env session is closed with the sid in the `domain` field, as the service expects
    assert posts[-1] == ("/env/close", {"domain": "s1"})


def test_stop_on_repeat(agent_loop_module, tok):
    """TAU2_STOP_ON_REPEAT=1: the episode ends at the repeat; the repeated turn is dropped."""
    M = agent_loop_module(stop_on_repeat=True)
    bye = "I'm glad I could help! Your return has been processed. Have a great day!"
    policy = ['<tool_call>\n{"name": "find_user_id_by_email", "arguments": {"email": "a@example.com"}}\n</tool_call>',
              "Done, your return is processed.", bye, bye + " ", "SHOULD_NEVER_BE_GENERATED"]
    user = ["I want to return an item. a@example.com", "Thanks!", "Thank you so much!", "Thanks again!"]
    loop, _ = _scripted_loop(M, tok, policy, user, tool_out="u1")

    async def go():
        loop.loop = asyncio.get_running_loop()
        return await loop.run({}, extra_info={"domain": "retail", "scenario": {"reason_for_call": "return"}})

    out = asyncio.run(go())
    text = tok.decode(out.response_ids)
    pol = tok.decode([t for t, k in zip(out.response_ids, out.response_mask) if k == 1])
    assert len(out.response_ids) == len(out.response_mask)
    assert loop.server_manager.i == 4
    assert pol.count("glad I could help") == 1 and text.count("glad I could help") == 1
    assert out.num_turns == 3 and out.extra_fields["stopped_on_repeat"] == 1


def test_agent_loop_requires_paths(monkeypatch, agent_loop_module, tok):
    M = agent_loop_module(stop_on_repeat=False)
    monkeypatch.setattr(M, "TAU2_CTX", "")
    with pytest.raises(RuntimeError, match="TAU2_CTX"):
        M.Tau2AgentLoop.init_class(_loop_cfg(), tok, None)


# -------------------------------------------------------------------- reward worker

WORKER_ENV = {"TAU2_SVC": "http://svc:1", "TAU2_K": "6", "TAU2_DIVTHRESH": "0.1", "TAU2_DIVW": "1.0",
              "TAU2_DIV_GROUP": "16", "TAU2_DUMP": "", "TAU2_SVC_TIMEOUT": "3600",
              "TAU2_GEN_TOKENIZER": "/x", "TAU2_EMB_MODEL": "/y"}


class RecordingEmbedder:
    """Stub embedder: unit vectors, one dimension per distinct text; records calls."""

    device = "cpu"

    def __init__(self):
        self.calls = []

    def embed(self, texts, bs=16):
        import torch
        self.calls.append((list(texts), bs))
        vocab = {t: i for i, t in enumerate(dict.fromkeys(texts))}
        E = torch.zeros(len(texts), max(len(vocab), 1))
        for i, t in enumerate(texts):
            E[i, vocab[t]] = 1.0
        return E


def test_worker_settings_required():
    pytest.importorskip("verl")
    from ask2distill.tau2.trainside.synrl_worker import WorkerSettings
    s = WorkerSettings.from_env(WORKER_ENV)
    assert (s.k, s.divthresh, s.divw, s.div_group, s.timeout) == (6, 0.1, 1.0, 16, 3600)
    with pytest.raises(RuntimeError, match="TAU2_K"):
        WorkerSettings.from_env({k: v for k, v in WORKER_ENV.items() if k != "TAU2_K"})


def test_diversity_group_16_valid_only():
    pytest.importorskip("verl")
    pytest.importorskip("sklearn")
    from ask2distill.tau2.trainside.synrl_worker import diversity_share
    emb = RecordingEmbedder()
    texts = ["a", "", "a", "b", "", "c"] + ["d"] * 10          # 16 rows, 14 parse-valid
    share = diversity_share(texts, 0.1, 16, emb)
    assert len(emb.calls) == 1 and emb.calls[0] == ([t for t in texts if t], 16)
    assert share[1] == 0.0 and share[4] == 0.0                   # invalid rows never charged
    assert share[0] == share[2] == pytest.approx(2 / 14)         # share over the VALID cards
    assert share[3] == pytest.approx(1 / 14) and share[6] == pytest.approx(10 / 14)
    # a smaller group slices the rank batch by position; a group of < 2 valid is skipped
    emb2 = RecordingEmbedder()
    share2 = diversity_share(["a", "a", "", "b", "c", ""], 0.1, 3, emb2)
    assert [c[0] for c in emb2.calls] == [["a", "a"], ["b", "c"]]
    assert share2 == [1.0, 1.0, 0.0, 0.5, 0.5, 0.0]
    # fewer than two valid cards: no call, no penalty (the original's len(valid) >= 2 gate)
    emb3 = RecordingEmbedder()
    assert diversity_share(["a", "", ""], 0.1, 16, emb3) == [0.0, 0.0, 0.0] and not emb3.calls


def test_worker_compute_rm_score(tok, monkeypatch, tmp_path):
    """Decoding, parsing, domain extraction, score placement, metrics, dumps, global step."""
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    W = importlib.import_module("ask2distill.tau2.trainside.synrl_worker")
    from verl import DataProto

    class _Resp:
        def __init__(self, body):
            self._b = body

        def json(self):
            return {"results": [{"R": 0.75, "rejected": None, "mean_turns": 9.0, "student_fail": 6}
                                if "actions" in c else {"R": 0.0, "rejected": "wrong_shape"}
                                for c in self._b["cands"]]}

    sent = {}

    def post(url, json=None, timeout=None):
        sent.update(url=url, body=json, timeout=timeout)
        return _Resp(json)

    monkeypatch.setattr(W, "requests", types.SimpleNamespace(post=post, get=lambda *a, **k: None))
    monkeypatch.setattr(W.Tau2RewardWorker, "rank", 0, raising=False)
    w = W.Tau2RewardWorker.__new__(W.Tau2RewardWorker)
    w.tok = tok
    w.s = W.WorkerSettings.from_env({**WORKER_ENV, "TAU2_DUMP": str(tmp_path / "dump.jsonl")})
    w.step, w._emb = 0, RecordingEmbedder()

    card = json.dumps({"scenario": {"reason_for_call": "x"},
                       "actions": [{"name": "get_order_details", "arguments": {"order_id": "#W1"}}]})
    texts = [card, "```json\n" + card + "\n```", "no json here"]
    ids = [tok.encode(t, add_special_tokens=False) for t in texts]
    L = max(len(x) for x in ids) + 2
    resp = torch.full((3, L), tok.pad_token_id or 0, dtype=torch.long)
    attn = torch.zeros((3, 64 + L), dtype=torch.long)
    attn[:, :64] = 1
    for i, x in enumerate(ids):
        resp[i, :len(x)] = torch.tensor(x)
        attn[i, 64:64 + len(x)] = 1
    ei = np.array([{"domain": "retail", "target": "t1"}, {"domain": "airline"}, {"domain": "retail"}], dtype=object)
    data = DataProto.from_dict(tensors={"responses": resp, "attention_mask": attn},
                               non_tensors={"extra_info": ei}, meta_info={"global_steps": 7})
    out = w.compute_rm_score(data)
    sc, m = out.batch["rm_scores"], out.meta_info["metrics"]
    assert sent["url"] == "http://svc:1/score" and sent["body"]["k"] == 6 and sent["timeout"] == 3600
    assert sent["body"]["cands"][0]["target"] == "t1" and sent["body"]["cands"][2] == {"domain": "retail"}
    nz = [(i, int(torch.nonzero(sc[i]).max()), float(sc[i].max())) for i in range(3) if sc[i].abs().sum() > 0]
    # two identical scenarios cluster together: share 2/2, so R = 0.75 * max(0, 1 - 1) = 0
    assert w.step == 7 and nz == []
    assert abs(m["tau2/parse_frac"] - 2 / 3) < 1e-6 and abs(m["tau2/valid_frac"] - 2 / 3) < 1e-6
    rows = [json.loads(x) for x in (tmp_path / "dump.jsonl.r0").read_text().splitlines()]
    assert [r["step"] for r in rows] == [7, 7, 7] and rows[0]["R_raw"] == 0.75 and rows[0]["share"] == 1.0
    assert set(rows[0]) == {"step", "domain", "R", "R_raw", "share", "rejected", "n_actions", "turns",
                            "teacher_ok", "target", "card"}

    # without the diversity term the scalar sits on each row's last response token
    w._emb = None
    out = w.compute_rm_score(data)
    sc = out.batch["rm_scores"]
    nz = [(i, int(torch.nonzero(sc[i]).max()), float(sc[i].max())) for i in range(3) if sc[i].abs().sum() > 0]
    assert len(nz) == 2 and all(abs(s - 0.75) < 1e-6 for _, _, s in nz)
    assert nz[0][1] == len(ids[0]) - 1 and nz[1][1] == len(ids[1]) - 1


def test_parse_card():
    pytest.importorskip("verl")
    from ask2distill.tau2.envside.harvest import parse_card as harvest_parse
    from ask2distill.tau2.trainside.synrl_worker import parse_card
    good = {"scenario": {"a": 1}, "actions": [], "plan": {"kind": "compound"}, "preference": "x"}
    for p in (parse_card, harvest_parse):
        assert p("prose " + json.dumps(good) + " more") == {"scenario": {"a": 1}, "actions": [],
                                                            "plan": {"kind": "compound"}}
        assert p('{"scenario": [], "actions": []}') is None and p("nothing") is None


# ------------------------------------------------------------------- synthesizer pick

def _write_dump(prefix: Path, rows_by_step: dict[int, list[tuple[float, str | None]]], ranks: int = 2):
    for r in range(ranks):
        with open(f"{prefix}.r{r}", "w") as fh:
            for step, rows in rows_by_step.items():
                for i, (R, rej) in enumerate(rows):
                    if i % ranks == r:
                        fh.write(json.dumps({"step": step, "R_raw": R, "R": R, "rejected": rej}) + "\n")


def test_select_window_rule(tmp_path):
    from ask2distill.tau2 import select_synthesizer as S
    rows = {s: [(0.5, None), (0.0, "x")] for s in range(1, 21)}
    rows[8] = [(1.0, None), (1.0, None)]             # lifts the windows ending at 8, 9, 10, 11
    rows[10] = [(0.9, None), (0.2, "pref_dump")]     # rejected rows never count, even with R > 0
    _write_dump(tmp_path / "d", rows)
    by = S.load_dump(tmp_path / "d")
    table = S.score_table(by, save_freq=5, total_steps=20, window=4)
    assert [r["step"] for r in table] == [5, 10, 15, 20]
    assert table[1]["rows"] == 8
    assert table[1]["score"] == pytest.approx((0.5 + 2.0 + 0.5 + 0.9) / 8)
    assert S.pick(table) == 10
    assert S.pick([{"step": 5, "score": 0.3}, {"step": 10, "score": 0.3}]) == 5


# ------------------------------------------------------------------------- harvest

def test_harvest_jobs_and_completion(tmp_path):
    from ask2distill.tau2.envside import harvest as H
    targets = {"retail": ["v7:1", "v7:2", "v7:2", "v7:3"], "airline": []}
    jobs = H.jobs(targets, 3)
    assert len(jobs) == 12 and jobs[:4] == [("retail", "v7:1")] * 3 + [("retail", "v7:2")]
    pool = tmp_path / "pool.jsonl"
    rows = [{"idx": 0, "rejected": None}, {"idx": 1, "rejected": "rollout_error:x"},
            {"idx": 2, "rejected": "gen_error:y"}, {"idx": 3, "rejected": "teacher_unsolvable"},
            {"idx": 1, "rejected": None}]
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert H.done_indices(str(pool)) == {0, 1, 3}
    assert H.progress(str(pool), 4) == (3, 4)


# --------------------------------------------------------------------------- bank

def _card(scenario, actions, **extra):
    return {"scenario": scenario, "actions": actions, **extra}


def test_synth_rows(tmp_path):
    from ask2distill.tau2 import bank as B
    w = {"name": "cancel_reservation", "arguments": {"reservation_id": "ABC"}}
    rd = {"name": "get_user_details", "arguments": {"user_id": "u"}}
    pool = tmp_path / "pool.jsonl"
    rows = [
        {"idx": 0, "domain": "airline", "target": "cancel_reservation", "rejected": None,
         "card": _card({"reason_for_call": "cancel ABC", "unknown_info": None}, [rd, w])},
        # same write with another phrasing -> duplicate task
        {"idx": 1, "domain": "airline", "target": "cancel_reservation", "rejected": None,
         "card": _card({"reason_for_call": "please cancel"}, [w])},
        {"idx": 2, "domain": "airline", "target": "refuse:cancel_reservation", "rejected": None,
         "card": _card({"reason_for_call": "cancel XYZ", "task_instructions": "be nice"}, [rd],
                       forbidden_action={"name": "cancel_reservation", "arguments": {"reservation_id": "XYZ"}})},
        {"idx": 3, "domain": "retail", "target": "v7:1", "rejected": None,
         "card": _card({"reason_for_call": "swap", "task_instructions": "hi"}, [],
                       plan={"parts": [{"kind": "cancel", "order_id": "#W1"}], "persona": "Be terse."})},
        {"idx": 4, "domain": "retail", "target": "v7:1", "rejected": "plan_dump", "card": _card({}, [])},
        # a redone row supersedes the earlier one for the same idx
        {"idx": 4, "domain": "retail", "target": "v7:1", "rejected": None,
         "card": _card({"reason_for_call": "other"}, [], plan={"parts": [{"kind": "cancel", "order_id": "#W2"}]})},
    ]
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out, n = B.synth_rows([pool])
    assert n == {"pool_idx": 5, "valid": 5, "rows": 4}
    assert [r["extra_info"]["task_id"] for r in out] == ["syn0", "syn2", "syn3", "syn4"]
    sc = [json.loads(r["extra_info"]["scenario"]) for r in out]
    assert sc[0] == {"reason_for_call": "cancel ABC", "domain": "airline"}          # None dropped
    assert sc[1]["task_instructions"] == B.REFUSE_PERSONA
    assert sc[2]["task_instructions"] == "hi Be terse."
    assert out[0]["prompt"] == [{"role": "user", "content": "cancel ABC"}]
    assert set(out[0]) == {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    assert list(out[0]["extra_info"]) == ["index", "domain", "task_id", "target", "scenario"]
    only_retail, _ = B.synth_rows([pool], ["retail"])
    assert [r["extra_info"]["task_id"] for r in only_retail] == ["syn3", "syn4"]


def test_mix_reindexes(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from ask2distill.tau2 import bank as B
    for name, n in (("a", 2), ("b", 3)):
        B.write_rows([{"data_source": "tau2_opd", "prompt": [{"role": "user", "content": f"{name}{i}"}],
                       "ability": "agentic", "reward_model": {"style": "model", "ground_truth": ""},
                       "extra_info": {"index": i, "domain": name, "task_id": f"syn{i}", "target": None,
                                      "scenario": "{}"}} for i in range(n)], tmp_path / f"{name}.parquet")
    assert B.mix([tmp_path / "a.parquet", tmp_path / "b.parquet"], tmp_path / "m.parquet") == 5
    df = pd.read_parquet(tmp_path / "m.parquet")
    assert [e["index"] for e in df["extra_info"]] == [0, 1, 2, 3, 4]
    assert [e["domain"] for e in df["extra_info"]] == ["a", "a", "b", "b", "b"]


def _canon(df) -> list[str]:
    import numpy as np

    def plain(x):
        if isinstance(x, np.ndarray):
            return [plain(v) for v in x]
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        return x
    return [json.dumps({c: plain(row[c]) for c in df.columns}, sort_keys=True, default=str)
            for _, row in df.iterrows()]


@pytest.mark.equiv
def test_released_pools_rebuild_bank(tmp_path):
    """The released pools rebuild the released banks exactly."""
    pd = pytest.importorskip("pandas")
    data = Path(os.environ.get("A2D_DATA", "/nonexistent"))
    pools = [data / "tau2" / "pool_v6s50.jsonl", data / "tau2" / "pool_v7s60.jsonl"]
    released = data / "banks" / f"{AGENT_QWEN3_4B}.parquet"
    if not all(p.is_file() for p in [*pools, released]):
        pytest.skip("released tau2 pools / bank not present under $A2D_DATA")
    from ask2distill.tau2 import bank as B
    assert B.build_synth([pools[0]], tmp_path / "air.parquet", ["airline"]) == 976
    assert B.build_synth([pools[1]], tmp_path / "ret.parquet", ["retail"]) == 1326
    B.mix([tmp_path / "air.parquet", tmp_path / "ret.parquet"], tmp_path / "mix.parquet")
    assert _canon(pd.read_parquet(tmp_path / "mix.parquet")) == _canon(pd.read_parquet(released))


# ----------------------------------------------------------------------- summarize

def _write_results(root: Path, model: str, domain: str, rewards: dict[str, list[float]]):
    d = root / f"{model}__{domain}"
    d.mkdir(parents=True, exist_ok=True)
    k = 0
    for task, rs in rewards.items():
        for r in rs:
            (d / f"{k}.json").write_text(json.dumps({"task_id": task, "reward": r}))
            k += 1


def test_summarize(tmp_path):
    from ask2distill.tau2 import summarize as S
    _write_results(tmp_path, "Qwen3-m", "airline", {"a": [1, 0, 1, 1], "b": [0.9999, 0.5, 0, 0], "c": [1, 1]})
    _write_results(tmp_path, "Qwen3-m", "retail", {"a": [1, 1, 1, 1, 0]})
    rec = S.record(tmp_path, "Qwen3-m")
    assert rec["airline"] == {"pass1": 50.0, "pass4": 0.0, "tasks": 2}      # task c has < 4 trials
    assert rec["retail"] == {"pass1": 100.0, "pass4": 100.0, "tasks": 1}    # only the first 4 trials
    assert rec["telecom"] == {"pass1": None, "pass4": None, "tasks": 0} and rec["mean3"] is None
    _write_results(tmp_path, "Qwen3-m", "telecom", {"a": [0, 0, 0, 1]})
    flat = S.summarize(tmp_path, "Qwen3-m")
    assert (flat["airline"], flat["retail"], flat["telecom"], flat["mean"]) == (50.0, 100.0, 25.0, 58.3)
    assert "(2t)" in S.table({"Qwen3-m": S.record(tmp_path, "Qwen3-m")})
