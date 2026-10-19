"""The reward worker's rank-batch logic (CPU, stubbed models).

Generation, the student / teacher forwards and the embedder are replaced by
deterministic stubs, so everything in between runs for real: the
>=20-character question gate, the <10-token rollout rule, the ``trunc_zero``
drop, the mean over scored rollouts, the per-token scoring, the scalar on the
last valid token, the diversity multiplier (``bleu`` over the full batch,
``emb`` per group), the local step counter, and the dump / metrics files.
"""
from __future__ import annotations

import json
import os
import random
import zlib

import numpy as np
import pytest
import torch

from conftest import CODE_QWEN3_30B, MATH_QWEN3_1P7B

RLEN = 96

# 16 rows = two diversity groups of 8. Shared first words make embedding clusters;
# TINY / TRUNC / MIXED steer the generation stub (see _gen_stub).
QUESTIONS = [
    "alpha: find all integers n such that n^2 + 1 divides 2^n + 1.",
    "alpha: find all integers m such that m^2 + 3 divides 3^m + 3.",
    "short one?",
    "<think>scratch work</think>beta: a train leaves at 3pm at 60 km/h; when does it arrive?",
    "<think>this block never closes and the budget runs out",
    "",
    "gamma: un café vend 120 croissants; find the expected profit given costs.",
    "alpha: find all integers k such that k^2 + 5 divides 5^k + 5.",
    "delta TINY: every rollout of this question is shorter than ten tokens.",
    "delta TRUNC: every rollout of this question hits the solve budget.",
    "epsilon MIXED: one rollout is too short, the other one is scored.",
    "zeta: count lattice paths from (0,0) to (7,7) avoiding the diagonal.",
    "zeta: count lattice paths from (0,0) to (6,6) avoiding the diagonal.",
    "eta: prove that the sum of reciprocals of primes diverges, give a bound.",
    "theta: a 3x3 grid is colored with two colors; count symmetric colorings.",
    "zeta: count lattice paths from (0,0) to (5,5) avoiding the diagonal.",
]


class CharTok:
    """Character-level tokenizer: id = code point (capped), no BOS."""

    bos_token = None
    eos_token_id = 1
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        s = "".join(f"<{m['role']}>{m['content']}" for m in msgs)
        if kw.get("enable_thinking") is False:
            s += "<think></think>"
        return s + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=True, return_tensors=None, padding=False):
        return {"input_ids": torch.tensor([[min(ord(c), 1000) for c in text]])}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids)


def _gen_stub(rolls: int, calls: list | None = None):
    """Deterministic rollouts ``(ids, truncated)`` per question."""
    def gen(questions):
        if calls is not None:
            calls.append(list(questions))
        out = []
        for q in questions:
            rng = random.Random(zlib.crc32(q.encode()))
            rs = []
            for r in range(rolls):
                n = rng.randint(12, 60)
                trunc = rng.random() < 0.3
                if "TINY" in q or ("MIXED" in q and r == 0):
                    n, trunc = 6, False
                if "TRUNC" in q:
                    trunc = True
                rs.append(([rng.randint(5, 200) for _ in range(n)], trunc))
            out.append(rs)
        return out
    return gen


def _logits_stub(ids, plen):
    keep = int(ids.shape[1] - plen + 1)
    g = torch.Generator().manual_seed(zlib.crc32(str(ids.tolist()).encode()))
    return torch.randn(keep, 64, generator=g) * 3, torch.randn(keep, 72, generator=g) * 3


def _fake_embed_np(texts):
    rows = []
    for t in texts:
        key = (t.split() or [""])[0]
        v = np.random.default_rng(zlib.crc32(key.encode())).standard_normal(8)
        v = v + np.random.default_rng(zlib.crc32(t.encode())).standard_normal(8) * 0.05
        rows.append((v / np.linalg.norm(v)).astype(np.float32))
    return np.stack(rows) if rows else np.zeros((0, 8), np.float32)


class RecordingEmbedder:
    device = "cpu"

    def __init__(self):
        self.calls: list[tuple[int, int]] = []

    def embed(self, texts, bs=16):
        self.calls.append((len(texts), bs))
        return torch.from_numpy(_fake_embed_np(list(texts)))


def _batch():
    from verl import DataProto
    B = len(QUESTIONS)
    plen = 5
    resp = torch.zeros(B, RLEN, dtype=torch.long)
    attn = torch.zeros(B, plen + RLEN, dtype=torch.long)
    attn[:, :plen] = 1
    for i, q in enumerate(QUESTIONS):
        ids = [ord(c) for c in q]
        if ids:
            resp[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[i, plen:plen + len(ids)] = 1
    return DataProto.from_dict(tensors={"responses": resp, "attention_mask": attn})


@pytest.fixture(scope="module")
def rw():
    """``ask2distill.synrl.reward_worker`` imported without leaking its allocator setting."""
    prev = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    import ask2distill.synrl.reward_worker as mod
    if prev is None:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = prev
    return mod


def _reward_cfg(pair: str, tmp_path, **over) -> dict:
    from ask2distill import config
    cfg = config.to_container(config.load(pair), "reward")
    cfg["dump"] = str(tmp_path / f"new_{pair}" / "dump.jsonl")
    for k, v in over.items():
        if k.startswith("div_"):
            cfg["diversity"][k[4:]] = v
        else:
            cfg[k] = v
    return cfg


def _new_worker(rw, cfg, calls=None):
    sc = rw.Scorer(cfg, "cpu")
    sc.tok = CharTok()
    sc.generate = _gen_stub(sc.rolls, calls)
    sc.scored_logits = _logits_stub
    sc.embedder = RecordingEmbedder()
    w = object.__new__(rw.SynRewardWorker)
    w.reward_cfg, w.scorer, w.dump_path, w.step, w._rank = cfg, sc, cfg["dump"], 0, 0
    os.makedirs(os.path.dirname(cfg["dump"]), exist_ok=True)
    return w


def _run_new(rw, cfg, calls=None):
    w = _new_worker(rw, cfg, calls)
    out = w.compute_rm_score(_batch())
    rows = [json.loads(x) for x in open(w.dump_path + ".r0", encoding="utf-8")]
    return w, out.batch["rm_scores"], rows


def test_gates_and_last_token(rw, tmp_path):
    calls: list = []
    w, scores, rows = _run_new(rw, _reward_cfg(CODE_QWEN3_30B, tmp_path), calls)
    solved = [q for c in calls for q in c]
    stripped = [rw.strip_question(q) for q in QUESTIONS]
    # only questions of >= 20 characters (after stripping a thinking block) are solved
    assert solved == [q for q in stripped if len(q) >= 20]
    lens = [len(q) for q in QUESTIONS]
    for i, row in enumerate(rows):
        assert row["problem"] == stripped[i]
        nz = scores[i].nonzero().flatten().tolist()
        assert nz in ([], [lens[i] - 1])                      # scalar on the last valid token only
        if len(stripped[i]) < 20:
            assert row["R"] == 0.0 and set(row) == {"step", "rank", "R", "n_docs", "problem"}
    by_q = {QUESTIONS[i]: rows[i] for i in range(len(rows))}
    tiny = by_q["delta TINY: every rollout of this question is shorter than ten tokens."]
    assert tiny["R"] == 0.0 and "delta_mean" not in tiny and "share" not in tiny
    trunc = by_q["delta TRUNC: every rollout of this question hits the solve budget."]
    assert trunc["R"] == 0.0 and "delta_mean" not in trunc        # code: trunc_zero drops them


def test_trunc_kept_without_trunc_zero(rw, tmp_path):
    _, _, rows = _run_new(rw, _reward_cfg(MATH_QWEN3_1P7B, tmp_path))
    trunc = next(r for r in rows if "TRUNC" in r["problem"])
    assert "delta_mean" in trunc and trunc["R"] > 0                # math scores truncated prefixes


def test_mean_over_scored_rollouts(rw, tmp_path):
    cfg = _reward_cfg(CODE_QWEN3_30B, tmp_path, div_pen=0.0)
    w, _, rows = _run_new(rw, cfg)
    q = next(x for x in QUESTIONS if "MIXED" in x)
    (ids0, tr0), (ids1, tr1) = _gen_stub(2)([q])[0]
    r0, d0 = w.scorer.score_one(q, ids0, tr0)
    r1, d1 = w.scorer.score_one(q, ids1, tr1)
    assert d0 == {} and r0 == 0.0                               # 6 tokens < 10: not scored
    row = next(r for r in rows if r["problem"] == q)
    if tr1:
        assert row["R"] == 0.0
    else:
        assert row["R"] == pytest.approx(r1, rel=1e-6) and row["L"] == float(d1["L"])
        assert isinstance(row["L"], float)                     # averaged over scored rollouts


def test_bleu_share_over_the_batch_without_embedder(rw, tmp_path):
    from ask2distill.synrl.diversity import bleu_cluster_share
    cfg = _reward_cfg(MATH_QWEN3_1P7B, tmp_path)
    assert cfg["diversity"]["mode"] == "bleu"
    w, scores, rows = _run_new(rw, cfg)
    assert w.scorer.embedder.calls == []                        # bleu never embeds
    want, mean_sim = bleu_cluster_share([rw.strip_question(q) for q in QUESTIONS], 0.1)
    for i, row in enumerate(rows):
        if "share" in row:
            assert row["share"] == pytest.approx(want[i])
    assert w.scorer.mean_sim == pytest.approx(mean_sim)
    # τ = 0.1 on BLEU distance clusters only near-verbatim repeats: the "zeta"
    # rewordings differ in two tokens and stay apart, every share is 1/16
    assert want == pytest.approx([1 / 16] * 16)


def test_bleu_share_uses_the_gathered_batch(rw, tmp_path):
    from ask2distill.synrl.diversity import bleu_cluster_share
    sc = _new_worker(rw, _reward_cfg(MATH_QWEN3_1P7B, tmp_path)).scorer
    other = ["zeta: count lattice paths from (0,0) to (4,4) avoiding the diagonal.",
             "omega: an unrelated question about the roots of x^5 - x + 1."]
    local = QUESTIONS[8:]
    batch = other + local
    full, _ = bleu_cluster_share(batch, 0.1)
    assert sc.diversity(local, batch, len(other)) == pytest.approx(full[len(other):])
    alone, _ = bleu_cluster_share(local, 0.1)
    assert sc.diversity(local) == pytest.approx(alone)


def test_diversity_groups_embedded_separately(rw, tmp_path):
    w, _, rows = _run_new(rw, _reward_cfg(MATH_QWEN3_1P7B, tmp_path, div_mode="emb"))
    # per group: cluster_share embed, then the mean-cos embed; batches of 16
    assert w.scorer.embedder.calls == [(8, 16), (8, 16), (8, 16), (8, 16)]
    shares = [r["share"] for r in rows if "share" in r]
    assert shares and all(0 < s <= 1 for s in shares)
    alpha = [r for r in rows[:8] if r["problem"].startswith("alpha:") and "share" in r]
    assert all(r["share"] == pytest.approx(len(alpha) / 8) for r in alpha)


def test_whole_batch_group(rw, tmp_path):
    w, _, _ = _run_new(rw, _reward_cfg(MATH_QWEN3_1P7B, tmp_path, div_mode="emb", div_group=0))
    assert w.scorer.embedder.calls == [(16, 16), (16, 16)]


def test_no_scored_row_writes_no_metrics(rw, tmp_path):
    cfg = _reward_cfg(MATH_QWEN3_1P7B, tmp_path, min_question_chars=10_000)
    w = _new_worker(rw, cfg)
    out = w.compute_rm_score(_batch())
    assert out.meta_info["metrics"] == {}
    assert not os.path.exists(w.dump_path + ".metrics.r0")
    assert torch.count_nonzero(out.batch["rm_scores"]) == 0


def test_reward_config_is_required(rw, tmp_path, monkeypatch):
    from ask2distill.synrl.reward_config import RewardConfigError, load_reward_config
    monkeypatch.delenv("A2D_REWARD_CONFIG", raising=False)
    with pytest.raises(RewardConfigError):
        load_reward_config()
    cfg = _reward_cfg(MATH_QWEN3_1P7B, tmp_path)
    del cfg["len_pen"]
    p = tmp_path / "rc.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(RewardConfigError, match="len_pen"):
        load_reward_config(p)
