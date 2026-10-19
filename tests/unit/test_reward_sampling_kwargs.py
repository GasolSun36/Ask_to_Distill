"""Solver sampling in the reward worker: what reaches HF ``generate`` and how rollouts are cut.

The final runs passed only temperature / top_p (plus top_k when
``sample_top_k > 0``) and never ``repetition_penalty``, so the model's
``generation_config`` supplied the rest -- Coder-3B's ``repetition_penalty
1.05`` included. Truncation is "no ``tok.eos_token_id`` in the ids"; a rollout
ending on another stop id counts as truncated unless ``fix_eos`` is set.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from conftest import CODE_CODER_32B, CODE_PAIRS, MATH_PAIRS

EOS, ENDOFTEXT, PAD = 1, 3, 0
PLEN = 4


class Enc(dict):
    def to(self, device):
        return self


class BatchTok:
    bos_token = None
    eos_token_id = EOS
    pad_token_id = PAD

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "|".join(m["content"] for m in msgs)

    def __call__(self, prompts, return_tensors="pt", padding=True, add_special_tokens=True):
        n = len(prompts)
        return Enc(input_ids=torch.full((n, PLEN), 7), attention_mask=torch.ones(n, PLEN, dtype=torch.long))


# Rollout tails, cycled over the generated rows: EOS mid-way, no stop id at all,
# <|endoftext|> (in generation_config but not tok.eos_token_id), EOS after it.
TAILS = [[11, 12, EOS, PAD, PAD], [11, 12, 13, 14, 15], [11, ENDOFTEXT, PAD, PAD, PAD],
         [11, ENDOFTEXT, 12, EOS, PAD]]


class RecordingSolver:
    generation_config = SimpleNamespace(eos_token_id=[EOS, ENDOFTEXT], top_k=20,
                                        repetition_penalty=1.05)

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, **kw):
        self.calls.append(kw)
        rows = kw["input_ids"].shape[0] * kw["num_return_sequences"]
        out = [[7] * PLEN + TAILS[(len(self.calls) + r) % len(TAILS)] for r in range(rows)]
        return torch.tensor(out)


@pytest.fixture(scope="module")
def rw():
    prev = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    import ask2distill.synrl.reward_worker as mod
    if prev is None:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    return mod


def _scorer(rw, pair=CODE_CODER_32B, **over):
    from ask2distill import config
    cfg = config.to_container(config.load(pair), "reward")
    cfg.update(over)
    sc = rw.Scorer(cfg, "cpu")
    sc.tok, sc.solver = BatchTok(), RecordingSolver()
    return sc


QS = [f"question number {i} with enough characters" for i in range(5)]


@pytest.mark.parametrize("pair", MATH_PAIRS + CODE_PAIRS)
def test_final_configs_pass_neither_top_k_nor_repetition_penalty(rw, pair):
    sc = _scorer(rw, pair)
    sc.generate(QS)
    for kw in sc.solver.calls:
        assert "repetition_penalty" not in kw and "top_k" not in kw
        assert set(kw) == {"input_ids", "attention_mask", "do_sample", "temperature", "top_p",
                           "max_new_tokens", "num_return_sequences", "pad_token_id"}
        assert kw["do_sample"] is True
        assert (kw["temperature"], kw["top_p"]) == (0.7, 0.95)
        assert kw["max_new_tokens"] == sc.max_new and kw["num_return_sequences"] == sc.rolls
        assert kw["pad_token_id"] == EOS                    # pad id 0 is falsy -> eos


@pytest.mark.parametrize("top_k,expected", [(20, 20), (40, 40), (-1, 0)])
def test_top_k_only_when_requested(rw, top_k, expected):
    sc = _scorer(rw, sample_top_k=top_k)
    sc.generate(QS[:2])
    assert sc.solver.calls[0]["top_k"] == expected            # HF's "no top-k" is 0
    assert "repetition_penalty" not in sc.solver.calls[0]


@pytest.mark.parametrize("gen_bs,sizes", [(0, [5]), (2, [2, 2, 1]), (8, [5])])
def test_gen_bs_chunks_the_batch(rw, gen_bs, sizes):
    sc = _scorer(rw, gen_bs=gen_bs)
    out = sc.generate(QS)
    assert [c["input_ids"].shape[0] for c in sc.solver.calls] == sizes
    assert len(out) == len(QS) and all(len(r) == sc.rolls for r in out)


def test_truncation_uses_tokenizer_eos_only(rw):
    sc = _scorer(rw, gen_bs=0, rolls=4)
    got = sc.generate(QS[:1])[0]
    seen = {tuple(ids): tr for ids, tr in got}
    assert seen[(11, 12)] is False                             # cut at EOS
    assert seen[(11, 12, 13, 14, 15)] is True                  # no stop id: truncated
    assert seen[(11, ENDOFTEXT, PAD, PAD, PAD)] is True        # <|endoftext|> only: truncated (kept quirk)
    assert seen[(11, ENDOFTEXT, 12)] is False


def test_fix_eos_checks_every_stop_id(rw):
    sc = _scorer(rw, gen_bs=0, rolls=4, fix_eos=True)
    got = dict((tuple(ids), tr) for ids, tr in sc.generate(QS[:1])[0])
    assert got[(11,)] is False                                 # endoftext now ends the rollout
    assert got[(11, 12, 13, 14, 15)] is True
