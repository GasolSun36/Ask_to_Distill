"""The shared divergence code: vocabulary cut, token scores s_t and the rollout reward.
"""
from __future__ import annotations

import itertools

import pytest
import torch

from ask2distill.common import divergence as dv


L, VS, VT, TOPK = 37, 64, 72, 16


def _logits(seed: int, n: int = L, vs: int = VS, vt: int = VT):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, vs, generator=g) * 3, torch.randn(n, vt, generator=g) * 3


def test_vocab_cut_to_common_size():
    ls, lt = _logits(3, vs=70, vt=74)          # heads padded to different sizes (Coder-3B vs -32B)
    delta, omega = dv.per_token(ls, lt, 8)
    delta2, omega2 = dv.per_token(ls, lt[:, :70], 8)
    assert torch.equal(delta, delta2) and torch.equal(omega, omega2)
    # the cut happens BEFORE the softmax: the teacher's raw mass is renormalised over 70 ids
    pt70 = torch.softmax(lt[:, :70].float(), -1)
    A = torch.softmax(ls.float(), -1).topk(8, -1).indices
    assert torch.allclose(omega, pt70.gather(1, A).sum(-1))


def test_token_scores_cap_and_use_omega():
    delta = torch.tensor([0.0, 1.0, 5.0, 100.0])
    omega = torch.tensor([1.0, 0.5, 0.9, 0.2])
    s = dv.token_scores(delta, omega, 2.0, True)
    assert torch.allclose(s, torch.tensor([0.0, 0.25, 0.9, 0.2]))
    assert torch.allclose(dv.token_scores(delta, omega, 2.0, False), torch.tensor([0.0, 0.5, 1.0, 1.0]))


@pytest.mark.parametrize("alpha,lam,max_new", list(itertools.product((0.5, 1.0, 0.0), (0.0, 0.3), (8, 4096))))
def test_rollout_reward_formula(alpha, lam, max_new):
    s = torch.rand(16, generator=torch.Generator().manual_seed(0))
    r_roll, r_raw = dv.rollout_reward(s, alpha, lam, max_new)
    expect_raw = float(s.sum() / 16.0 ** alpha)
    assert r_raw == expect_raw
    assert r_roll == pytest.approx(expect_raw * (1 - lam * min(16 / max_new, 1.0)))
