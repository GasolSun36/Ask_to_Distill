"""Per-token teacher-student divergence and the rollout-level question reward (paper §2.3).

Per position t of a student solution y (prefix c_t = (q, y_<t)), with the
teacher teacher-forced on the SAME token ids:

  A_t = Top-K(p_S), B_t = Top-K(p_T), U_t = A_t u B_t          (K = 16, <= 32 tokens)
  δ_t = sum_{v in U_t} p̄_T(v) log( p̄_T(v) / p̄_S(v) )          KL(p̄_T || p̄_S), Eq. 4
        with p̄ renormalised WITHIN U_t for each side
  ω_t = sum_{v in A_t} p_T(v)                                   raw full-vocab mass, Eq. 5
  s_t = min( δ_t / (δ_ref + ε), 1 ) * ω_t                       Eq. 6, ε = 1e-8

  R_raw  = sum_t s_t / L**α                                     Eq. 7
  r_roll = R_raw * (1 - λ_len * min(L / L_max, 1))              Eq. 8

Configured values: math α 0.5, λ_len 0.3, L_max 7168; code α 1.0, λ_len 0,
L_max 4096 (configs/base/{math,code}.yaml).

Three details that are easy to get wrong and are deliberate here:
  * δ_t is KL(p̄_T || p̄_S) -- teacher first. That is the scoring direction only; the
    OPD loss keeps its own direction.
  * ω_t does NOT renormalise the teacher: it asks what fraction of the teacher's
    total probability the student's own top-16 already covers.
  * The per-position product is formed BEFORE averaging. Averaging δ and ω
    separately and multiplying would pair a high-divergence position with a
    high-compatibility position that is somewhere else entirely.

This is the single implementation behind δ_ref calibration and the reward
worker.
"""
from __future__ import annotations

import torch

EPS = 1e-8


def per_token(s_logits: torch.Tensor, t_logits: torch.Tensor, topk: int
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """δ_t and ω_t for a chunk of positions. Inputs are raw logits ``[n, V]``."""
    # Two models of the same family can pad their output head to different
    # multiples -- Qwen2.5-Coder-3B declares vocab_size 151936, the 32B 152064 --
    # while sharing one tokenizer whose real vocabulary is 151,665. The extra
    # rows are ids the tokenizer never emits, so both sides are cut to the common
    # length BEFORE the softmax, which keeps the two normalisations comparable.
    V = min(s_logits.shape[-1], t_logits.shape[-1])
    ls, lt = s_logits[..., :V], t_logits[..., :V]
    ps_log = torch.log_softmax(ls.float(), -1)
    pt_log = torch.log_softmax(lt.float(), -1)
    ps, pt = ps_log.exp(), pt_log.exp()
    A = ps.topk(topk, -1).indices                       # student's top-k
    B = pt.topk(topk, -1).indices                       # teacher's top-k
    omega = pt.gather(1, A).sum(-1)                     # raw teacher mass on A
    mask = torch.zeros_like(ps, dtype=torch.bool)
    mask.scatter_(1, A, True)
    mask.scatter_(1, B, True)                           # union, <= 2*topk
    psU = torch.where(mask, ps, torch.zeros_like(ps))
    ptU = torch.where(mask, pt, torch.zeros_like(pt))
    psU = psU / psU.sum(-1, keepdim=True).clamp_min(1e-12)
    ptU = ptU / ptU.sum(-1, keepdim=True).clamp_min(1e-12)
    # sum only inside U_t; outside, ptU is 0 and contributes nothing
    term = ptU * (ptU.clamp_min(1e-12).log() - psU.clamp_min(1e-12).log())
    delta = torch.where(mask, term, torch.zeros_like(term)).sum(-1)
    return delta, omega


def per_token_chunked(s_logits: torch.Tensor, t_logits: torch.Tensor, topk: int, chunk: int,
                      start: int = 0, stop: int | None = None
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """``per_token`` over rows ``start..stop-1`` in blocks of ``chunk`` rows.

    Chunking bounds the ``[chunk, V]`` float32 softmax temporaries. Row r of the
    logits predicts token r+1, so for a full-sequence forward the response is
    scored with ``start = plen - 1, stop = L - 1`` (calibration); the reward
    worker asks the model for the response positions only and uses
    ``start = 0, stop = L - plen``.
    """
    stop = s_logits.shape[0] if stop is None else stop
    deltas, omegas = [], []
    for s0 in range(start, stop, chunk):
        s1 = min(s0 + chunk, stop)
        delta, omega = per_token(s_logits[s0:s1], t_logits[s0:s1], topk)
        deltas.append(delta)
        omegas.append(omega)
    return torch.cat(deltas), torch.cat(omegas)


def token_scores(delta: torch.Tensor, omega: torch.Tensor, delta_ref: float, use_omega: bool) -> torch.Tensor:
    """s_t = min(δ_t / (δ_ref + 1e-8), 1) * ω_t (ω_t replaced by 1 when ``use_omega`` is off)."""
    return torch.clamp(delta / (delta_ref + EPS), max=1.0) * (omega if use_omega else torch.ones_like(omega))


def rollout_reward(s: torch.Tensor, alpha: float, len_pen: float, max_new: int
                   ) -> tuple[float, float]:
    """``(r_roll, r_raw)`` for one rollout from its token scores ``s``.

    The length L is the number of scored response tokens (EOS already cut).
    """
    r_raw = float(s.sum() / max(float(s.numel()), 1.0) ** alpha)
    frac = min(int(s.numel()) / max(max_new, 1), 1.0)
    return r_raw * (1.0 - len_pen * frac), r_raw


def score_rollout(delta: torch.Tensor, omega: torch.Tensor, *, delta_ref: float, use_omega: bool,
                  alpha: float, len_pen: float, max_new: int) -> tuple[float, dict]:
    """``(r_roll, diag)`` for one solver rollout.

    ``diag`` keys and their order are part of the reward-dump format:
    ``delta_mean, omega_mean, L, R_raw, len_frac``.
    """
    s = token_scores(delta, omega, delta_ref, use_omega)
    r_roll, r_raw = rollout_reward(s, alpha, len_pen, max_new)
    n_tok = int(delta.numel())
    frac = min(n_tok / max(max_new, 1), 1.0)
    return r_roll, {"delta_mean": float(delta.mean()), "omega_mean": float(omega.mean()),
                    "L": n_tok, "R_raw": r_raw, "len_frac": frac}
