"""BLEU diversity share (paper §2.3, Eq. 9-10): the R-Zero clustering, over the full batch.

* ``bleu_distance_matrix`` / ``cluster_share`` equal R-Zero's
  ``_bleu_distance_matrix`` / ``cluster_share_per_problem`` (re-stated below
  from its reward function) on the same texts and threshold;
* the penalty is max(0, 1 - λ_div · share);
* ``gather_batch`` over a 2-rank CPU ``gloo`` group hands every rank the full
  batch in rank order and its own offset, so each rank's shares equal the
  shares of a single-process clustering of the whole batch.
"""
from __future__ import annotations

import socket
from collections import Counter

import numpy as np
import pytest
import torch.multiprocessing as mp

from ask2distill.synrl import diversity as D

TEXTS = [
    "Find all integers n such that n^2 + 1 divides 2^n + 1.",
    "Find all integers m such that m^2 + 3 divides 3^m + 3.",
    "Find all integers n such that n^2 + 1 divides 2^n + 1.",
    "A train leaves at 3pm at 60 km/h; when does it arrive 180 km away?",
    "Count lattice paths from (0,0) to (7,7) that never cross the diagonal.",
    "Count lattice paths from (0,0) to (6,6) that never cross the diagonal.",
    "Prove that the sum of the reciprocals of the primes diverges.",
    "",
    "A 3x3 grid is colored with two colors; count the colorings fixed by a rotation.",
    "Let f be a polynomial with integer coefficients such that f(1) = 5 and f(5) = 1.",
    "Count lattice paths from (0,0) to (7,7) that never cross the diagonal.",
    "Given n points in the plane, no three collinear, count the convex quadrilaterals.",
]


def _rzero_shares(problems, distance_threshold, linkage="average"):
    """R-Zero's cluster_share_per_problem (examples/reward_function/caller_penalty.py)."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from sklearn.cluster import AgglomerativeClustering
    n = len(problems)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [problems[j].split()]
                hyp = problems[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold,
                                     metric="precomputed", linkage=linkage).fit_predict(dist)
    size = Counter(labels)
    return [size[lab] / n for lab in labels]


@pytest.mark.parametrize("thresh", [0.1, 0.5])
def test_matches_rzero(thresh):
    share, mean_sim = D.bleu_cluster_share(TEXTS, thresh)
    assert share == pytest.approx(_rzero_shares(TEXTS, thresh))
    assert -1.0 < mean_sim <= 1.0


def test_duplicates_share_a_cluster_and_singletons_get_one_over_n():
    share, _ = D.bleu_cluster_share(TEXTS, 0.1)
    n = len(TEXTS)
    assert share[0] == share[2] == pytest.approx(2 / n)        # exact duplicates
    assert share[4] == share[10] == pytest.approx(2 / n)
    assert share[3] == pytest.approx(1 / n)                    # unique question
    assert D.penalty(share[3], 1.0) == pytest.approx(1 - 1 / n)
    assert D.penalty(1.0, 1.0) == 0.0 and D.penalty(0.75, 2.0) == 0.0


def test_degenerate_batches():
    assert D.bleu_cluster_share([], 0.1)[0] == []
    assert D.bleu_cluster_share(["only one"], 0.1) == ([1.0], -1.0)
    same = ["the same question text"] * 4
    assert D.bleu_cluster_share(same, 0.1)[0] == [1.0] * 4


def test_gather_without_group_is_local():
    assert D.gather_batch(["a", "b"]) == (["a", "b"], 0)
    assert D.init_text_group(0, 1) is False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rank_main(rank, world, port, chunks, out):
    import torch.distributed as dist
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world)
    try:
        batch, offset = D.gather_batch(chunks[rank])
        share, _ = D.bleu_cluster_share(batch, 0.1)
        out[rank] = (batch, offset, share[offset:offset + len(chunks[rank])])
    finally:
        dist.destroy_process_group()


def test_two_rank_gather_equals_single_process_batch():
    world = 2
    chunks = [TEXTS[:5], TEXTS[5:]]                           # uneven slices on purpose
    with mp.Manager() as m:
        out = m.dict()
        mp.spawn(_rank_main, args=(world, _free_port(), chunks, out), nprocs=world, join=True)
        got = dict(out)
    full, _ = D.bleu_cluster_share(TEXTS, 0.1)
    assert got[0][0] == got[1][0] == TEXTS                     # every rank sees the whole batch, in order
    assert (got[0][1], got[1][1]) == (0, 5)
    assert got[0][2] + got[1][2] == pytest.approx(full)
