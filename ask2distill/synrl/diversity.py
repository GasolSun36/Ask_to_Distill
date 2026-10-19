"""BLEU-based cluster share of a scoring batch: the diversity factor of the reward.

Paper §2.3 (Eq. 9-10), following the cluster-based repetition penalty of
R-Zero. For the full batch B = {q_1 .. q_N} of questions scored in one
synthesizer step:

    dist(i, j) = 1 - sentence_BLEU(hypothesis = q_i.split(), reference = [q_j.split()])
                 for i < j (mirrored to j > i), 0 on the diagonal;
    labels     = average-linkage agglomerative clustering of dist, cut at τ;
    share(q_i) = |{j : label_j = label_i}| / N;
    P_div(q_i) = max(0, 1 - λ_div · share(q_i)).

BLEU is NLTK's ``sentence_bleu`` with ``SmoothingFunction().method1`` on
whitespace tokens, the R-Zero implementation. No embedding model is involved.

The batch is the WHOLE step (``synrl.train_batch_size x synrl.rollout_n``
questions, 64 for the configured runs), but verl hands each reward rank only
its slice. ``gather_batch`` therefore all-gathers the question texts over the
reward ranks (a CPU ``gloo`` group, see ``init_text_group``) before
clustering; every rank clusters the same full batch and keeps the shares of
its own slice. Rank r's slice is chunk r of verl's ``DP_COMPUTE_PROTO``
split, so concatenating the gathered lists in rank order restores the batch
order. Every row takes part, including ones too short to be solved (their
reward is 0 anyway), as in the per-group embedding mode.
"""
from __future__ import annotations

import warnings
from collections import Counter
from typing import Sequence

import numpy as np


def bleu_distance_matrix(texts: Sequence[str]) -> np.ndarray:
    """``1 - sentence BLEU`` between every pair of texts (R-Zero's ``_bleu_distance_matrix``)."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    n = len(texts)
    toks = [str(t).split() for t in texts]
    smoother = SmoothingFunction().method1
    dist = np.zeros((n, n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n):
            for j in range(i + 1, n):
                score = sentence_bleu([toks[j]], toks[i], smoothing_function=smoother)
                dist[i, j] = dist[j, i] = 1.0 - score
    return dist


def cluster_share(dist: np.ndarray, thresh: float) -> list[float]:
    """Per-item cluster share under average-linkage clustering of ``dist`` cut at ``thresh``."""
    n = int(dist.shape[0])
    if n < 2:
        return [1.0] * n
    from sklearn.cluster import AgglomerativeClustering
    labels = AgglomerativeClustering(n_clusters=None, distance_threshold=thresh,
                                     metric="precomputed", linkage="average").fit_predict(dist)
    size = Counter(labels)
    return [size[x] / n for x in labels]


def bleu_cluster_share(texts: Sequence[str], thresh: float) -> tuple[list[float], float]:
    """``(share per text, mean off-diagonal BLEU similarity)`` of one batch.

    The mean similarity is a logging-only monitor; -1.0 for fewer than two texts.
    """
    dist = bleu_distance_matrix(texts)
    n = len(texts)
    mean_sim = float((1.0 - dist)[~np.eye(n, dtype=bool)].mean()) if n > 1 else -1.0
    return cluster_share(dist, thresh), mean_sim


def penalty(share: float, pen: float) -> float:
    """P_div = max(0, 1 - λ_div · share)."""
    return max(0.0, 1.0 - pen * share)


# ------------------------------------------------------------ distributed batch

def init_text_group(rank: int, world_size: int) -> bool:
    """Join a CPU ``gloo`` process group over the reward ranks; False when there is one rank.

    Uses the ``MASTER_ADDR`` / ``MASTER_PORT`` that verl's worker group sets
    for every worker. A process that already belongs to a group keeps it.
    """
    if world_size <= 1:
        return False
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    return True


def gather_batch(local: Sequence[str]) -> tuple[list[str], int]:
    """``(full batch in rank order, offset of this rank's slice)``.

    Without an initialised process group (one reward rank, or unit tests) the
    local slice is the batch.
    """
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return list(local), 0
    parts: list[list[str] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(parts, list(local))
    rank = dist.get_rank()
    offset = sum(len(p or []) for p in parts[:rank])
    return [q for p in parts for q in (p or [])], offset
