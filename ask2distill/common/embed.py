"""Qwen3-Embedding-4B: semantic dedup, the checkpoint pick and the ``emb`` diversity mode.

The frozen embedding model serves the post-generation semantic deduplication
(``greedy_dedup``, paper §2.5), the diversity statistics of the checkpoint pick,
and the optional ``reward.diversity.mode: emb`` share. The default diversity
share of the reward is BLEU-based and needs no embedding model
(``synrl.diversity``).

The embedding is NOT Qwen's official last-token pooling: it is a bf16 mean pool
over the first 512 tokens, L2-normalised, with the tokenizer's default (right)
padding and batches of 16. The thresholds (0.1 clustering, 0.95 dedup, 0.85
pick) were set on exactly this embedding, and bf16 results depend on the batch
shapes near those thresholds, so the call structure is part of the contract:

* ``cluster_share`` embeds EACH group in its own ``embed`` call (one call per
  diversity group) and skips the call for a group of fewer than two texts.
* ``greedy_dedup`` embeds ALL texts in one call, then walks them greedily in
  blocks of 2048 on the embedder's device.

No function here has a default threshold; callers pass the configured value.
"""
from __future__ import annotations

from collections import Counter
from typing import Protocol, Sequence

import numpy as np
import torch

EMBED_BS = 16
EMBED_MAX_LEN = 512
DEDUP_BLOCK = 2048


class SupportsEmbed(Protocol):
    """Anything with the ``Embedder.embed`` signature (tests use a recording stub)."""

    device: str

    def embed(self, texts: Sequence[str], bs: int = EMBED_BS) -> torch.Tensor: ...


class Embedder:
    """Mean-pooled, L2-normalised sentence embeddings from a causal-LM backbone."""

    def __init__(self, model_path: str, device: str = "cuda", max_len: int = EMBED_MAX_LEN,
                 dtype: torch.dtype = torch.bfloat16) -> None:
        from transformers import AutoModel, AutoTokenizer
        self.model_path = str(model_path)
        self.device = device
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, torch_dtype=dtype,
                                               trust_remote_code=True).to(device).eval()

    @torch.no_grad()
    def embed(self, texts: Sequence[str], bs: int = EMBED_BS) -> torch.Tensor:
        """``[n, d]`` float32 CPU tensor, one unit vector per text."""
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok(list(texts[i:i + bs]), padding=True, truncation=True,
                         max_length=self.max_len, return_tensors="pt").to(self.device)
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1)
            v = (h * m).sum(1) / m.sum(1).clamp_min(1)
            out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu())
        return torch.cat(out)


def _as_numpy(E: torch.Tensor | np.ndarray) -> np.ndarray:
    return E.numpy() if isinstance(E, torch.Tensor) else np.asarray(E)


def _share_one_group(texts: Sequence[str], thresh: float, embedder: SupportsEmbed) -> list[float]:
    """Share of the group each text's cluster occupies, on cosine distance."""
    from sklearn.cluster import AgglomerativeClustering
    n = len(texts)
    if n < 2:
        return [1.0 / max(n, 1)] * n
    E = _as_numpy(embedder.embed(list(texts), bs=EMBED_BS))
    dist = np.clip(1.0 - E @ E.T, 0, None)
    np.fill_diagonal(dist, 0.0)
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=thresh,
                                  metric="precomputed", linkage="average").fit_predict(dist)
    size = Counter(lab)
    return [size[x] / n for x in lab]


def _groups(n: int, group: int | None) -> list[slice]:
    g = group or n            # 0 / None = the whole batch
    return [slice(i, i + g) for i in range(0, n, max(g, 1))]


def cluster_share(texts: Sequence[str], thresh: float, group: int | None,
                  embedder: SupportsEmbed) -> list[float]:
    """Per-text cluster share within consecutive groups of ``group`` texts.

    Average-linkage agglomerative clustering on cosine distance, cut at
    ``thresh``; a text's share is its cluster's size over its group's size, so a
    fully unique text in a group of 8 still has share 1/8. ``group`` larger than
    the batch (or 0/None) means the whole batch is one group.
    """
    share: list[float] = []
    for sl in _groups(len(texts), group):
        share += _share_one_group(texts[sl], thresh, embedder)
    return share


def mean_cos(texts: Sequence[str], group: int | None, embedder: SupportsEmbed) -> float:
    """Mean off-diagonal cosine over the groups (the worker's ``synrm/mean_cos``).

    Logging only. Embeds each group in a separate call, as the worker did right
    after ``cluster_share``; groups of one text contribute nothing. -1.0 when
    no group has two texts.
    """
    vals = []
    for sl in _groups(len(texts), group):
        sub = list(texts[sl])
        E = _as_numpy(embedder.embed(sub, bs=EMBED_BS))
        if len(sub) > 1:
            C = E @ E.T
            vals.append(float(C[~np.eye(len(sub), dtype=bool)].mean()))
    return sum(vals) / len(vals) if vals else -1.0


def greedy_dedup(texts: Sequence[str], thr: float, embedder: SupportsEmbed,
                 block: int = DEDUP_BLOCK) -> list[int]:
    """Indices kept by one greedy pass: keep a text if its cosine to every
    already-kept text is below ``thr``.

    O(n*k) with k = kept, which stays tractable because k saturates -- that
    saturation is exactly what the saturation filter measures.
    """
    if not texts:
        return []
    E = _as_numpy(embedder.embed(list(texts), bs=EMBED_BS))
    device = getattr(embedder, "device", "cuda")
    keep: list[int] = []
    kept_vecs = None
    for i in range(0, len(texts), block):
        chunk = torch.tensor(E[i:i + block]).to(device)
        for j in range(chunk.shape[0]):
            v = chunk[j:j + 1]
            if kept_vecs is None or kept_vecs.shape[0] == 0:
                keep.append(i + j)
                kept_vecs = v.clone()
                continue
            if float((kept_vecs @ v.T).max()) < thr:
                keep.append(i + j)
                kept_vecs = torch.cat([kept_vecs, v], 0)
    return keep
