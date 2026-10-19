"""Embedding helpers: cluster share, greedy dedup, required thresholds (no model needed).

bf16 embeddings depend on batch shapes near the thresholds, so the number of
``embed`` calls, their sizes and the batch size are part of the contract:
``cluster_share`` makes one call per diversity group (none for a group of one),
``greedy_dedup`` one call for all texts, always with bs 16.
"""
from __future__ import annotations

import zlib

import numpy as np
import torch

from ask2distill.common import embed as E


def _vectors(texts):
    """Deterministic unit vectors; texts sharing a prefix before '#' are near-duplicates."""
    out = []
    for t in texts:
        key, _, var = t.partition("#")
        base = np.random.default_rng(zlib.crc32(key.encode())).standard_normal(32)
        noise = np.random.default_rng(zlib.crc32(t.encode())).standard_normal(32) * (0.05 if var else 0)
        v = base + noise
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, dtype=np.float32)


class Recorder:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def embed(self, texts, bs=16):
        self.calls.append((list(texts), bs))
        return torch.from_numpy(_vectors(texts))


def test_cluster_share_values():
    rec = Recorder()
    texts = ["a", "a#1", "a#2", "b", "c", "d", "e", "f"]
    share = E.cluster_share(texts, 0.1, 8, rec)
    assert share[:3] == [3 / 8] * 3 and share[3:] == [1 / 8] * 5


def test_greedy_dedup_empty():
    rec = Recorder()
    assert E.greedy_dedup([], 0.95, rec) == []
    assert rec.calls == []


def test_no_default_thresholds():
    import inspect
    for fn in (E.cluster_share, E.greedy_dedup):
        sig = inspect.signature(fn)
        assert all(p.default is inspect.Parameter.empty for name, p in sig.parameters.items()
                   if name in ("thresh", "thr"))
