"""Saturation funnel and cross-slice aggregation on synthetic shards, on CPU.

The embedding model is replaced by a deterministic stub (questions that differ
only in their numbers embed identically, so the 0.95 dedup has work to do).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from ask2distill.filter import aggregate as AG
from ask2distill.filter import saturation as SAT


class StubEmbedder:
    device = "cpu"

    def embed(self, texts, bs=16):
        rows = []
        for t in texts:
            key = re.sub(r"\d+", "#", t)
            seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(16).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return torch.from_numpy(np.stack(rows))


WORDS = ["triangle", "prime", "sequence", "circle", "polynomial", "matrix", "grid", "cube",
         "probability", "function", "integer", "divisor", "remainder", "square", "graph"]


def _math_raw(rng: random.Random, i: int) -> str:
    nums = " ".join(str(rng.randint(1, 99)) for _ in range(6))
    w = rng.sample(WORDS, 4)
    kind = i % 10
    body = (f"Consider a {w[0]} and a {w[1]} defined by the values {nums}. "
            f"Find the {w[2]} of the {w[3]} formed by these numbers?")
    if kind == 0:
        return "<think>draft</think>" + body
    if kind == 1:
        return body + " Answer: 12"
    if kind == 2:
        return body + " Wait, let me reconsider the numbers once more."
    if kind == 3:
        return "short?"
    if kind == 4:
        return body.replace("?", "")
    return body


def _code_raw(rng: random.Random, i: int) -> str:
    nums = " ".join(str(rng.randint(1, 99)) for _ in range(6))
    w = rng.sample(WORDS, 4)
    body = (f"You are given a {w[0]} with n cells and a {w[1]} of weights {nums}. For each query "
            f"report the {w[2]} over a range of cells and the {w[3]} of the answers so far.\n\n"
            "Input\nThe first line contains n and q. Each of the next q lines contains a query.\n\n"
            "Output\nPrint the answer to every query on its own line.")
    kind = i % 8
    if kind == 0:
        return "<think>x</think>" + body
    if kind == 1:
        return body + "\n\nWait, the constraints may be too small."
    if kind == 2:
        return "Problem 1: " + body
    return body


def _write_shards(tmp_path: Path, domain: str, shards: int = 2, per: int = 600) -> list[Path]:
    rng = random.Random(7 if domain == "math" else 11)
    make = _math_raw if domain == "math" else _code_raw
    files = []
    for s in range(shards):
        f = tmp_path / f"raw.s{s}.jsonl"
        rows = [make(rng, s * per + i) for i in range(per)]
        rows += rows[:25]                                  # exact repeats inside a shard
        f.write_text("".join(json.dumps({"problem": q}, ensure_ascii=False) + "\n" for q in rows))
        files.append(f)
    return files


@pytest.mark.parametrize("domain", ["math", "code"])
def test_run_slice_funnel(domain, tmp_path):
    files = _write_shards(tmp_path, domain)
    res = SAT.run_slice(files, tmp_path / "sat.s0.json", domain, 0.95, StubEmbedder())
    assert res["raw"] == 1250
    assert res["raw"] > res["fmt"] >= res["jac"] >= res["emb"] > 0
    assert [c[0] for c in res["curve"]] == [1000]
    qs = [json.loads(line)["problem"] for line in (tmp_path / "sat.s0_questions.jsonl").open()]
    assert len(qs) == res["emb"]
    assert json.loads((tmp_path / "sat.s0.json").read_text()) == res
    assert all("<think>" not in q for q in qs)


def test_questions_path():
    assert SAT.questions_path("/x/sat.s10.json").name == "sat.s10_questions.jsonl"
    assert SAT.questions_path("/x/saturation_g9.json").name == "saturation_g9_questions.jsonl"


def test_aggregate_keeps_order_and_dedups(tmp_path):
    a, b = tmp_path / "sat.s0_questions.jsonl", tmp_path / "sat.s1_questions.jsonl"
    a.write_text("".join(json.dumps({"problem": q}) + "\n" for q in ["alpha 1", "beta 2"]))
    b.write_text("".join(json.dumps({"problem": q}) + "\n" for q in ["alpha 7", "gamma 3"]))
    n = AG.aggregate([b, a], tmp_path / "pool.jsonl", 0.95, StubEmbedder())
    pool = [json.loads(line)["problem"] for line in (tmp_path / "pool.jsonl").open()]
    assert n == 3 and pool == ["alpha 7", "gamma 3", "beta 2"]
