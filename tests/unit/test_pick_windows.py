"""Checkpoint pick: the local-step window fix, the ``min_rows`` skip and the gates.

``windows`` must map restarted reward-worker counters back onto global steps
for save_freq 5 and 10, as the pick of the final runs did.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ask2distill.select import pick as PK


def _vec(text: str, dim: int = 32) -> np.ndarray:
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def stub_embed(texts):
    return np.stack([_vec(t) for t in texts]).astype(np.float32)


class StubEmbedder:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def embed(self, texts, bs=16):
        self.calls.append((len(texts), bs))
        return torch.from_numpy(stub_embed(list(texts)))


RESTARTS = [list(range(1, 24)), list(range(1, 23)), list(range(1, 21))]
RESTARTS_10 = [list(range(1, 18)), list(range(1, 31)), list(range(1, 21))]


def _flat(windows):
    return [k for w in windows for k in w]


def test_windows_save_freq_5():
    got = PK.windows(_flat(RESTARTS), 5)
    assert got == list(range(1, 24)) + list(range(21, 43)) + list(range(41, 61))


def test_windows_save_freq_10():
    got = PK.windows(_flat(RESTARTS_10), 10)
    assert got == list(range(1, 18)) + list(range(11, 41)) + list(range(41, 61))
    # the same counters under save_freq 5 resume elsewhere
    assert PK.windows(_flat(RESTARTS_10), 5)[17] == 16


def test_windows_no_restart_is_identity():
    assert PK.windows(list(range(1, 61)), 5) == list(range(1, 61))


# --------------------------------------------------------------------------- fixtures

def _write_metrics(prefix: Path, ranks: int, counters, value, extra=None):
    for r in range(ranks):
        with open(f"{prefix}.metrics.r{r}", "w") as fh:
            for i, k in enumerate(counters):
                row = {"step": k, "synrm/delta_mean": 1.0, "synrm/R_raw": value(i, k, r)}
                if extra:
                    row.update(extra(i, k))
                fh.write(json.dumps(row) + "\n")
            fh.write("not json\n")


def _diverse(step: int, n: int) -> list[str]:
    return [f"step {step} question {i}: compute the {i}-th prime of a grid" for i in range(n)]


def _make_inputs(tmp_path: Path, save_freq: int):
    dump = tmp_path / "synrl" / "synrl_dump_x.jsonl"
    dump.parent.mkdir()
    counters = _flat(RESTARTS if save_freq == 5 else RESTARTS_10)
    # reward rises to a peak around global step 40, then falls
    glob_steps = PK.windows(counters, save_freq)
    _write_metrics(dump, 8, counters,
                   lambda i, k, r: 0.10 + 0.002 * min(glob_steps[i], 40) - 0.003 * max(glob_steps[i] - 40, 0)
                   + 0.0001 * r)
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    sizes = {save_freq * j: 60 for j in range(1, 60 // save_freq + 1)}
    sizes[2 * save_freq] = 49                          # under min_rows: skipped
    for st, n in sizes.items():
        qs = _diverse(st, n) if st != 3 * save_freq else ["the same question"] * n   # collapsed
        (sweep / f"sweep_step{st}.jsonl").write_text(
            "".join(json.dumps({"problem": q}) + "\n" for q in qs))
    return dump, sweep


PCFG = {"cos_max": 0.85, "min_eff_clusters": 5, "eff_cluster_cos": 0.9, "reward_frac": 0.95,
        "reward_window": 2, "min_rows": 50, "allow_empty": False}


@pytest.mark.parametrize("save_freq", [5, 10])
def test_pick_gates_and_min_rows(tmp_path, save_freq):
    dump, sweep = _make_inputs(tmp_path, save_freq)
    res = PK.pick(dump, sweep, dict(PCFG, save_freq=save_freq), StubEmbedder(), domain="math")
    steps = [r["step"] for r in res["table"]]
    assert 2 * save_freq not in steps                  # min_rows skip
    collapsed = next(r for r in res["table"] if r["step"] == 3 * save_freq)
    assert not collapsed["passed"] and collapsed["reason"] == "cos_over_max"
    assert res["step"] in steps and res["fallback"] is False
    assert all(r["reward"] == r["reward"] for r in res["table"])      # no NaN: windows fixed


def test_global_step_field_used(tmp_path):
    dump = tmp_path / "d.jsonl"
    # local counters say 1, 2, 3 but the rows carry their true global steps
    _write_metrics(dump, 1, [1, 2, 3], lambda i, k, r: float(k),
                   extra=lambda i, k: {"global_step": 100 + k})
    assert PK.reward_by_step(dump, 5) == {101: 1.0, 102: 2.0, 103: 3.0}


def test_allow_empty_fallback(tmp_path):
    dump = tmp_path / "d.jsonl"
    _write_metrics(dump, 1, [1, 2, 3, 4, 5], lambda i, k, r: 0.1)
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    for st in (3, 5):
        (sweep / f"sweep_step{st}.jsonl").write_text("\n".join([json.dumps({"problem": "same"})] * 8)
                                                     + "\n")
    pcfg = dict(PCFG, save_freq=5, min_rows=8)
    assert PK.pick(dump, sweep, pcfg, StubEmbedder())["step"] is None
    res = PK.pick(dump, sweep, dict(pcfg, allow_empty=True), StubEmbedder())
    assert res["step"] == 5 and res["fallback"] is True


# ------------------------------------------------------------------------ equivalence
