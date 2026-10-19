"""Topic lists of the math / code question synthesizers and their seeded rotation.

Every synthesizer prompt names one topic (``prompts_math.synthesizer_messages`` /
``prompts_code.synthesizer_messages``). The lists are small YAML files shipped in
the repository (``configs/topics/{math,code}.yaml``, selected by
``synrl.topics``), so every prompt set can be rebuilt from the repository alone.

Assignment is a seeded rotation, not independent draws: each block of
``len(topics)`` consecutive prompts is a fresh ``rng.shuffle`` of the whole
list. Every topic therefore appears equally often (to within one block), and
the same ``random.Random(seed)`` always yields the same sequence. The prompt
parquet uses ``synrl.prompt_seed``; the checkpoint sweep re-seeds with
``select.sweep.topic_seed`` for every checkpoint (all checkpoints answer the
same topics); harvest shard ``i`` uses ``harvest.seed_formula(i)``.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Sequence

import yaml

from ask2distill.paths import REPO_ROOT

TOPICS_DIR = REPO_ROOT / "configs" / "topics"


def default_topics_file(domain: str) -> Path:
    """``configs/topics/<domain>.yaml``."""
    return TOPICS_DIR / f"{domain}.yaml"


def load_topics(path: str | os.PathLike) -> list[str]:
    """The ``topics`` list of a topic file, in file order (stripped, non-empty, unique)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"topic file {path} not found (synrl.topics)")
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    raw = doc.get("topics") if isinstance(doc, dict) else None
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a top-level 'topics:' list")
    topics = [str(t).strip() for t in raw if str(t).strip()]
    if not topics:
        raise ValueError(f"{path}: the topic list is empty")
    dup = sorted({t for t in topics if topics.count(t) > 1})
    if dup:
        raise ValueError(f"{path}: duplicate topics {dup}")
    return topics


def rotation(topics: Sequence[str], n: int, rng: random.Random) -> list[str]:
    """``n`` topics: consecutive shuffled copies of ``topics`` drawn from ``rng``."""
    if not topics:
        raise ValueError("no topics to rotate")
    out: list[str] = []
    while len(out) < n:
        block = list(topics)
        rng.shuffle(block)
        out.extend(block)
    return out[:n]
