"""Stage 2a: merge and sample every saved synthesizer checkpoint.

The final checkpoint can reach the reward it was optimised for while collapsing
onto one family of questions, with far fewer effective clusters than the
untrained synthesizer. The useful checkpoint is wherever that trade turns bad, which
cannot be read off the training curve -- the v4 synthesizer passed every training
metric and then produced 300 copies of one question -- so every checkpoint is
sampled.

For each step in ``select.sweep.steps``:

1. merge ``<synrl.ckpt_dir>/global_step_<N>/actor`` into
   ``merge.hf_dir(cfg, synrl.experiment, N)`` unless already merged (the
   harvest and the yield comparison later read these merged directories);
2. sample ``n`` questions under the training conditioning, with topics rotated
   by ``random.Random(topic_seed)`` afresh for every checkpoint, so all
   checkpoints answer the same topics;
3. write ``sweep_dir/sweep_step<N>.jsonl`` with ``o.outputs[0].text.strip()``
   only. Thinking blocks are NOT stripped here (the harvest strips them); the
   pick's cosine statistics were computed on exactly this raw text.

``max_new`` must equal the max_response_length the synthesizer was TRAINED
under, or the sample comes from a distribution it was never optimised on.

Incremental by construction: a step whose sweep file already holds at least
90% of ``n`` rows is skipped, and a step whose checkpoint is not saved yet is
reported and left for a later call (Code_Qwen3-4B_Qwen3-4B-Instruct-2507 swept each
checkpoint as it landed). ``select.pick`` counts the sweep files rather than testing that one
exists: the sweep writes one file per step as it goes, so "a file exists" is
true after the first of twelve, and picking then sees a single checkpoint and
dutifully selects it -- which is how Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct once chose
step 5 of 60 and started
a 96k-question harvest from a barely-trained synthesizer.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig

from ask2distill import config as a2d_config
from ask2distill.common.io import count_lines, write_jsonl
from ask2distill.harvest import sampling

DONE_FRACTION = 0.9           # a sweep file with >= 90% of its rows counts as done


def sweep_file(sweep_dir: str | Path, step: int) -> Path:
    """``<sweep_dir>/sweep_step<N>.jsonl``."""
    return Path(sweep_dir) / f"sweep_step{step}.jsonl"


def sweep_records(texts: Iterable[str], topics: Iterable[str] | None = None) -> list[dict]:
    """``{"problem": text.strip(), "topic": topic}`` for every non-empty completion.

    Raw text on purpose: no ``strip_thinking`` (see the module docstring).
    ``topic`` is the prompt's topic (omitted when ``topics`` is None).
    """
    texts = list(texts)
    out = []
    for text, topic in zip(texts, topics if topics is not None else [None] * len(texts)):
        t = text.strip()
        if t:
            out.append({"problem": t} if topic is None else {"problem": t, "topic": topic})
    return out


def is_done(path: Path, n: int) -> bool:
    """A sweep file counts as done once it holds at least 90% of ``n`` rows."""
    return path.is_file() and count_lines(path) >= n * DONE_FRACTION


def merged_dir(cfg: DictConfig, step: int) -> Path:
    """Where the merged synthesizer of ``step`` lives (owned by ``ask2distill.merge``)."""
    from ask2distill import merge
    return Path(merge.hf_dir(cfg, str(cfg.synrl.experiment), int(step)))


def actor_dir(cfg: DictConfig, step: int) -> Path:
    """The FSDP actor shards of synthesizer step ``step``."""
    return Path(str(cfg.synrl.ckpt_dir)) / f"global_step_{step}" / "actor"


def ensure_merged(cfg: DictConfig, step: int) -> Path | None:
    """The merged HF dir of ``step``, merging it first if needed; ``None`` on failure."""
    hf = merged_dir(cfg, step)
    if (hf / "config.json").is_file():
        return hf
    src = actor_dir(cfg, step)
    if not src.is_dir():
        print(f"[sweep] step {step}: no checkpoint at {src} yet, skip", flush=True)
        return None
    from ask2distill import merge
    print(f"[sweep] merging step {step}", flush=True)
    try:
        merge.merge_fsdp(src, hf)
    except Exception as e:           # a failed merge skips the step, as the original did
        print(f"[sweep] step {step} merge FAILED ({type(e).__name__}: {e}), skip", flush=True)
        return None
    if not (hf / "config.json").is_file():
        print(f"[sweep] step {step} merge FAILED, skip", flush=True)
        return None
    return hf


def sweep_step(cfg: DictConfig, step: int, model: Path, topics: list[str]) -> Path:
    """Sample ``select.sweep.n`` questions from ``model`` and write the sweep file."""
    s = cfg.select.sweep
    out = sweep_file(s.out_dir, step)
    tok = sampling.load_tokenizer(model)
    rng = random.Random(int(s.topic_seed))        # same topics for every checkpoint
    prompts, drawn = sampling.build_prompts(tok, int(s.n), rng, str(cfg.domain), topics)
    texts = sampling.generate(model, prompts, max_new=int(s.max_new), prompt_head=int(s.prompt_head),
                              temperature=float(s.temperature), top_p=float(s.top_p),
                              gpu_memory_utilization=float(s.gpu_memory_utilization))
    write_jsonl(out, sweep_records(texts, drawn))
    print(f"[sweep] step {step} -> {count_lines(out)}", flush=True)
    return out


def main(cfg: DictConfig, args: argparse.Namespace | None = None) -> int:
    """``a2d select sweep --pair P``."""
    a2d_config.require(cfg, "select.sweep")
    s = cfg.select.sweep
    if s.strip_thinking is not False:
        raise a2d_config.ConfigError("select.sweep.strip_thinking must be false: the sweep keeps "
                                     "the raw text the pick thresholds were set on")
    steps = [int(x) for x in s.steps]
    todo = [st for st in steps if not is_done(sweep_file(s.out_dir, st), int(s.n))]
    for st in sorted(set(steps) - set(todo)):
        print(f"[sweep] step {st} already done", flush=True)
    if getattr(args, "dry_run", False):
        for st in todo:
            print(f"[sweep] would merge {actor_dir(cfg, st)} and sample {s.n} questions "
                  f"(max_new {s.max_new}, T {s.temperature}) -> {sweep_file(s.out_dir, st)}", flush=True)
        return 0
    if todo:
        sampling.export_stage_env("select")
        Path(str(s.out_dir)).mkdir(parents=True, exist_ok=True)
        topics = sampling.load_topics(cfg)
        for st in todo:
            model = ensure_merged(cfg, st)
            if model is not None:
                sweep_step(cfg, st, model, topics)
    have = sum(sweep_file(s.out_dir, st).is_file() for st in steps)
    print(f"SWEEP_DONE {have}/{len(steps)} steps swept", flush=True)
    return 0
