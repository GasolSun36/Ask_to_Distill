"""Stage 0: calibrate δ_ref, the κ-th percentile of the per-token divergence δ_t.

δ_ref is the fixed scale of the token score s_t = min(δ_t / (δ_ref + ε), 1) · ω_t
(paper §2.3), so it is measured under exactly the conditions the reward later
scores: questions written by the synthesizer, the same solve prompt
(``calib.solve_template``, shared with the reward worker through
``synrl.solve``), the same chat-template switch, and the same solve budget as
the OPD run the questions are for (7168 math, 4096 code). A δ_ref measured on
a different budget is measured on a different δ_t distribution.

Procedure (no external questions are used):

1. **Questions.** The INITIAL synthesizer (``models.synthesizer_init``, i.e.
   the untrained student or its no-think variant) writes ``calib.n_questions``
   (N_cal) questions under the synthesizer-RL prompt: topics from the seeded
   rotation of ``synrl.topics`` (``calib.question_sampling.topic_seed``),
   vLLM sampling as in the RL rollouts, thinking blocks stripped and
   questions shorter than ``reward.min_question_chars`` dropped, exactly as
   the reward worker would. The questions are written to ``calib.questions``
   (with their topics) and reused when that file already holds N_cal rows,
   so a rerun only repeats the scoring.
2. **Rollouts.** The student samples ``rolls`` solutions per question with
   vLLM (temperature 0.7, top_p 0.95, no top_k, ``max_tokens = max_new``).
   The token ids are used as returned, EOS included.
3. **Divergence.** Student and teacher (HF, bf16, sdpa) are run on prompt +
   solution; the teacher goes on ``calib.teacher_device`` when one card cannot
   hold both (a 30B/32B code teacher plus the student is ~68GB of weights).
   δ_t is collected over every response position of every rollout of at least
   ``min_tokens`` tokens (truncated rollouts included) and pooled.

Output ``calib.out``: ``percentile`` (κ), ``delta_ref`` (the κ-th percentile),
``delta_ref_rounded`` (rounded to ``round_decimals``, the value a run trains
with), ``n_questions``, ``n_rollouts``, ``n_tokens`` and the p50 / p75 / p90 /
p99 / mean of δ_t. Copy ``delta_ref_rounded`` into the pair's
``reward.delta_ref``; the run prints its distance to
``calib.expected_delta_ref``. vLLM sampling is not bit-reproducible, so a
rerun agrees to a few percent, not exactly.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from omegaconf import DictConfig

from ask2distill import config as a2d_config
from ask2distill import runtime
from ask2distill.common.io import atomic_write_text, read_jsonl, write_jsonl
from ask2distill.common.text import strip_question
from ask2distill.paths import get_paths
from ask2distill.synrl.solve import render_solve_prompt, solve_template

PERCENTILES = (50, 75, 90, 99)


# ------------------------------------------------------------------ questions

def load_questions(path: str | os.PathLike) -> list[str]:
    """The ``problem`` field of every row of a calibration jsonl, in file order."""
    return [r["problem"] for r in read_jsonl(path)]


def question_records(texts: Sequence[str], topics: Sequence[str], min_chars: int) -> list[dict]:
    """``{"problem", "topic"}`` for every completion that is a question to the reward worker.

    The completion is cleaned as the reward worker cleans a synthesizer
    rollout (thinking block stripped, then ``.strip()``); one shorter than
    ``min_chars`` would score 0 there without being solved, so it is dropped.
    """
    out = []
    for text, topic in zip(texts, topics):
        q = strip_question(text)
        if len(q) >= min_chars:
            out.append({"problem": q, "topic": topic})
    return out


def sample_questions(cfg: DictConfig) -> list[dict]:
    """``calib.n_questions`` questions of the initial synthesizer (see the module docstring).

    One vLLM call per round; a round asks for the missing count plus a margin
    of ``oversample``, and rounds continue the same topic rotation, so the
    result depends only on the config (and on vLLM's sampling).
    """
    from ask2distill.harvest import sampling
    c, q = cfg.calib, cfg.calib.question_sampling
    model = str(cfg.models.synthesizer_init)
    need, min_chars = int(c.n_questions), int(cfg.reward.min_question_chars)
    topics = sampling.load_topics(cfg)
    tok = sampling.load_tokenizer(model)
    rng = random.Random(int(q.topic_seed))
    rows: list[dict] = []
    for rnd in range(int(q.max_rounds)):
        missing = need - len(rows)
        if missing <= 0:
            break
        n = missing + math.ceil(missing * float(q.oversample))
        prompts, drawn = sampling.build_prompts(tok, n, rng, str(cfg.domain), topics)
        texts = sampling.generate(model, prompts, max_new=int(q.max_new), prompt_head=int(q.prompt_head),
                                  temperature=float(q.temperature), top_p=float(q.top_p),
                                  gpu_memory_utilization=float(q.gpu_memory_utilization))
        got = question_records(texts, drawn, min_chars)
        print(f"[calib] round {rnd}: {len(got)}/{n} completions are questions", flush=True)
        rows += got
    if len(rows) < need:
        raise RuntimeError(f"the initial synthesizer {model} wrote only {len(rows)} usable questions "
                           f"in {q.max_rounds} rounds (need {need}); check its chat template and "
                           "calib.question_sampling")
    return rows[:need]


def ensure_questions(cfg: DictConfig) -> list[str]:
    """The calibration questions: ``calib.questions`` if it holds N_cal rows, else sampled now."""
    c = cfg.calib
    path, need = Path(str(c.questions)), int(c.n_questions)
    if path.is_file():
        have = load_questions(path)
        if len(have) >= need:
            print(f"[calib] reusing {need} questions of {path}", flush=True)
            return have[:need]
        print(f"[calib] {path} has {len(have)} < {need} questions; sampling afresh", flush=True)
    rows = sample_questions(cfg)
    write_jsonl(path, rows)
    print(f"[calib] {len(rows)} questions of {cfg.models.synthesizer_init} -> {path}", flush=True)
    return [r["problem"] for r in rows]


# ------------------------------------------------------------------- scoring

def calib_prompt(tok: Any, question: str, c: DictConfig) -> str:
    """The solve prompt as calibration renders it (BOS and literal DeepSeek BOS stripped)."""
    return render_solve_prompt(tok, question, template=solve_template(c.solve_template),
                               system=str(c.solve_system), qwen3_nothink=bool(c.qwen3_nothink),
                               strip_deepseek_bos=True)


def sample_rollouts(student: str, prompts: Sequence[str], c: DictConfig) -> list[list[list[int]]]:
    """``rolls`` vLLM samples per prompt, as token-id lists (EOS kept)."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=student, max_model_len=int(c.max_new) + int(c.prompt_head),
              gpu_memory_utilization=float(c.gpu_memory_utilization), trust_remote_code=True)
    outs = llm.generate(list(prompts), SamplingParams(n=int(c.rolls), temperature=float(c.temperature),
                                                      top_p=float(c.top_p), max_tokens=int(c.max_new)))
    rollouts = [[list(o_.token_ids) for o_ in o.outputs] for o in outs]
    del llm
    return rollouts


def collect_divergence(stu: Any, tea: Any, prompt_ids: Sequence[list[int]],
                       rollouts: Sequence[Sequence[list[int]]], *, topk: int, chunk: int,
                       min_tokens: int, device: str, teacher_device: str) -> list[np.ndarray]:
    """δ_t (float32 numpy, one array per scored rollout) over every response position.

    Row r of a full-sequence forward predicts token r+1, so the response is
    read from rows ``plen-1 .. L-2``.
    """
    import torch

    from ask2distill.common.divergence import per_token_chunked
    all_delta: list[np.ndarray] = []
    with torch.inference_mode():
        for pids, rs in zip(prompt_ids, rollouts):
            for rids in rs:
                if len(rids) < min_tokens:
                    continue
                ids = list(pids) + list(rids)
                L, plen = len(ids), len(pids)
                t = torch.tensor([ids], device=device)
                ls_all = stu(t).logits[0]
                lt_all = tea(t if teacher_device == device else t.to(teacher_device)).logits[0]
                if teacher_device != device:
                    lt_all = lt_all.to(device)
                delta, _ = per_token_chunked(ls_all, lt_all, topk, chunk, start=plen - 1, stop=L - 1)
                del ls_all, lt_all
                all_delta.append(delta.float().cpu().numpy())
    return all_delta


def summarize(all_delta: Sequence[np.ndarray], percentile: float, round_decimals: int) -> dict:
    """The calibration record: δ_ref = the ``percentile``-th percentile of the pooled δ_t."""
    if not 0 < float(percentile) < 100:
        raise a2d_config.ConfigError(f"calib.percentile={percentile} must lie in (0, 100)")
    if not all_delta:
        raise RuntimeError("no rollout was long enough to score (calib.min_tokens)")
    v = np.concatenate(all_delta)
    out: dict[str, Any] = {"percentile": float(percentile),
                           "delta_ref": float(np.percentile(v, percentile)),
                           "n_rollouts": len(all_delta), "n_tokens": int(v.size)}
    for p in PERCENTILES:
        out[f"p{p}"] = float(np.percentile(v, p))
    out["mean"] = float(v.mean())
    out["delta_ref_rounded"] = round(out["delta_ref"], int(round_decimals))
    return out


def run(cfg: DictConfig, out_path: Path) -> dict:
    """Sample questions and rollouts, teacher-force, and write the record to ``out_path``."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    c = cfg.calib
    student, teacher = str(cfg.models.student), str(cfg.models.teacher)
    print(f"[calib] synthesizer={cfg.models.synthesizer_init}\n[calib] student={student}\n"
          f"[calib] teacher={teacher}", flush=True)
    questions = ensure_questions(cfg)
    gc.collect()
    torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(student, trust_remote_code=True)
    prompts = [calib_prompt(tok, q, c) for q in questions]
    rollouts = sample_rollouts(student, prompts, c)
    gc.collect()
    torch.cuda.empty_cache()

    dev = "cuda"
    # Empty keeps student and teacher on one device (the math pairs).
    tdev = str(c.teacher_device or "") or dev
    stu = AutoModelForCausalLM.from_pretrained(student, torch_dtype=torch.bfloat16,
                                               attn_implementation="sdpa").to(dev).eval()
    tea = AutoModelForCausalLM.from_pretrained(teacher, torch_dtype=torch.bfloat16,
                                               attn_implementation="sdpa").to(tdev).eval()
    print(f"[calib] student_dev={dev} teacher_dev={tdev}", flush=True)
    prompt_ids = [tok(p, add_special_tokens=True)["input_ids"] for p in prompts]
    all_delta = collect_divergence(stu, tea, prompt_ids, rollouts, topk=int(c.topk), chunk=int(c.chunk),
                                   min_tokens=int(c.min_tokens), device=dev, teacher_device=tdev)
    out = {"n_questions": len(questions), **summarize(all_delta, c.percentile, c.round_decimals)}
    print("[calib] " + json.dumps(out), flush=True)
    atomic_write_text(out_path, json.dumps(out, indent=1) + "\n")
    expected = c.expected_delta_ref
    if expected is not None:
        diff = abs(out["delta_ref_rounded"] - float(expected))
        print(f"[calib] delta_ref={out['delta_ref_rounded']} expected={expected} "
              f"|diff|={diff:.4f} ({100 * diff / float(expected):.2f}%)", flush=True)
    return out


def _apply_env(stage: str) -> None:
    """Export the stage hygiene environment into this process (vLLM reads it at import)."""
    for k, v in runtime.base_env(get_paths(), stage=stage).items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def main(cfg: DictConfig, args: argparse.Namespace | None = None) -> dict:
    """``a2d calibrate --pair P``: write ``calib.out`` and return the record."""
    for section in ("calib", "models", "synrl"):
        a2d_config.require(cfg, section)
    c = cfg.calib
    out_path = Path(str(c.out))
    if getattr(args, "dry_run", False):
        q = c.question_sampling
        print(f"[calib] would sample {c.n_questions} questions from {cfg.models.synthesizer_init} "
              f"(topics {cfg.synrl.topics}, seed {q.topic_seed}, temperature {q.temperature}, "
              f"top_p {q.top_p}, max_new {q.max_new}) -> {c.questions}", flush=True)
        print(f"[calib] would calibrate {cfg.models.student} <- {cfg.models.teacher} "
              f"(rolls {c.rolls}, max_new {c.max_new}, template {c.solve_template}, "
              f"percentile {c.percentile}, teacher on {c.teacher_device or 'cuda'}) -> {out_path}",
              flush=True)
        return {}
    _apply_env("calib")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(str(c.questions)).parent.mkdir(parents=True, exist_ok=True)
    return run(cfg, out_path)
