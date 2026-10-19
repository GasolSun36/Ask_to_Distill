"""Sample n responses per problem of ONE math task with vLLM on ONE GPU.

Writes ``<out_dir>/<task>_t<T>_p<P>_n<N>-MNT<M>.jsonl`` (task lower-cased,
e.g. ``aime24_t0.7_p0.95_n16-MNT31744.jsonl``), one row per (rollout, problem)::

    {"example_id", "prompt", "answer", "seed", "response"}

``seed`` is the rollout index, bookkeeping only. The file name encodes the
settings, so runs at different budgets coexist; an existing file means the task
is done and is skipped (``--replace`` regenerates). Writes are atomic: the
graders treat any existing ``<task>_t*.jsonl`` as finished.

Kept exactly as the final runs had it:

* the prompt is ``"{problem} Please reason step by step, and put your final
  answer within \\boxed{}."`` rendered as the only user turn with
  ``enable_thinking=False`` (``--disable-thinking``; templates that do not know
  the switch ignore it);
* **stop ids** are the FIRST token of ``encode("<|im_end|>")`` and of
  ``encode("<|endoftext|>")``. For Qwen tokenizers these are the real special
  tokens (151645, 151643); the R1-Distill tokenizer has neither, so both
  encode to ``<`` first and token 27 (``<``) stops generation for the two
  DeepSeek-R1-Distill pairs -- see docs/DEVIATIONS.md;
* every rollout index is a separate ``llm.generate`` call over all problems,
  with no per-request seed (per-request RNGs can force vLLM off the
  FlashInfer sampler); the engine seed is vLLM's default unless
  ``--engine-seed`` is given;
* ``enforce_eager`` on by default (see ``--no-enforce-eager``).

Dropped from the original: the LCB / DOCMATH tasks, multi-GPU fan-out inside one
process (``run`` starts one process per task and GPU instead), and saving a
partial file when a rollout crashes (a crash now writes nothing and exits 1).

Usage::

    python -m ask2distill.eval.math.generate --model DIR --task AIME24 \\
        --data DIR/AIME24/test.parquet --n 16 --out-dir OUT [--limit k]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ask2distill.common.io import atomic_write_text
from ask2distill.common.prompts_math import eval_prompt

STOP_STRINGS = ("<|im_end|>", "<|endoftext|>")


def output_name(task: str, temperature: float, top_p: float, n: int, max_tokens: int) -> str:
    """``aime24_t0.7_p0.95_n16-MNT31744.jsonl`` (floats formatted as Python prints them)."""
    return f"{task.lower()}_t{float(temperature)}_p{float(top_p)}_n{int(n)}-MNT{int(max_tokens)}.jsonl"


def stop_token_ids(tokenizer: Any) -> list[int]:
    """First token of each of ``STOP_STRINGS`` -- NOT necessarily a special token.

    Kept as run: under the R1-Distill tokenizer this yields ``[27, 27]``.
    """
    ids: list[int] = []
    for stop in STOP_STRINGS:
        try:
            if hasattr(tokenizer, "encode"):
                encoded = tokenizer.encode(stop, add_special_tokens=False)
                if encoded:
                    ids.append(encoded[0])
        except Exception:
            pass
    return ids


def load_samples(filepath: str | Path, limit: int | None = None) -> list[dict]:
    """Problems of a test parquet: ``{"example_id", "prompt", "answer"}``.

    The layout is picked from the columns (raw ``problem``/``answer`` dumps or
    verl format), not from the file name.
    """
    import pandas as pd
    df = pd.read_parquet(filepath)
    if "problem" in df.columns and "answer" in df.columns:
        samples = [{"example_id": i,
                    "prompt": str(df.at[i, "problem"]).strip(),
                    "answer": str(df.at[i, "answer"]).strip()} for i in range(len(df))]
    else:
        samples = [{"example_id": i,
                    "prompt": df.at[i, "prompt"][0]["content"].strip(),
                    "answer": df.at[i, "reward_model"]["ground_truth"].strip()}
                   for i in range(len(df))]
    if limit:
        samples = samples[:limit]
    print(f"Total unique samples: {len(samples)}", flush=True)
    return samples


def format_prompt(tokenizer: Any, prompt_text: str, enable_thinking: bool) -> str:
    """Chat-template one user turn; tokenizers without ``enable_thinking`` get the plain call."""
    msgs = [{"role": "user", "content": prompt_text}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=enable_thinking)
    except TypeError:
        # Some tokenizers (e.g. DeepSeek-R1-Distill) don't accept the
        # enable_thinking kwarg; fall back to the plain template.
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def generate(model: str, samples: list[dict], n: int, *, temperature: float, top_p: float,
             max_tokens: int, gpu_memory_utilization: float, enforce_eager: bool,
             enable_thinking: bool, engine_seed: int | None = None) -> list[dict]:
    """All rollouts of all samples; rollout-major order (rollout 0 for every problem first)."""
    import torch
    from vllm import LLM, SamplingParams

    # enforce_eager=True disables vLLM's torch.compile/inductor path, which can
    # intermittently crash a worker with "InductorError: CUDA driver error: file
    # not found" (a flaky compile issue, esp. with NFS-backed triton caches).
    llm = LLM(model=model, trust_remote_code=True, gpu_memory_utilization=gpu_memory_utilization,
              tensor_parallel_size=1, enforce_eager=enforce_eager,
              **({"seed": int(engine_seed)} if engine_seed is not None else {}))
    try:
        tokenizer = llm.get_tokenizer()
        stops = stop_token_ids(tokenizer)
        print(f"[gen] stop_token_ids={stops} enable_thinking={enable_thinking}", flush=True)
        prompts = [format_prompt(tokenizer, s["prompt"], enable_thinking) for s in samples]
        results: list[dict] = []
        for rollout_id in range(n):
            sampling = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                      stop_token_ids=stops if stops else None)
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            for sample, out in zip(samples, outputs):
                results.append({"example_id": sample["example_id"], "prompt": sample["prompt"],
                                "answer": sample["answer"], "seed": rollout_id,
                                "response": out.outputs[0].text})
            print(f"[gen] rollout {rollout_id + 1}/{n} done", flush=True)
        return results
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def run_task(*, model: str, task: str, data: str | Path, n: int, out_dir: str | Path,
             temperature: float, top_p: float, max_tokens: int, gpu_memory_utilization: float,
             enforce_eager: bool, enable_thinking: bool, engine_seed: int | None = None,
             limit: int | None = None, replace: bool = False) -> Path:
    """Generate one task into ``out_dir`` (skipped when its file exists); returns the file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name(task, temperature, top_p, n, max_tokens)
    if not replace and out_path.exists():
        print(f"Result file already exists at '{out_path}'. Skipping.", flush=True)
        return out_path
    samples = load_samples(data, limit)
    for s in samples:
        s["prompt"] = eval_prompt(s["prompt"])
    if samples:
        print("Example prompt after formatting:\n" + samples[0]["prompt"], flush=True)
    rows = generate(model, samples, n, temperature=temperature, top_p=top_p,
                    max_tokens=max_tokens, gpu_memory_utilization=gpu_memory_utilization,
                    enforce_eager=enforce_eager, enable_thinking=enable_thinking,
                    engine_seed=engine_seed)
    if not rows:
        raise RuntimeError(f"no generations for {task}")
    atomic_write_text(out_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"Saved {len(rows)} generations for {task} to {out_path}", flush=True)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate math eval rollouts for one task with vLLM.")
    ap.add_argument("--model", required=True, help="merged HF model directory")
    ap.add_argument("--task", required=True, help="task name, e.g. AIME24 (lower-cased in the file name)")
    ap.add_argument("--data", required=True, help="the task's test.parquet")
    ap.add_argument("--n", type=int, required=True, help="rollouts per problem")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--top-p", type=float, required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, required=True)
    ap.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, required=True)
    think = ap.add_mutually_exclusive_group(required=True)
    think.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    think.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    ap.add_argument("--engine-seed", type=int, default=None,
                    help="vLLM engine seed (unset in the final runs)")
    ap.add_argument("--limit", type=int, default=None, help="first k problems only (smoke)")
    ap.add_argument("--replace", action="store_true", help="regenerate an existing file")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        run_task(model=a.model, task=a.task, data=a.data, n=a.n, out_dir=a.out_dir,
                 temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens,
                 gpu_memory_utilization=a.gpu_memory_utilization, enforce_eager=a.enforce_eager,
                 enable_thinking=a.enable_thinking, engine_seed=a.engine_seed, limit=a.limit,
                 replace=a.replace)
    except Exception as e:  # noqa: BLE001 -- report and fail the task, never leave a partial file
        print(f"GEN_FAILED {a.task}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
