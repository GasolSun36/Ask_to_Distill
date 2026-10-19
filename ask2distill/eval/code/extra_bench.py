"""Generate and grade one extra code benchmark (APPS / CodeContests) with LCB's grader.

The question goes through the training-time prompt (``prompts_code.solve_prompt``,
LiveCodeBench's stdin/stdout template) as the user turn, after the system turn
"You are a helpful assistant.". Sampling: n 1, T 0.2, top_p 0.95, top_k 20,
4096 new tokens, ``max_model_len = max_new + 4096``. Code is extracted with
LiveCodeBench's ``extract_code`` (``CodeQwenInstruct`` rules) and graded with
its ``codegen_metrics``, so the extraction matches the LCB runs.

Writes ``{"overall": pass@1 (fraction), "n_problems", "by_difficulty": {d:
{"pass@1": percent, "n"}}}``; ``--detail`` adds per-problem pass, generation
lengths, truncation flags, the no-code fraction and the settings.

Run one benchmark per process: vLLM's engine workers and the grader's forked
sandboxes can outlive the engine and keep GPU memory, so the caller
(``run_code_eval``) starts each benchmark in its own process group and kills the
group afterwards.

Usage::

    python -m ask2distill.eval.code.extra_bench --bench-file B.jsonl --model DIR --out B.json \\
        --n 1 --temperature 0.2 --top-p 0.95 --top-k 20 --max-new 4096 \\
        --max-model-len-extra 4096 --tp 2 --util 0.9 --system-prompt "You are a helpful assistant." \\
        --nothink --extraction CodeQwenInstruct --num-process-evaluate 24 --timeout 6
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ask2distill.common.io import atomic_write_text
from ask2distill.common.prompts_code import solve_prompt


def build_prompt(tok: Any, question: str, system_prompt: str, nothink: bool) -> str:
    """Rendered prompt for one question, BOS stripped (vLLM adds it again)."""
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user", "content": solve_prompt(question)}]
    kw = {"enable_thinking": False} if nothink else {}
    try:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    except TypeError:
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    b = tok.bos_token or ""
    # A thinking-only model (Qwen3-*-Thinking-2507) ignores enable_thinking and
    # opens a <think> block in its generation prompt. Closing it here is the
    # only way to score its direct-answer distribution -- the same prefill the
    # hybrid models' template emits for enable_thinking=False.
    if nothink and s.rstrip().endswith("<think>"):
        s = s.rstrip() + "\n\n</think>\n\n"
    return s[len(b):] if b and s.startswith(b) else s


def _lcb_on_path() -> None:
    from ask2distill.paths import get_paths
    lcb = str(get_paths().third_party / "LiveCodeBench")
    if lcb not in sys.path:
        sys.path.insert(0, lcb)


def summarize(rows: list[dict], codes: list[list[str]], metrics: dict, results: dict,
              *, detail: dict | None = None) -> dict:
    """The output record (``detail`` holds what ``--detail`` adds, already computed)."""
    per: dict[str, list[float]] = {}
    for i, r in enumerate(rows):
        ok = results.get(i) or results.get(str(i))
        if ok:
            per.setdefault(r["difficulty"], []).append(
                sum(all(x > 0 for x in s) for s in ok) / len(ok))
    out = {"overall": metrics.get("pass@1"), "n_problems": len(rows),
           "by_difficulty": {k: {"pass@1": 100 * sum(v) / len(v), "n": len(v)}
                             for k, v in sorted(per.items())}}
    if detail is not None:
        pp = []
        for i in range(len(rows)):
            ok = results.get(i) or results.get(str(i))
            pp.append(sum(all(x > 0 for x in s_) for s_ in ok) / len(ok) if ok else 0.0)
        out["per_problem"] = pp
        out["gen_tokens"] = detail["gen_tokens"]
        out["truncated"] = detail["truncated"]
        out["no_code_frac"] = (sum(1 for cc in codes for c in cc if not c.strip())
                               / max(sum(len(cc) for cc in codes), 1))
        out["settings"] = detail["settings"]
    return out


def run(a: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    with open(a.bench_file) as f:
        rows = [json.loads(line) for line in f]
    if a.limit:
        rows = rows[:a.limit]
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    prompts = [build_prompt(tok, r["question"], a.system_prompt, a.nothink) for r in rows]
    print(f"[xb] {len(prompts)} problems, n={a.n} temp={a.temperature}", flush=True)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.util,
              max_model_len=a.max_new + a.max_model_len_extra, trust_remote_code=True)
    outs = llm.generate(prompts, SamplingParams(n=a.n, temperature=a.temperature, top_p=a.top_p,
                                                top_k=a.top_k, max_tokens=a.max_new))
    gens = [[c.text for c in o.outputs] for o in outs]
    glen = [[len(c.token_ids) for c in o.outputs] for o in outs]
    gtrunc = [[c.finish_reason == "length" for c in o.outputs] for o in outs]
    # vLLM keeps its KV pool alive until the engine's worker processes exit, and
    # this runs once per benchmark inside one job: without an explicit shutdown
    # the next engine found 0.2 of 79.33 GiB free and refused to start.
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # LCB's extractor and grader, so the code-extraction rules match the LCB runs.
    _lcb_on_path()
    from lcb_runner.evaluation import codegen_metrics
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code
    style = LMStyle[a.extraction]
    codes = [[extract_code(g, style) for g in gg] for gg in gens]
    samples = [{"input_output": r["input_output"]} for r in rows]
    metrics, results, _ = codegen_metrics(samples, codes, num_process_evaluate=a.num_process_evaluate,
                                          timeout=a.timeout)
    detail = None
    if a.detail:
        detail = {"gen_tokens": glen, "truncated": gtrunc,
                  "settings": {"model": a.model, "n": a.n, "temp": a.temperature,
                               "max_new": a.max_new, "nothink": a.nothink}}
    out = summarize(rows, codes, metrics, results, detail=detail)
    atomic_write_text(Path(a.out), json.dumps(out, indent=1))
    print("[xb] RESULT " + json.dumps(out), flush=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="APPS / CodeContests generation + LCB grading.")
    ap.add_argument("--bench-file", required=True, help="<bench>.jsonl (question, input_output, difficulty)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--top-p", type=float, required=True)
    ap.add_argument("--top-k", type=int, required=True)
    ap.add_argument("--max-new", type=int, required=True)
    ap.add_argument("--max-model-len-extra", type=int, required=True,
                    help="max_model_len = max_new + this")
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("--util", type=float, required=True)
    ap.add_argument("--system-prompt", required=True)
    ap.add_argument("--nothink", action=argparse.BooleanOptionalAction, required=True,
                    help="enable_thinking=False and close a dangling <think>")
    ap.add_argument("--extraction", required=True, help="LMStyle name for extract_code")
    ap.add_argument("--num-process-evaluate", type=int, required=True)
    ap.add_argument("--timeout", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first k problems (smoke); 0 = all")
    ap.add_argument("--detail", action="store_true", help="per-problem pass, lengths, truncation")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
