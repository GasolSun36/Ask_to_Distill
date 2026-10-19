"""``a2d eval math``: merge a checkpoint, generate every task, grade, print avg6.

    a2d eval math --pair Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500 --arm ours --step <N> [--tasks AIME24 MATH500] [--limit 5]
    a2d eval math --model gasolsun/Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500   # a ready model

With ``--model`` (a local HF directory or an HF repo id, ``ask2distill.eval.model_source``)
step 1 is replaced by resolving / downloading the model, and outputs go to
``eval.out_dir/<model name>/``.

1. **merge** ``$A2D_CKPT/<exp>/global_step_<N>/actor`` -> ``$A2D_CKPT/<exp>/hf/step_<N>``
   (``ask2distill.merge``; skipped when already merged, locked against
   concurrent evals of the same checkpoint).
2. **generate** each task in its own process on its own GPU
   (``python -m ask2distill.eval.math.generate``), ``eval.tasks`` by default.
   A 1-GPU job fits in any scheduling gap, and the tasks of a checkpoint run
   side by side when more GPUs are visible, so the checkpoint finishes in
   max(task) rather than sum(task). Tasks whose output file exists are skipped,
   so a killed job resumes where it stopped.
3. **grade** the directory once, after every task is present (two graders on
   one directory would race on ``grading_results.json``), under a 30-minute
   timeout because sympy can hang on adversarial answers.

Outputs go to ``eval.out_dir/<exp>_step<N>/``. Settings come from the ``eval``
section (``configs/base/eval_math.yaml``): avg@16 (MATH500 avg@4), T 0.7,
top_p 0.95, 31744 new tokens, thinking disabled, eager mode, stop ids per the
documented quirk (``generate.stop_token_ids``).

``avg_output_length`` in the grading results is counted with
``eval.len_tokenizer`` (``$A2D_MODELS/DeepSeek-R1-Distill-Qwen-1.5B``, what the
final runs used for every pair); it is a statistic only and never enters a score.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ask2distill import runtime
from ask2distill.config import ConfigError, require
from ask2distill.eval.math.generate import output_name
from ask2distill.eval.math.grade import RESULT_NAME

GRADE_TIMEOUT_S = 1800
STOP_IDS_RULE = "first_token_of_im_end_and_endoftext"
DEFAULT_LEN_TOKENIZER = "DeepSeek-R1-Distill-Qwen-1.5B"


# ------------------------------------------------------------------------- settings

def experiment_of(cfg: DictConfig, arm: str) -> str:
    """``opd.arms.<arm>.experiment``: names the checkpoint dir and the eval dir."""
    from ask2distill.opd.launch import arm_config
    return str(arm_config(cfg, arm).experiment)


def eval_dir(cfg: DictConfig, experiment: str, step: int) -> Path:
    """``eval.out_dir/<experiment>_step<step>``."""
    return Path(str(cfg.eval.out_dir)) / f"{experiment}_step{int(step)}"


def task_n(cfg: DictConfig, task: str) -> int:
    """Rollouts per problem: ``eval.n[task]``, else ``eval.n.default``."""
    n = cfg.eval.n
    return int(n[task]) if task in n else int(n.default)


def task_data(cfg: DictConfig, task: str) -> Path:
    """``eval.data_dir/<eval.task_dirs[task]>/test.parquet`` (MATH500 lives in ``MATH-500``)."""
    if task not in cfg.eval.task_dirs:
        raise ConfigError(f"unknown math task {task!r}; known: {', '.join(cfg.eval.task_dirs)}")
    return Path(str(cfg.eval.data_dir)) / str(cfg.eval.task_dirs[task]) / "test.parquet"


def task_output(cfg: DictConfig, out_dir: Path, task: str) -> Path:
    e = cfg.eval
    return out_dir / output_name(task, e.temperature, e.top_p, task_n(cfg, task), e.max_tokens)


def len_tokenizer(cfg: DictConfig) -> str:
    value = OmegaConf.select(cfg, "eval.len_tokenizer", default=None)
    return str(value) if value else str(Path(str(cfg.paths.models)) / DEFAULT_LEN_TOKENIZER)


def generate_cmd(cfg: DictConfig, model: Path, out_dir: Path, task: str,
                 limit: int | None) -> list[str]:
    """The ``generate`` command line for one task."""
    e = cfg.eval
    cmd = [sys.executable, "-m", "ask2distill.eval.math.generate",
           "--model", str(model), "--task", task, "--data", str(task_data(cfg, task)),
           "--n", str(task_n(cfg, task)), "--out-dir", str(out_dir),
           "--max-tokens", str(int(e.max_tokens)), "--temperature", str(float(e.temperature)),
           "--top-p", str(float(e.top_p)),
           "--gpu-memory-utilization", str(float(e.gpu_memory_utilization)),
           "--enforce-eager" if e.enforce_eager else "--no-enforce-eager",
           "--disable-thinking" if e.disable_thinking else "--enable-thinking"]
    if e.engine_seed is not None:
        cmd += ["--engine-seed", str(int(e.engine_seed))]
    if limit:
        cmd += ["--limit", str(int(limit))]
    return cmd


def grade_cmd(cfg: DictConfig, out_dir: Path) -> list[str]:
    cmd = [sys.executable, "-m", "ask2distill.eval.math.grade", str(out_dir),
           "--len-tokenizer", len_tokenizer(cfg)]
    if cfg.eval.strict_boxed:
        cmd.append("--strict-boxed")
    return cmd


def visible_gpus() -> list[str]:
    """GPU ids to place one task on each (``CUDA_VISIBLE_DEVICES``, else ``nvidia-smi -L``)."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout
        n = sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.CalledProcessError):
        n = 0
    return [str(i) for i in range(max(n, 1))]


# ---------------------------------------------------------------------------- stages

def _eval_env() -> dict[str, str | None]:
    env = runtime.base_env(stage="eval_math")
    env["TOKENIZERS_PARALLELISM"] = "false"          # as in the final evaluation jobs
    return env


# VLLM_PORT .. VLLM_PORT+9 lie inside the job's port block (runtime._PORT_OFFSETS:
# VLLM_PORT 10, MASTER_PORT 20).
_VLLM_PORT_SLOTS = 10


def slot_env(gpu: str, slot: int, inherited_vllm_port: str | None) -> dict[str, str | None]:
    """Environment of the generation process on ``gpu`` (slot = its index in the GPU list).

    Side-by-side vLLM engines that inherit the same ``VLLM_PORT`` both probe it
    as free and then race to bind it for torch.distributed (EADDRINUSE killed
    MATH500 in a smoke run). Each slot therefore starts its probe at its
    own port, ``VLLM_PORT + slot``; with no inherited ``VLLM_PORT`` vLLM picks
    an OS-assigned port and nothing is set.
    """
    env = _eval_env()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    if inherited_vllm_port:
        env["VLLM_PORT"] = str(int(inherited_vllm_port) + slot % _VLLM_PORT_SLOTS)
    return env


def run_generation(cmds: dict[str, list[str]], log_dir: Path, gpus: list[str]) -> dict[str, int]:
    """Run each task's command on its own GPU, ``len(gpus)`` at a time; returns exit codes."""
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(cmds.items())
    free = list(gpus)
    running: dict[str, tuple[subprocess.Popen, str, object]] = {}
    codes: dict[str, int] = {}
    while pending or running:
        while pending and free:
            task, cmd = pending.pop(0)
            gpu = free.pop(0)
            env = slot_env(gpu, gpus.index(gpu), os.environ.get("VLLM_PORT"))
            fh = open(log_dir / f"gen_{task}.log", "ab")
            print(f"[eval-math] gen {task} on GPU {gpu} -> {fh.name}", flush=True)
            proc = subprocess.Popen(cmd, env=runtime.merged_env(env), stdout=fh,
                                    stderr=subprocess.STDOUT)
            running[task] = (proc, gpu, fh)
        for task in list(running):
            proc, gpu, fh = running[task]
            if proc.poll() is None:
                continue
            fh.close()
            codes[task] = proc.returncode
            free.append(gpu)
            del running[task]
            print(f"[eval-math] {'done' if proc.returncode == 0 else 'GEN_FAILED'} {task} "
                  f"(exit {proc.returncode})", flush=True)
        if running:
            time.sleep(5)
    return codes


def main(cfg: DictConfig, args: argparse.Namespace) -> int:
    """``a2d eval math --pair P (--arm A --step N | --model M) [--tasks ...] [--limit k]``."""
    from ask2distill import merge
    from ask2distill.eval import collect, model_source

    if cfg.domain != "math":
        raise SystemExit(f"a2d eval math: pair {cfg.pair} is a {cfg.domain} pair")
    require(cfg, "eval")
    if str(cfg.eval.stop_ids_rule) != STOP_IDS_RULE:
        raise ConfigError(f"eval.stop_ids_rule must be {STOP_IDS_RULE!r} (the only rule implemented)")
    dry = bool(getattr(args, "dry_run", False))
    tasks = list(getattr(args, "tasks", None) or cfg.eval.tasks)
    for t in tasks:
        task_data(cfg, t)                           # validates the name
    limit = getattr(args, "limit", None) or cfg.eval.limit

    spec = model_source.check_args(args)
    if spec:
        model = model_source.resolve(cfg, spec, dry_run=dry)
        out_dir = Path(str(cfg.eval.out_dir)) / model_source.model_name(spec)
        print(f"[eval-math] {spec}: tasks={tasks} -> {out_dir}", flush=True)
    else:
        exp = experiment_of(cfg, args.arm)
        model = merge.hf_dir(cfg, exp, args.step)
        out_dir = eval_dir(cfg, exp, args.step)
        print(f"[eval-math] {exp} step {args.step}: tasks={tasks} -> {out_dir}", flush=True)
        try:
            merge.merge_fsdp(merge.actor_dir(cfg, exp, args.step), model, dry_run=dry)
        except (FileNotFoundError, merge.MergeError) as e:
            print(str(e), file=sys.stderr, flush=True)
            if not dry:
                return 1

    cmds = {t: generate_cmd(cfg, model, out_dir, t, limit) for t in tasks
            if dry or not task_output(cfg, out_dir, t).exists()}
    for t in tasks:
        if t not in cmds:
            print(f"[eval-math] {t}: {task_output(cfg, out_dir, t).name} exists, skipped", flush=True)
    if dry:
        for t, cmd in cmds.items():
            print(f"[dry-run] gen {t}: " + shlex.join(cmd), flush=True)
        print("[dry-run] grade: " + shlex.join(grade_cmd(cfg, out_dir)), flush=True)
        return 0

    codes = run_generation(cmds, out_dir / "logs", visible_gpus()) if cmds else {}
    missing = [t for t in tasks if not task_output(cfg, out_dir, t).exists()]
    if missing:
        print(f"[eval-math] not grading: missing {missing} (exit codes {codes})", file=sys.stderr,
              flush=True)
        return 1

    env = _eval_env()
    try:
        rc = subprocess.run(grade_cmd(cfg, out_dir), env=runtime.merged_env(env),
                            timeout=GRADE_TIMEOUT_S).returncode
    except subprocess.TimeoutExpired:
        print(f"[eval-math] GRADE_TIMEOUT after {GRADE_TIMEOUT_S}s", file=sys.stderr, flush=True)
        return 1
    if rc != 0:
        print(f"[eval-math] GRADE_FAILED (exit {rc})", file=sys.stderr, flush=True)
        return 1

    keys = collect.math_keys(cfg)
    summary = collect.math_avg6(out_dir / RESULT_NAME, keys)
    print(collect.format_math_row(out_dir.name, summary, keys), flush=True)
    return 0
