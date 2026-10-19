"""``a2d eval code``: merge a checkpoint and run the code benchmarks of the main table.

    a2d eval code --pair Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507 --arm ours --step <N> [--curve] [--smoke] [--limit k]
    a2d eval code --model gasolsun/Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507   # a ready model

Results land in ``eval.out_dir/<TAG>/`` with ``TAG = <experiment>_step<N>``:

=================================  ===========================================
``evalplus/<dataset>/*.jsonl``     EvalPlus samples and results
``evalplus_<dataset>.log``         EvalPlus console output (pass@1 lines)
``livecodebench_v5_v6_eval.json``  LiveCodeBench metrics (copied from LCB's output)
``evalplus/mbpp/lenient.jsonl``    MBPP samples after the entry-point repair
``evalplus_mbpp_lenient.log``      their MBPP / MBPP+ pass@1
``xb/<bench>.json``                APPS intro / interview1k / competition, CodeContests
=================================  ===========================================

Order, as the final runs had it: EvalPlus (cheap) then LiveCodeBench, then the
lenient MBPP re-score, then the extra benches. A failure in either of the first
two stops the run before the lenient and extra-bench steps (those ran as
dependent jobs). ``--curve`` evaluates a training-curve point: MBPP only (HE+ is
saturated for these pairs) and no extra benches. ``--smoke`` runs LCB with
``--debug`` (15 problems) and HumanEval+ only (EvalPlus asserts that a sample
file covers every problem, so it cannot be subset).

Settings (``configs/base/eval_code.yaml``):

* the prompt style is detected from the checkpoint's own chat template
  (``prompt_style.detect``, thinking off) and must equal ``eval.prompt_style``;
* LiveCodeBench v5_v6 (342 problems), n 4, non-thinking sampling T 0.7 /
  top_p 0.8 / top_k 20 / min_p 0, ``max_model_len`` 16384 capped at the
  model's context, ``max_tokens`` 8192 capped at ``max_model_len - 4096``,
  TP 2, ``--continue_existing``. LCB writes ``output/<TAG>/`` under
  ``eval.lcb.cwd``; the dataset revision is pinned through
  ``LCB_DATASET_REVISION``;
* EvalPlus official protocol: greedy, assistant prefill, 1024 new tokens,
  ``max_model_len`` 4096;
* vLLM sizes its KV cache to hit ``gpu_memory_utilization`` counting only
  weights + KV + activations, then captures CUDA graphs on top of that budget.
  On an 80GB card a 32B model at 0.9 overshoots by ~8GB and dies during
  capture, while a 4B model has room to spare. Above 10 GB of weights per GPU
  the util drops to 0.85 and the model runs eager (upstream LCB's default);
  otherwise 0.9 with CUDA graphs. The extra benches only lower the util.

**The tag must be unique per checkpoint.** LCB's ``--continue_existing`` reuses
whatever generations sit under ``output/<TAG>/``; a tag derived from a
directory basename (``step_150-nothink``) is shared by every run's step 150 and
silently scores another run's generations. ``render_tag`` refuses anything but
``<experiment>_step<N>``.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig

from ask2distill import runtime
from ask2distill.config import ConfigError, require
from ask2distill.eval.code.prompt_style import StyleInfo

GIB = 1073741824
_TAG = re.compile(r"^(?P<experiment>[A-Za-z0-9][A-Za-z0-9._+-]*)_step(?P<step>\d+)$")
# Headroom the final runs kept between LCB's prompt and its generation budget.
LCB_PROMPT_HEADROOM = 4096
XB_GPU_FREE_MIB = 2000
XB_GPU_WAIT = (12, 5)        # polls x seconds


class TagError(ValueError):
    """An eval tag that could collide with another run's cached generations."""


def render_tag(template: str, experiment: str, step: int) -> str:
    """``eval.tag`` rendered for one checkpoint; always ``<experiment>_step<N>``."""
    tag = template.format(experiment=experiment, step=int(step))
    validate_tag(tag, experiment, step)
    return tag


def validate_tag(tag: str, experiment: str | None = None, step: int | None = None) -> str:
    """Raise ``TagError`` unless ``tag`` is ``<experiment>_step<N>`` (and matches the given ones)."""
    m = _TAG.match(tag)
    if not m:
        raise TagError(f"eval tag {tag!r} is not <experiment>_step<N>; basename-style tags such "
                       "as 'step_150-nothink' collide across runs under LCB --continue_existing")
    if experiment is not None and m["experiment"] != experiment:
        raise TagError(f"eval tag {tag!r} does not name experiment {experiment!r}")
    if step is not None and int(m["step"]) != int(step):
        raise TagError(f"eval tag {tag!r} does not name step {step}")
    return tag


# ------------------------------------------------------------------------- planning

@dataclass(frozen=True)
class Plan:
    """Every resolved setting of one code eval (printed by ``--dry-run``)."""

    tag: str
    model: Path
    out: Path
    style: str
    ctx: int
    weight_bytes: int
    lcb_util: float
    lcb_eager: bool
    lcb_max_model_len: int
    lcb_max_tokens: int
    ep_util: float
    ep_eager: bool
    ep_datasets: tuple[str, ...]
    xb_util: float
    run_xb: bool
    debug: bool


def big_model(cfg: DictConfig, weight_bytes: int, tp: int) -> tuple[float, bool]:
    """``(gpu_memory_utilization, enforce_eager)`` from the per-GPU weight footprint."""
    rule = cfg.eval.big_model_rule
    gb_per_gpu = weight_bytes // GIB // int(tp)          # bash integer arithmetic, as run
    if gb_per_gpu > int(rule.weights_gb_per_gpu_gt):
        return float(rule.util), bool(rule.enforce_eager)
    return float(rule.default_util), False


def lcb_lengths(cfg: DictConfig, ctx: int) -> tuple[int, int]:
    """``(max_model_len, max_tokens)``: never more context than the checkpoint declares
    (Qwen2.5-Coder stops at 32k; Qwen3-30B-A3B-Instruct-2507 claims 262k, which would
    eat the whole KV cache), and generation leaves room for the prompt."""
    want_len, want_gen = int(cfg.eval.lcb.max_model_len), int(cfg.eval.lcb.max_tokens)
    max_model_len = min(want_len, int(ctx))
    if want_gen > max_model_len - LCB_PROMPT_HEADROOM:
        want_gen = max_model_len - LCB_PROMPT_HEADROOM
    return max_model_len, want_gen


def make_plan(cfg: DictConfig, model: Path, tag: str, info: StyleInfo, *, curve: bool = False,
              smoke: bool = False) -> Plan:
    e = cfg.eval
    lcb_util, lcb_eager = big_model(cfg, info.weight_bytes, e.lcb.tp)
    ep_util, ep_eager = big_model(cfg, info.weight_bytes, e.evalplus.tp)
    rule = e.big_model_rule
    xb_util = (float(rule.util) if info.weight_bytes // GIB // int(e.extra_bench.tp)
               > int(rule.weights_gb_per_gpu_gt) else float(e.extra_bench.util))
    max_model_len, max_tokens = lcb_lengths(cfg, info.ctx)
    if smoke:
        datasets = ("humaneval",)
    elif curve:
        datasets = tuple(str(d) for d in e.curve_evalplus_datasets)
    else:
        datasets = tuple(str(d) for d in e.evalplus.datasets)
    return Plan(tag=tag, model=Path(model), out=Path(str(e.out_dir)) / tag, style=info.style,
                ctx=info.ctx, weight_bytes=info.weight_bytes, lcb_util=lcb_util,
                lcb_eager=lcb_eager, lcb_max_model_len=max_model_len, lcb_max_tokens=max_tokens,
                ep_util=ep_util, ep_eager=ep_eager, ep_datasets=datasets, xb_util=xb_util,
                run_xb=not curve, debug=smoke)


# ------------------------------------------------------------------------- commands

def lcb_cmd(cfg: DictConfig, plan: Plan) -> list[str]:
    lc = cfg.eval.lcb
    cmd = [sys.executable, "-m", "lcb_runner.runner.main",
           "--model", plan.tag, "--model_style", plan.style,
           "--local_model_path", str(plan.model), "--scenario", "codegeneration",
           "--release_version", str(lc.version), "--n", str(int(lc.n)),
           "--temperature", str(float(lc.temperature)), "--top_p", str(float(lc.top_p)),
           "--top_k", str(int(lc.top_k)), "--min_p", str(lc.min_p),
           "--max_tokens", str(plan.lcb_max_tokens), "--max_model_len", str(plan.lcb_max_model_len),
           "--tensor_parallel_size", str(int(lc.tp)),
           "--gpu_memory_utilization", str(plan.lcb_util)]
    if not plan.lcb_eager:
        cmd.append("--cuda_graph")
    cmd += ["--stop", str(lc.stop), "--num_process_evaluate", str(int(lc.num_process_evaluate)),
            "--timeout", str(int(lc.timeout))]
    if lc.continue_existing:
        cmd.append("--continue_existing")
    cmd.append("--evaluate")
    if plan.debug:
        cmd.append("--debug")
    return cmd


def lcb_result(cfg: DictConfig, tag: str) -> Path:
    """Where LCB writes the metrics file: ``<cwd>/output/<TAG>/Scenario.codegeneration_<n>_<T>_eval.json``."""
    lc = cfg.eval.lcb
    return (Path(str(lc.cwd)) / "output" / tag
            / f"Scenario.codegeneration_{int(lc.n)}_{float(lc.temperature)}_eval.json")


def evalplus_bin() -> str:
    """The ``evalplus.evaluate`` console script of this interpreter's environment."""
    local = Path(sys.executable).parent / "evalplus.evaluate"
    return str(local) if local.exists() else (shutil.which("evalplus.evaluate") or "evalplus.evaluate")


def evalplus_cmd(cfg: DictConfig, plan: Plan, dataset: str) -> list[str]:
    ep = cfg.eval.evalplus
    if not ep.greedy:
        raise ConfigError("eval.evalplus.greedy must be true (the official greedy protocol; "
                          "the reasoning mode was never used)")
    return [evalplus_bin(), "--model", str(plan.model), "--dataset", dataset, "--backend", "vllm",
            "--tp", str(int(ep.tp)), "--root", str(plan.out / "evalplus"),
            "--parallel", str(int(ep.parallel)), "--gpu-memory-utilization", str(plan.ep_util),
            "--enforce-eager", "True" if plan.ep_eager else "False",
            "--greedy", "--max-new-tokens", str(int(ep.max_new_tokens)),
            "--max-model-len", str(int(ep.max_model_len))]


def xb_cmd(cfg: DictConfig, plan: Plan, bench: str, limit: int | None) -> list[str]:
    xb = cfg.eval.extra_bench
    cmd = [sys.executable, "-m", "ask2distill.eval.code.extra_bench",
           "--bench-file", str(Path(str(xb.data_dir)) / f"{bench}.jsonl"),
           "--model", str(plan.model), "--out", str(plan.out / "xb" / f"{bench}.json"),
           "--n", str(int(xb.n)), "--temperature", str(float(xb.temperature)),
           "--top-p", str(float(xb.top_p)), "--top-k", str(int(xb.top_k)),
           "--max-new", str(int(xb.max_new)), "--max-model-len-extra", str(int(xb.max_model_len_extra)),
           "--tp", str(int(xb.tp)), "--util", str(plan.xb_util),
           "--system-prompt", str(xb.system_prompt),
           "--nothink" if xb.nothink_close else "--no-nothink",
           "--extraction", str(xb.extraction),
           "--num-process-evaluate", str(int(xb.num_process_evaluate)),
           "--timeout", str(int(xb.timeout))]
    if limit:
        cmd += ["--limit", str(int(limit))]
    if xb.detail:
        cmd.append("--detail")
    return cmd


def lenient_source(mbpp_dir: Path) -> Path | None:
    """The MBPP samples file EvalPlus wrote (not ``*.raw.jsonl``, not our ``lenient.jsonl``)."""
    for p in sorted(Path(mbpp_dir).glob("*.jsonl")):
        if not p.name.endswith(".raw.jsonl") and p.name != "lenient.jsonl":
            return p
    return None


def code_env(cfg: DictConfig) -> dict[str, str | None]:
    """Stage env plus LiveCodeBench on ``PYTHONPATH`` and its pinned dataset revision."""
    env = runtime.base_env(stage="eval_code")
    lcb = str(Path(str(cfg.paths.third_party)) / "LiveCodeBench")
    old = os.environ.get("PYTHONPATH", "")
    env["PYTHONPATH"] = lcb + (os.pathsep + old if old else "")
    env["LCB_DATASET_REVISION"] = str(cfg.eval.lcb.dataset_revision)
    return env


# ----------------------------------------------------------------------------- steps

def run_evalplus(cfg: DictConfig, plan: Plan, env: dict, dry: bool) -> int:
    cwd = Path(str(cfg.paths.third_party)) / "evalplus"
    for ds in plan.ep_datasets:
        log = plan.out / f"evalplus_{ds}.log"
        print(f"===== EvalPlus {ds} =====", flush=True)
        if not dry and log.exists():
            log.unlink()                      # the original tee'd (overwrote) this log
        if runtime.run(evalplus_cmd(cfg, plan, ds), env, log=None if dry else log,
                       dry_run=dry, cwd=cwd) != 0:
            print(f"EVALPLUS_FAILED {ds}", flush=True)
            return 1
    return 0


def run_lcb(cfg: DictConfig, plan: Plan, env: dict, dry: bool) -> int:
    cwd = Path(str(cfg.eval.lcb.cwd))
    if not dry:
        cwd.mkdir(parents=True, exist_ok=True)
    print(f"===== LiveCodeBench {cfg.eval.lcb.version} n={cfg.eval.lcb.n} "
          f"max_tokens={plan.lcb_max_tokens} =====", flush=True)
    if runtime.run(lcb_cmd(cfg, plan), env, log=None if dry else plan.out / "logs" / "lcb.log",
                   dry_run=dry, cwd=cwd) != 0:
        print("LCB_FAILED", flush=True)
        return 1
    if dry:
        return 0
    res = lcb_result(cfg, plan.tag)
    if not res.is_file():
        print(f"NO LCB RESULTS at {res}", flush=True)
        return 0                              # as run: a missing file is reported, not fatal
    dst = plan.out / f"livecodebench_{cfg.eval.lcb.version}_eval.json"
    shutil.copyfile(res, dst)
    print(f"[eval-code] {res} -> {dst}", flush=True)
    return 0


def run_lenient(cfg: DictConfig, plan: Plan, env: dict, dry: bool) -> int:
    """Rename entry points in the MBPP samples, re-evaluate, keep the pass@1 lines."""
    mbpp = plan.out / "evalplus" / "mbpp"
    src = lenient_source(mbpp) if not dry else mbpp / "<samples>.jsonl"
    if src is None:
        print(f"[eval-code] no MBPP samples under {mbpp}; lenient re-score skipped", flush=True)
        return 0
    lenient = mbpp / "lenient.jsonl"
    rescore = [sys.executable, "-m", "ask2distill.eval.code.mbpp_lenient",
               "--samples", str(src), "--out", str(lenient)]
    evaluate = [evalplus_bin(), "--dataset", "mbpp", "--samples", str(lenient)]
    if dry:
        runtime.run(rescore, env, dry_run=True)
        runtime.run(evaluate, env, dry_run=True)
        return 0
    if runtime.run(rescore, env) != 0:
        return 1
    proc = subprocess.run(evaluate, env=runtime.merged_env(env), capture_output=True)
    text = (proc.stdout + proc.stderr).decode(errors="replace")
    kept = [line for line in text.splitlines() if re.match(r"^mbpp|^pass@1", line)]
    log = plan.out / "evalplus_mbpp_lenient.log"
    log.write_text("".join(line + "\n" for line in kept))
    print("".join(line + "\n" for line in kept), end="", flush=True)
    return 0 if proc.returncode == 0 else 1


def _gpu_mem_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return max(int(x) for x in out.split())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def run_extra_benches(cfg: DictConfig, plan: Plan, env: dict, limit: int | None, dry: bool) -> int:
    """One process group per benchmark; the group is killed once the parent returns.

    Each benchmark is its own python process, but its descendants -- vLLM's
    EngineCore workers and the grader's forked sandboxes, which inherit the CUDA
    context -- can outlive it and keep the GPUs. Waiting was not enough: after
    Thinking-2507's first benchmark 79GB stayed allocated for 120s and the next
    three engines all died at init.
    """
    rc = 0
    cwd = Path(str(cfg.eval.lcb.cwd))
    for bench in cfg.eval.extra_bench.benches:
        out = plan.out / "xb" / f"{bench}.json"
        if not dry and out.is_file() and out.stat().st_size > 0:
            print(f"skip {bench} (done)", flush=True)
            continue
        cmd = xb_cmd(cfg, plan, str(bench), limit)
        print(f"########## {plan.tag} / {bench} ##########", flush=True)
        if dry:
            runtime.run(cmd, env, dry_run=True, cwd=cwd)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        cwd.mkdir(parents=True, exist_ok=True)
        log = plan.out / "logs" / f"xb_{bench}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as fh:
            proc = subprocess.Popen(cmd, env=runtime.merged_env(env), cwd=cwd, stdout=fh,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            code = proc.wait()
        if code != 0:
            print(f"XB_FAILED {bench} (exit {code}, log {log})", flush=True)
            rc = 1
        for sig, pause in ((signal.SIGTERM, 3), (signal.SIGKILL, 0)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            time.sleep(pause)
        used, waited = None, 0
        for i in range(1, XB_GPU_WAIT[0] + 1):
            used, waited = _gpu_mem_used_mib(), i * XB_GPU_WAIT[1]
            if used is None or used < XB_GPU_FREE_MIB:
                break
            time.sleep(XB_GPU_WAIT[1])
        print(f"[xb] gpu mem after {bench}: {used if used is not None else '?'} MiB "
              f"(waited {waited}s)", flush=True)
    return rc


# ------------------------------------------------------------------------------ main

def experiment_of(cfg: DictConfig, arm: str) -> str:
    from ask2distill.opd.launch import arm_config
    return str(arm_config(cfg, arm).experiment)


def main(cfg: DictConfig, args: argparse.Namespace) -> int:
    """``a2d eval code --pair P (--arm A --step N | --model M) [--curve] [--smoke] [--limit k]``."""
    from ask2distill import merge
    from ask2distill.eval import collect, model_source
    from ask2distill.eval.code.prompt_style import detect

    if cfg.domain != "code":
        raise SystemExit(f"a2d eval code: pair {cfg.pair} is a {cfg.domain} pair")
    require(cfg, "eval")
    if cfg.eval.think:
        raise ConfigError("eval.think=true is not supported: only the non-thinking protocol "
                          "(the one every main-table code number used) is configured")
    dry = bool(getattr(args, "dry_run", False))
    curve, smoke = bool(getattr(args, "curve", False)), bool(getattr(args, "smoke", False))
    limit = getattr(args, "limit", None) or cfg.eval.extra_bench.limit

    spec = model_source.check_args(args)
    if spec:
        # The model's own name is the tag: unique per repo id / absolute path (model_source).
        tag = model_source.model_name(spec)
        model = model_source.resolve(cfg, spec, dry_run=dry)
    else:
        exp = experiment_of(cfg, args.arm)
        tag = render_tag(str(cfg.eval.tag), exp, args.step)
        model = merge.hf_dir(cfg, exp, args.step)
        try:
            merge.merge_fsdp(merge.actor_dir(cfg, exp, args.step), model, dry_run=dry)
        except (FileNotFoundError, merge.MergeError) as e:
            print(str(e), file=sys.stderr, flush=True)
            if not dry:
                return 1

    if (model / "config.json").is_file():
        info = detect(str(model), think=False)
    elif dry:
        info = StyleInfo(str(cfg.eval.prompt_style), int(cfg.eval.lcb.max_model_len), True, 0)
        print("[dry-run] model not merged yet: style/ctx/weights taken from the config", flush=True)
    else:
        print(f"NOT_AN_HF_DIR: {model}", file=sys.stderr, flush=True)
        return 1
    if info.style != str(cfg.eval.prompt_style):
        raise ConfigError(f"{model}: detected LCB prompt style {info.style}, but "
                          f"eval.prompt_style is {cfg.eval.prompt_style} -- wrong model for this pair?")
    plan = make_plan(cfg, model, tag, info, curve=curve, smoke=smoke)
    print(f"[eval-code] tag={plan.tag} style={plan.style} ctx={plan.ctx} "
          f"weights={plan.weight_bytes // GIB}GB lcb_util={plan.lcb_util} eager={plan.lcb_eager} "
          f"lcb_len={plan.lcb_max_model_len}/{plan.lcb_max_tokens} evalplus={list(plan.ep_datasets)} "
          f"-> {plan.out}", flush=True)
    if not dry:
        plan.out.mkdir(parents=True, exist_ok=True)

    env = code_env(cfg)
    rc = 0
    rc |= run_evalplus(cfg, plan, env, dry)      # EvalPlus first: it is the cheap one
    rc |= run_lcb(cfg, plan, env, dry)
    if rc:
        print(f"EVAL_FAILED {tag}", file=sys.stderr, flush=True)
        return 1
    if cfg.eval.mbpp_lenient and "mbpp" in plan.ep_datasets:
        rc |= run_lenient(cfg, plan, env, dry)
    if plan.run_xb:
        rc |= run_extra_benches(cfg, plan, env, limit, dry)
    if dry:
        return 0
    summary = collect.code_avg(plan.out, items=tuple(str(i) for i in cfg.eval.avg_items),
                               lcb_version=str(cfg.eval.lcb.version))
    print(collect.format_code_row(tag, summary, tuple(str(i) for i in cfg.eval.avg_items)), flush=True)
    print(f"EVALSTEP_DONE {tag} (rc={rc})", flush=True)
    return rc
