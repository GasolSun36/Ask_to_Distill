"""Self-check for the a2d environments.

    python env/check_env.py --stage imports            # CPU: versions vs lock, vendored packages, console scripts
    python env/check_env.py --stage core               # 1 GPU: torch/CUDA, flash-attn fwd+bwd, nvcc, flashinfer JIT
    python env/check_env.py --stage verl               # verl imports + OPD-specific pieces
    python env/check_env.py --stage vllm --model DIR   # real model load + generate with top-k logprobs
    python env/check_env.py --tau2                     # every tau2 envside module imports under $A2D_TAU2_PYTHON

Run the GPU stages on a compute node after ``export CUDA_HOME=$CONDA_PREFIX``.
Exit code 0 means every check in the selected stage(s) passed.

The model path for ``--stage vllm`` is an argument.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(os.path.abspath(__file__)).parents[1]
THIRD_PARTY = REPO_ROOT / "third_party"
A2D_LOCK = REPO_ROOT / "env" / "requirements-a2d.lock.txt"

# Installed outside the lock by env/build_a2d_env.sh.
EXPLICIT_VERSIONS = {"flash-attn": "2.8.1", "verl": "0.7.0.dev0", "evalplus": "0.4.0.dev44"}
CONSOLE_SCRIPTS = ["evalplus.evaluate", "evalplus.codegen", "a2d"]
# The tau2 envside processes must stay importable without the training stack.
TAU2_FORBIDDEN = ("omegaconf", "torch", "pyarrow")

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str) -> Callable[[Callable[[], str | None]], Callable[[], str | None]]:
    """Run the decorated function once, record PASS/FAIL, never raise."""

    def wrap(fn: Callable[[], str | None]) -> Callable[[], str | None]:
        try:
            detail = fn()
            RESULTS.append((True, name, detail or ""))
            print(f"PASS  {name:38s} {detail or ''}", flush=True)
        except Exception as e:
            RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
            print(f"FAIL  {name:38s} {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
        return fn

    return wrap


def normalize(name: str) -> str:
    """PEP 503 project-name normalisation (``Foo_Bar`` -> ``foo-bar``)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(path: Path) -> dict[str, str]:
    """``{normalized name: version}`` for every ``name==version`` line of a lock file."""
    pins: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise ValueError(f"{path.name}: not an exact pin: {raw!r}")
        name, version = line.split("==", 1)
        name = name.split("[", 1)[0].strip()
        pins[normalize(name)] = version.strip()
    return pins


def _under(path: str, root: Path) -> bool:
    return os.path.abspath(path).startswith(os.path.abspath(root) + os.sep)


# --------------------------------------------------------------------------- stages


def stage_imports() -> None:
    from importlib import metadata

    @check("a2d lock versions installed")
    def _():
        pins = parse_lock(A2D_LOCK)
        installed = {normalize(d.metadata["Name"]): d.version for d in metadata.distributions()}
        wrong = [f"{n} {installed.get(n, 'missing')} != {v}" for n, v in pins.items() if installed.get(n) != v]
        assert not wrong, f"{len(wrong)} mismatches: {wrong[:20]}"
        return f"{len(pins)} pins match"

    @check("explicitly installed packages")
    def _():
        bad = []
        for name, want in EXPLICIT_VERSIONS.items():
            try:
                got = metadata.version(name)
            except metadata.PackageNotFoundError:
                got = "missing"
            if got != want:
                bad.append(f"{name} {got} != {want}")
        assert not bad, bad
        return ", ".join(f"{k} {v}" for k, v in EXPLICIT_VERSIONS.items())

    @check("verl resolves to third_party/verl")
    def _():
        import verl

        assert _under(verl.__file__, THIRD_PARTY / "verl"), f"verl imported from {verl.__file__}"
        return os.path.dirname(verl.__file__)

    @check("evalplus resolves to third_party/evalplus")
    def _():
        import evalplus

        assert _under(evalplus.__file__, THIRD_PARTY / "evalplus"), f"evalplus imported from {evalplus.__file__}"
        return evalplus.__version__

    @check("console scripts on PATH")
    def _():
        missing = [s for s in CONSOLE_SCRIPTS if shutil.which(s) is None]
        assert not missing, f"missing: {missing}"
        return ", ".join(CONSOLE_SCRIPTS)

    @check("LiveCodeBench imports from any cwd")
    def _():
        # The a2d LCB patch makes the few-shot files cwd-independent and pins the dataset revision.
        code = (
            "from lcb_runner.prompts import code_generation as cg; "
            "from lcb_runner.benchmarks import code_generation as b; "
            "print(len(cg.func), len(cg.stdin), b._LCB_DATASET_REVISION)"
        )
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            [str(THIRD_PARTY / "LiveCodeBench"), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run([sys.executable, "-c", code], cwd=tmp, env=env,
                                 capture_output=True, text=True, check=True)
        return out.stdout.strip()

    @check("ask2distill importable")
    def _():
        import ask2distill.paths as p

        assert _under(p.__file__, REPO_ROOT), f"ask2distill imported from {p.__file__}"
        return str(p.REPO_ROOT)


def stage_core() -> None:
    import torch

    @check("node glibc")
    def _():
        # flash-attn wheels are glibc-sensitive; record what this node actually has.
        return f"glibc {platform.libc_ver()[1]} on {platform.node()}"

    @check("torch + CUDA")
    def _():
        assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        _, total = torch.cuda.mem_get_info(0)
        return (f"torch {torch.__version__} cuda {torch.version.cuda} | "
                f"{torch.cuda.device_count()}x {name} sm{cap[0]}{cap[1]} | {total / 2**30:.0f}GiB")

    @check("bf16 matmul on GPU")
    def _():
        a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        err = ((a @ b).float() - a.float() @ b.float()).abs().max().item()
        assert err < 2.0, f"bf16 matmul error too large: {err}"
        return f"max abs err {err:.3f}"

    @check("autograd on GPU")
    def _():
        x = torch.randn(256, 256, device="cuda", requires_grad=True)
        (x * x).sum().backward()
        assert torch.allclose(x.grad, 2 * x, atol=1e-5)
        return "grad ok"

    @check("NCCL")
    def _():
        assert torch.distributed.is_nccl_available()
        return ".".join(map(str, torch.cuda.nccl.version()))

    @check("torch SDPA attention")
    def _():
        import torch.nn.functional as F

        q, k, v = (torch.randn(2, 8, 256, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert out.shape == q.shape and torch.isfinite(out).all()
        return "sdpa fwd ok"

    @check("flash-attn kernel (fwd+bwd)")
    def _():
        import flash_attn
        from flash_attn import flash_attn_func

        b, s, h, d = 2, 256, 8, 64
        q, k, v = (torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16,
                               requires_grad=True) for _ in range(3))
        out = flash_attn_func(q, k, v, causal=True)
        out.sum().backward()
        # Compare against naive causal attention.
        qf, kf, vf = q.float(), k.float(), v.float()
        scores = (qf.transpose(1, 2) @ kf.transpose(1, 2).transpose(-1, -2)) / d**0.5
        mask = torch.triu(torch.ones(s, s, device="cuda", dtype=torch.bool), diagonal=1)
        ref = (scores.masked_fill(mask, float("-inf")).softmax(-1) @ vf.transpose(1, 2)).transpose(1, 2)
        err = (out.float() - ref).abs().max().item()
        assert err < 0.05, f"flash-attn deviates from reference: {err}"
        assert q.grad is not None and torch.isfinite(q.grad).all()
        return f"flash_attn {flash_attn.__version__} | max err {err:.4f}"

    @check("flash-attn varlen + bert_padding")
    def _():
        # The entry points verl uses for use_remove_padding=True.
        from flash_attn import flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa: F401

        cu = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
        qq, kk, vv = (torch.randn(256, 8, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        o = flash_attn_varlen_func(qq, kk, vv, cu, cu, 128, 128, causal=True)
        torch.cuda.synchronize()
        assert torch.isfinite(o).all()
        return f"shape {tuple(o.shape)}"

    @check("nvcc (flashinfer JIT needs it)")
    def _():
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        cuda_home = os.environ.get("CUDA_HOME")
        assert cuda_home, "CUDA_HOME is unset; export CUDA_HOME=$CONDA_PREFIX"
        return f"{out.strip().splitlines()[-2].strip()} | CUDA_HOME={cuda_home}"

    @check("flashinfer JIT sampling kernel")
    def _():
        import flashinfer
        from flashinfer.sampling import top_k_top_p_sampling_from_probs

        probs = torch.softmax(torch.randn(4, 1024, device="cuda"), dim=-1)
        ids = top_k_top_p_sampling_from_probs(probs, top_k=20, top_p=0.95)
        torch.cuda.synchronize()
        assert ids.shape[0] == 4 and int(ids.max()) < 1024
        return f"flashinfer {flashinfer.__version__}"


def stage_verl() -> None:
    @check("verl resolves to third_party/verl")
    def _():
        import verl

        path = os.path.dirname(verl.__file__)
        assert _under(path, THIRD_PARTY / "verl"), f"verl resolved to {path}, not the vendored tree"
        return f"{verl.__version__} @ {path}"

    @check("verl core modules import")
    def _():
        import importlib

        # verl/models, verl/utils/dataset and verl/utils/checkpoint were once dropped by an
        # unanchored .gitignore (models/, dataset/, checkpoint/) -- keep watching them.
        mods = [
            "verl.trainer.main_ppo",
            "verl.models.registry",
            "verl.models.transformers.monkey_patch",
            "verl.models.weight_loader_registry",
            "verl.utils.dataset.rl_dataset",
            "verl.utils.checkpoint.fsdp_checkpoint_manager",
            "verl.experimental.dataset.sampler",
            "verl.third_party.torch.distributed.checkpoint.state_dict",
            "verl.workers.fsdp_workers",
            "verl.workers.actor.dp_actor",
            "verl.trainer.ppo.ray_trainer",
            "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd",
            "verl.workers.reward_manager",
            "verl.model_merger",
        ]
        bad = []
        for m in mods:
            try:
                importlib.import_module(m)
            except Exception as e:
                bad.append(f"{m} ({type(e).__name__})")
        assert not bad, f"failed: {bad}"
        return f"{len(mods)} modules"

    @check("OPD advantage estimators registered")
    def _():
        from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY

        need = ["token_reward_direct", "token_reward_direct_plus_grpo", "token_grpo", "grpo"]
        missing = [n for n in need if n not in ADV_ESTIMATOR_REGISTRY]
        assert not missing, f"missing: {missing}"
        return ", ".join(need)

    @check("dp_actor carries the OPD top-k patch")
    def _():
        import inspect

        from verl.workers.actor import dp_actor

        src = inspect.getsource(dp_actor)
        for k in ["student_top_k_ids", "log_softmax"]:
            assert k in src, f"{k} missing from dp_actor"
        return "student_top_k_ids + chunked log_softmax present"

    @check("math-verify scoring")
    def _():
        from math_verify import parse, verify

        assert verify(parse("$\\frac{1}{2}$"), parse("$0.5$"))
        assert not verify(parse("$\\frac{1}{2}$"), parse("$3$"))
        return "equal/unequal both correct"


def stage_vllm(model: str | None) -> None:
    @check("vllm import")
    def _():
        import vllm

        return vllm.__version__

    @check("vllm load+generate")
    def _():
        if not model:
            raise ValueError("pass --model <local HF model dir> (a small instruct model is enough)")
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model,
            dtype="bfloat16",
            gpu_memory_utilization=0.45,
            max_model_len=2048,
            enforce_eager=True,
            disable_log_stats=True,
        )
        # logprobs=k exercises the top-k path OPD uses for token-level rewards.
        sp = SamplingParams(temperature=0.0, max_tokens=32, logprobs=16)
        out = llm.generate(["What is 2 + 2? Answer with just the number."], sp)
        text = out[0].outputs[0].text
        lp = out[0].outputs[0].logprobs
        assert text.strip(), "vllm returned empty text"
        assert lp and len(lp[0]) >= 16, f"top-k logprobs missing (got {len(lp[0]) if lp else 0})"
        print(f"      model output: {text.strip()[:100]!r}", flush=True)
        return f"{os.path.basename(model.rstrip('/'))} | {len(text)} chars | top-{len(lp[0])} logprobs/token"


def envside_modules(repo: Path = REPO_ROOT) -> list[str]:
    """Dotted names of every ``ask2distill.tau2.envside`` module, sorted."""
    pkg = repo / "ask2distill" / "tau2" / "envside"
    return sorted(f"ask2distill.tau2.envside.{p.stem}" for p in pkg.glob("*.py") if p.stem != "__init__")


_TAU2_PROBE = """
import importlib, json, sys
bad = {}
for m in sys.argv[1:]:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad[m] = f"{type(e).__name__}: {e}"
print(json.dumps({"bad": bad, "loaded": sorted(sys.modules)}))
"""


def stage_tau2() -> None:
    @check("tau2 envside modules import")
    def _():
        from ask2distill import config
        from ask2distill.paths import get_paths
        from ask2distill.tau2 import spawn

        paths = get_paths()
        if paths.tau2_python is None:
            raise RuntimeError("A2D_TAU2_PYTHON is unset (see env/build_tau2_env.sh)")
        mods = envside_modules(paths.repo)
        assert mods, "no envside modules found"
        env = spawn.env(config.load(pair=config.domain_pair("tau2")), "synrl")
        # LLM_CALL creates ./keys at import time, so probe from a throwaway directory.
        with tempfile.TemporaryDirectory() as tmp:
            out = subprocess.run([str(paths.tau2_python), "-c", _TAU2_PROBE, *mods], cwd=tmp, env=env,
                                 capture_output=True, text=True, check=True)
        report = json.loads(out.stdout.strip().splitlines()[-1])
        leaked = [m for m in TAU2_FORBIDDEN if m in report["loaded"]]
        assert not report["bad"], f"import failures: {report['bad']}"
        assert not leaked, f"envside pulled in {leaked}"
        return f"{len(mods)} modules under {paths.tau2_python}"


STAGES: dict[str, Callable[[], None]] = {"imports": stage_imports, "core": stage_core, "verl": stage_verl}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["imports", "core", "verl", "vllm"])
    ap.add_argument("--model", help="local HF model directory for --stage vllm")
    ap.add_argument("--tau2", action="store_true", help="import every envside module under $A2D_TAU2_PYTHON")
    args = ap.parse_args(argv)
    if not args.stage and not args.tau2:
        ap.error("give --stage and/or --tau2")

    print(f"\n{'=' * 70}\nSTAGE: {args.stage or ''}{' tau2' if args.tau2 else ''}   "
          f"python={sys.executable}\n{'=' * 70}", flush=True)
    if args.stage == "vllm":
        stage_vllm(args.model)
    elif args.stage:
        STAGES[args.stage]()
    if args.tau2:
        stage_tau2()

    failed = [n for ok, n, _ in RESULTS if not ok]
    print(f"\n--- {len(RESULTS) - len(failed)}/{len(RESULTS)} passed ---", flush=True)
    if failed:
        print("FAILED CHECKS: " + ", ".join(failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
