"""Process environment, per-job Ray, subprocess runner and run bookkeeping.

Environment. ``base_env`` is the process-wide hygiene the final runs' launchers
exported (caches off the small HOME quota, NCCL / Ray / hydra settings). It is
exported by whoever starts Ray, before ``ray start``, so every Ray worker
inherits it; run-specific values travel separately through
``+ray_kwargs.ray_init.runtime_env.env_vars`` (see the launchers).

Ray. The final runs' jobs ran ``ray stop --force`` and a head on the default port
6379. On shared single-node partitions that kills sibling jobs' clusters,
collides on ports (vLLM's VLLM_PORT hit EADDRINUSE there too) and trips over a
``/tmp/ray`` owned by another user, and a launcher that restarts Ray inside an
sbatch-built multi-node cluster tears that cluster down.
So every job gets a private Ray: a short temp dir and a port block derived from
its job id, an attach-only path when ``RAY_ADDRESS`` is already set, and a
shutdown that stops only the processes of its own head.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Iterator, Mapping

from ask2distill.paths import Paths, get_paths

# Stages whose process environment differs from the default (see base_env).
_TOKENIZERS_PARALLELISM = {"tau2_synrl": "false", "eval_code": "false"}

PORT_BLOCK = 1000
PORT_BASE = 20000
N_BLOCKS = 40
# Offsets inside a job's port block.
_PORT_OFFSETS = {
    "ray_port": 0,
    "node_manager_port": 1,
    "object_manager_port": 2,
    "dashboard_agent_listen_port": 3,
    "ray_client_server_port": 4,
    "dashboard_agent_grpc_port": 5,
    "runtime_env_agent_port": 6,
    "metrics_export_port": 7,
    "VLLM_PORT": 10,
    "MASTER_PORT": 20,
    "serve_port_start": 30,       # vLLM OpenAI servers (tau2.serve) scan 30..99
    "serve_port_end": 99,
    "min_worker_port": 100,
    "max_worker_port": 999,
}


# ----------------------------------------------------------------------- environment

def base_env(paths: Paths | None = None, *, stage: str) -> dict[str, str | None]:
    """Environment variables every process of ``stage`` should run with.

    Returns only the variables to set (merge it over ``os.environ``; ``run``
    does that). ``PYTORCH_CUDA_ALLOC_CONF`` is mapped to ``None`` when an
    inherited value enables expandable segments, which ``run`` turns into an
    unset.
    """
    paths = paths or get_paths()
    cache = paths.cache
    env: dict[str, str | None] = {
        # HOME sits on a small quota (EDQUOT killed a 16k wave via triton cache
        # writes); route every compiler/runtime cache to the project filesystem.
        "XDG_CACHE_HOME": str(cache),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
        "VLLM_CACHE_ROOT": str(cache / "vllm"),
        "HF_HOME": str(cache / "hf"),
        "EVALPLUS_CACHE": str(cache / "evalplus"),
        # A fresh directory per call, as the launchers did with $(uuidgen).
        "OUTLINES_CACHE_DIR": str(cache / "outlines" / uuid.uuid4().hex),
        "PYTHONUNBUFFERED": "1",
        "RAY_USAGE_STATS_ENABLED": "0",
        "RAY_memory_usage_threshold": "0.99",
        "TORCH_NCCL_BLOCKING_WAIT": "1",
        "NCCL_TIMEOUT": "7200",
        "HYDRA_FULL_ERROR": "1",
        "TOKENIZERS_PARALLELISM": _TOKENIZERS_PARALLELISM.get(stage, "true"),
    }
    # flashinfer JIT-compiles vLLM's sampling kernels at runtime and needs the
    # CUDA dev headers under $CONDA_PREFIX/targets/x86_64-linux/include. Without
    # this the vLLM engine dies at startup with "FileNotFoundError: 'nvcc'" /
    # "fatal error: curand.h".
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CONDA_PREFIX")
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
    if stage == "opd":
        env["NCCL_DEBUG"] = "WARN"
        env["SWANLAB_MODE"] = "local"
    if stage == "synrl":
        env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    # NOTE: never set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True job-wide --
    # it is incompatible with vLLM's CuMemAllocator (used to offload rollout
    # weights) and makes vLLM init crash with an AssertionError. The math/code
    # reward worker needs it and sets it for its own processes only.
    if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        env["PYTORCH_CUDA_ALLOC_CONF"] = None
    return env


def merged_env(env: Mapping[str, str | None] | None) -> dict[str, str]:
    """``os.environ`` overlaid with ``env``; a ``None`` value unsets the variable."""
    out = dict(os.environ)
    for k, v in (env or {}).items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = str(v)
    return out


# ------------------------------------------------------------------------ ports, Ray

_LOCAL_KEY: str | None = None


def job_key() -> str:
    """``$SLURM_JOB_ID`` when inside SLURM, else a short random id for this process."""
    global _LOCAL_KEY
    jid = os.environ.get("SLURM_JOB_ID", "").strip()
    if jid:
        return jid
    if _LOCAL_KEY is None:
        _LOCAL_KEY = uuid.uuid4().hex[:8]
    return _LOCAL_KEY


def _block_index(jobkey: str) -> int:
    try:
        return int(jobkey) % N_BLOCKS
    except ValueError:
        return int(uuid.uuid5(uuid.NAMESPACE_OID, jobkey).hex, 16) % N_BLOCKS


def _ports_for(index: int) -> dict[str, int]:
    base = PORT_BASE + index * PORT_BLOCK
    return {"base": base, **{k: base + off for k, off in _PORT_OFFSETS.items()}}


def port_free(port: int, host: str = "0.0.0.0") -> bool:
    """True if a TCP socket can bind ``port`` right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def job_ports(jobkey: str | None = None, *, check_free: bool = True) -> dict[str, int]:
    """The job's port block: ``BASE = 20000 + (jobkey mod 40) * 1000``.

    Holds the Ray head ports, ``VLLM_PORT``, ``MASTER_PORT`` and a range for
    OpenAI-compatible vLLM servers. With ``check_free`` the fixed head ports of
    the block are probed and, if one is taken (another job whose id is equal
    mod 40, or a stray process), the next block is tried; launchers/slurm calls
    this same function so shell and Python agree.
    """
    idx = _block_index(jobkey or job_key())
    for k in range(N_BLOCKS if check_free else 1):
        ports = _ports_for((idx + k) % N_BLOCKS)
        fixed = [v for name, v in ports.items()
                 if name not in ("base", "min_worker_port", "max_worker_port",
                                 "serve_port_start", "serve_port_end")]
        if not check_free or all(port_free(p) for p in fixed):
            return ports
    raise RuntimeError("no free port block in 20000-59999")


def ray_temp_dir(jobkey: str | None = None) -> str:
    """``/tmp/ray_<user>_<jobkey>``.

    Must stay short and off the shared filesystem: Ray puts AF_UNIX sockets under it and
    socket paths are limited to ~107 bytes.
    """
    user = (os.environ.get("USER") or "u")[:12]
    return f"/tmp/ray_{user}_{(jobkey or job_key())[:12]}"


def node_ip() -> str:
    """This node's IP as other nodes see it (``hostname -i`` in the originals)."""
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


def ray_start_cmd(num_gpus: int, ports: Mapping[str, int], temp_dir: str, ip: str,
                  num_cpus: int | None = None) -> list[str]:
    """``ray start --head`` for a private, single-job cluster."""
    cmd = ["ray", "start", "--head", f"--node-ip-address={ip}", f"--temp-dir={temp_dir}",
           f"--port={ports['ray_port']}",
           f"--node-manager-port={ports['node_manager_port']}",
           f"--object-manager-port={ports['object_manager_port']}",
           f"--dashboard-agent-listen-port={ports['dashboard_agent_listen_port']}",
           f"--ray-client-server-port={ports['ray_client_server_port']}",
           f"--dashboard-agent-grpc-port={ports['dashboard_agent_grpc_port']}",
           f"--runtime-env-agent-port={ports['runtime_env_agent_port']}",
           f"--metrics-export-port={ports['metrics_export_port']}",
           f"--min-worker-port={ports['min_worker_port']}",
           f"--max-worker-port={ports['max_worker_port']}",
           "--include-dashboard=false", "--disable-usage-stats",
           f"--num-gpus={num_gpus}"]
    if num_cpus:
        cmd.append(f"--num-cpus={num_cpus}")
    return cmd


def _own_ray_pids(temp_dir: str) -> list[int]:
    """PIDs of this user's processes whose command line mentions ``temp_dir``."""
    pids = []
    uid = os.getuid()
    me = os.getpid()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            if d.stat().st_uid != uid:
                continue
            cmdline = (d / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        if temp_dir.encode() in cmdline:
            pids.append(int(d.name))
    return pids


def stop_own_head(temp_dir: str, grace_s: float = 15.0) -> None:
    """Stop the Ray processes started with ``--temp-dir=temp_dir`` and nothing else.

    ``ray stop`` has no way to target one cluster: it signals every Ray process
    of the user on the node, which is exactly what killed sibling jobs. The
    head's daemons (gcs_server, raylet, ...) carry the session dir under
    ``temp_dir`` on their command line, so they can be found by it.
    """
    pids = _own_ray_pids(temp_dir)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, sig)
        deadline = time.time() + grace_s
        while time.time() < deadline:
            pids = [p for p in pids if Path(f"/proc/{p}").exists()]
            if not pids:
                return
            time.sleep(0.5)


def _wait_ray(address: str, env: Mapping[str, str], timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = subprocess.run(["ray", "status", f"--address={address}"], env=dict(env),
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            return
        time.sleep(3)
    raise RuntimeError(f"RAY_CLUSTER_FAILED: no healthy Ray head at {address} after {timeout_s:.0f}s")


@contextlib.contextmanager
def ray_session(num_gpus: int, env: Mapping[str, str | None], *,
                cuda_visible: str | None = None, dry_run: bool = False) -> Iterator[str]:
    """Yield a Ray address for this job.

    If ``RAY_ADDRESS`` is set (in ``env`` or the process environment) the
    cluster belongs to whoever built it -- typically ``launchers/slurm/common.sh``
    on a multi-node job -- and this only attaches: no start, no stop. Otherwise
    a private head is started on this node with the job's temp dir and port
    block, ``RAY_ADDRESS`` is exported for the duration, and on exit only that
    head is stopped. ``/tmp/ray/ray_current_cluster`` is never touched.
    """
    address = (env or {}).get("RAY_ADDRESS") or os.environ.get("RAY_ADDRESS")
    if address:
        yield address
        return
    key = job_key()
    temp_dir = ray_temp_dir(key)
    ports = job_ports(key, check_free=not dry_run)
    ip = node_ip()
    address = f"{ip}:{ports['ray_port']}"
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    cmd = ray_start_cmd(num_gpus, ports, temp_dir, ip, int(cpus) if cpus else None)
    start_env = merged_env(env)
    if cuda_visible is not None:
        start_env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    if dry_run:
        print("[ray] " + shlex.join(cmd), flush=True)
        yield address
        return
    subprocess.run(cmd, env=start_env, check=True)
    prev = os.environ.get("RAY_ADDRESS")
    os.environ["RAY_ADDRESS"] = address
    try:
        _wait_ray(address, start_env)
        print(f"[ray] private head at {address} (temp dir {temp_dir})", flush=True)
        yield address
    finally:
        if prev is None:
            os.environ.pop("RAY_ADDRESS", None)
        else:
            os.environ["RAY_ADDRESS"] = prev
        stop_own_head(temp_dir)


# --------------------------------------------------------------------------- running

def run(cmd: list[str], env: Mapping[str, str | None] | None = None, log: Path | None = None,
        dry_run: bool = False, cwd: Path | None = None) -> int:
    """Run ``cmd`` with ``env`` merged over ``os.environ``; tee output to ``log``.

    With ``dry_run`` the command and the environment changes are printed and
    nothing runs.
    """
    if dry_run:
        print("[dry-run] env:", flush=True)
        for k in sorted(env or {}):
            v = (env or {})[k]
            print(f"  {k}={'<unset>' if v is None else v}", flush=True)
        if cwd:
            print(f"[dry-run] cwd: {cwd}", flush=True)
        print("[dry-run] " + shlex.join(map(str, cmd)), flush=True)
        return 0
    full = merged_env(env)
    if log is None:
        return subprocess.run(list(map(str, cmd)), env=full, cwd=cwd).returncode
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        proc = subprocess.Popen(list(map(str, cmd)), env=full, cwd=cwd,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()
        return proc.wait()


def new_run_dir(cfg, stage: str) -> Path:
    """Create ``$A2D_OUT/<pair>/runs/<stage>_<UTC time>_<jobkey>/`` and record the config.

    The resolved config goes to ``resolved_config.yaml``; launchers add
    ``env.json`` and ``command.txt`` with ``write_run_record``.
    """
    from ask2distill.config import save_resolved
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(str(cfg.paths.out)) / str(cfg.pair) / "runs" / f"{stage}_{stamp}_{job_key()}"
    out.mkdir(parents=True, exist_ok=False)
    save_resolved(cfg, out)
    return out


def write_run_record(run_dir: Path, env: Mapping[str, str | None], cmd: list[str]) -> None:
    """``env.json`` (the variables this run set) and ``command.txt`` (shell-quoted)."""
    from ask2distill.common.io import atomic_write_json, atomic_write_text
    atomic_write_json(Path(run_dir) / "env.json", {k: env[k] for k in sorted(env)})
    atomic_write_text(Path(run_dir) / "command.txt", shlex.join(map(str, cmd)) + "\n")


# ------------------------------------------------------------------- memory estimate

_BYTES = {"fp32": 4, "float32": 4, "bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2}


def _model_numbers(model_dir: str) -> dict | None:
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        return None
    c = json.loads(cfg_path.read_text())
    c = c.get("text_config", c)
    idx = Path(model_dir) / "model.safetensors.index.json"
    n_params = None
    if idx.is_file():
        total = json.loads(idx.read_text()).get("metadata", {}).get("total_size")
        dtype = c.get("torch_dtype", "bfloat16")
        if total:
            n_params = total / _BYTES.get(str(dtype), 2)
    if n_params is None:
        h, L = c.get("hidden_size", 0), c.get("num_hidden_layers", 0)
        inter, V = c.get("intermediate_size", 4 * h), c.get("vocab_size", 0)
        n_params = L * (4 * h * h + 3 * h * inter) + 2 * V * h
    return {"params": float(n_params), "vocab": int(c.get("vocab_size", 0))}


def mem_preflight(cfg, stage: str) -> dict:
    """Rough per-GPU memory estimate (GB) for a training stage, for ``--dry-run``.

    Counts the FSDP-sharded actor state (master weights + grads in
    ``model_dtype``, Adam moments in fp32; zero on the GPU when offloaded) and
    the float32 logits of one micro-batch (``ppo_max_token_len_per_gpu`` tokens
    x vocab, twice for the log-softmax). Activations, the vLLM KV cache and
    the frozen models are not included. Good enough to catch "fp32 4B on one
    GPU" before a 20-minute queue wait, not more.
    """
    from omegaconf import OmegaConf
    sel = lambda k, d=None: OmegaConf.select(cfg, k, default=d)  # noqa: E731
    if stage in ("synrl",):
        model = sel("models.synthesizer_init") or sel("models.student")
        dtype = sel("synrl.model_dtype", "fp32")
        n_gpus = sel("synrl.layout.actor_gpus") or sel("synrl.n_gpus_per_node", 8)
        offload = bool(sel("synrl.offload", False))
        tokens = sel("synrl.ppo_max_token_len_per_gpu", 0)
    elif stage in ("opd",):
        model = sel("opd.student") or sel("models.student")
        dtype = sel("opd.model_dtype", "fp32")
        n_gpus = sel("opd.n_gpus_per_node", 8) * sel("opd.nnodes", 1)
        offload = bool(sel("opd.offload.actor_param", True))
        arms = sel("opd.arms") or {}
        tokens = max([a.get("ppo_max_token_len_per_gpu", 0) or 0 for a in arms.values()]
                     + [sel("opd.ppo_max_token_len_per_gpu", 0) or 0])
    else:
        return {"stage": stage, "note": "no estimate for this stage"}
    if not isinstance(tokens, int):
        # tau2 renders it as max_prompt_length + max_response_length per version
        versions = sel("synrl.versions") or {}
        tokens = max([v.get("max_prompt_length", 0) for v in versions.values()] or [0]) \
            + int(sel("synrl.max_response_length", 0))
    nums = _model_numbers(str(model)) if model else None
    if nums is None:
        return {"stage": stage, "model": str(model), "note": "model config.json not found"}
    b = _BYTES.get(str(dtype), 4)
    state = nums["params"] * (2 * b + 8) / n_gpus / 2**30
    logits = 2 * tokens * nums["vocab"] * 4 / 2**30
    return {"stage": stage, "model": str(model), "params_b": round(nums["params"] / 1e9, 2),
            "dtype": dtype, "gpus": n_gpus, "offload": offload,
            "sharded_state_gb": 0.0 if offload else round(state, 1),
            "logits_gb": round(logits, 1),
            "total_gb": round((0.0 if offload else state) + logits, 1)}


# ----------------------------------------------------------------- shell interface

def shell_exports(stage: str, jobkey: str | None = None) -> str:
    """``export`` lines for launchers/slurm/common.sh: base_env, the job's port
    block (``A2D_PORT_*``, ``VLLM_PORT``, ``MASTER_PORT``) and ``A2D_RAY_TMPDIR``.

    The shell launchers source this output, so they and ``ray_session`` derive
    the same ports and temp dir from the same job id.
    """
    key = jobkey or job_key()
    lines = []
    for k, v in sorted(base_env(stage=stage).items()):
        lines.append(f"unset {k}" if v is None else f"export {k}={shlex.quote(v)}")
    for k, v in job_ports(key).items():
        name = k if k in ("VLLM_PORT", "MASTER_PORT") else f"A2D_PORT_{k.upper()}"
        lines.append(f"export {name}={v}")
    lines.append(f"export A2D_RAY_TMPDIR={shlex.quote(ray_temp_dir(key))}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``python -m ask2distill.runtime exports --stage S [--jobkey K]``."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m ask2distill.runtime")
    ap.add_argument("what", choices=["exports"])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--jobkey")
    args = ap.parse_args(argv)
    print(shell_exports(args.stage, args.jobkey))
    return 0


if __name__ == "__main__":
    sys.exit(main())
