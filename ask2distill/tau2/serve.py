"""Serve the frozen tau2 models as vLLM OpenAI servers and write an endpoints file.

One ``vllm serve`` per replica, one GPU per replica (TP 1), the hermes tool
parser and auto tool choice -- as the final runs served them -- for four
plans:

=========  ========================================================================
plan       servers (kind: model)
=========  ========================================================================
synrl      stu: student, usr: user simulator / judge, tea: teacher, with the
           replica map ``synrl.serve.node_maps[$A2D_SERVE_NODE_INDEX]``
harvest    gen: the merged synthesizer of ``--version``, plus stu / usr / tea,
           ``harvest.serve.replicas_per_model`` each
opd        usr: the user simulator, ``opd.user_sim_serve.map`` (per-arm override)
eval       stu: the checkpoint under test (``--model``/``--served-name`` or
           ``--arm``/``--step``), ``eval.serve.gpus`` replicas; a model served
           under the name ``eval.user_llm`` is the local user simulator (kind usr)
=========  ========================================================================

Served names are explicit (``--served-model-name``): the env service, the
harvest client and ToolOrchestra's eval client address models by name, and
the eval client routes a model to vLLM only when its name contains "qwen", so
eval names are checked. Chat templates are chosen by ROLE (``models.templates``
-- student, synthesizer, teacher, user), never by served name; ``user: null``
means the model's own template (Instruct-2507 has no thinking mode). A role
whose template is configured but missing is a hard error: vLLM's fallback, the
model's own template, silently re-enables thinking for Qwen3 and Thinking-2507.
Non-thinking is forced through the template because ToolOrchestra's client
sends no ``chat_template_kwargs``.

No ``--generation-config`` or sampling flags are passed: requests sample with
the model's own generation_config defaults (Qwen3-4B: top_p 0.95, top_k 20),
which is what the final runs did.

Servers start one at a time, each waiting until it answers ``/v1/models``
with its own served name before the next starts (a busy port answered by
another job's server once produced a false ready). Ports come from the job's
port block (``runtime.job_ports``), stepping past ports already taken. When
more than one server shares a GPU, ``--enforce-eager`` is added (CUDA-graph
capture needs headroom co-located servers do not leave). The endpoints file
(``kind url served_name``) is written atomically once every server is ready;
the process then stays up until it is terminated, and takes its servers down
with it.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from ask2distill import runtime
from ask2distill.tau2.envside import endpoints as E

PLANS = ("synrl", "harvest", "opd", "eval")
NODE_INDEX_ENV = "A2D_SERVE_NODE_INDEX"
READY_TIMEOUT_S = 1800
_POLL_S = 10


class ServeError(RuntimeError):
    """A server could not be planned or did not come up."""


@dataclass(frozen=True)
class ServerSpec:
    """One served model: ``replicas`` vLLM servers of ``tp`` GPUs each."""

    kind: str
    model_path: str
    served_name: str
    template: str | None
    replicas: int
    tp: int = 1
    gpu_util: float = 0.9
    max_len: int = 32768


# ------------------------------------------------------------------- templates, names

def template_for(cfg: DictConfig, role: str) -> str | None:
    """Template of a role; ``None`` (own template) only when configured null."""
    templates = OmegaConf.select(cfg, "models.templates", throw_on_missing=True)
    if templates is None or role not in templates:
        raise ServeError(f"models.templates has no role {role!r}")
    value = templates[role]
    if value is None:
        return None
    if not Path(str(value)).is_file():
        raise ServeError(f"chat template for role {role!r} not found: {value} (refusing to fall "
                         "back to the model's own template, which enables thinking)")
    return str(value)


def check_eval_name(name: str) -> str:
    """Eval served names must contain "qwen" (ToolOrchestra routes agent models by name)."""
    if "qwen" not in name.lower():
        raise ServeError(f"eval served name {name!r} must contain 'qwen': ToolOrchestra only routes "
                         "such names to the vLLM client")
    return name


def eval_served_name(cfg: DictConfig, arm: str, step: int) -> str:
    """``eval.served_name`` rendered for an OPD arm checkpoint (e.g. Agent_Qwen3-4B_Qwen3-4B-Thinking-2507-ours-s<N>)."""
    short = OmegaConf.select(cfg, f"opd.arms.{arm}.short_name", throw_on_missing=True)
    if short is None:
        raise ServeError(f"opd.arms.{arm}.short_name is not set")
    return check_eval_name(str(cfg.eval.served_name).format(short_exp=short, step=int(step)))


def parse_map(text: str) -> dict[str, int]:
    """``"A=2|B=3"`` -> ``{"A": 2, "B": 3}`` (``|``-joined: sbatch --export splits on commas)."""
    out: dict[str, int] = {}
    for item in str(text).split("|"):
        if item.strip():
            k, _, v = item.partition("=")
            out[k.strip()] = int(v)
    return out


def _frozen_roles(cfg: DictConfig) -> dict[str, tuple[str, str, str]]:
    """served name -> (kind, model path, template role) for the frozen models."""
    re_ = cfg.reward_env
    return {str(re_.TAU2_STUDENT_MODEL): ("stu", str(cfg.models.student), "student"),
            str(re_.TAU2_USER_MODEL): ("usr", str(cfg.models.user_sim_train), "user"),
            str(re_.TAU2_TEACHER_MODEL): ("tea", str(cfg.models.teacher), "teacher")}


def _from_map(cfg: DictConfig, rep_map: dict[str, int], *, gpu_util: float, max_len: int,
              tp: int = 1) -> list[ServerSpec]:
    roles = _frozen_roles(cfg)
    specs = []
    for name, n in rep_map.items():
        if name not in roles:
            raise ServeError(f"replica map names {name!r}; known models: {sorted(roles)}")
        kind, path, role = roles[name]
        specs.append(ServerSpec(kind, path, name, template_for(cfg, role), n, tp, gpu_util, max_len))
    return specs


def _no_sampling_overrides(node: DictConfig, where: str) -> None:
    if str(node.sampling_overrides) != "none":
        raise ServeError(f"{where}.sampling_overrides must be 'none' (the final runs passed no "
                         "--generation-config or sampling flags)")


# ----------------------------------------------------------------------------- plans

def plan_servers(cfg: DictConfig, plan: str, *, node_index: int = 0, version: str | None = None,
                 arm: str | None = None, model: str | None = None, served_name: str | None = None,
                 step: int | None = None) -> list[ServerSpec]:
    """The servers one node runs for ``plan``."""
    if plan == "synrl":
        s = cfg.synrl.serve
        _no_sampling_overrides(s, "synrl.serve")
        maps = list(s.node_maps)
        if not 0 <= node_index < len(maps):
            raise ServeError(f"node index {node_index} out of range for {len(maps)} synrl.serve.node_maps")
        return _from_map(cfg, parse_map(maps[node_index]), gpu_util=float(s.gpu_util),
                         max_len=int(s.max_len), tp=int(s.tp))
    if plan == "harvest":
        if version is None:
            raise ServeError("--plan harvest needs --version")
        h = cfg.harvest
        hv = OmegaConf.select(cfg, f"harvest.{version}", throw_on_missing=True)
        if hv is None:
            raise ServeError(f"no harvest.{version}")
        n, util, max_len = int(h.serve.replicas_per_model), float(h.serve.gpu_util), int(h.serve.max_len)
        gen = ServerSpec("gen", str(hv.synthesizer), str(hv.served_name), template_for(cfg, "synthesizer"),
                         n, 1, util, max_len)
        frozen = _from_map(cfg, {name: n for name in _frozen_roles(cfg)}, gpu_util=util, max_len=max_len)
        return [gen, *frozen]
    if plan == "opd":
        u = cfg.opd.user_sim_serve
        rep_map = parse_map(u.map)
        if arm is not None:
            arm_map = OmegaConf.select(cfg, f"opd.arms.{arm}.user_sim_serve.map", default=None)
            if arm_map is not None:
                rep_map = parse_map(arm_map)
        return _from_map(cfg, rep_map, gpu_util=float(u.gpu_util), max_len=int(u.max_len))
    if plan == "eval":
        ev = cfg.eval
        _no_sampling_overrides(ev.serve, "eval.serve")
        if model is None:
            if arm is None or step is None:
                raise ServeError("--plan eval needs --model and --served-name, or --arm and --step")
            from ask2distill.merge import hf_dir
            exp = OmegaConf.select(cfg, f"opd.arms.{arm}.experiment", throw_on_missing=True)
            model = str(hf_dir(cfg, str(exp), int(step)))
            served_name = served_name or eval_served_name(cfg, arm, int(step))
        if not served_name:
            raise ServeError("--plan eval with --model needs --served-name")
        if served_name == str(ev.user_llm):
            # local stand-in for the OpenAI user simulator (smoke): own template, kind usr
            return [ServerSpec("usr", model, served_name, template_for(cfg, "user"), 1, 1,
                               float(ev.serve.gpu_util), int(ev.serve.max_len))]
        return [ServerSpec("stu", model, check_eval_name(served_name),
                           template_for(cfg, str(ev.template_role)), int(ev.serve.gpus), 1,
                           float(ev.serve.gpu_util), int(ev.serve.max_len))]
    raise ServeError(f"unknown plan {plan!r}; expected one of {PLANS}")


# ---------------------------------------------------------------------------- GPUs

def visible_gpus() -> list[str]:
    """GPU ids this process may use: ``$CUDA_VISIBLE_DEVICES``, else nvidia-smi's list."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        return [g for g in cvd.split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    gpus = [x.strip() for x in out.splitlines() if x.strip()]
    if not gpus:
        raise ServeError("no GPUs visible (set CUDA_VISIBLE_DEVICES or pass --gpus)")
    return gpus


def assign_gpus(specs: list[ServerSpec], gpus: list[str]) -> list[tuple[ServerSpec, int, list[str]]]:
    """(spec, replica, gpu ids) per server, filling GPUs in order and wrapping when they run out."""
    slots, k = [], 0
    for spec in specs:
        for r in range(spec.replicas):
            slots.append((spec, r, [gpus[(k + j) % len(gpus)] for j in range(spec.tp)]))
            k += spec.tp
    return slots


# ------------------------------------------------------------------------- serving

def vllm_command(spec: ServerSpec, port: int, *, eager: bool) -> list[str]:
    """``vllm serve`` for one replica, with the flags of the final runs."""
    cmd = ["vllm", "serve", spec.model_path,
           "--served-model-name", spec.served_name,
           "--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    if spec.template:
        cmd += ["--chat-template", spec.template]
    cmd += ["--max-model-len", str(spec.max_len), "--gpu-memory-utilization", str(spec.gpu_util),
            "--tensor-parallel-size", str(spec.tp)]
    if eager:
        cmd.append("--enforce-eager")
    cmd += ["--port", str(port)]
    return cmd


def serves(url: str, name: str) -> bool:
    """Ready == answers AND serves the model this line claims (another job's server
    on a colliding port must not count)."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=10) as r:
            data = json.loads(r.read().decode())
    except Exception:
        return False
    return any(m.get("id") == name for m in data.get("data", []))


def _free_port(start: int, end: int, taken: set[int]) -> int:
    for p in range(start, end + 1):
        if p not in taken and runtime.port_free(p):
            return p
    raise ServeError(f"no free port in {start}-{end}")


def start(plan: list[ServerSpec], gpus: list[str], out_endpoints: Path, *,
          served_names: dict[str, str] | None = None, templates: dict[str, str | None] | None = None,
          sequential: bool = True, log_dir: Path | None = None, dry_run: bool = False,
          ready_timeout_s: float = READY_TIMEOUT_S) -> list[subprocess.Popen]:
    """Start every server of ``plan`` and write ``out_endpoints`` when all are ready.

    ``served_names`` / ``templates`` (keyed by kind) override the specs'. With
    ``sequential`` each server must be ready before the next one starts.
    """
    specs = []
    for s in plan:
        name = (served_names or {}).get(s.kind, s.served_name)
        tmpl = (templates or {}).get(s.kind, s.template)
        specs.append(ServerSpec(s.kind, s.model_path, name, tmpl, s.replicas, s.tp, s.gpu_util, s.max_len))
    slots = assign_gpus(specs, gpus)
    load = Counter(g for _, _, gs in slots for g in gs)
    shared = any(n > 1 for n in load.values())
    ports = runtime.job_ports(check_free=not dry_run)
    ip = runtime.node_ip()
    key = runtime.job_key()
    log_dir = Path(log_dir or Path(out_endpoints).parent / "serve_logs")
    base = runtime.base_env(stage="tau2_serve")
    procs: list[subprocess.Popen] = []
    lines: list[E.Endpoint] = []
    pending: list[tuple[subprocess.Popen, E.Endpoint]] = []
    taken: set[int] = set()
    try:
        for i, (spec, r, gs) in enumerate(slots):
            port = (ports["serve_port_start"] + i if dry_run else
                    _free_port(ports["serve_port_start"], ports["serve_port_end"], taken))
            taken.add(port)
            cmd = vllm_command(spec, port, eager=shared)
            env = dict(base)
            # Each replica needs its own compile cache or they race on the same dir.
            env["VLLM_CACHE_ROOT"] = str(Path(base["VLLM_CACHE_ROOT"]) / f"tau2_{key}_{i}")
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gs)
            ep = E.Endpoint(spec.kind, f"http://{ip}:{port}/v1", spec.served_name)
            print(f"[serve] {spec.kind} {spec.served_name} replica {r} on GPU {','.join(gs)} :{port}"
                  + (" (eager: shared GPU)" if shared else ""), flush=True)
            if dry_run:
                print("[dry-run] CUDA_VISIBLE_DEVICES=" + env["CUDA_VISIBLE_DEVICES"] + " "
                      + shlex.join(cmd), flush=True)
                lines.append(ep)
                continue
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / f"vllm_{spec.served_name}_{r}.log", "ab") as fh:
                proc = subprocess.Popen(cmd, env=runtime.merged_env(env), stdout=fh,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            procs.append(proc)
            pending.append((proc, ep))
            if sequential:
                _wait_ready(pending, ready_timeout_s)
                lines += [e for _, e in pending]
                pending = []
        if pending:
            _wait_ready(pending, ready_timeout_s)
            lines += [e for _, e in pending]
    except BaseException:
        stop_all(procs)
        raise
    if not dry_run:
        from ask2distill.common.io import atomic_write_text
        Path(out_endpoints).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out_endpoints, E.format_lines(lines))
    print(f"ALL_READY {out_endpoints}: " + " ".join(f"{k}={n}" for k, n in
                                                     Counter(e.kind for e in lines).items()), flush=True)
    return procs


def _wait_ready(pending: list[tuple[subprocess.Popen, E.Endpoint]], timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    todo = list(pending)
    while todo:
        for proc, ep in list(todo):
            if proc.poll() is not None:
                raise ServeError(f"vLLM for {ep.served_name} at {ep.url} exited with code {proc.returncode}")
            if serves(ep.url, ep.served_name):
                todo.remove((proc, ep))
        if todo:
            if time.time() > deadline:
                raise ServeError("SERVE_TIMEOUT: " + ", ".join(f"{e.served_name}@{e.url}" for _, e in todo))
            time.sleep(_POLL_S)


def stop_all(procs: list[subprocess.Popen], grace_s: float = 30.0) -> None:
    """Terminate every server's process group."""
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + grace_s
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main(cfg: DictConfig, args: argparse.Namespace) -> int:
    """``a2d tau2 serve --plan P [--gpus 0,1] [--version V | --arm A --step N | --model D --served-name S] [--out F]``."""
    dry = bool(getattr(args, "dry_run", False))
    node_index = int(os.environ.get(NODE_INDEX_ENV, "0"))
    specs = plan_servers(cfg, args.plan, node_index=node_index, version=getattr(args, "version", None),
                         arm=getattr(args, "arm", None), model=getattr(args, "model", None),
                         served_name=getattr(args, "served_name", None), step=getattr(args, "step", None))
    gpus = [g for g in str(args.gpus).split(",") if g.strip()] if getattr(args, "gpus", None) else (
        ["0"] if dry else visible_gpus())
    out = Path(args.out) if getattr(args, "out", None) else (
        Path(str(cfg.paths.out)) / "tau2" / "serve" / f"{args.plan}_{runtime.job_key()}" / "endpoints.txt")
    procs = start(specs, gpus, out, dry_run=dry)
    if dry:
        return 0

    def _term(signum, _frame):
        stop_all(procs)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    # Stay up with the servers; a server that dies takes the fleet down, so the
    # consumer sees the failure instead of a silently shrunken replica set.
    while all(p.poll() is None for p in procs):
        time.sleep(30)
    dead = [p.args[2] for p in procs if p.poll() is not None]
    print(f"[serve] a server exited ({dead}); stopping the fleet", flush=True)
    stop_all(procs)
    return 1
