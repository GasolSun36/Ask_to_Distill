"""tau2-bench evaluation client: the official test split against served checkpoints.

Runs on a host that can reach the user simulator (gpt-4o-mini over the OpenAI
API; compute nodes have no route to it, so this is the login node), against
vLLM servers started on compute nodes by ``a2d tau2 serve --plan eval``.

Protocol (as the reported evaluations ran it): for every model
of kind ``stu`` in the endpoints file and every domain, ``tau2/cli.py`` runs the
task file ``<tasks_dir>/<domain>.json`` with ``trials`` trials, ``max_steps``,
``concurrency`` parallel simulations and the user simulator ``user_llm``. The
fork addresses the agent model by its served name and routes it to the vLLM
client only when the name contains "qwen" (its ``LLM_CALL.py`` / ``llm_utils.py``),
so names are checked up front.

Resuming and the trial top-up: tau2 skips any task that has a result file, so
before each pass every task with fewer than ``trials`` results is deleted and
rerun whole. tau2's vLLM client retries forever (server gone, context
overflow), so a watchdog ends a pass whose server stopped answering or that
wrote nothing for ``watchdog_min`` minutes (checked after the first
``watchdog_min``); passes repeat until every task is complete.

OpenAI key handling. The key is read from ``eval.openai_env_file``
(``$A2D_OPENAI_ENV_FILE``) or ``$OPENAI_API_KEY`` and is never copied
anywhere persistent: a 0600 temporary env file holding ``OPENAI_API_KEY`` and an
explicit ``OPENAI_BASE_URL`` is written, ``TAU2_OPENAI_ENV`` points at it, and it
is deleted on exit. The fork's ``get_direct_openai_client`` (TAU2_OPENAI_DIRECT=1)
loads that file into ``os.environ``, which overrides a relay ``OPENAI_BASE_URL``
inherited from a shell profile. With ``eval.openai_base_url: local`` (smoke) the
user simulator is the endpoint of kind ``usr`` served under the name
``user_llm``, with the key ``EMPTY`` -- the fork's gpt-4o-mini branch is taken
unchanged, and the numbers are not comparable to the paper's.

The fork writes ``cut_error_message.json`` into its working directory, so each
run gets a dedicated one under the outputs.

    python -m ask2distill.tau2.envside.run_eval --endpoints endpoints.txt [--tag test4]
"""
from __future__ import annotations

import argparse
import atexit
import glob
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from ask2distill.tau2.envside import endpoints as E
from ask2distill.tau2.envside import repo, settings

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
POLL_S = 120


# ------------------------------------------------------------------- OpenAI key file

def read_env_file(path: str | Path) -> dict[str, str]:
    """``KEY=VALUE`` lines (``export`` prefix and quotes allowed), as the fork parses them."""
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line[len("export "):] if line.startswith("export ") else line
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def openai_credentials(ev: dict, eps: list[E.Endpoint]) -> tuple[str, str]:
    """(api key, base url) for the user simulator."""
    base = ev.get("openai_base_url") or DEFAULT_OPENAI_BASE_URL
    if base == "local":
        user = [e for e in E.by_kind(eps, "usr") if e.served_name == ev["user_llm"]]
        if not user:
            raise SystemExit(f"openai_base_url=local needs an endpoint of kind 'usr' served as "
                             f"{ev['user_llm']!r} (the local user simulator)")
        return "EMPTY", user[0].url
    key = None
    env_file = ev.get("openai_env_file")
    if env_file:
        key = read_env_file(env_file).get("OPENAI_API_KEY")
    key = key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("no OpenAI key: set A2D_OPENAI_ENV_FILE (a file with OPENAI_API_KEY=...) "
                         "or OPENAI_API_KEY")
    return key, base


def write_temp_env(key: str, base_url: str) -> str:
    """A 0600 env file with the key and an explicit base URL; removed at exit."""
    fd, path = tempfile.mkstemp(prefix="a2d_openai_", suffix=".env")
    os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as fh:
        fh.write(f"OPENAI_API_KEY={key}\nOPENAI_BASE_URL={base_url}\n")

    def _cleanup() -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    atexit.register(_cleanup)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, _frame: sys.exit(128 + signum))
    return path


# ---------------------------------------------------------------------- task files

def task_ids(tasks_dir: str, domain: str, num_tasks: int | None) -> list[str]:
    with open(f"{tasks_dir}/{domain}.json") as fh:
        ids = [t["id"] for t in json.load(fh)]
    return ids[:num_tasks] if num_tasks else ids


def result_dir(out_root: Path, model: str, domain: str) -> Path:
    """Directory of per-simulation json files (tau2 strips the ``.json`` of --output_file)."""
    return out_root / f"{model}__{domain}"


def prune_partial(out_root: Path, model: str, domains: list[str], tasks_dir: str,
                  trials: int, num_tasks: int | None) -> int:
    """Delete results of tasks with fewer than ``trials`` runs; return how many tasks are left."""
    left = 0
    for d in domains:
        files = defaultdict(list)
        for f in glob.glob(str(result_dir(out_root, model, d) / "*.json")):
            try:
                with open(f) as fh:
                    files[json.load(fh)["task_id"]].append(f)
            except Exception:
                os.remove(f)
        for i in task_ids(tasks_dir, d, num_tasks):
            if len(files[i]) < trials:
                for f in files[i]:
                    os.remove(f)
                left += 1
    return left


def n_results(out_root: Path, model: str, domain: str) -> int:
    return len(glob.glob(str(result_dir(out_root, model, domain) / "*.json")))


def newest_result(out_root: Path, model: str) -> float:
    ts = [os.path.getmtime(f) for f in glob.glob(str(out_root / f"{model}__*" / "*.json"))]
    return max(ts) if ts else 0.0


# ---------------------------------------------------------------------- servers

def alive(url: str, timeout: float = 20.0) -> bool:
    """True if ``<url>/models`` answers."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------- one pass

def tau2_command(tau2_dir: str, model: str, domain: str, ev: dict, task_path: str,
                 output_file: Path, config_path: Path) -> list[str]:
    cmd = [sys.executable, f"{tau2_dir}/tau2/cli.py",
           "--domain", domain,
           "--agent-llm", model,
           "--user-llm", str(ev["user_llm"]),
           "--num-trials", str(ev["trials"]),
           "--task_path", task_path,
           "--max-steps", str(ev["max_steps"]),
           "--max-concurrency", str(ev["concurrency"]),
           "--output_file", str(output_file),
           "--model_config_path", str(config_path)]
    if ev.get("num_tasks"):
        cmd += ["--num-tasks", str(ev["num_tasks"])]
    return cmd


def run_domain(cmd: list[str], env: dict, cwd: Path, log: Path, model: str, out_root: Path,
               urls: list[str], watchdog_s: float) -> int:
    """Run one ``tau2/cli.py``; kill it when its server dies or it goes stale."""
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "a") as fh:
        proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        while proc.poll() is None:
            time.sleep(POLL_S)
            if proc.poll() is not None:
                break
            stale = (time.time() - t0 > watchdog_s
                     and time.time() - newest_result(out_root, model) > watchdog_s)
            gone = not any(alive(u) for u in urls)
            if stale or gone:
                print(f"[eval] {model}: ending this pass (server gone={gone}, stale={stale})", flush=True)
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ask2distill.tau2.envside.run_eval")
    ap.add_argument("--endpoints", required=True, help="endpoints file: kind 'stu' = models under test")
    ap.add_argument("--tag", default=None, help="output tag (default: eval.tag)")
    ap.add_argument("--models", nargs="+", default=None, help="subset of served names to evaluate")
    ap.add_argument("--domains", nargs="+", default=None)
    ap.add_argument("--max-passes", type=int, default=20)
    a = ap.parse_args(argv)

    ev = settings.role_settings()["eval"]
    eps = E.read(a.endpoints)
    agents = list(dict.fromkeys(e.served_name for e in E.by_kind(eps, "stu")))
    if a.models:
        agents = [m for m in agents if m in a.models]
    if not agents:
        raise SystemExit("no model of kind 'stu' to evaluate in the endpoints file")
    bad = [m for m in agents if "qwen" not in m.lower()]
    if bad:
        # ToolOrchestra routes a model to the vLLM client only when its name contains
        # "qwen" (its LLM_CALL.py); any other name falls through and the call fails.
        raise SystemExit(f"served names must contain 'qwen': {bad}")
    domains = a.domains or list(ev["domains"])
    tag = a.tag or str(ev["tag"])
    out_root = Path(ev["out_dir"]) / tag
    out_root.mkdir(parents=True, exist_ok=True)
    cwd = out_root / "_cwd"
    cwd.mkdir(exist_ok=True)
    trials, num_tasks = int(ev["trials"]), (int(ev["num_tasks"]) if ev.get("num_tasks") else None)
    watchdog_s = 60.0 * float(ev["watchdog_min"])

    agent_eps = [e for e in eps if e.kind == "stu" and e.served_name in agents]
    for e in agent_eps:
        if not alive(e.url):
            raise SystemExit(f"server not ready: {e.served_name} {e.url}")
    config_path = out_root / f"model_config.{os.getpid()}.json"
    config_path.write_text(json.dumps(E.model_config(agent_eps, str(config_path)), indent=2))

    key, base_url = openai_credentials(ev, eps)
    env = dict(os.environ)
    env.update({
        "TAU2_OPENAI_DIRECT": str(ev["TAU2_OPENAI_DIRECT"]),
        "TAU2_OPENAI_ENV": write_temp_env(key, base_url),
        "OPENAI_BASE_URL": base_url,
        "TAU2_ORCHESTRATOR_PROMPT": str(ev["TAU2_ORCHESTRATOR_PROMPT"]),
        "TAU2_MAX_TOKENS": str(ev["TAU2_MAX_TOKENS"]),
        "TOKENIZERS_PARALLELISM": "false",
    })
    env.pop("OPENAI_API_KEY", None)
    tau2_dir = f"{repo.repo_path()}/evaluation/tau2-bench"
    print(f"[eval] tag {tag}: models {agents}, domains {domains}, trials {trials}, "
          f"user {ev['user_llm']} via {base_url}", flush=True)

    for model in agents:
        urls = [e.url for e in agent_eps if e.served_name == model]
        for p in range(1, a.max_passes + 1):
            left = prune_partial(out_root, model, domains, ev["tasks_dir"], trials, num_tasks)
            print(f"[eval] {model} pass {p}: {'complete' if left == 0 else f'left {left}'}", flush=True)
            if left == 0:
                break
            for d in domains:
                want = len(task_ids(ev["tasks_dir"], d, num_tasks)) * trials
                have = n_results(out_root, model, d)
                if have >= want:
                    continue
                cmd = tau2_command(tau2_dir, model, d, ev, f"{ev['tasks_dir']}/{d}.json",
                                   out_root / f"{model}__{d}.json", config_path)
                print(f"[eval] ===== {model} / {d} / {tag}  ({have}/{want})", flush=True)
                run_domain(cmd, env, cwd, out_root / "logs" / f"run_{model}_{d}.log", model,
                           out_root, urls, watchdog_s)
        else:
            left = prune_partial(out_root, model, domains, ev["tasks_dir"], trials, num_tasks)
            if left:
                print(f"[eval] {model}: {left} tasks still incomplete after {a.max_passes} passes",
                      flush=True)
                return 3
    print("EVAL_TEST_ALL_DONE", flush=True)
    print("summarize with: a2d tau2 summarize --outputs-dir", out_root, "--models", *agents, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
