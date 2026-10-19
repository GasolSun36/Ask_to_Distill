"""Start env-side modules under the tau2 interpreter (``$A2D_TAU2_PYTHON``).

``a2d`` always runs in the a2d env; tau2 code runs in a separate interpreter
that has tau2's dependencies but neither torch nor pyarrow. Every env-side
process is started as ``$A2D_TAU2_PYTHON -m ask2distill.tau2.envside.<module>``
with the environment built by ``env``:

* ``PYTHONPATH=<repo>:$REPO_PATH:$REPO_PATH/evaluation/tau2-bench`` -- tau2 is
  not pip-installed and ask2distill is not installed in the tau2 env;
* ``REPO_PATH`` -- the fork's ``tau2/utils/utils.py`` raises ``TypeError``
  without it;
* ``HF_HOME=$A2D_CACHE/hf`` and ``HF_HUB_OFFLINE=1`` -- the fork loads the
  ``Qwen/Qwen3-8B`` tokenizer at import (``llm_utils.py``); it is
  pre-cached by ``a2d data build tau2``;
* the role's ``TAU2_*`` variables (``ask2distill.tau2.env.render``) and, for
  harvest / eval, ``A2D_TAU2_SETTINGS``.

Inherited ``TAU2_*`` variables are dropped first, so a value left in the
user's shell can never leak into a run, and ``TAU2_DATA`` (a DB-directory
variable name that collides with another setting) is never passed.
"""
from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import IO, Mapping

from omegaconf import DictConfig

from ask2distill.paths import REPO_ROOT, require_tau2_python
from ask2distill.tau2 import env as tau2_env

ENVSIDE = "ask2distill.tau2.envside"
_DROP_PREFIXES = ("TAU2_", "A2D_TAU2_")


def _load_default_cfg() -> DictConfig:
    from ask2distill import config
    return config.load(config.domain_pair("tau2"))


def env(cfg: DictConfig | None, role: str, extra: Mapping[str, str] | None = None,
        *, version: str | None = None, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Full environment for an env-side process of ``role``.

    ``cfg`` ``None`` loads the shipped tau2-domain pair (``a2d data build tau2``
    runs without a pair). ``base`` defaults to ``os.environ``.
    """
    cfg = cfg if cfg is not None else _load_default_cfg()
    repo_path = str(cfg.tau2_data.repo_path)
    out = {k: v for k, v in (os.environ if base is None else base).items()
           if not k.startswith(_DROP_PREFIXES) and k != "PYTHONPATH"}
    out.update({
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), repo_path, f"{repo_path}/evaluation/tau2-bench"]),
        "REPO_PATH": repo_path,
        "HF_HOME": str(cfg.tau2_data.tokenizer_cache),
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    })
    out.update(tau2_env.render(cfg, role, version))
    out.update(tau2_env.settings_env(cfg, role))
    out.update({k: str(v) for k, v in (extra or {}).items()})
    return out


def command(module: str, argv: list[str], python: str | os.PathLike | None = None) -> list[str]:
    """``[$A2D_TAU2_PYTHON, -m, ask2distill.tau2.envside.<module>, *argv]``."""
    py = str(python or require_tau2_python())
    return [py, "-m", f"{ENVSIDE}.{module}", *map(str, argv)]


def run(module: str, argv: list[str], env: Mapping[str, str], *, log: Path | None = None,
        cwd: Path | None = None) -> subprocess.Popen:
    """Start an env-side module; stdout/stderr go to ``log`` when given."""
    stdout: IO | None = None
    if log is not None:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        stdout = open(log, "ab")
    try:
        return subprocess.Popen(command(module, argv), env=dict(env), cwd=cwd,
                                stdout=stdout, stderr=subprocess.STDOUT if stdout else None)
    finally:
        if stdout is not None:
            stdout.close()


def wait_http(url: str, proc: subprocess.Popen | None = None, timeout_s: float = 300.0,
              interval_s: float = 5.0) -> None:
    """Poll ``url`` until it answers 200; fail early if ``proc`` exits."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"process exited with code {proc.returncode} before {url} answered")
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(interval_s)
    raise TimeoutError(f"{url} did not answer within {timeout_s:.0f}s")


def start_env_service(cfg: DictConfig, role: str, *, port: int, workers: int,
                      extra: Mapping[str, str] | None = None, version: str | None = None,
                      log: Path | None = None, timeout_s: float = 300.0) -> subprocess.Popen:
    """Start ``envside.env_service`` and wait for ``/health``."""
    e = env(cfg, role, extra, version=version)
    proc = run("env_service", ["--port", str(port), "--workers", str(workers)], e, log=log)
    try:
        wait_http(f"http://127.0.0.1:{port}/health", proc, timeout_s)
    except Exception:
        stop(proc)
        raise
    return proc


def stop(proc: subprocess.Popen | None, grace_s: float = 15.0) -> None:
    """Terminate a spawned process, killing it after ``grace_s``."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
