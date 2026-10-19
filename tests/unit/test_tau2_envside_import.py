"""Every env-side module imports under the tau2 interpreter with the spawn environment.

Runs ``$A2D_TAU2_PYTHON`` (skipped when unset) in a subprocess per module, with
``ask2distill.tau2.spawn.env`` -- the environment ``a2d`` gives env-side
processes -- and checks that none of omegaconf, torch or pyarrow got imported:
the tau2 env does not have them, and a module that pulled one in would only
fail on a machine where the tau2 env is clean.

Modules that touch tau2 or its data at import (the prompt builders, the
rewards, the solvers) need ``a2d data build tau2`` to have run (ToolOrchestra
``data/tau2``, ``domain_ctx.json``, the pre-cached Qwen3-8B tokenizer); they
are skipped when that data is absent.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import AGENT_QWEN3_4B

pytestmark = pytest.mark.tau2env

LIGHT = ["endpoints", "settings", "repo", "official_split", "harvest", "run_eval", "build_prompts",
         "dump_domain_ctx", "env_service"]
HEAVY = ["scenario_prompt", "scenario_prompt_v5", "scenario_prompt_v6", "scenario_prompt_v7",
         "pref_solver", "retail_v7", "reward_base", "reward_v5", "reward_v6", "reward_v7"]
FORBIDDEN = ("omegaconf", "torch", "pyarrow")

PROBE = ("import importlib, json, sys; importlib.import_module(sys.argv[1]); "
         "print(json.dumps([m for m in %r if m in sys.modules]))" % (FORBIDDEN,))


@pytest.fixture(scope="module")
def tau2_python() -> str:
    py = os.environ.get("A2D_TAU2_PYTHON", "").strip()
    if not py or not Path(py).exists():
        pytest.skip("A2D_TAU2_PYTHON is not set")
    return py


@pytest.fixture(scope="module")
def spawn_env():
    from ask2distill import config
    from ask2distill.tau2 import spawn
    cfg = config.load(AGENT_QWEN3_4B)
    return cfg, spawn.env(cfg, "synrl")


def _import(py: str, env: dict, module: str) -> list[str]:
    r = subprocess.run([py, "-c", PROBE, f"ask2distill.tau2.envside.{module}"], env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"{module}: {r.stderr[-3000:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", LIGHT)
def test_light_modules_import(module, tau2_python, spawn_env):
    _, env = spawn_env
    assert _import(tau2_python, env, module) == []


@pytest.mark.parametrize("module", HEAVY)
def test_heavy_modules_import(module, tau2_python, spawn_env):
    cfg, env = spawn_env
    needed = [Path(str(cfg.tau2_data.data_dir)) / "domains", Path(str(cfg.tau2_data.domain_ctx)),
              Path(str(cfg.tau2_data.tokenizer_cache))]
    if not all(p.exists() for p in needed):
        pytest.skip("tau2 data not built (a2d data build tau2)")
    assert _import(tau2_python, env, module) == []


def test_package_init_is_import_free():
    """ask2distill/__init__ and ask2distill/tau2/__init__ are imported by the tau2 interpreter too."""
    from ask2distill.paths import REPO_ROOT
    for init in (REPO_ROOT / "ask2distill" / "__init__.py", REPO_ROOT / "ask2distill" / "tau2" / "__init__.py",
                 REPO_ROOT / "ask2distill" / "tau2" / "envside" / "__init__.py"):
        code = [ln for ln in init.read_text().splitlines() if ln.strip().startswith(("import ", "from "))]
        assert code == [], f"{init} imports {code}"


def test_envside_sources_import_nothing_forbidden():
    """Static check: env-side modules import only stdlib, tau2, web/openai deps and other env-side modules."""
    import ast

    from ask2distill.paths import REPO_ROOT
    allowed_third_party = {"tau2", "fastapi", "pydantic", "uvicorn", "openai", "requests"}
    for f in sorted((REPO_ROOT / "ask2distill" / "tau2" / "envside").glob("*.py")):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for n in names:
                top = n.split(".")[0]
                if n.startswith("ask2distill"):
                    assert n.startswith("ask2distill.tau2.envside"), f"{f.name} imports {n}"
                else:
                    import sys
                    assert top in sys.stdlib_module_names or top in allowed_third_party, f"{f.name} imports {n}"
