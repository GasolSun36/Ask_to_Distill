#!/usr/bin/env bash
# Build the `a2d-tau2` conda environment that runs the tau2 side of the agent pipeline:
# the env/reward HTTP service, card harvest, official split, domain context dump and the
# tau2 evaluation client.
#
#   bash env/build_tau2_env.sh
#   ENV_NAME=myenv FORCE=1 bash env/build_tau2_env.sh
#
# Environment knobs:
#   ENV_NAME             conda env name (default: a2d-tau2)
#   FORCE=1              remove an existing env of that name first
#   CONDA_SH             path to conda's etc/profile.d/conda.sh (default: from `conda info --base`)
#   A2D_TAU2_PY_VERSION  Python for the env (default: 3.12.14, the frozen interpreter)
#
# Neither tau2 nor ask2distill is installed here. `a2d` (always run from the a2d env) spawns
# `$A2D_TAU2_PYTHON -m ask2distill.tau2.envside.<module>` with PYTHONPATH covering the repo,
# third_party/toolorchestra_tau2 and its evaluation/tau2-bench (see ask2distill/tau2/spawn.py).
set -euo pipefail

ENV_NAME=${ENV_NAME:-a2d-tau2}
PY_VERSION=${A2D_TAU2_PY_VERSION:-3.12.14}
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOCK="$REPO_DIR/env/requirements-tau2.lock.txt"

if [ -z "${CONDA_SH:-}" ]; then
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
fi
# shellcheck disable=SC1090
source "$CONDA_SH"

echo "===== STEP 0: create env $ENV_NAME (python $PY_VERSION) ====="
if [ "${FORCE:-0}" = "1" ]; then
    conda env remove -y -n "$ENV_NAME" || true
fi
if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi
set +u
conda activate "$ENV_NAME"
set -u
echo "python: $(command -v python) ($(python --version 2>&1))"

echo "===== STEP 1: locked pip packages ====="
python -m pip install --upgrade pip
# Exactly the frozen closure; --no-deps keeps every pin (litellm==1.82.6, no brotli).
python -m pip install --no-deps -r "$LOCK"

echo "===== STEP 2: verify ====="
python - <<'PY'
import importlib
import importlib.util
import sys

mods = ["litellm", "openai", "fastapi", "uvicorn", "pydantic", "loguru", "deepdiff",
        "addict", "toml", "dotenv", "transformers", "requests", "httpx"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m:14s} {getattr(mod, '__version__', '')}")
    except Exception as e:  # report every failure, not just the first
        bad.append(m)
        print(f"FAIL {m:14s} {type(e).__name__}: {e}")
# brotli must stay absent: with an old brotli, httpx dies on br-encoded responses with
# "APIConnectionError: Connection error".
if importlib.util.find_spec("brotli") is not None:
    bad.append("brotli (must not be installed)")
    print("FAIL brotli is installed; the frozen tau2 env has none")
for forbidden in ("torch", "pyarrow", "omegaconf"):
    if importlib.util.find_spec(forbidden) is not None:
        print(f"NOTE {forbidden} is importable here; envside code must not depend on it")
print("MISSING:", bad if bad else "none")
sys.exit(1 if bad else 0)
PY

PY_BIN=$(command -v python)
cat <<EOF

BUILD_DONE: conda env '$ENV_NAME'.
Point the a2d CLI at this interpreter (configs/paths.env):

    export A2D_TAU2_PYTHON=$PY_BIN

Then, from the a2d env: python env/check_env.py --tau2
EOF
