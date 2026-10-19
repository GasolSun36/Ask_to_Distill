#!/usr/bin/env bash
# Build the `a2d` conda environment: training (verl OPD + synthesizer RL), vLLM inference,
# math/code evaluation and the `a2d` CLI.
#
# The env recipe behind every final run (base build plus the packages installed by
# hand afterwards), with the measured versions taken from the frozen `pip freeze` in
# env/requirements-a2d.lock.txt instead of open-ended specifiers.
#
#   bash env/build_a2d_env.sh                      # env name `a2d`
#   ENV_NAME=myenv FORCE=1 bash env/build_a2d_env.sh
#
# Environment knobs:
#   ENV_NAME               conda env name (default: a2d)
#   FORCE=1                remove an existing env of that name first
#   CONDA_SH               path to conda's etc/profile.d/conda.sh (default: from `conda info --base`)
#   A2D_PYTHON_VERSION     Python for the env (default: 3.12.13, the frozen interpreter)
#   A2D_FLASH_ATTN_WHEEL   local flash-attn wheel to install instead of the upstream prebuilt one;
#                          use the output of env/build_flash_attn.sh on old-driver/old-glibc nodes
#
# Only prebuilt wheels are used, so this runs on a GPU-less login node. It is re-runnable.
set -euo pipefail

ENV_NAME=${ENV_NAME:-a2d}
PY_VERSION=${A2D_PYTHON_VERSION:-3.12.13}
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOCK="$REPO_DIR/env/requirements-a2d.lock.txt"

# Upstream prebuilt flash-attn matching torch 2.8 / CUDA 12 / cp312 / C++11 ABI (torch>=2.7 PyPI
# wheels use the new ABI). The final runs used a source build of the same 2.8.1 release made with
# env/build_flash_attn.sh; kernels are identical, only fatbin compression differs.
FLASH_ATTN_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

# evalplus derives its version with setuptools-scm, which fails without a .git directory.
# 0.4.0.dev44 is what `git describe` gave for the vendored commit (v0.3.1-44-g26d6d00).
EVALPLUS_VERSION=0.4.0.dev44

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
# conda's activate scripts reference unset variables.
set +u
conda activate "$ENV_NAME"
set -u
echo "python: $(command -v python) ($(python --version 2>&1))"

echo "===== STEP 1: CUDA toolchain (nvcc + curand headers) ====="
# vLLM 0.11 lets flashinfer JIT-compile its sampling kernels at RUNTIME, which needs
# nvcc and the CUDA dev headers. verl's docker image ships a full CUDA toolkit, so
# upstream never hits this; in a bare conda env the engine dies with
# "FileNotFoundError: 'nvcc'" and then "fatal error: curand.h: No such file".
# Callers must also export CUDA_HOME=$CONDA_PREFIX at run time.
conda install -y -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8 libcurand-dev=10.3.9

echo "===== STEP 2: locked pip packages ====="
python -m pip install --upgrade pip
# The lock is the complete frozen closure, so dependency resolution is skipped: every
# package lands at exactly the version the final runs used.
python -m pip install --no-deps -r "$LOCK"

echo "===== STEP 3: flash-attn 2.8.1 ====="
# OPD needs flash-attn: the run configs set actor_rollout_ref.model.use_remove_padding=True,
# whose varlen path goes through flash_attn. vLLM itself does not -- it ships its own.
if [ -n "${A2D_FLASH_ATTN_WHEEL:-}" ]; then
    python -m pip install --no-deps --force-reinstall "$A2D_FLASH_ATTN_WHEEL"
else
    python -m pip install --no-deps "$FLASH_ATTN_URL"
fi

echo "===== STEP 4: vendored packages (editable, --no-deps keeps the locked pins) ====="
# verl's setup.py pins numpy<2.0.0; the env runs numpy 2.2.6 (the CUDA stack vLLM 0.11
# pulls in has dropped numpy 1.x, and verl uses no removed numpy-1 API), hence --no-deps.
python -m pip install --no-deps -e "$REPO_DIR/third_party/verl"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_EVALPLUS=$EVALPLUS_VERSION \
    python -m pip install --no-deps -e "$REPO_DIR/third_party/evalplus"
if ! command -v evalplus.evaluate >/dev/null 2>&1; then
    echo "ERROR: the evalplus.evaluate console script is missing after installing evalplus" >&2
    exit 1
fi
# LiveCodeBench is not installed: ask2distill runs `python -m lcb_runner.runner.main` with
# third_party/LiveCodeBench on PYTHONPATH.

echo "===== STEP 5: ask2distill (editable, provides the \`a2d\` CLI) ====="
if [ -f "$REPO_DIR/pyproject.toml" ]; then
    python -m pip install --no-deps -e "$REPO_DIR"
else
    echo "WARNING: $REPO_DIR/pyproject.toml not found; skipping the ask2distill install" >&2
fi

echo "===== STEP 6: verify ====="
# pip check reports verl's numpy<2 pin (expected, see STEP 4); it is informational only.
python -m pip check || true
export CUDA_HOME="$CONDA_PREFIX"
python "$REPO_DIR/env/check_env.py" --stage imports

cat <<EOF

BUILD_DONE: conda env '$ENV_NAME'.
Before running anything in it, export the CUDA toolchain for the flashinfer JIT:

    conda activate $ENV_NAME && export CUDA_HOME=\$CONDA_PREFIX

GPU self-checks (run on a compute node with one GPU):

    python env/check_env.py --stage core
    python env/check_env.py --stage verl
    python env/check_env.py --stage vllm --model <small HF model dir, e.g. Qwen2.5-0.5B-Instruct>

If the flash-attn check fails with a GLIBC_2.32 import error or CUDA error 200
(CUDA_ERROR_INVALID_IMAGE), build the fallback wheel with env/build_flash_attn.sh and
re-run this script with A2D_FLASH_ATTN_WHEEL=<that wheel>.
EOF
