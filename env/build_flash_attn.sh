#!/usr/bin/env bash
# FALLBACK: build flash-attn 2.8.1 from source with fatbin compression DISABLED, for clusters
# where the upstream prebuilt wheel (the default in env/build_a2d_env.sh) cannot load.
# This is how the wheel used by every final run was made.
#
# Run it inside the a2d env, ON A COMPUTE NODE with one GPU (not the login node: the build
# must link against the compute nodes' glibc), e.g.
#
#   conda activate a2d
#   srun <your one-GPU allocation flags> bash env/build_flash_attn.sh
#   A2D_FLASH_ATTN_WHEEL=$(ls -t $A2D_CACHE/wheels/flash_attn_nocomp/flash_attn-2.8.1-*.whl | head -1) \
#       bash env/build_a2d_env.sh        # or: pip install --no-deps --force-reinstall <wheel>
#
# Two independent things break the published wheels on old clusters:
#
#   1. GLIBC. Every published wheel (2.7.4.post1 and 2.8.1 alike, torch2.6/2.7/2.8)
#      needs GLIBC >= 2.32; GPU nodes running Ubuntu 20.04 have 2.31 (a login node with
#      2.35 imports the wheel fine, then a compute node dies with
#      "ImportError: ... version `GLIBC_2.32' not found"). Fix: build on a compute node.
#
#   2. Fatbin compression. With driver 535.104.12 (CUDA 12.2) and nvcc 12.8, flash-attn's
#      setup.py passes "-Xfatbin -compress-all" (plus a "-compress-mode=size" flag that only
#      exists in CUDA 12.8 -- itself a sign that 12.8 changed fatbin compression). The 12.2
#      driver cannot decompress the result, so loading fails with CUDA error 200
#      CUDA_ERROR_INVALID_IMAGE -- an invalid *module*, not the arch mismatch that error 209
#      would signal. Nothing else in the stack sets those flags, which is why torch's own
#      cu128 kernels and flashinfer's JIT-built kernels work on the same node.
#      Fix: strip the compression flags. Costs binary size only (62MB -> 200MB), never
#      correctness.
#
# This also avoids a CUDA-12.2 downgrade, which kept hitting version walls: 12.2 lacks
# -compress-mode, and nvcc 12.2 refuses host compilers newer than gcc 12 while the env
# ships gcc 14.3.
#
# Environment knobs:
#   A2D_FLASH_ATTN_SRC   existing flash-attention checkout to build (default: clone v2.8.1)
#   A2D_FLASH_ATTN_OUT   wheel output directory (default: $A2D_CACHE/wheels/flash_attn_nocomp)
#   FLASH_ATTN_CUDA_ARCHS  default 80 (A100 / sm_80, the only arch the final runs used)
#   MAX_JOBS             default 28
#   INSTALL=0            build the wheel only, do not install it into the active env
set -euo pipefail

FA_TAG=v2.8.1
if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: activate the a2d env first (conda activate a2d)" >&2
    exit 1
fi
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CACHE=${A2D_CACHE:-$REPO_DIR/.cache}    # same default as ask2distill.paths
OUT=${A2D_FLASH_ATTN_OUT:-$CACHE/wheels/flash_attn_nocomp}
WORK=$CACHE/build/flash_attn_nocomp
mkdir -p "$OUT" "$WORK"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found; this should run on a GPU compute node" >&2
fi

# The env's own 12.8 toolchain (installed by build_a2d_env.sh) -- the one that already
# builds working kernels.
export CUDA_HOME=$CONDA_PREFIX
command -v nvcc >/dev/null || { echo "ERROR: nvcc not found in $CONDA_PREFIX/bin" >&2; exit 1; }
nvcc --version | tail -2

if [ -n "${A2D_FLASH_ATTN_SRC:-}" ]; then
    SRC=$A2D_FLASH_ATTN_SRC
else
    SRC=$WORK/flash-attention
    if [ ! -d "$SRC/.git" ]; then
        git clone --depth 1 --branch "$FA_TAG" --recurse-submodules --shallow-submodules \
            https://github.com/Dao-AILab/flash-attention.git "$SRC"
    fi
fi

echo "=========== strip the fatbin compression flags from setup.py ==========="
[ -f "$SRC/setup.py.orig" ] || cp "$SRC/setup.py" "$SRC/setup.py.orig"
python - "$SRC/setup.py" <<'PY'
import re
import sys

path = sys.argv[1]
src = open(path).read()
out = re.sub(r'^[ \t]*"(-Xfatbin|-compress-all|-compress-mode=size)",[ \t]*\n', "", src, flags=re.M)
open(path, "w").write(out)
PY
if grep -nE '"-Xfatbin"|"-compress-all"|"-compress-mode' "$SRC/setup.py"; then
    echo "STILL_PRESENT -- abort" >&2
    exit 1
fi
echo "  none present, good"

export MAX_JOBS=${MAX_JOBS:-28}
export NVCC_THREADS=${NVCC_THREADS:-2}
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
export FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS:-80}

echo "=========== BUILD (nvcc 12.8, sm_${FLASH_ATTN_CUDA_ARCHS}, no fatbin compression) ==========="
rm -rf "$SRC/build"
(cd "$SRC" && python -m pip wheel . --no-build-isolation --no-deps -w "$OUT")
WHEEL=$(ls -t "$OUT"/flash_attn-*.whl | head -1)
echo "built $WHEEL"

if [ "${INSTALL:-1}" = "1" ]; then
    python -m pip install --force-reinstall --no-deps "$WHEEL"
    echo "=========== VERIFY (neutral cwd) ==========="
    (cd "$WORK" && python "$REPO_DIR/env/check_env.py" --stage core)
fi
echo "NOCOMP_BUILD_OK: $WHEEL"
