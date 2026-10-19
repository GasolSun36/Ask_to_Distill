#!/bin/bash
# One OPD arm (math or code) on one 8-GPU node.
#
#   launchers/slurm/submit.sh opd Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500 --arm ours
#   -> sbatch ... opd_1node.sbatch opd --arm ours
#
# One epoch of a real-data arm does not fit in a 4h wall clock, so arms are
# resubmitted until they finish (resubmit.sh): the experiment name is stable,
# resume_mode=auto continues from the newest checkpoint, and math arms
# force-save just before the job is killed (opd.esi_save; the end time comes
# from squeue %L here). `a2d opd` starts its own private Ray head.
#
#SBATCH --job-name=a2d-opd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
set -o pipefail
# shellcheck disable=SC1091
source "${A2D_REPO:?submit through launchers/slurm/submit.sh}/launchers/slurm/common.sh"
[ $# -gt 0 ] || a2d_die "no a2d command given"

a2d_job_setup opd
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader || true
WALL=$(a2d_wall_clock_end)
a2d_run "$@" ${WALL:+--wall-clock-end "$WALL"}
rc=$?
a2d_log "exit code: $rc"
exit $rc
