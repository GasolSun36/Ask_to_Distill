#!/bin/bash
# Small single-node job (0-2 GPUs): calibration, sweep, pick, yield, harvest
# shard, filter slice, aggregate, bank, merge, math / code eval.
# Submit through submit.sh, which sets account, partition, time, the GPU count
# for the stage and A2D_CONFIG; the batch arguments are the a2d command line:
#
#   launchers/slurm/submit.sh harvest Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500 --shard 3
#   -> sbatch ... gpu_small.sbatch harvest --shard 3
#
#SBATCH --job-name=a2d-small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
set -o pipefail
# shellcheck disable=SC1091
source "${A2D_REPO:?submit through launchers/slurm/submit.sh}/launchers/slurm/common.sh"
[ $# -gt 0 ] || a2d_die "no a2d command given"

a2d_job_setup "$(a2d_stage_of "$@")"
a2d_run "$@"
rc=$?
a2d_log "exit code: $rc"
exit $rc
