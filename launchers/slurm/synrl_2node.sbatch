#!/bin/bash
# Synthesizer RL (math / code): GRPO-train S_theta against R(q) = divergence x
# compatibility, computed by a reward worker that holds the frozen student and
# teacher. Two nodes, split by resource pool -- this is why verl can be reused:
#   node0 (global_pool, 8 GPU): the synthesizer S_theta, the only model that updates
#   node1 (reward_pool, 8 GPU): student + teacher (+ the embedder in diversity mode emb), frozen
#
#   launchers/slurm/submit.sh synrl Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500
#   -> sbatch ... synrl_2node.sbatch synrl
#
# The job builds one Ray cluster over both nodes and exports RAY_ADDRESS, so
# `a2d synrl` only attaches. No --wall-clock-end: synthesizer RL does not
# force-save before the wall clock (a resubmit loses up to save_freq steps).
#
#SBATCH --job-name=a2d-synrl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
set -o pipefail
# shellcheck disable=SC1091
source "${A2D_REPO:?submit through launchers/slurm/submit.sh}/launchers/slurm/common.sh"
[ $# -gt 0 ] || a2d_die "no a2d command given"

a2d_job_setup synrl
a2d_nodes
trap a2d_ray_teardown EXIT
a2d_ray_cluster 2 8
a2d_run "$@"
rc=$?
a2d_log "exit code: $rc"
exit $rc
