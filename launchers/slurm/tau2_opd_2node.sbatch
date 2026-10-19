#!/bin/bash
# ONE tau2 OPD arm in a 2-node job: node 1 serves the user simulator
# (opd.user_sim_serve, per-arm map), node 0 runs the trainer with its env
# service and private Ray head. Serve and trainer share the job's lifetime, so
# the user-simulator endpoints never go stale mid-run.
#
#   launchers/slurm/submit.sh tau2-opd Agent_Qwen3-4B_Qwen3-4B-Thinking-2507 --arm ours
#   -> sbatch ... tau2_opd_2node.sbatch tau2 opd --arm ours
#
#SBATCH --job-name=a2d-tau2-opd
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
ARM=$(a2d_arg_value --arm "$@") || a2d_die "tau2 opd needs --arm"

a2d_job_setup opd
a2d_nodes
[ "${#A2D_NODES[@]}" -ge 2 ] || a2d_die "need 2 nodes, got ${#A2D_NODES[@]}"
trap a2d_stop_serves EXIT
a2d_log "train=${A2D_NODES[0]} user-sim serve=${A2D_NODES[1]}"
EP=$A2D_JOB_DIR/user_endpoints.node0.txt
a2d_serve_on_node 0 "${A2D_NODES[1]}" "$EP" --plan opd --arm "$ARM"
a2d_wait_endpoints 1800 "$EP"
EPF=$A2D_JOB_DIR/user_endpoints.txt
a2d_collect_endpoints "$EPF" "$EP"

WALL=$(a2d_wall_clock_end)
# SLURM_NNODES etc. describe the whole 2-node job; the trainer owns exactly one
# 8-GPU node and must be told so.
export SLURM_NNODES=1 SLURM_JOB_NUM_NODES=1
a2d_run "$@" --user-endpoints "$EPF" ${WALL:+--wall-clock-end "$WALL"}
rc=$?
a2d_log "trainer exit code: $rc"
exit $rc
