#!/bin/bash
# tau2 synthesizer RL for one version in ONE 3-node job: node 0 trains, nodes 1
# and 2 both serve the frozen student / user simulator / teacher (replica maps
# synrl.serve.node_maps[0] and [1]). At 8 serve GPUs a step took 849s, bound by
# the k=6 student rollouts per card and the teacher gate, hence two serve nodes.
# Serves and trainer share the job's lifetime, so endpoints never go stale
# mid-run; resubmit.sh resubmits across wall clocks.
#
#   launchers/slurm/submit.sh tau2-synrl Agent_Qwen3-4B_Qwen3-4B-Thinking-2507 --version v7
#   -> sbatch ... tau2_synrl_3node.sbatch tau2 synrl --version v7
#
#SBATCH --job-name=a2d-tau2-synrl
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --mem=0
set -o pipefail
# shellcheck disable=SC1091
source "${A2D_REPO:?submit through launchers/slurm/submit.sh}/launchers/slurm/common.sh"
[ $# -gt 0 ] || a2d_die "no a2d command given"
VERSION=$(a2d_arg_value --version "$@") || a2d_die "tau2 synrl needs --version"

a2d_job_setup tau2_synrl
a2d_nodes
[ "${#A2D_NODES[@]}" -ge 3 ] || a2d_die "need 3 nodes, got ${#A2D_NODES[@]}"
trap a2d_stop_serves EXIT
a2d_log "train=${A2D_NODES[0]} serve=${A2D_NODES[1]},${A2D_NODES[2]}"
EPS=()
for i in 0 1; do
    EPS+=("$A2D_JOB_DIR/endpoints.node$i.txt")
    a2d_serve_on_node "$i" "${A2D_NODES[$((i + 1))]}" "${EPS[$i]}" --plan synrl --version "$VERSION"
done
a2d_wait_endpoints 1800 "${EPS[@]}"
EPF=$A2D_JOB_DIR/endpoints.txt
a2d_collect_endpoints "$EPF" "${EPS[@]}"

WALL=$(a2d_wall_clock_end)
# SLURM_NNODES etc. describe the whole 3-node job; the trainer owns exactly one
# 8-GPU node and must be told so.
export SLURM_NNODES=1 SLURM_JOB_NUM_NODES=1
a2d_run "$@" --endpoints "$EPF" ${WALL:+--wall-clock-end "$WALL"}
rc=$?
a2d_log "trainer exit code: $rc"
exit $rc
