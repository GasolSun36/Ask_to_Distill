#!/bin/bash
# tau2 card harvest (v6 / v7) in ONE 2-node job: every node serves the merged
# synthesizer, student, user simulator and teacher (harvest.serve, two replicas
# each), and the harvest client runs on node 0's CPUs against all of them.
# The client resumes from its pool file and redoes rows that failed on
# infrastructure (rollout_error / gen_error), so a job killed at the wall clock
# loses nothing; resubmit.sh is not needed -- resubmit until the client reports
# the pool complete.
#
#   launchers/slurm/submit.sh tau2-harvest Agent_Qwen3-4B_Qwen3-4B-Thinking-2507 --version v7
#   -> sbatch ... tau2_harvest_2node.sbatch tau2 harvest --version v7
#
#SBATCH --job-name=a2d-tau2-harvest
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
VERSION=$(a2d_arg_value --version "$@") || a2d_die "tau2 harvest needs --version"

a2d_job_setup tau2_serve
a2d_nodes
[ "${#A2D_NODES[@]}" -ge 2 ] || a2d_die "need 2 nodes, got ${#A2D_NODES[@]}"
trap a2d_stop_serves EXIT
EPS=()
for i in 0 1; do
    EPS+=("$A2D_JOB_DIR/endpoints.node$i.txt")
    a2d_serve_on_node "$i" "${A2D_NODES[$i]}" "${EPS[$i]}" --plan harvest --version "$VERSION"
done
a2d_wait_endpoints 1800 "${EPS[@]}"
EPF=$A2D_JOB_DIR/endpoints.txt
a2d_collect_endpoints "$EPF" "${EPS[@]}"

a2d_run "$@" --endpoints "$EPF"
rc=$?
a2d_log "harvest client exit code: $rc"
exit $rc
