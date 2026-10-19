#!/bin/bash
# One smoke job: 1 node, 2 GPUs, under the 4 h wall clock. Submit with
#   tests/smoke/submit.sh math|code|tau2
# A resubmitted job resumes after the last stage that passed (stage markers).
#SBATCH --job-name=a2d-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
set -o pipefail
DOMAIN=${1:?usage: smoke.sbatch math|code|tau2}
# shellcheck disable=SC1091
source "${A2D_REPO:?submit through tests/smoke/submit.sh}/launchers/slurm/common.sh"
a2d_load_cluster_env
a2d_activate
# Process environment and this job's private port block / Ray temp dir; the
# smoke stages start their own Ray heads from it, never a shared one.
a2d_job_exports "smoke"
exec bash "$A2D_REPO/tests/smoke/run_$DOMAIN.sh"
