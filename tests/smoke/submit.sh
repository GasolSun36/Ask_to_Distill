#!/bin/bash
# Submit a smoke job (2x A100-80GB, one node) with the settings of
# launchers/slurm/cluster.env (PARTITION_SMALL, ACCOUNT, TIME):
#
#   tests/smoke/submit.sh math|code|tau2 [extra sbatch args...]
#
# Prints the job id; the log goes to $A2D_OUT/smoke/<domain>/slurm-<id>.out.
set -o pipefail
DOMAIN=${1:-}
case "$DOMAIN" in math|code|tau2) shift ;; *) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;; esac
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# shellcheck disable=SC1091
source "$REPO/launchers/slurm/common.sh"
a2d_load_cluster_env
a2d_activate
OUT=$(python -c 'from ask2distill.paths import get_paths; print(get_paths().out)')/smoke/$DOMAIN
mkdir -p "$OUT"
# shellcheck disable=SC2206
EXTRA=(${EXTRA_SBATCH_ARGS:-})
sbatch --parsable --job-name="a2d-smoke-$DOMAIN" --account="$ACCOUNT" --partition="$PARTITION_SMALL" \
    --time="$TIME" --output="$OUT/slurm-%j.out" --error="$OUT/slurm-%j.out" --export=ALL \
    "${EXTRA[@]}" "$@" "$REPO/tests/smoke/smoke.sbatch" "$DOMAIN"
