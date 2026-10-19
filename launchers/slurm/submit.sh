#!/bin/bash
# Submit one pipeline stage as a SLURM job.
#
#   launchers/slurm/submit.sh <stage> <pair | config.yaml> [a2d options] [key=value ...]
#
# Examples:
#   launchers/slurm/submit.sh synrl Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500
#   launchers/slurm/submit.sh harvest Code_Qwen3-4B_Qwen3-4B-Instruct-2507 --shard 3
#   launchers/slurm/submit.sh opd Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500 --arm ours
#   launchers/slurm/submit.sh opd configs/smoke/math_smoke.yaml --arm ours
#   launchers/slurm/submit.sh eval-math Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500 --arm ours --step <N> --tasks AIME24
#   launchers/slurm/submit.sh tau2-opd Agent_Qwen3-4B_Qwen3-4B-Thinking-2507 --arm ours
#
# The config (pair file + key=value overrides) is resolved HERE, written to
# $A2D_OUT/<pair>/submit/<stage>_<time>/resolved_config.yaml, and only its path
# travels to the job, as A2D_CONFIG through --export. sbatch's --export list is
# comma-separated and cannot carry a value with commas, quotes or spaces, so
# lists never travel through it; the a2d options travel as batch-script
# arguments, which have no such limit.
#
# Prints the job id. Settings come from launchers/slurm/cluster.env.
set -o pipefail
A2D_SLURM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$A2D_SLURM_DIR/common.sh"

usage() {
    sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    echo "stages: calibrate synrl sweep pick yield harvest filter aggregate bank prompts"
    echo "        opd merge eval-math eval-code tau2-synrl tau2-opd tau2-harvest"
    exit 2
}
[ $# -ge 2 ] || usage
STAGE=$1 TARGET=$2; shift 2

# stage -> batch script, a2d command words, partition class, GPUs (small jobs)
case "$STAGE" in
    calibrate)    SCRIPT=gpu_small.sbatch;         WORDS=(calibrate);       CLASS=small; GPUS=domain ;;
    synrl)        SCRIPT=synrl_2node.sbatch;       WORDS=(synrl);           CLASS=big ;;
    sweep)        SCRIPT=gpu_small.sbatch;         WORDS=(select sweep);    CLASS=small; GPUS=2 ;;
    pick)         SCRIPT=gpu_small.sbatch;         WORDS=(select pick);     CLASS=small; GPUS=1 ;;
    yield)        SCRIPT=gpu_small.sbatch;         WORDS=(select yield);    CLASS=small; GPUS=1 ;;
    harvest)      SCRIPT=gpu_small.sbatch;         WORDS=(harvest);         CLASS=small; GPUS=1 ;;
    filter)       SCRIPT=gpu_small.sbatch;         WORDS=(filter);          CLASS=small; GPUS=1 ;;
    aggregate)    SCRIPT=gpu_small.sbatch;         WORDS=(aggregate);       CLASS=small; GPUS=1 ;;
    bank)         SCRIPT=gpu_small.sbatch;         WORDS=(bank);            CLASS=small; GPUS=0 ;;
    prompts)      SCRIPT=gpu_small.sbatch;         WORDS=(prompts build);   CLASS=small; GPUS=0 ;;
    opd)          SCRIPT=opd_1node.sbatch;         WORDS=(opd);             CLASS=big ;;
    merge)        SCRIPT=gpu_small.sbatch;         WORDS=(merge);           CLASS=small; GPUS=1 ;;
    eval-math)    SCRIPT=gpu_small.sbatch;         WORDS=(eval math);       CLASS=small; GPUS=1 ;;
    eval-code)    SCRIPT=gpu_small.sbatch;         WORDS=(eval code);       CLASS=small; GPUS=2 ;;
    tau2-synrl)   SCRIPT=tau2_synrl_3node.sbatch;  WORDS=(tau2 synrl);      CLASS=big ;;
    tau2-opd)     SCRIPT=tau2_opd_2node.sbatch;    WORDS=(tau2 opd);        CLASS=big ;;
    tau2-harvest) SCRIPT=tau2_harvest_2node.sbatch; WORDS=(tau2 harvest);   CLASS=big ;;
    -h|--help)    usage ;;
    *) a2d_die "unknown stage '$STAGE'" ;;
esac

OVERRIDES=() ARGS=()
for a in "$@"; do
    if [[ $a != -* && $a == *=* ]]; then OVERRIDES+=("$a"); else ARGS+=("$a"); fi
done

a2d_load_cluster_env
a2d_activate

# Resolve and record the config; print "<path> <pair> <domain>".
read -r CONFIG PAIR DOMAIN < <(python - "$TARGET" "$STAGE" "${OVERRIDES[@]}" <<'PY'
import datetime, os, sys
from pathlib import Path
from ask2distill import config
target, stage, *overrides = sys.argv[1:]
try:
    if target.endswith((".yaml", ".yml")):
        cfg = config.load(None, target, overrides)
    else:
        cfg = config.load(target, None, overrides)
except config.ConfigError as e:
    sys.exit(f"config error:\n{e}")
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
run = Path(str(cfg.paths.out)) / str(cfg.pair) / "submit" / f"{stage}_{stamp}_{os.getpid()}"
print(config.save_resolved(cfg, run), cfg.pair, cfg.domain)
PY
) || a2d_die "could not resolve the config"
[ -s "$CONFIG" ] || a2d_die "no resolved config written"
case "$CONFIG" in *,*) a2d_die "config path $CONFIG contains a comma; sbatch --export cannot carry it" ;; esac

if [ "$CLASS" = big ]; then
    SHAPE=(--partition="$PARTITION_BIG" --cpus-per-task="$CPUS_PER_TASK")
else
    [ "$GPUS" = domain ] && { [ "$DOMAIN" = code ] && GPUS=2 || GPUS=1; }   # code calib: teacher on cuda:1
    SHAPE=(--partition="$PARTITION_SMALL" --gpus-per-node="$GPUS"
           --cpus-per-task="$SMALL_CPUS_PER_TASK" --mem="$SMALL_MEM")
fi
NAME="a2d-$STAGE-$PAIR"
ARM=$(a2d_arg_value --arm "${ARGS[@]}") && NAME="$NAME-$ARM"
OUTDIR=$(dirname "$CONFIG")

# shellcheck disable=SC2206
EXTRA=(${EXTRA_SBATCH_ARGS:-})
export A2D_REPO
JOB=$(sbatch --parsable --job-name="$NAME" --account="$ACCOUNT" --time="$TIME" "${SHAPE[@]}" \
        --output="$OUTDIR/slurm-%j.out" --error="$OUTDIR/slurm-%j.err" \
        --export=ALL,A2D_CONFIG="$CONFIG" "${EXTRA[@]}" \
        "$A2D_SLURM_DIR/$SCRIPT" "${WORDS[@]}" "${ARGS[@]}") || a2d_die "sbatch failed"
JOB=${JOB%%;*}
echo "$JOB"
echo "[a2d] submitted $NAME as job $JOB (config $CONFIG, log $OUTDIR/slurm-$JOB.out)" >&2
