#!/bin/bash
# Resubmit a training stage across wall clocks until its checkpoint reaches a step.
#
#   launchers/slurm/resubmit.sh --exp EXPERIMENT --target STEP [options] -- <submit.sh args>
#
#   P=Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500
#   nohup launchers/slurm/resubmit.sh --exp opd_${P}_ours --target 90 \
#       -- opd $P --arm ours > resubmit_${P}_ours.log 2>&1 &
#
# Progress is read from $A2D_CKPT/<EXPERIMENT>/latest_checkpointed_iteration.txt
# -- the tracker verl's resume reads, which it writes only after a save
# completes -- not from the highest-numbered global_step_* directory: a save
# interrupted mid-write (a job once died during the step-240 save, leaving 25
# of 32 files) exists on disk but is correctly ignored by resume.
#
# Options:
#   --exp NAME        experiment (checkpoint directory name under $A2D_CKPT)
#   --ckpt-dir DIR    checkpoint directory instead of --exp
#   --target N        stop once the tracker reaches N
#   --max-fast K      stop after K consecutive jobs shorter than --fast-secs (default 3)
#   --fast-secs S     (default 600)
#   --poll S          seconds between checks (default 120)
set -o pipefail
A2D_SLURM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
source "$A2D_SLURM_DIR/common.sh"

EXP="" CKPT="" TARGET="" MAX_FAST=3 FAST_SECS=600 POLL=120
while [ $# -gt 0 ]; do
    case "$1" in
        --exp) EXP=$2; shift 2 ;;
        --ckpt-dir) CKPT=$2; shift 2 ;;
        --target) TARGET=$2; shift 2 ;;
        --max-fast) MAX_FAST=$2; shift 2 ;;
        --fast-secs) FAST_SECS=$2; shift 2 ;;
        --poll) POLL=$2; shift 2 ;;
        --) shift; break ;;
        *) a2d_die "unknown option $1 (submit.sh arguments go after --)" ;;
    esac
done
[ -n "$TARGET" ] || a2d_die "--target is required"
[ $# -ge 2 ] || a2d_die "give the submit.sh arguments after --"
a2d_load_cluster_env
a2d_activate
if [ -z "$CKPT" ]; then
    [ -n "$EXP" ] || a2d_die "--exp or --ckpt-dir is required"
    CKPT=$(python -c "from ask2distill.paths import get_paths; print(get_paths().ckpt)")/$EXP
fi
TRACKER=$CKPT/latest_checkpointed_iteration.txt

fast=0 last=""
while :; do
    step=$(cat "$TRACKER" 2>/dev/null || echo 0)
    if [ "${step:-0}" -ge "$TARGET" ]; then
        a2d_log "RESUBMIT_DONE $CKPT at step $step (target $TARGET)"
        exit 0
    fi
    if [ -z "$last" ] || [ -z "$(squeue -h -j "$last" -o %i 2>/dev/null)" ]; then
        if [ -n "$last" ]; then
            el=$(sacct -j "$last" -X -n -o ElapsedRaw 2>/dev/null | head -1 | tr -d ' ')
            if [ -n "$el" ] && [ "$el" -lt "$FAST_SECS" ]; then fast=$((fast + 1)); else fast=0; fi
            if [ "$fast" -ge "$MAX_FAST" ]; then
                a2d_log "$fast consecutive jobs shorter than ${FAST_SECS}s (last $last); stopping"
                exit 1
            fi
        fi
        last=$("$A2D_SLURM_DIR/submit.sh" "$@") || a2d_die "submit failed"
        a2d_log "submitted $last (step $step / $TARGET)"
    fi
    sleep "$POLL"
done
