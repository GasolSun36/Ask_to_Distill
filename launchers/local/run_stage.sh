#!/bin/bash
# Run one a2d stage on this machine with its output logged.
#
#   launchers/local/run_stage.sh <a2d command line>
#   launchers/local/run_stage.sh opd --pair Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500 --arm ours
#   launchers/local/run_stage.sh eval math --config configs/smoke/math_smoke.yaml --arm ours --step 2
#
# Output goes to the terminal and to $A2D_OUT/logs/<UTC time>_<command>.log;
# the exit code is a2d's. Activate the a2d env (and source configs/paths.env)
# first. Training stages start a private Ray head unless RAY_ADDRESS is set.
set -o pipefail
[ $# -gt 0 ] || { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
command -v a2d >/dev/null || { echo "a2d is not on PATH: activate the a2d env" >&2; exit 1; }

OUT=$(python -c "from ask2distill.paths import get_paths; print(get_paths().out)") || exit 1
words=()
for a in "$@"; do
    case "$a" in -*|*=*) break ;; esac
    words+=("$a")
done
name=$(IFS=_; echo "${words[*]:-a2d}")
LOG=$OUT/logs/$(date -u +%Y%m%dT%H%M%SZ)_${name}.log
mkdir -p "$(dirname "$LOG")"
echo "[run_stage] a2d $* -> $LOG" >&2
{ echo "# a2d $*"; echo "# host $(hostname -s)  start $(date -u +%FT%TZ)"; } > "$LOG"
a2d "$@" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "# exit $rc  end $(date -u +%FT%TZ)" >> "$LOG"
exit "$rc"
