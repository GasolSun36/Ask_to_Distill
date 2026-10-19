#!/bin/bash
# Helpers for the smoke scripts (tests/smoke/run_<domain>.sh). Sourced, never executed.
#
# A smoke run checks PLUMBING ONLY: every stage exits 0, writes its declared
# outputs and logs finite metrics. No number is compared with the paper (2 GPUs
# change packing, dtypes and model sizes; see configs/smoke/*.yaml).
#
# Each stage leaves a marker in $SMOKE_STATE; a resubmitted job skips the
# stages that already passed, so a run longer than the wall clock resumes.
# Outputs are isolated from real runs: A2D_OUT and A2D_CKPT are redirected to
# <A2D_OUT>/smoke/<domain> and <A2D_CKPT>/smoke/<domain>; data, models and
# caches are shared.
set -o pipefail

SMOKE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SMOKE_DIR/../.." && pwd)

smoke_die() { echo "[smoke] FAIL: $*" >&2; exit 1; }
smoke_log() { echo "[smoke $(date -u +%H:%M:%SZ)] $*"; }

# smoke_init <domain>: isolate outputs, create the state directory, check the env.
smoke_init() {
    SMOKE_DOMAIN=$1
    command -v a2d >/dev/null || smoke_die "a2d is not on PATH (activate the a2d env)"
    local paths
    paths=$(python -c 'from ask2distill.paths import get_paths as g; p = g(); print(p.out, p.ckpt, p.data, p.models)') \
        || smoke_die "cannot resolve A2D_* paths"
    read -r BASE_OUT BASE_CKPT A2D_DATA_DIR A2D_MODELS_DIR <<< "$paths"
    export A2D_OUT=${A2D_SMOKE_OUT:-$BASE_OUT/smoke/$SMOKE_DOMAIN}
    export A2D_CKPT=${A2D_SMOKE_CKPT:-$BASE_CKPT/smoke/$SMOKE_DOMAIN}
    SMOKE_STATE=$A2D_OUT/_state
    mkdir -p "$SMOKE_STATE" "$A2D_CKPT"
    export HYDRA_FULL_ERROR=1
    smoke_log "domain=$SMOKE_DOMAIN out=$A2D_OUT ckpt=$A2D_CKPT host=$(hostname -s) job=${SLURM_JOB_ID:-local}"
    if command -v nvidia-smi >/dev/null; then
        local n
        n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
        [ "$n" -ge 2 ] || smoke_die "needs 2 GPUs, found $n"
    else
        smoke_die "no nvidia-smi: run on a GPU node"
    fi
}

# smoke_require_file <path> <hint>: an input the smoke needs (models, released data).
smoke_require_file() { [ -e "$1" ] || smoke_die "missing input $1 ($2)"; }

# stage <name> <command...>: run once; log to $SMOKE_STATE/<name>.log; mark on success.
stage() {
    local name=$1; shift
    local marker=$SMOKE_STATE/$name.done log=$SMOKE_STATE/$name.log t0
    if [ -f "$marker" ]; then
        smoke_log "skip  $name (passed earlier)"
        return 0
    fi
    smoke_log "start $name: $*"
    t0=$(date +%s)
    "$@" > "$log" 2>&1
    local rc=$?
    if [ $rc -ne 0 ]; then
        tail -n 40 "$log" >&2
        smoke_die "stage $name exited $rc (log $log)"
    fi
    date -u +%FT%TZ > "$marker"
    smoke_log "done  $name in $(( $(date +%s) - t0 ))s"
}

# expect <glob>...: every glob matches at least one non-empty file.
expect() {
    local g f found
    for g in "$@"; do
        found=0
        for f in $g; do [ -s "$f" ] && found=1 && break; done
        [ $found = 1 ] || smoke_die "expected output missing or empty: $g"
    done
}

# expect_finite <file> [<file>...]: every number in the JSON / TSV files is finite.
expect_finite() {
    python - "$@" <<'PY' || smoke_die "non-finite metric in $*"
import json, math, re, sys
bad = []
def walk(x, where):
    if isinstance(x, float) and not math.isfinite(x):
        bad.append(where)
    elif isinstance(x, dict):
        for k, v in x.items():
            walk(v, f"{where}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            walk(v, f"{where}[{i}]")
for path in sys.argv[1:]:
    text = open(path).read()
    if path.endswith(".json"):
        walk(json.loads(text), path)
    else:
        for tok in re.split(r"[\s,]+", text):
            if tok.lower() in ("nan", "inf", "-inf", "+inf"):
                bad.append(f"{path}: {tok}")
if bad:
    sys.exit("non-finite: " + ", ".join(bad[:10]))
PY
}

# smoke_report <harvest shard> <dump prefix> <sat.s*.json>...: print three harvested
# questions with their topics and the filter yield, and check the synthesizer-RL
# diversity values: every per-question share finite and in (0, 1], the per-step
# mean pairwise similarity finite and not constant across steps.
smoke_report() {
    python - "$@" <<'PY' || smoke_die "smoke report failed"
import glob, json, math, sys
shard, dump, sats = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = [json.loads(line) for line in open(shard, encoding="utf-8")]
assert rows and all(r.get("topic") for r in rows), "harvest rows carry no topic"
for i, r in enumerate(rows[:3], 1):
    q = " ".join(r["problem"].split())
    print(f"[smoke] harvested {i} (topic: {r['topic']}): {q[:300]}{' ...' if len(q) > 300 else ''}")
raw = sum(json.load(open(s))["raw"] for s in sats)
kept = sum(json.load(open(s))["emb"] for s in sats)
print(f"[smoke] filter yield: {kept}/{raw} = {100 * kept / max(raw, 1):.1f}% of the harvested questions kept")
shares = [json.loads(line)["share"] for f in sorted(glob.glob(dump + ".r*")) if ".metrics." not in f
          for line in open(f, encoding="utf-8") if "share" in json.loads(line)]
sims = [json.loads(line)["synrm/mean_sim"] for f in sorted(glob.glob(dump + ".metrics.r*"))
        for line in open(f, encoding="utf-8")]
assert shares, f"no diversity share in {dump}.r*"
assert all(math.isfinite(s) and 0 < s <= 1 for s in shares), "share outside (0, 1]"
assert sims and all(math.isfinite(s) for s in sims), "non-finite synrm/mean_sim"
assert len(set(sims)) > 1, f"synrm/mean_sim is constant across steps: {sims}"
print(f"[smoke] bleu diversity: {len(shares)} shares in [{min(shares):.4f}, {max(shares):.4f}], "
      f"mean_sim per step {[round(s, 4) for s in sims]}")
PY
}

# wait_file <path> <timeout_s> [<pid>]: wait for a file (an endpoints list) to appear.
wait_file() {
    local f=$1 timeout=$2 pid=${3:-} deadline
    deadline=$(( $(date +%s) + timeout ))
    while [ ! -s "$f" ]; do
        [ -z "$pid" ] || kill -0 "$pid" 2>/dev/null || smoke_die "process $pid exited before $f appeared"
        [ "$(date +%s)" -lt "$deadline" ] || smoke_die "$f did not appear within ${timeout}s"
        sleep 15
    done
}

# Background servers started by a script are stopped on exit.
SMOKE_BG_PIDS=()
smoke_stop_bg() {
    local p
    for p in "${SMOKE_BG_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
    for p in "${SMOKE_BG_PIDS[@]:-}"; do [ -n "$p" ] && wait "$p" 2>/dev/null; done
    SMOKE_BG_PIDS=()
}
trap smoke_stop_bg EXIT
