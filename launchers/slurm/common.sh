#!/bin/bash
# Shared helpers for the SLURM launchers. Sourced, never executed:
#
#   source "$A2D_REPO/launchers/slurm/common.sh"
#
# What lives here and why:
#
# * Per-job Ray. Every job gets a private Ray: a short temp dir under /tmp and a
#   port block derived from the job id (`python -m ask2distill.runtime exports`,
#   the same arithmetic ask2distill/runtime.py uses, so shell and Python agree).
#   A fixed head port (6379) plus `ray stop --force` would, on shared nodes,
#   kill sibling jobs' clusters and collide on ports. Nothing here ever
#   stops a cluster it did not start.
# * Process environment (caches, NCCL / Ray / hydra settings) is exported before
#   `ray start` on every node, so every Ray worker inherits it; run-specific
#   values travel through `+ray_kwargs.ray_init.runtime_env.env_vars` instead.
# * Wall-clock end time from squeue's REMAINING time (%L), for verl's
#   save-before-kill path.
# * Opt-in NUMA interleave (A2D_NUMA_INTERLEAVE=1).

A2D_SLURM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
A2D_REPO=${A2D_REPO:-$(cd "$A2D_SLURM_DIR/../.." && pwd)}
export A2D_REPO

A2D_SERVE_PIDS=()
A2D_RAY_WORKER_PIDS=()

a2d_die() { echo "[a2d] FATAL: $*" >&2; exit 1; }
a2d_log() { echo "[a2d $(date -u +%H:%M:%SZ)] $*"; }

# ---------------------------------------------------------------- configuration

# Load launchers/slurm/cluster.env (or $A2D_CLUSTER_ENV) and check what is required.
a2d_load_cluster_env() {
    local f=${A2D_CLUSTER_ENV:-$A2D_SLURM_DIR/cluster.env}
    [ -f "$f" ] || a2d_die "no $f -- copy cluster.env.example to cluster.env and fill it in"
    # shellcheck disable=SC1090
    source "$f"
    local v
    for v in ACCOUNT PARTITION_BIG PARTITION_SMALL TIME CONDA_SH A2D_CONDA_ENV; do
        [ -n "${!v:-}" ] || a2d_die "$v is not set in $f"
        case "${!v}" in "<"*">"*) a2d_die "$v in $f still holds the placeholder ${!v}";; esac
    done
    CPUS_PER_TASK=${CPUS_PER_TASK:-64}
    SMALL_CPUS_PER_TASK=${SMALL_CPUS_PER_TASK:-16}
    SMALL_MEM=${SMALL_MEM:-200G}
}

# Activate the a2d env and the user's A2D_* paths.
a2d_activate() {
    # shellcheck disable=SC1090
    source "$CONDA_SH" || a2d_die "cannot source $CONDA_SH"
    conda activate "$A2D_CONDA_ENV" || a2d_die "cannot activate conda env $A2D_CONDA_ENV"
    if [ -n "${A2D_PATHS_ENV:-}" ]; then
        # shellcheck disable=SC1090
        source "$A2D_PATHS_ENV" || a2d_die "cannot source $A2D_PATHS_ENV"
    fi
    # flashinfer JIT-compiles vLLM's sampling kernels and needs the CUDA dev
    # headers of the env; without CUDA_HOME the vLLM engine dies at startup
    # with "FileNotFoundError: 'nvcc'" / "fatal error: curand.h".
    export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
    command -v a2d >/dev/null || a2d_die "a2d is not on PATH in env $A2D_CONDA_ENV (pip install -e . --no-deps)"
}

# Export the process environment of a stage plus this job's port block
# (A2D_PORT_*, VLLM_PORT, MASTER_PORT) and A2D_RAY_TMPDIR.
a2d_job_exports() {
    local stage=$1 out
    out=$(python -m ask2distill.runtime exports --stage "$stage" --jobkey "${SLURM_JOB_ID:?}") \
        || a2d_die "ask2distill.runtime exports failed"
    eval "$out"
}

# Everything a batch script does before its work: cluster env, conda, exports.
a2d_job_setup() {
    local stage=$1
    a2d_load_cluster_env
    a2d_activate
    a2d_job_exports "$stage"
    A2D_JOB_DIR=$(dirname "${A2D_CONFIG:?submit through launchers/slurm/submit.sh}")/job_${SLURM_JOB_ID}
    mkdir -p "$A2D_JOB_DIR"
    export A2D_JOB_DIR
    a2d_log "job $SLURM_JOB_ID stage=$stage config=$A2D_CONFIG host=$(hostname -s)"
}

# The base_env stage for an a2d command line (first one or two words).
a2d_stage_of() {
    case "$1 ${2:-}" in
        "synrl "*) echo synrl ;;
        "opd "*) echo opd ;;
        "eval code") echo eval_code ;;
        "eval math") echo eval_math ;;
        "tau2 synrl") echo tau2_synrl ;;
        "tau2 opd") echo opd ;;
        "tau2 "*) echo tau2_serve ;;
        *) echo "$1" ;;
    esac
}

# Value following a flag in an argument list: a2d_arg_value --version "$@"
a2d_arg_value() {
    local flag=$1; shift
    while [ $# -gt 0 ]; do
        case "$1" in
            "$flag") echo "${2:-}"; return 0 ;;
            "$flag="*) echo "${1#*=}"; return 0 ;;
        esac
        shift
    done
    return 1
}

# ---------------------------------------------------------------------- nodes

a2d_nodes() { mapfile -t A2D_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST"); }

a2d_node_ip() {
    local ip
    ip=$(getent hosts "$1" | awk '{print $1}' | head -1)
    echo "${ip:-$1}"
}

# ------------------------------------------------------------------------ Ray

# Build this job's Ray cluster over the first N nodes: head here (node 0),
# workers on nodes 1..N-1. Exports RAY_ADDRESS for the a2d launcher, which then
# only attaches.
a2d_ray_cluster() {
    local nnodes=$1 gpus=$2 head_ip i cpus=$((CPUS_PER_TASK - 4))
    [ "${#A2D_NODES[@]}" -ge "$nnodes" ] || a2d_die "need $nnodes nodes, got ${#A2D_NODES[@]}"
    head_ip=$(a2d_node_ip "${A2D_NODES[0]}")
    local ports=(--node-manager-port="$A2D_PORT_NODE_MANAGER_PORT"
                 --object-manager-port="$A2D_PORT_OBJECT_MANAGER_PORT"
                 --dashboard-agent-listen-port="$A2D_PORT_DASHBOARD_AGENT_LISTEN_PORT"
                 --dashboard-agent-grpc-port="$A2D_PORT_DASHBOARD_AGENT_GRPC_PORT"
                 --runtime-env-agent-port="$A2D_PORT_RUNTIME_ENV_AGENT_PORT"
                 --metrics-export-port="$A2D_PORT_METRICS_EXPORT_PORT"
                 --min-worker-port="$A2D_PORT_MIN_WORKER_PORT"
                 --max-worker-port="$A2D_PORT_MAX_WORKER_PORT")
    a2d_log "ray head ${A2D_NODES[0]} ($head_ip:$A2D_PORT_RAY_PORT), temp dir $A2D_RAY_TMPDIR"
    # The batch script already runs on node 0, so the head starts here directly.
    # An `srun ... ray start --block` on the head would hold that node's only job
    # step and every later srun sits in "step creation still disabled" forever --
    # which is exactly how the first attempt spun for two hours without training
    # a thing. Only the remote workers need srun, and they take --overlap so they
    # can share the allocation.
    ray start --head --node-ip-address="$head_ip" --port="$A2D_PORT_RAY_PORT" \
        --temp-dir="$A2D_RAY_TMPDIR" --ray-client-server-port="$A2D_PORT_RAY_CLIENT_SERVER_PORT" \
        "${ports[@]}" --include-dashboard=false --disable-usage-stats \
        --num-cpus="$cpus" --num-gpus="$gpus" || a2d_die "RAY_START_FAILED"
    A2D_RAY_HEAD_STARTED=1
    A2D_RAY_WORKER_PIDS=()
    for ((i = 1; i < nnodes; i++)); do
        srun --nodes=1 --ntasks=1 -w "${A2D_NODES[$i]}" --overlap \
            ray start --address="$head_ip:$A2D_PORT_RAY_PORT" --temp-dir="$A2D_RAY_TMPDIR" \
            "${ports[@]}" --disable-usage-stats --num-cpus="$cpus" --num-gpus="$gpus" --block &
        A2D_RAY_WORKER_PIDS+=($!)
    done
    export RAY_ADDRESS="$head_ip:$A2D_PORT_RAY_PORT"
    local t want=$((nnodes * gpus)) have
    for t in $(seq 1 60); do
        have=$(python - <<'PY' 2>/dev/null
import ray
ray.init(address="auto", logging_level="ERROR")
print(int(ray.cluster_resources().get("GPU", 0)))
PY
)
        [ "${have:-0}" -ge "$want" ] && break
        sleep 5
    done
    [ "${have:-0}" -ge "$want" ] || a2d_die "RAY_CLUSTER_FAILED: $have of $want GPUs joined"
    a2d_log "ray cluster up: $have GPUs on $nnodes node(s), RAY_ADDRESS=$RAY_ADDRESS"
}

# Stop the processes of THIS job's head (matched by its temp dir) and the worker
# steps. Never `ray stop --force`: that signals every Ray process of the user on
# the node, sibling jobs included.
a2d_ray_teardown() {
    local p
    for p in "${A2D_RAY_WORKER_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
    if [ "${A2D_RAY_HEAD_STARTED:-0}" = 1 ]; then
        python -c "from ask2distill.runtime import stop_own_head; stop_own_head('$A2D_RAY_TMPDIR')" \
            2>/dev/null || true
        A2D_RAY_HEAD_STARTED=0
    fi
}

# ------------------------------------------------------------------ wall clock

# Seconds in squeue's %L field: [D-]HH:MM:SS, MM:SS or SS.
a2d_rem_secs() {
    local d=0 t=$1 h=0 m=0 s=0
    [[ $t == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
    IFS=: read -r -a parts <<<"$t"
    case ${#parts[@]} in
        3) h=${parts[0]}; m=${parts[1]}; s=${parts[2]} ;;
        2) m=${parts[0]}; s=${parts[1]} ;;
        1) s=${parts[0]} ;;
    esac
    echo $(( d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

# Unix time this job is killed, from squeue's REMAINING time -- not scontrol's
# EndTime, which is printed in the cluster's local zone while `date -d` reads it
# as UTC: one restart logged a wall clock "ending" three hours before the job
# began, and verl would have seen negative remaining time and force-saved every
# step. Prints nothing when unknown.
a2d_wall_clock_end() {
    local rem
    rem=$(squeue -h -j "${SLURM_JOB_ID:?}" -o "%L" 2>/dev/null | head -1)
    case "$rem" in ""|INVALID|UNLIMITED|NOT_SET) return 0 ;; esac
    echo $(( $(date +%s) + $(a2d_rem_secs "$rem") ))
}

# ---------------------------------------------------------------- execution

# Command prefix that interleaves host memory over all NUMA nodes when
# A2D_NUMA_INTERLEAVE=1. A job step that only gets the CPUs of some NUMA nodes
# lets first-touch put ~1 TB of offloaded weights on them; those nodes filled,
# kswapd thrashed and the trainer stalled 30+ minutes while the other nodes had
# 1 TB free. No numactl on the nodes: set_mempolicy(MPOL_INTERLEAVE) via syscall
# 238 (x86_64), inherited across fork/exec by Ray and every worker.
a2d_exec_prefix() {
    A2D_EXEC_PREFIX=()
    [ "${A2D_NUMA_INTERLEAVE:-0}" = 1 ] || return 0
    A2D_EXEC_PREFIX=(python -c '
import ctypes, os, sys
n = len([d for d in os.listdir("/sys/devices/system/node") if d.startswith("node") and d[4:].isdigit()])
mask = ctypes.c_ulong((1 << n) - 1)
rc = ctypes.CDLL(None, use_errno=True).syscall(238, 3, ctypes.byref(mask), 64)
print(f"[a2d] set_mempolicy interleave over {n} NUMA nodes rc={rc}", flush=True)
os.execvp(sys.argv[1], sys.argv[1:])')
}

# Run an a2d command line, through the NUMA prefix when enabled.
a2d_run() {
    a2d_exec_prefix
    a2d_log "a2d $*"
    "${A2D_EXEC_PREFIX[@]}" a2d "$@"
}

# ---------------------------------------------------------------- tau2 serving

# Start `a2d tau2 serve` on one node as a background step. The serve writes its
# endpoints file ("kind url served_name" lines) once every server is ready.
# A2D_SERVE_NODE_INDEX tells the plan which node map applies (synrl uses
# synrl.serve.node_maps[i]).
#   a2d_serve_on_node <index> <node> <endpoints-file> <a2d tau2 serve args...>
a2d_serve_on_node() {
    local idx=$1 node=$2 out=$3; shift 3
    rm -f "$out"
    srun --nodes=1 --ntasks=1 -w "$node" --gpus-per-node=8 --cpus-per-task="$CPUS_PER_TASK" \
        --mem=0 --exact \
        env A2D_SERVE_NODE_INDEX="$idx" a2d tau2 serve --out "$out" "$@" \
        > "$A2D_JOB_DIR/serve.node$idx.log" 2>&1 &
    A2D_SERVE_PIDS+=($!)
    a2d_log "serve step on $node (index $idx) -> $out"
}

# Wait until every endpoints file exists and is non-empty, or a serve step dies.
#   a2d_wait_endpoints <timeout_s> <file>...
a2d_wait_endpoints() {
    local timeout=$1; shift
    local deadline=$(( $(date +%s) + timeout )) f ready p
    while :; do
        ready=1
        for f in "$@"; do [ -s "$f" ] || ready=0; done
        [ "$ready" = 1 ] && return 0
        for p in "${A2D_SERVE_PIDS[@]}"; do
            kill -0 "$p" 2>/dev/null || a2d_die "a serve step exited before it was ready (see $A2D_JOB_DIR/serve.node*.log)"
        done
        [ "$(date +%s)" -lt "$deadline" ] || a2d_die "serve not ready after ${timeout}s (see $A2D_JOB_DIR/serve.node*.log)"
        sleep 30
    done
}

# Concatenate per-node endpoints files into one.
#   a2d_collect_endpoints <out> <file>...
a2d_collect_endpoints() {
    local out=$1; shift
    cat "$@" > "$out.tmp" && mv "$out.tmp" "$out"
    a2d_log "endpoints: $(awk '{print $1}' "$out" | sort | uniq -c | tr '\n' ' ')"
}

a2d_stop_serves() {
    local p
    for p in "${A2D_SERVE_PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
    wait 2>/dev/null || true
}
