#!/bin/bash
# tau2 smoke (pair Agent_Qwen3-4B_Qwen3-4B-Thinking-2507;
# configs/smoke/tau2_smoke.yaml) on ONE node with 2x A100-80GB.
#
#   bash tests/smoke/run_tau2.sh            # inside a 2-GPU allocation
#   bash tests/smoke/submit.sh tau2         # as a SLURM job
#
# GPU partitioning: GPU0 runs the vLLM fleet (servers start one at a time,
# --enforce-eager when they share the card), GPU1 runs Ray and the trainer; the
# env service runs on CPU under $A2D_TAU2_PYTHON. Served roles are the real 4B
# models; trained actors are Qwen3-1.7B in bf16 with the `student` template.
#
# Stages (about 80 min): tau2 data -> template pre-check -> synthesizer RL v7
# (2 steps) -> select + merge with the synthesizer template -> harvest v7 (8
# cards) -> banks (smoke pool, plus the CPU check that the released pools
# rebuild the pair's released bank exactly) -> OPD Ours (2 steps on the released
# bank, user simulator on GPU0) -> merge -> eval (the checkpoint as
# Qwen3-smoke-s2 on GPU1, Instruct-2507 served as `gpt-4o-mini` on GPU0, 2 tasks
# per domain, 1 trial) -> summarize. The eval numbers are meaningless and are
# labelled non-comparable: the user simulator is not OpenAI's.
set -o pipefail
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
CFG=$REPO/configs/smoke/tau2_smoke.yaml
PAIR=Agent_Qwen3-4B_Qwen3-4B-Thinking-2507
smoke_init tau2

[ -n "${A2D_TAU2_PYTHON:-}" ] || smoke_die "A2D_TAU2_PYTHON is not set (env/build_tau2_env.sh)"
for m in Qwen3-1.7B Qwen3-4B Qwen3-4B-Instruct-2507 Qwen3-4B-Thinking-2507 Qwen3-Embedding-4B; do
    smoke_require_file "$A2D_MODELS_DIR/$m/config.json" "a2d models download --pair $PAIR (and --pair Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500)"
done
for f in "banks/$PAIR.parquet" prompts/GENRL_TAU2_V7.parquet tau2/pool_v6s50.jsonl \
         tau2/pool_v7s60.jsonl eval/math/AIME24/test.parquet; do
    smoke_require_file "$A2D_DATA_DIR/$f" "a2d data download / a2d data build math"
done

IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
GPU0=${GPUS[0]}
GPU1=${GPUS[1]:?need two GPUs in CUDA_VISIBLE_DEVICES}
EPD=$A2D_OUT/endpoints
mkdir -p "$EPD"

# serve_bg <endpoints-file> <a2d tau2 serve args...>: start a fleet, wait until it is ready.
serve_bg() {
    local ep=$1; shift
    rm -f "$ep"
    a2d tau2 serve --config "$CFG" --out "$ep" "$@" > "$ep.log" 2>&1 &
    local pid=$!
    SMOKE_BG_PIDS+=("$pid")
    wait_file "$ep" 1800 "$pid"
    echo "[smoke] serving: $(awk '{print $1"="$3}' "$ep" | tr '\n' ' ')"
}

data() {
    if [ -s "$A2D_DATA_DIR/tau2/domain_ctx.json" ] && [ -s "$A2D_DATA_DIR/tau2/tasks_test/airline.json" ] \
            && python -c 'import sys; from ask2distill import config; from pathlib import Path; \
sys.exit(0 if (Path(str(config.load(None, sys.argv[1]).tau2_data.data_dir)) / "domains").is_dir() else 1)' "$CFG"; then
        echo "tau2 data present"
    else
        a2d data build tau2
    fi
}

template_check() {
    python - "$A2D_MODELS_DIR" <<'PY'
import json, sys
m = sys.argv[1]
a = json.load(open(f"{m}/Qwen3-1.7B/tokenizer_config.json"))["chat_template"]
b = json.load(open(f"{m}/Qwen3-4B/tokenizer_config.json"))["chat_template"]
assert a == b, "Qwen3-1.7B and Qwen3-4B chat templates differ: the `student` template would not fit"
print("Qwen3-1.7B chat_template == Qwen3-4B chat_template")
PY
}

synrl_v7() {
    serve_bg "$EPD/synrl.txt" --plan synrl --gpus "$GPU0" || return 1
    CUDA_VISIBLE_DEVICES=$GPU1 a2d tau2 synrl --config "$CFG" --version v7 --endpoints "$EPD/synrl.txt"
    local rc=$?
    smoke_stop_bg
    return $rc
}

select_merge() {
    a2d tau2 select --config "$CFG" --version v7 || return 1
    CUDA_VISIBLE_DEVICES=$GPU1 a2d merge --config "$CFG" \
        --actor-dir "$A2D_CKPT/synthesizer_${PAIR}_v7_smoke/global_step_2/actor" \
        --target-dir "$A2D_CKPT/synthesizer_${PAIR}_v7_smoke/hf/step_2" --chat-template synthesizer
}

harvest_v7() {
    serve_bg "$EPD/harvest.txt" --plan harvest --version v7 --gpus "$GPU0" || return 1
    a2d tau2 harvest --config "$CFG" --version v7 --endpoints "$EPD/harvest.txt"
    local rc=$?
    smoke_stop_bg
    return $rc
}

banks() {
    # Released pools -> released banks, then the smoke pool -> a parquet.
    a2d tau2 bank --config "$CFG" || return 1
    python - "$CFG" "$A2D_OUT/$PAIR/banks" "$A2D_DATA_DIR/banks" "$A2D_OUT/$PAIR/harvest/pool_v7_smoke.jsonl" <<'PY'
import hashlib, sys
from pathlib import Path
from ask2distill import config
from ask2distill.tau2.bank import build_synth
cfg = config.load(None, sys.argv[1])
built, released, pool = map(Path, sys.argv[2:])
for name in [str(p.out) for p in cfg.bank.parts] + [str(cfg.bank.mix)]:
    a, b = built / f"{name}.parquet", released / f"{name}.parquet"
    if b.is_file():
        same = hashlib.md5(a.read_bytes()).digest() == hashlib.md5(b.read_bytes()).digest()
        assert same, f"{a} differs from the released {b}"
        print(f"{name}: identical to the release")
n = build_synth([pool], built / "smoke_pool.parquet")
assert n > 0, "the smoke pool produced no bank rows"
print(f"smoke pool -> {n} rows")
PY
}

opd_ours() {
    serve_bg "$EPD/user.txt" --plan opd --arm ours --gpus "$GPU0" || return 1
    CUDA_VISIBLE_DEVICES=$GPU1 a2d tau2 opd --config "$CFG" --arm ours --user-endpoints "$EPD/user.txt"
    local rc=$?
    smoke_stop_bg
    return $rc
}

smoke_tasks() {
    # First eval.num_tasks tasks per domain, nl_assertions removed (see eval.tasks_dir
    # in tau2_smoke.yaml); refuses a task whose reward basis includes NL_ASSERTION.
    python - "$CFG" <<'PY'
import json, sys
from pathlib import Path
from ask2distill import config
cfg = config.load(None, sys.argv[1])
src, dst, n = Path(str(cfg.tau2_data.tasks_test)), Path(str(cfg.eval.tasks_dir)), int(cfg.eval.num_tasks)
assert src.resolve() != dst.resolve(), "eval.tasks_dir must not be the released test split"
dst.mkdir(parents=True, exist_ok=True)
for d in cfg.eval.domains:
    tasks = json.loads((src / f"{d}.json").read_text())[:n]
    for t in tasks:
        ec = t.get("evaluation_criteria") or {}
        if ec.get("nl_assertions"):
            assert "NL_ASSERTION" not in (ec.get("reward_basis") or []), f"{d} task {t['id']} is scored on NL assertions"
            ec["nl_assertions"] = None
    (dst / f"{d}.json").write_text(json.dumps(tasks, indent=2))
    print(f"{d}: tasks {[t['id'] for t in tasks]} -> {dst / (d + '.json')}")
PY
}

eval_smoke() {
    smoke_tasks || return 1
    serve_bg "$EPD/eval_stu.txt" --plan eval --gpus "$GPU1" \
        --model "$A2D_CKPT/opd_${PAIR}_smoke_ours/hf/step_2" --served-name Qwen3-smoke-s2 || return 1
    serve_bg "$EPD/eval_usr.txt" --plan eval --gpus "$GPU0" \
        --model "$A2D_MODELS_DIR/Qwen3-4B-Instruct-2507" --served-name gpt-4o-mini || return 1
    cat "$EPD/eval_stu.txt" "$EPD/eval_usr.txt" > "$EPD/eval.txt"
    a2d tau2 eval --config "$CFG" --endpoints "$EPD/eval.txt" --tag smoke
    local rc=$?
    smoke_stop_bg
    return $rc
}

stage data            data
stage template_check  template_check
stage synrl_v7        synrl_v7
expect "$A2D_CKPT/synthesizer_${PAIR}_v7_smoke/global_step_2/actor/*"
stage select_merge    select_merge
expect "$A2D_CKPT/synthesizer_${PAIR}_v7_smoke/hf/step_2/config.json" "$A2D_CKPT/synthesizer_${PAIR}_v7_smoke/hf/step_2/*.jinja"
stage harvest_v7      harvest_v7
expect "$A2D_OUT/$PAIR/harvest/pool_v7_smoke.jsonl"
stage banks           banks
expect "$A2D_OUT/$PAIR/banks/$PAIR.parquet"
stage opd_ours        opd_ours
expect "$A2D_CKPT/opd_${PAIR}_smoke_ours/global_step_2/actor/*"
stage merge           a2d merge --config "$CFG" --arm ours --step 2 --chat-template student
expect "$A2D_CKPT/opd_${PAIR}_smoke_ours/hf/step_2/config.json"
stage eval            eval_smoke
expect "$A2D_OUT/tau2/outputs/smoke/Qwen3-smoke-s2__*"
stage summarize       a2d tau2 summarize --config "$CFG" --models Qwen3-smoke-s2 \
                          --outputs-dir "$A2D_OUT/tau2/outputs/smoke" --out "$A2D_OUT/tau2_smoke_summary"
expect "$A2D_OUT/tau2_smoke_summary.json" "$A2D_OUT/tau2_smoke_summary.md"
expect_finite "$A2D_OUT/tau2_smoke_summary.json"

smoke_log "PASS tau2 smoke (plumbing only; the eval used a local user simulator: NOT comparable)"
