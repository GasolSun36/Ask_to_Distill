#!/bin/bash
# Code smoke (pair Code_Qwen3-4B_Qwen3-4B-Instruct-2507;
# configs/smoke/code_smoke.yaml) on ONE node with 2x A100-80GB.
#
#   bash tests/smoke/run_code.sh            # inside a 2-GPU allocation
#   bash tests/smoke/submit.sh code         # as a SLURM job
#
# It exercises the same code path as Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507
# (Qwen3 no-think, code domain, trunc_zero, 2 rolls); the 30B MoE and 32B
# teachers of the other two code pairs are covered by the render tests. Stages
# (about 90 min): calibrate (16 questions of the initial Qwen3-1.7B
# synthesizer, teacher on GPU1) -> synthesizer RL (Qwen3-1.7B actor; student
# Qwen3-4B and teacher Instruct-2507 in bf16 on GPU1, bleu diversity; 3 steps)
# -> sweep + pick -> harvest (2 x 64) -> 2 one-shard slices -> aggregate in
# agg_order [1, 0] -> report (three harvested questions with their topics, the
# filter yield, the diversity values) -> bank (32) -> OPD Ours (Qwen3-4B bf16,
# 2 steps on the pair's released bank) and Vanilla (1 step on
# OpenCodeReasoning-10k) -> merge -> eval --smoke (LCB --debug, HumanEval+,
# extra benches --limit 5) -> collect.
set -o pipefail
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
CFG=$REPO/configs/smoke/code_smoke.yaml
PAIR=Code_Qwen3-4B_Qwen3-4B-Instruct-2507
smoke_init code

for m in Qwen3-1.7B Qwen3-4B Qwen3-4B-Instruct-2507 Qwen3-Embedding-4B; do
    smoke_require_file "$A2D_MODELS_DIR/$m/config.json" "a2d models download --pair $PAIR (and --pair Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500 for Qwen3-1.7B)"
done
smoke_require_file "$A2D_DATA_DIR/banks/$PAIR.parquet" "a2d data download"
smoke_require_file "$A2D_DATA_DIR/real/OpenCodeReasoning-10k.parquet" "a2d data build ocr && a2d data build code-real"
smoke_require_file "$A2D_DATA_DIR/eval/code/extra_bench/apps_introductory.jsonl" "a2d data build code-eval"

P=$A2D_OUT/$PAIR

stage calibrate      a2d calibrate --config "$CFG"
expect "$P/calib/calib_delta_ref.json"
expect_finite "$P/calib/calib_delta_ref.json"

stage synrl          a2d synrl --config "$CFG"
expect "$P/prompts/SYNRL_TOPICS_CODE.parquet"
expect "$A2D_CKPT/synthesizer_${PAIR}_smoke/global_step_3/actor/*"

stage sweep          a2d select sweep --config "$CFG"
expect "$P/sweep/sweep_step3.jsonl"

stage pick           a2d select pick --config "$CFG"
expect "$P/pick.json"

stage harvest        a2d harvest --config "$CFG"
expect "$P/harvest/raw.s0.jsonl" "$P/harvest/raw.s1.jsonl"

stage filter         a2d filter --config "$CFG"
expect "$P/filter/sat.s0_questions.jsonl" "$P/filter/sat.s1_questions.jsonl"

stage aggregate      a2d aggregate --config "$CFG"
expect "$P/pool.jsonl"

stage report         smoke_report "$P/harvest/raw.s0.jsonl" "$P/synrl/synrl_dump_synthesizer_${PAIR}_smoke.jsonl" \
                         "$P/filter/sat.s0.json" "$P/filter/sat.s1.json"
cat "$SMOKE_STATE/report.log"

stage bank           a2d bank --config "$CFG" --allow-short
expect "$P/banks/${PAIR}_smoke.parquet"

stage opd_ours       a2d opd --config "$CFG" --arm ours
expect "$A2D_CKPT/opd_${PAIR}_smoke_ours/global_step_2/actor/*"

stage opd_vanilla    a2d opd --config "$CFG" --arm vanilla "opd.extra_overrides=[trainer.total_training_steps=1]"

stage merge          a2d merge --config "$CFG" --arm ours --step 2
expect "$A2D_CKPT/opd_${PAIR}_smoke_ours/hf/step_2/config.json"

stage eval           a2d eval code --config "$CFG" --arm ours --step 2 --smoke
RUN=$A2D_OUT/eval/code/opd_${PAIR}_smoke_ours_step2
expect "$RUN/livecodebench_v5_v6_eval.json" "$RUN/xb/*.json"

stage collect        a2d collect code --runs "$RUN" --out "$A2D_OUT/collect_code.tsv"
expect "$A2D_OUT/collect_code.tsv"
expect_finite "$RUN/livecodebench_v5_v6_eval.json"

smoke_log "PASS code smoke (plumbing only; numbers are not comparable to the paper)"
