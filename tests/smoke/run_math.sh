#!/bin/bash
# Math smoke (pair Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500;
# configs/smoke/math_smoke.yaml) on ONE node with 2x A100-80GB.
#
#   bash tests/smoke/run_math.sh            # inside a 2-GPU allocation
#   bash tests/smoke/submit.sh math         # as a SLURM job (launchers/slurm/cluster.env)
#
# Stages (about 75 min): calibrate (16 questions of the initial synthesizer,
# student vLLM on GPU0, teacher on GPU1) -> synthesizer RL (topic prompts built
# on the fly, bleu diversity; 1 actor GPU + 1 reward GPU, 3 steps) -> sweep +
# pick (relaxed gates) -> harvest (2 x 64) -> one-slice filter -> report (three
# harvested questions with their topics, the filter yield, the diversity
# values) -> bank (32, short allowed) -> OPD Ours (2 steps on the pair's
# released bank) and Vanilla (1 step on DAPO-Math-17k, no save) -> merge -> eval
# (AIME24 + MATH500, 5 problems, n 2) -> collect. Requires: Qwen3-1.7B,
# Qwen3-4B-Non-Thinking-RL-Math-Step500 and Qwen3-Embedding-4B (a2d models
# download --pair <pair>), the pair's bank (a2d data download) and
# DAPO-Math-17k / math test sets (a2d data build math).
set -o pipefail
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
CFG=$REPO/configs/smoke/math_smoke.yaml
PAIR=Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500
smoke_init math

smoke_require_file "$A2D_MODELS_DIR/Qwen3-1.7B/config.json" "a2d models download --pair $PAIR"
smoke_require_file "$A2D_MODELS_DIR/Qwen3-4B-Non-Thinking-RL-Math-Step500/config.json" "a2d models download --pair $PAIR"
smoke_require_file "$A2D_MODELS_DIR/Qwen3-Embedding-4B/config.json" "a2d models download --pair $PAIR"
smoke_require_file "$A2D_DATA_DIR/banks/$PAIR.parquet" "a2d data download"
smoke_require_file "$A2D_DATA_DIR/real/DAPO-Math-17k.parquet" "a2d data build math"
smoke_require_file "$A2D_DATA_DIR/eval/math/AIME24/test.parquet" "a2d data build math"

P=$A2D_OUT/$PAIR

stage calibrate      a2d calibrate --config "$CFG"
expect "$P/calib/calib_delta_ref.json"
expect_finite "$P/calib/calib_delta_ref.json"

stage synrl          a2d synrl --config "$CFG"
expect "$P/prompts/SYNRL_TOPICS_MATH.parquet"
expect "$A2D_CKPT/synthesizer_${PAIR}_smoke/global_step_3/actor/*"

stage sweep          a2d select sweep --config "$CFG"
expect "$P/sweep/sweep_step3.jsonl"

stage pick           a2d select pick --config "$CFG"
expect "$P/pick.json"
expect_finite "$P/pick.json"

stage harvest        a2d harvest --config "$CFG"
expect "$P/harvest/raw.s0.jsonl" "$P/harvest/raw.s1.jsonl"

stage filter         a2d filter --config "$CFG"
expect "$P/filter/sat.s0.json" "$P/filter/sat.s0_questions.jsonl"

stage report         smoke_report "$P/harvest/raw.s0.jsonl" "$P/synrl/synrl_dump_synthesizer_${PAIR}_smoke.jsonl" \
                         "$P/filter/sat.s0.json"
cat "$SMOKE_STATE/report.log"

stage bank           a2d bank --config "$CFG" --allow-short
expect "$P/banks/${PAIR}_smoke.parquet"

stage opd_ours       a2d opd --config "$CFG" --arm ours
expect "$A2D_CKPT/opd_${PAIR}_smoke_ours/global_step_2/actor/*"

stage opd_vanilla    a2d opd --config "$CFG" --arm vanilla "opd.extra_overrides=[trainer.total_training_steps=1]"

stage merge          a2d merge --config "$CFG" --arm ours --step 2
expect "$A2D_CKPT/opd_${PAIR}_smoke_ours/hf/step_2/config.json"

stage eval           a2d eval math --config "$CFG" --arm ours --step 2
RUN=$A2D_OUT/eval/math/opd_${PAIR}_smoke_ours_step2
expect "$RUN/grading_results.json" "$RUN/*.jsonl"

stage collect        a2d collect math --runs "$RUN" --out "$A2D_OUT/collect_math.tsv"
expect "$A2D_OUT/collect_math.tsv"
# Only the per-task scores must be finite: avg6 over a 2-task smoke is undefined.
expect_finite "$RUN/grading_results.json"

smoke_log "PASS math smoke (plumbing only; numbers are not comparable to the paper)"
