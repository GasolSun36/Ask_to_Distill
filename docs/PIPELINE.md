# Pipeline

Every stage is one `a2d` command. Locally it runs in the current allocation; on SLURM,
`launchers/slurm/submit.sh <stage> <pair> [options]` resolves the configuration to a file, submits
the matching batch script and passes only that file's path to the job. Long stages resume across
the 4-hour wall clock: `launchers/slurm/resubmit.sh` resubmits a training stage until its
checkpoint tracker reaches a target step.

```
 [0 calibrate]   the initial synthesizer writes N_cal = 200 questions, the student answers them,
       |         student and teacher are teacher-forced on the answers; δ_ref = p95(δ_t)
 [1 synrl]       synthesizer S_θ (initialised from the student) writes 64 questions per step on
       |         rotated topics; reward pool (frozen π_S + π_T, 8 ranks): the student solves, both
       |         models score the same tokens, s_t = min(δ_t / δ_ref, 1) * ω_t, aggregated with a
       |         length term, x (1 - λ_div * share) over the step's BLEU clusters
       |         -> GRPO, 60 steps, a checkpoint every 5 (10 for the Qwen3-4B-Instruct code pair)
 [2 select]      merge + sample 300 questions per checkpoint (sweep) -> pick (diversity gate,
       |         reward gate, lowest cosine) -> math: saturation yield of frozen candidates
 [3 harvest]     the chosen S_θ samples 40k-96k questions (1 GPU per shard)
 [3 filter]      per slice: format gate (+ math anti-leak) -> Jaccard 0.7 -> quality filter ->
       |         embedding dedup 0.95; across slices: embedding dedup 0.95 in filter.agg_order
 [3 bank]        first N questions -> verl parquet (DAPO prefix / LCB template), ground_truth "N/A"
       |
 [4 opd]         top-16 on-policy distillation on the bank (8 GPUs); Vanilla: same recipe on
       |         real questions
 [5 merge]       FSDP checkpoint -> HF directory
 [6 eval]        math: avg@16 (MATH500 avg@4) on 6 tasks; code: LCB v5_v6 + MBPP+ + APPS x 3 +
                 CodeContests; tau2: pass^1 on the official test split
 [7 collect]     aggregate metrics (math avg6, code Avg, tau2 mean)
```

The released banks (`a2d data download`) are the OPD default, so stage 4 onwards reproduces the
main table without stages 0-3. A new pair runs every stage (README, "Using your own
student-teacher pair").

## Math and code (the `Math_*` and `Code_*` pairs)

| Stage | Command | SLURM | Inputs -> outputs | Resources (final) |
|---|---|---|---|---|
| 0 calibration | `a2d calibrate --pair P` | `submit.sh calibrate P` | synthesizer init, student, teacher -> `$A2D_OUT/P/calib/calib_questions.jsonl` (N_cal questions with their topics; reused when present), `$A2D_OUT/P/calib/calib_delta_ref.json` (percentile, delta_ref, delta_ref_rounded, n_questions, n_rollouts, n_tokens, p50, p75, p90, p99, mean) | 1 GPU (math); 2 GPUs (code: teacher on `cuda:1`) |
| prompts | `a2d prompts build --pair P` (also run by `a2d synrl` when the file is missing) | `submit.sh prompts P` | `synrl.topics`, `synrl.prompt_seed` -> `$A2D_OUT/P/prompts/SYNRL_TOPICS_<DOMAIN>.parquet` (`synrl.prompt_rows` rows, one rotated topic each) | CPU |
| 1 synthesizer RL | `a2d synrl --pair P` | `submit.sh synrl P` + `resubmit.sh` | prompts, synthesizer init, student, teacher -> `$A2D_CKPT/<exp>/global_step_*`, reward dumps `$A2D_OUT/P/synrl/synrl_dump_<exp>.jsonl.r*` (+ `.metrics.r*`) | 2 nodes x 8 GPUs: node 0 trains S_θ, node 1 is the reward pool |
| 2a sweep | `a2d select sweep --pair P` | `submit.sh sweep P` | `global_step_*` -> `$A2D_CKPT/<exp>/hf/step_N`, `$A2D_OUT/P/sweep/sweep_stepN.jsonl` (raw text, `.strip()` only) | 2 GPUs |
| 2b pick | `a2d select pick --pair P` | `submit.sh pick P` | dumps + sweeps -> `$A2D_OUT/P/pick.json` | 1 GPU (embedder) |
| 2c yield (two math pairs, see below) | `a2d select yield --pair P` | `submit.sh yield P` | frozen `select.yield.candidates` -> `$A2D_OUT/P/yield.json` | 1 GPU per candidate |
| 3a harvest | `a2d harvest --pair P [--shard i]` | `submit.sh harvest P --shard i` | `harvest.synthesizer` (default `hf/step_<chosen_step>`) -> `$A2D_OUT/P/harvest/raw.s<i>.jsonl` (question and topic per row) | 1 GPU per shard |
| 3b filter | `a2d filter --pair P [--slice k]` | `submit.sh filter P --slice k` | shards of `filter.slices[k]` -> `$A2D_OUT/P/filter/sat.s<k>.json` + `_questions.jsonl` | 1 GPU per slice |
| 3c aggregate | `a2d aggregate --pair P` (none for `Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500`) | `submit.sh aggregate P` | slice question files in `filter.agg_order` -> `$A2D_OUT/P/pool.jsonl` | 1 GPU |
| 3d bank | `a2d bank --pair P` | `submit.sh bank P` | `bank.pool` -> `$A2D_OUT/P/banks/P.parquet` (`Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507` re-filters the pool first) | CPU |
| 4 OPD | `a2d opd --pair P --arm ours\|vanilla` | `submit.sh opd P --arm A` + `resubmit.sh` | bank (default: released) or real parquet -> `$A2D_CKPT/<exp>/global_step_*` | 1 node x 8 GPUs |
| 5 merge | `a2d merge --pair P --arm A --step N` | `submit.sh merge P --arm A --step N` | `global_step_N/actor` -> `$A2D_CKPT/<exp>/hf/step_N` | a compute node (a 4B fp32 merge exceeds a 24 GB login-node limit) |
| 6 eval, math | `a2d eval math --pair P --arm A --step N [--tasks T ...]` | `submit.sh eval-math P --arm A --step N --tasks T` | merged model, `$A2D_DATA/eval/math/<dir>/test.parquet` -> `$A2D_OUT/eval/math/<exp>_step<N>/{<task>_t0.7_p0.95_n16-MNT31744.jsonl, grading_results.json}` | 1 GPU per task |
| 6 eval, code | `a2d eval code --pair P --arm A --step N [--curve]` | `submit.sh eval-code P --arm A --step N` | merged model, LCB cache, extra_bench -> `$A2D_OUT/eval/code/<exp>_step<N>/{livecodebench_v5_v6_eval.json, evalplus/, evalplus_mbpp_lenient.log, xb/<bench>.json}` | 2 GPUs (TP 2) |
| 7 collect | `a2d collect math\|code --runs DIR ... [--out FILE]` | | eval directories -> TSV / JSON with avg6 / Avg | CPU |

Per pair, the stage parameters that differ (all in `configs/pairs/<pair>.yaml`):

| Pair | δ_ref | Synthesizer step | Harvest | Slices | Bank rows | OPD Ours steps |
|---|---|---|---|---|---|---|
| `Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500` | 1.9049 | 50 | 16 x 5000, seed `i` | one 80k pass | 2,889 | 90 |
| `Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B` | 0.4411 | 40 | 16 x 5000, seed `i` | 4 x 4 shards | 12,800 | 400 |
| `Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B` | 0.4867 | 55 | 8 x 5000, seed `i` | 2 x 4 | 12,800 | 400 |
| `Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507` | 3.9756 | 60 | 12 x 8000, seed `7919i+104729` | 4 x 3 | 10,000 | 156 |
| `Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct` | 1.3629 | 40 | 12 x 8000, seed `7919i+104729` | 4 x 3 | 10,000 | 155 |
| `Code_Qwen3-4B_Qwen3-4B-Instruct-2507` | 3.8221 | 50 | 12 x 8000, seed `7919i+104729` | 12 x 1 (lexicographic aggregation) | 10,000 | 156 |

The bank of each pair is named after it (`banks/<pair>.parquet`). The two `Math_DeepSeek-*`
synthesizers start from the `*-SynNoThink` bases (`a2d models syn-nothink`); the OPD arms of
`Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct` run with KL 0.005 and entropy 0.005 (without
them the student collapsed).

Synthesizer step, harvest, slices, bank rows and OPD training steps are those of the released
banks and checkpoints; the δ_ref column is `reward.delta_ref` (`a2d calibrate`).

**Re-harvesting needs a synthesizer.** The math synthesizers at `select.chosen_step` were kept
(docs/DATA.md); point `harvest.synthesizer` at one to skip stages 1-2. The code and tau2 synthesizers were not
kept and must be retrained (about 60 steps x 350-490 s on 16 GPUs per code pair).

## tau2

See docs/TAU2.md for the two-env layout, serving and the login-node evaluation. `P` is the
agent pair, `Agent_Qwen3-4B_Qwen3-4B-Thinking-2507`.

| Stage | Command | Process |
|---|---|---|
| data | `a2d data build tau2` | a2d env; spawns `envside.dump_domain_ctx` under `$A2D_TAU2_PYTHON` |
| synthesizer RL | `a2d tau2 synrl --pair P --version vX --endpoints FILE` (`submit.sh tau2-synrl P --version vX`) | trainer (a2d env) + env service (tau2 env) + vLLM fleet |
| select + merge | `a2d tau2 select --pair P --version vX`; `a2d merge ... --chat-template synthesizer` | |
| harvest | `a2d tau2 harvest --pair P --version v6\|v7 --endpoints FILE` (`submit.sh tau2-harvest`) | env side |
| bank | `a2d tau2 bank --pair P` | a2d env |
| OPD | `a2d tau2 opd --pair P --arm ours\|vanilla --user-endpoints FILE` (`submit.sh tau2-opd P --arm A`) | trainer + env service + user-simulator fleet |
| eval | `a2d tau2 serve --plan eval ...` (compute node), `a2d tau2 eval --endpoints FILE` (login node) | |
| summarize | `a2d tau2 summarize --pair P --models NAME ...` | a2d env, pure JSON |

## Running on SLURM

1. `cp launchers/slurm/cluster.env.example launchers/slurm/cluster.env` and fill in the account,
   partitions, conda activation and (optionally) `A2D_PATHS_ENV=configs/paths.env`.
2. Submit: `launchers/slurm/submit.sh opd Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500 --arm ours`.
   The job id is printed; logs go next to the resolved config under `$A2D_OUT/<pair>/submit/`.
3. For training stages, let `resubmit.sh` resubmit them:
   `nohup launchers/slurm/resubmit.sh --exp opd_${P}_ours --target 90 -- opd $P --arm ours &`
   (`P` = the pair; the experiment is `opd.arms.<arm>.experiment`, here `opd_<pair>_ours`).
   Resumption uses `trainer.resume_mode=auto` and verl's `latest_checkpointed_iteration.txt`.

Every job gets its own Ray: a short temp dir under `/tmp` and a port block derived from the job id
(`VLLM_PORT` and `MASTER_PORT` too). Nothing ever runs `ray stop --force`; a launcher that finds
`RAY_ADDRESS` set (a multi-node job built the cluster) only attaches. Math OPD arms save before
the wall clock kills them (verl's ESI path, end time from squeue's remaining time); code arms do
not (`opd.esi_save: false`), as in the released runs. NUMA interleaving for big offloaded teachers is
opt-in (`A2D_NUMA_INTERLEAVE=1`).

Disk: fp32 OPD checkpoints with optimizer state are hundreds of GB per arm (one ESI over-save once
reached 883 GB); keep `save_freq` and checkpoint retention as configured.

## Smoke tests

`tests/smoke/run_{math,code,tau2}.sh` run a scaled-down pipeline on one node with 2 x A100-80GB
(`tests/smoke/submit.sh math|code|tau2` submits one), with the overrides in `configs/smoke/`. They
check plumbing only -- every stage exits 0, writes its outputs, logs finite metrics; the math and
code smokes also print three harvested questions with their topics, the filter yield and the
diversity values of synthesizer RL -- and resume after the last passed stage when resubmitted.
Outputs go to `$A2D_OUT/smoke/<domain>` and `$A2D_CKPT/smoke/<domain>`.

## Tests

```bash
pytest -m "not gpu and not smoke"                 # CPU unit tests (login node is fine)
pytest -m equiv tests/equiv/test_calib_gpu.py     # calibration of a code pair on 2 GPUs
```

GPU tests are marked `gpu` (and skip without GPUs or models), tests that run the tau2 interpreter
`tau2env`. `tests/unit/test_custom_pair.py` dry-runs every stage of a pair built from
`configs/pairs/TEMPLATE_custom.yaml`.
