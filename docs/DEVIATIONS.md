# Deviations and preserved quirks

The OPD, bank and evaluation stages reproduce the runs behind the paper's main table bit for bit
where that is possible; the synthesizer stages (calibration, synthesizer RL, selection, harvest)
implement the method as the paper describes it. Abandoned branches are removed, but behaviour that
shapes the results is not silently "fixed". This page lists:

1. the implementation details that are kept on purpose;
2. open points: what is uncertain, or where the implementation is more specific than a short
   description of the method;
3. where the repository knowingly differs from the runs behind the released artefacts.

A fix is allowed only behind a flag that is off by default (for example `reward.fix_eos`).

## 1. Quirks kept on purpose

| # | Behaviour | Where it lives now | Effect |
|---|---|---|---|
| 1 | **Four different math solve prompts.** Synthesizer reward: `q` + `\n\nPlease reason step by step, and put your final answer within \boxed{}.`; synthetic OPD banks: `DAPO_PREFIX + q`; Vanilla OPD (DAPO-17k): `DAPO_PREFIX + q + "Remember ... Answer:"` tail; evaluation: prefix + tail + ` Please reason ... \boxed{}.` | `common/prompts_math.py` | Ours and Vanilla train on different wrappers; the reward is measured under a third. Code uses one template everywhere. |
| 2 | **Reward-worker EOS check uses `tok.eos_token_id` only.** Qwen3 / Coder also stop on `<\|endoftext\|>` (151643); such a rollout is flagged truncated. | `synrl/reward_worker.py` (`reward.fix_eos: false`) | Under `trunc_zero` (code) these rollouts are dropped. Frequency unmeasured. |
| 3 | **Truncated code rollouts are dropped from the mean**, not counted as 0; a question scores 0 only when every rollout is dropped. Rollouts under 10 tokens are dropped the same way. | `synrl/reward_worker.py` | With 2 rolls, one truncated + one normal rollout scores the normal one. A description saying truncated rollouts count as 0 would not match what ran. |
| 4 | **Diversity share over the whole scoring step.** In the default `bleu` mode the 64 questions of a synthesizer step are clustered together (1 − sentence BLEU, average linkage, τ = 0.1), every row included (also ones too short to solve); the reward ranks all-gather the question texts first, because verl hands each rank only its slice. A unique question's multiplier is 1 − 1/64. The `emb` mode instead clusters `diversity.group` consecutive rows of one reward rank's batch on Qwen3-Embedding-4B cosine distance. | `reward.diversity`, `synrl/diversity.py`, `synrl/reward_worker.py` | `balance_batch` splits the step into length-balanced rank batches, so an `emb` group is neither a GRPO group nor a random subset; `config.validate` warns when the group is not one rank batch. |
| 5 | **tau2 diversity runs over the parse-valid cards of the whole rank batch (16 cards)**, on `json.dumps(card["scenario"], ensure_ascii=False)`. | `tau2/trainside/synrl_worker.py`, `synrl.diversity` | A unique card's multiplier is 15/16. |
| 6 | **top_p is 1.0 during synthesizer GRPO (and for the calibration questions) and 0.95 during sweep and harvest.** | `synrl` / `calib.question_sampling` / `select.sweep` / `harvest` sections | The harvested distribution is slightly sharper than the trained one. |
| 7 | **Sweep keeps raw text (`.strip()` only); harvest applies `strip_thinking`.** | `select/sweep.py`, `harvest/harvest.py` | Pick statistics are computed on unstripped sweep rows. |
| 8 | **`data.shuffle=False` for math, code and synthesizer RL; `True` for tau2 OPD.** Synthesizer RL consumes only rows 0..479 of the 4,096 prompt rows. | `opd.shuffle`, `synrl` | The two OPD epochs of `Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500` replay the same order. |
| 9 | **The Vanilla arm of `Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B` ran at 32768 tokens per GPU / reward micro-batch 8**; every other arm used 16384 / 4. | `configs/pairs/Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B.yaml` `opd.arms.vanilla` | token-mean is computed per micro-batch, so the loss weighting differs. |
| 10 | **The aggregation of `Code_Qwen3-4B_Qwen3-4B-Instruct-2507` ran in lexicographic shard order** (s0, s10, s11, s1, ..., s9), and the bank takes the first N questions in that order. | `filter.agg_order` in `configs/pairs/Code_Qwen3-4B_Qwen3-4B-Instruct-2507.yaml` | s0 / s10 / s11 supplied 5,138 / 4,252 / 610 of its bank. Shards are i.i.d., so this should not matter, but it is what ran. |
| 11 | **tau2 OPD ran with the math validation files** (AIME25 / AMC23 / AIME24): `TEST_FILE` was never set, so the launcher fell back to its default. | `configs/base/tau2.yaml` `opd.test_files` | Inert: `test_freq` is -1, nothing is validated during training. |
| 12 | **Math eval stop ids are the first token of `encode("<\|im_end\|>")` and `encode("<\|endoftext\|>")`.** Under the R1-Distill tokenizer neither is a single token, so both encode to `<` and token 27 is a stop id for the two `Math_DeepSeek-*` pairs. | `eval/math/generate.py` | Their answers stop at the first `<`; reported numbers include this. |
| 13 | **Every math-eval rollout index is a separate `llm.generate` call with no per-request seed.** | `eval/math/generate.py` (`eval.engine_seed: null`) | Per-request RNGs would force vLLM off the FlashInfer sampler. |
| 14 | **The reward solver takes top_k and repetition_penalty from the model's `generation_config`.** Qwen3 / Coder top_k 20; R1-Distill has none, so HF's default 50 applies; Coder-3B ships `repetition_penalty 1.05`, so the reward rollouts of `Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct` used 1.05. | `reward.sample_top_k: 0` | Calibration (vLLM, no top_k) and the reward (HF, generation_config) sample differently. |
| 15 | **The math/code worker's step counter restarts at 1 every 4 h window**; `select.pick` undoes it with `save_freq`. | `reward.step_counter: local`, `select/pick.py` | Without the correction, rewards past the first window are misattributed to early steps. |
| 16 | **δ_ref is rounded to 4 decimals before training.** | `reward.delta_ref`, `config.validate` | |
| 17 | **tau2 env-service workers are 32 for v4 / v5 and 64 for v6 / v7.** | `synrl.versions.*.svc_workers` | Throughput only. |
| 18 | **The code Vanilla first windows inherited `VLLM_ATTENTION_BACKEND=FLASH_ATTN` and `TOKENIZERS_PARALLELISM=true`** from the synthesizer-RL job environment they were launched from. | documented only | Probably a no-op on A100. |
| 19 | **The code filter's known misfires**: `ships_solution` does not require the keyword inside the code block (false positives after a fenced sample, misses for short fenced code); `no_io_spec` accepts very loose inline phrasing. | `filter/code_filter.py` | `Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct` and `Code_Qwen3-4B_Qwen3-4B-Instruct-2507` were filtered with this version; the pool of `Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507` was re-filtered with it. |
| 20 | **When the tau2 env service fails, every card of that step scores 0.** | `tau2/trainside/synrl_worker.py` | |
| 21 | **tau2 agent loop:** `/env/close` is posted with `{"domain": sid}`; an episode stops at `TAU2_MAX_TURNS` assistant turns or when `response_length` tokens (8192) are used. | `tau2/trainside/agent_loop.py` | |

## 2. Open points

These come from checking the code, launch command lines, logs and reward dumps against the method
description; they matter when describing the method or comparing against it.

1. **Truncated code rollouts are dropped, not scored 0** (quirk 3). A description saying
   "truncated rollouts count as zero" would not match the code that ran.
2. **The EOS check may misclassify `<|endoftext|>` endings** (quirk 2); unmeasured.
3. **Math uses four solve prompts** (quirk 1); code uses one template in all four places.
4. **δ_ref is measured on questions of the initial synthesizer.** `a2d calibrate` samples
   N_cal = 200 questions from `models.synthesizer_init` under the synthesizer-RL prompt and scores
   one student solution per question. The three math values were measured on one shared set of
   200 questions written by the untrained Qwen3-1.7B synthesizer; a rerun for a DeepSeek pair
   samples that pair's own no-think synthesizer and can differ by more than sampling noise.
   Calibration samples with vLLM, no top_k, EOS kept and truncated rollouts counted; the worker
   uses HF, the model's top_k, cuts EOS and (code) drops truncated rollouts.
5. **ω_t is almost constant** in every run (math, code and tau2), so the reward is driven by δ_t.
6. **The rewards of the two `Math_DeepSeek-*` pairs mostly come from truncated prefixes**:
   most of their reward rollouts hit the 7,168 cap, where the length factor is a constant 0.7
   and the sqrt(L) normaliser is constant too (rarely the case for
   `Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500`).
7. **Synthesizer training and harvesting sample differently** (quirk 6). The prompt's "private
   scratch-pad" is inert because thinking is off.
8. **Other residue.** The math filter does not strip a leading design narrative (a minority of
    the rows of the `Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B` and
    `Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B` banks carry one or start with a markdown heading). The in-slice funnel numbers of
    `Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507` came from an earlier `code_filter` version that no longer exists (its pool was
    re-filtered with the current one, see `bank.refilter`). The math harvest seed is the shard
    index, not the code formula `7919*i+104729`. `a2d data build code-real` reruns the
    OpenCodeReasoning-10k selection with the current `code_filter`, so it differs from the file
    the released Vanilla code arms trained on (docs/DATA.md).
9. **Known drift.** The code synthesizer learned to append code to its questions (the share of
    questions with a code block grows during training). Neither δ_t nor ω_t penalises it; only the harvest filter's `ships_solution`
    catches it, with the misfires of quirk 19.

## 3. Where this repository differs from the runs behind the released artefacts

| Difference | Why | Effect on numbers |
|---|---|---|
| Ray is private per job (temp dir and port block from the job id) and the launcher attaches to an existing cluster; the final runs' launchers ran `ray stop --force` and a head on 6379. | Shared nodes: the old way killed sibling jobs and collided on ports. | None. |
| Run-specific environment reaches Ray workers through `+ray_kwargs.ray_init.runtime_env.env_vars`, and `hydra.run.dir` is set. | sbatch need not know the run directory; hydra no longer writes `outputs/<date>/` into the cwd. | None (both are stripped before the render / composition comparisons). |
| Only `A2D_CONFIG=<file>` crosses `sbatch --export`; the final runs passed shard and aggregation lists through side files instead. | `--export` splits on commas. | None. |
| LiveCodeBench's few-shot files load relative to the module, and the dataset is pinned to `0fe84c3` (`LCB_DATASET_REVISION`). | Loading depended on the cwd and on the floating `main` revision. | None on the same data (THIRD_PARTY.md). |
| tau2 synthesizer v4: the repo uses the steady-state `max_prompt_length 7168` with 16384 batched tokens; the first two v4 windows ran 5632 / 8192. | One configuration per version. | v4 is only an initialisation for v5; its weights are lost either way. |
| The Vanilla arm of `Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507` renders `trainer.is_plot=False`; its early windows (launched from inside the synthesizer-RL job) plotted. | Follows the launch settings of the later windows. | Plots only. |
| The airline part of the agent bank (`<pair>_part-airline`) is built by a reconstructed command. | | Byte-identical output (verified). |
| `GENRL_TAU2_V4..V7` rebuilt by `a2d tau2 build-prompts` are equivalent, not identical (retail rows depend on set iteration order). The released parquets are canonical. | | Use the released files to reproduce. |
| The `*-SynNoThink` bases, the apps_interview1k subset and the official tau2 split are built by scripts here. | | Byte-identical to the shipped files (`test_data_release.py`, `tau2_data`). |
| Smoke configs change models, dtypes, sizes, layouts and caps. | 2-GPU plumbing checks. | Smoke numbers are not comparable and are never compared. |

Lost artefacts that limit reproduction (see docs/DATA.md): the three code synthesizers, the tau2
synthesizers v4-v7, the tau2 Vanilla-arm checkpoint and the Vanilla-arm checkpoint of
`Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B` were not kept. The released banks let OPD and evaluation reproduce without them; re-harvesting needs a
retrained synthesizer.
