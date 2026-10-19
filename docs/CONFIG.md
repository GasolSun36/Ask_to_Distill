# Configuration reference

Every stage reads one resolved configuration: a pair file under `configs/pairs/`, its bases, an
optional extra file, and command-line overrides. This page explains how they compose, maps the
paper's symbols to keys, and then lists every key with its value and, where it is not obvious,
why.

## How a configuration is assembled

`a2d <command> --pair P [--config FILE] [key=value ...]` (`P` is a pair name,
`<Domain>_<Student>_<Teacher>`, e.g. `Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500`) (or `$A2D_CONFIG`, which is how
`launchers/slurm/submit.sh` hands a resolved file to a job) loads, in order:

1. the files named in the pair file's `_base_` list, e.g.
   `[../base/opd.yaml, ../base/math.yaml, ../base/eval_math.yaml]`;
2. `configs/pairs/<pair>.yaml`;
3. the `--config` file, for example `configs/smoke/math_smoke.yaml` (a smoke file names its pair in
   its own `_base_`, so `--config` alone is enough);
4. `key=value` overrides (OmegaConf dotlist; `+key=value` adds a key).

`config.load` then injects a `paths` node from the `A2D_*` variables (see below). Values refer to each
other with interpolation: `${paths.models}/Qwen3-1.7B`, `${bank.released}`, `${pair}`.

**No silent defaults.** The bases hold `???` for every per-pair value, and loading fails with the
list of every key still missing. The code has no fallback for a configured value (a default
length penalty, solve budget or δ_ref can never leak in). `config.validate` also enforces
cross-key guards:

- math/code `synrl.layout.enable_resource_pool` must be `true`: the reward worker sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` for itself, which would break vLLM's CuMemAllocator
  in a colocated actor;
- the questions of a synthesizer step (`synrl.train_batch_size` x `synrl.rollout_n`) must split
  evenly over the reward ranks; with `reward.diversity.mode: emb` a `diversity.group` that is not
  one reward-rank batch warns; a tau2 rank batch other than the released 8 ranks x 16 cards prints
  `NON-FAITHFUL: diversity grouping differs from the final runs`;
- `opd.forbidden_overrides` / `synrl.forbidden_overrides` must not appear among the extra
  overrides;
- `reward.delta_ref` must be rounded to 4 decimals (`null` -- a new pair before `a2d calibrate` --
  warns, and `a2d synrl` refuses to start), `filter.agg_order` must be a permutation of the
  slices, `harvest.seed_formula` must be a known formula;
- outside a smoke config (`smoke: true`), `select.pick.allow_empty` is rejected and
  `bank.allow_short` needs `bank.expected_rows` (the documented 2,889-row bank of
  `Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500`); capped
  `opd.train_max_samples` / `val_max_samples` warn;
- the tau2 eval served name must contain "qwen" (ToolOrchestra routes the agent model by name).

**Sections.** `models`, `calib`, `reward`, `synrl`, `select`, `harvest`, `filter`, `bank`, `opd`,
`eval`; the tau2 pair adds `tau2_data` and `reward_env` (the synthesizer reward's
`TAU2_*` environment, rendered completely by `ask2distill.tau2.env`).

**What reaches the trainers.** verl stays a hydra program. `synrl.launch` and `opd.launch` render
the `synrl` / `opd` sections into override strings in a fixed order (pinned by
`tests/golden/overrides.json`; tests/unit/test_synrl_render.py, test_opd_render.py), then add
`hydra.run.dir=<run_dir>/hydra` and `+ray_kwargs.ray_init.runtime_env.env_vars.*` for run-specific
environment. The reward worker reads only `<run_dir>/reward_config.json`, the resolved `reward`
section. Every run directory records `resolved_config.yaml`, `env.json` and `command.txt`.

## Paths

| Variable | Default | Holds |
|---|---|---|
| `A2D_DATA` | `<repo>/data` | the data release and built datasets |
| `A2D_MODELS` | `<repo>/models` | base models (`a2d models download`) |
| `A2D_CKPT` | `<repo>/checkpoints` | training checkpoints, merged HF directories |
| `A2D_OUT` | `<repo>/outputs` | stage outputs, run directories, eval results |
| `A2D_CACHE` | `<repo>/.cache` | HF hub, triton, inductor, vLLM, evalplus, outlines caches |
| `A2D_TAU2_PYTHON` | (none) | interpreter of the tau2 env (tau2 stages only) |
| `A2D_HF_DATA_REPO` | `gasolsun/Ask_to_Distill-data` | HF dataset repo of the release (`a2d data download`; `--repo-id` overrides it) |
| `A2D_OPENAI_ENV_FILE` | (none) | your own `OPENAI_API_KEY=...` file for the tau2 eval; read, never copied |

Copy `configs/paths.example.env` to `configs/paths.env` (gitignored) and `source` it. A relative
value is resolved against the repository root, never the current directory: Ray workers, the tau2
env service and LiveCodeBench run in other working directories. A path that is still relative
after resolution is an error.

Per-pair outputs land in `$A2D_OUT/<pair>/{calib,synrl,sweep,harvest,filter,banks,runs}/`,
`pick.json`, `yield.json` and `pool.jsonl` (the tau2 pair: `synrl/`, `harvest/`, `banks/`, `opd/`);
checkpoints in `$A2D_CKPT/<experiment>/global_step_<N>/`
and merged models in `$A2D_CKPT/<experiment>/hf/step_<N>/`; evaluations in
`$A2D_OUT/eval/{math,code}/<experiment>_step<N>/` and `$A2D_OUT/tau2/outputs/<tag>/`.

## Overriding a value

```bash
P=Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500
a2d opd --pair $P --arm ours opd.arms.ours.train=$A2D_OUT/$P/banks/$P.parquet   # your own bank
a2d synrl --pair Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B --dry-run              # print, run nothing
a2d eval math --pair $P --arm ours --step <N> eval.tasks='[AIME24]'
```

`--dry-run` prints the resolved configuration, the stage environment, a per-GPU memory estimate for
training stages and the exact command or hydra overrides. docs/DEVIATIONS.md lists the values
whose effect is known.

## Paper symbols

| Symbol (paper) | Key | Value (math / code) |
|---|---|---|
| S_θ (synthesizer), initialised from π_S | `models.synthesizer_init` | the student (DeepSeek pairs: its `-SynNoThink` copy) |
| K (candidate support) | `reward.topk`, `calib.topk` | 16 |
| δ_ref (calibration scale) | `reward.delta_ref` (from `a2d calibrate`) | per pair |
| κ (calibration percentile) | `calib.percentile` | 95 |
| N_cal (calibration questions) | `calib.n_questions` | 200 |
| ε (Eq. 6) | `common/divergence.py` `EPS` | 1e-8 |
| α (length exponent) | `reward.len_alpha` | 0.5 / 1.0 |
| λ_len (length penalty) | `reward.len_pen` | 0.3 / 0.0 |
| L_max (student response budget) | `reward.max_new` = `calib.max_new` = `opd.max_response_length` | 7168 / 4096 |
| L_q (question budget) | `synrl.max_response_length` | 4096 / 2048 |
| λ_div (diversity penalty) | `reward.diversity.pen` | 1.0 |
| τ (clustering threshold) | `reward.diversity.thresh` | 0.1 |
| g (group size), B (groups per step) | `synrl.rollout_n`, `synrl.train_batch_size` | 8, 8 |
| T_G (synthesizer steps) | `synrl.total_steps` | 60 |
| N (raw questions) | `harvest.shards` x `harvest.per_shard` | per pair |
| τ_lex (Jaccard dedup) | `filter.jaccard` | 0.7 |
| τ_sem (embedding dedup) | `filter.sat_thr`, `filter.agg_thr` | 0.95 |
| n_roll (OPD responses per question) | `opd.n_responses` | 4 |

The synthesizer's instructions name one topic each from `synrl.topics` (`configs/topics/math.yaml`,
`configs/topics/code.yaml`: plain lists, one topic per line), assigned by a seeded rotation:
`synrl.prompt_seed` for the instruction parquet, `calib.question_sampling.topic_seed` for the
calibration questions, `select.sweep.topic_seed` for the sweep, and `harvest.seed_formula` per
harvest shard.

## Key reference

Generated from the YAML files by `scripts/maintainer/make_config_doc.py` (rerun it after editing a
config; `tests/unit/test_docs.py` fails while this page is stale). The note column is the YAML
comment of the key.

<!-- BEGIN GENERATED: scripts/maintainer/make_config_doc.py -->

### `configs/base/math.yaml`

| Key | Value | Note |
|---|---|---|
| `domain` | `math` |  |
| `smoke` | `false` |  |
| `models.student` | `???` |  |
| `models.teacher` | `???` |  |
| `models.synthesizer_init` | `???` |  |
| `models.embedder` | `${paths.models}/Qwen3-Embedding-4B` | frozen: semantic dedup, checkpoint pick, diversity.mode emb |
| `models.qwen3_nothink` | `???` |  |
| `models.syn_nothink` | `false` | true: synthesizer_init is built by `a2d models syn-nothink` |
| `models.hf` | `???` | pinned revisions for `a2d models download` |
| `calib` |  | stage 0: δ_ref = κ-th percentile of δ_t (paper §2.3) |
| `calib.n_questions` | `200` | N_cal: questions written by the INITIAL synthesizer (no external data) |
| `calib.questions` | `${paths.out}/${pair}/calib/calib_questions.jsonl` | sampled by `a2d calibrate`; reused when it holds N_cal rows |
| `calib.question_sampling` |  | models.synthesizer_init under the synthesizer-RL prompt |
| `calib.question_sampling.topic_seed` | `1000` | Random(1000) topic rotation over synrl.topics |
| `calib.question_sampling.temperature` | `${synrl.rollout_temperature}` | as in the synthesizer-RL rollouts |
| `calib.question_sampling.top_p` | `1.0` | as in the synthesizer-RL rollouts (verl default) |
| `calib.question_sampling.max_new` | `${synrl.max_response_length}` | L_q, the question budget |
| `calib.question_sampling.prompt_head` | `${harvest.prompt_head}` | max_model_len = max_new + prompt_head |
| `calib.question_sampling.gpu_memory_utilization` | `0.85` |  |
| `calib.question_sampling.oversample` | `0.25` | a round asks for 25% more questions than it still needs |
| `calib.question_sampling.max_rounds` | `3` |  |
| `calib.rolls` | `1` | one student solution per question |
| `calib.max_new` | `7168` | = opd.max_response_length (L_max): δ_ref is measured on the OPD solve budget |
| `calib.qwen3_nothink` | `${models.qwen3_nothink}` |  |
| `calib.solve_template` | `math_boxed` | "{q}\n\nPlease reason step by step ... \boxed{}." |
| `calib.solve_system` | `""` | no system prompt |
| `calib.temperature` | `0.7` | student sampling for the calibration rollouts |
| `calib.top_p` | `0.95` | vLLM, no top_k, EOS kept |
| `calib.prompt_head` | `1024` | max_model_len = max_new + prompt_head |
| `calib.gpu_memory_utilization` | `0.85` |  |
| `calib.topk` | `16` | K: top-K support of δ_t (union of both models' top-16) |
| `calib.chunk` | `128` | positions per teacher-forced chunk (memory only) |
| `calib.min_tokens` | `10` | rollouts under 10 tokens are skipped; truncated ones are kept |
| `calib.percentile` | `95` | κ: δ_ref = the κ-th percentile of the pooled δ_t |
| `calib.round_decimals` | `4` | the trained δ_ref is the percentile rounded to 4 decimals |
| `calib.teacher_device` | `null` | student and teacher share one device for math |
| `calib.expected_delta_ref` | `???` |  |
| `calib.out` | `${paths.out}/${pair}/calib/calib_delta_ref.json` |  |
| `reward` |  | -> reward_config.json -> SynRewardWorker |
| `reward.domain` | `math` |  |
| `reward.solver` | `${models.student}` | the frozen, untrained student π_S solves each generated question |
| `reward.teacher` | `${models.teacher}` | the frozen teacher π_T |
| `reward.embedder` | `${models.embedder}` | loaded only in diversity.mode emb |
| `reward.qwen3_nothink` | `${models.qwen3_nothink}` |  |
| `reward.solve_template` | `math_boxed` |  |
| `reward.solve_system` | `""` | no system prompt |
| `reward.delta_ref` | `???` | δ_ref: the calibrated percentile, rounded to 4 decimals (`a2d calibrate`) |
| `reward.topk` | `16` | K: top-16 support, as in OPD |
| `reward.chunk` | `128` |  |
| `reward.use_omega` | `true` | s_t = min(δ_t / (δ_ref + ε), 1) * ω_t (false drops the ω_t factor) |
| `reward.len_alpha` | `0.5` | α: R_raw = sum_t s_t / L^α |
| `reward.len_pen` | `0.3` | λ_len: P_len = 1 - λ_len * min(L / L_max, 1) |
| `reward.max_new` | `7168` | L_max = opd.max_response_length: the question is scored on the budget OPD will use |
| `reward.rolls` | `1` | one solver rollout per question |
| `reward.temperature` | `0.7` | solver sampling |
| `reward.top_p` | `0.95` |  |
| `reward.sample_top_k` | `0` | 0 => pass neither top_k nor repetition_penalty, the model's generation_config supplies them (Qwen3 top_k 20, R1-Distill: HF default top_k 50) |
| `reward.trunc_zero` | `false` | truncated rollouts are scored on their prefix |
| `reward.gen_bs` | `0` | one generate call per rank batch |
| `reward.min_question_chars` | `20` |  |
| `reward.min_rollout_tokens` | `10` |  |
| `reward.fix_eos` | `false` | EOS check on tok.eos_token_id only (true: every stop id) |
| `reward.step_counter` | `local` | the dump step restarts at 1 every window; select.pick undoes it |
| `reward.diversity` |  | P_div = max(0, 1 - λ_div * share(q)) (paper Eq. 9-10) |
| `reward.diversity.mode` | `bleu` | bleu: 1 - sentence BLEU over the full scoring batch (paper); emb: embedding cosine per group |
| `reward.diversity.thresh` | `0.1` | τ: distance threshold of the average-linkage clustering |
| `reward.diversity.pen` | `1.0` | λ_div |
| `reward.diversity.group` | `8` | emb only: rows per clustering group (8 = one reward-rank batch with 8 reward ranks) |
| `reward.diversity.max_len` | `512` | emb only: embedder max tokens |
| `reward.diversity.embed_bs` | `16` | emb only |
| `reward.diversity.per_group_embed` | `true` | emb only: one embed call per group |
| `reward.dump` | `${paths.out}/${pair}/synrl/synrl_dump_${synrl.experiment}.jsonl` | per-rank dumps .r<rank>, .metrics.r<rank> |
| `synrl` |  | synthesizer RL: S_θ (initialised from the student) trained with GRPO |
| `synrl.experiment` | `???` |  |
| `synrl.prompts` | `${paths.out}/${pair}/prompts/SYNRL_TOPICS_MATH.parquet` | built by `a2d prompts build` (or by `a2d synrl` when absent); 60 steps read rows 0..479 |
| `synrl.topics` | `${paths.repo}/configs/topics/math.yaml` | one topic per prompt, seeded rotation (also used by sweep / harvest) |
| `synrl.prompt_rows` | `4096` | rows of the prompt parquet (>= total_steps x train_batch_size) |
| `synrl.prompt_seed` | `0` | Random(0) topic rotation of the prompt parquet |
| `synrl.total_steps` | `60` |  |
| `synrl.save_freq` | `5` |  |
| `synrl.total_epochs` | `1` |  |
| `synrl.train_batch_size` | `8` | instructions (GRPO groups) per step |
| `synrl.rollout_n` | `8` | g: questions per instruction; 8 x 8 = 64 scored per step |
| `synrl.ppo_mini_batch_size` | `8` |  |
| `synrl.ppo_micro_batch_size_per_gpu` | `1` |  |
| `synrl.lr` | `1.0e-6` |  |
| `synrl.entropy_coeff` | `0.01` |  |
| `synrl.use_kl_loss` | `true` |  |
| `synrl.kl_loss_coef` | `0.005` |  |
| `synrl.kl_loss_type` | `low_var_kl` |  |
| `synrl.max_prompt_length` | `1024` |  |
| `synrl.max_response_length` | `4096` | question budget; rollout max_model_len = prompt + response |
| `synrl.ppo_max_token_len_per_gpu` | `16384` |  |
| `synrl.max_num_batched_tokens` | `16384` |  |
| `synrl.offload` | `???` | actor params + optimizer offload |
| `synrl.gpu_memory_utilization` | `???` |  |
| `synrl.rollout_temperature` | `1.0` | top_p 1.0 / top_k -1 (verl defaults) |
| `synrl.rollout_tp` | `1` |  |
| `synrl.model_dtype` | `fp32` | master weights; FSDP compute is bf16 |
| `synrl.ref_param_offload` | `true` |  |
| `synrl.use_remove_padding` | `true` |  |
| `synrl.gradient_checkpointing` | `true` |  |
| `synrl.shuffle` | `false` |  |
| `synrl.filter_overlong_prompts` | `true` |  |
| `synrl.truncation` | `error` |  |
| `synrl.return_raw_chat` | `true` |  |
| `synrl.reward_micro_batch_size_per_gpu` | `1` |  |
| `synrl.critic_warmup` | `0` |  |
| `synrl.logger` | `console` |  |
| `synrl.project_name` | `opd_synrl` |  |
| `synrl.test_freq` | `-1` |  |
| `synrl.val_before_train` | `false` |  |
| `synrl.resume_mode` | `auto` |  |
| `synrl.custom_reward` | `{path: "${paths.repo}/ask2distill/synrl/rule_reward.py", name: compute_score}` | logging only; the reward worker's rm_scores win |
| `synrl.layout` | `{actor_gpus: 8, reward_gpus: 8, reward_nnodes: 1, enable_resource_pool: true}` | node 0: S_θ; node 1: reward pool (student, teacher) |
| `synrl.ckpt_dir` | `${paths.ckpt}/${synrl.experiment}` |  |
| `synrl.extra_overrides` | `[]` |  |
| `synrl.forbidden_overrides` | `[]` |  |
| `select.sweep.steps` | `[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]` | every saved checkpoint |
| `select.sweep.n` | `300` | questions sampled per checkpoint |
| `select.sweep.max_new` | `4096` | = the training question budget |
| `select.sweep.temperature` | `1.0` |  |
| `select.sweep.top_p` | `0.95` |  |
| `select.sweep.topic_seed` | `1` | Random(1) topic rotation, the same for every checkpoint |
| `select.sweep.prompt_head` | `1536` |  |
| `select.sweep.gpu_memory_utilization` | `0.85` |  |
| `select.sweep.strip_thinking` | `false` | the sweep keeps the raw text (the pick thresholds were set on it); the harvest strips |
| `select.sweep.out_dir` | `${paths.out}/${pair}/sweep` |  |
| `select.pick.cos_max` | `0.85` |  |
| `select.pick.min_eff_clusters` | `5` |  |
| `select.pick.eff_cluster_cos` | `0.9` |  |
| `select.pick.reward_frac` | `0.95` |  |
| `select.pick.reward_window` | `2` | +-2 steps around the checkpoint |
| `select.pick.save_freq` | `${synrl.save_freq}` | maps dump steps to checkpoints (the dump step counter restarts every window) |
| `select.pick.min_rows` | `50` |  |
| `select.pick.allow_empty` | `false` | smoke-only relaxation |
| `select.pick.out` | `${paths.out}/${pair}/pick.json` |  |
| `select.yield` | `???` |  |
| `select.chosen_step` | `???` |  |
| `harvest.synthesizer` | `${paths.ckpt}/${synrl.experiment}/hf/step_${select.chosen_step}` |  |
| `harvest.shards` | `???` |  |
| `harvest.per_shard` | `5000` |  |
| `harvest.seed_formula` | `index` | topic-rotation seed = shard index for the math pairs |
| `harvest.temperature` | `1.0` |  |
| `harvest.top_p` | `0.95` |  |
| `harvest.max_new` | `4096` |  |
| `harvest.prompt_head` | `1536` | max_model_len = max_new + prompt_head |
| `harvest.gpu_memory_utilization` | `0.85` |  |
| `harvest.strip_thinking` | `true` |  |
| `harvest.vllm_seed` | `0` | engine seed; the per-shard seed only moves the topic rotation |
| `harvest.out_dir` | `${paths.out}/${pair}/harvest` |  |
| `filter.domain` | `math` |  |
| `filter.slices` | `???` | shard indices per saturation slice, in listed order |
| `filter.jaccard` | `0.7` | token-overlap dedup inside a slice |
| `filter.sat_thr` | `0.95` | embedding dedup threshold of the saturation pass |
| `filter.embed_bs` | `16` |  |
| `filter.aggregate` | `???` |  |
| `filter.agg_thr` | `0.95` | cross-slice embedding dedup threshold |
| `filter.agg_order` | `???` |  |
| `filter.out_dir` | `${paths.out}/${pair}/filter` |  |
| `filter.pool` | `${paths.out}/${pair}/pool.jsonl` |  |
| `bank.name` | `???` |  |
| `bank.row_id_prefix` | `???` | extra_info.index = <row_id_prefix>_<i> (the released bank's) |
| `bank.n` | `???` |  |
| `bank.pool` | `???` |  |
| `bank.template` | `dapo_prefix_only` | DAPO_PREFIX + q, without the "Remember..." tail of the DAPO-17k rows |
| `bank.refilter` | `null` |  |
| `bank.allow_short` | `false` |  |
| `bank.expected_rows` | `null` |  |
| `bank.out` | `${paths.out}/${pair}/banks/${bank.name}.parquet` |  |
| `bank.released` | `${paths.data}/banks/${bank.name}.parquet` |  |
| `opd` |  | math arms |
| `opd.student` | `${models.student}` |  |
| `opd.teacher` | `${models.teacher}` |  |
| `opd.max_prompt_length` | `1024` |  |
| `opd.max_response_length` | `7168` |  |
| `opd.max_val_response_length` | `7168` | explicit; feeds max_model_len and val_kwargs.max_tokens |
| `opd.test_files` | `${paths.data}/eval/math/AIME25/test.parquet, ${paths.data}/eval/math/AMC23/test…` | The three AIME/AMC parquets are loaded as val_files but never scored -- val_before_train=False and test_freq=-1 -- so they are only a startup gate. |
| `opd.custom_reward` | `{path: "${paths.third_party}/verl/verl/utils/reward_score/ttrl_math/__init__.py…` | logging only |
| `opd.is_plot` | `true` | diagnostic figures |
| `opd.esi_save` | `true` | math arms force-save before the wall clock |

### `configs/base/code.yaml`

| Key | Value | Note |
|---|---|---|
| `domain` | `code` |  |
| `smoke` | `false` |  |
| `models.student` | `???` |  |
| `models.teacher` | `???` |  |
| `models.synthesizer_init` | `${models.student}` |  |
| `models.embedder` | `${paths.models}/Qwen3-Embedding-4B` | frozen: semantic dedup, checkpoint pick, diversity.mode emb |
| `models.qwen3_nothink` | `???` |  |
| `models.syn_nothink` | `false` |  |
| `models.hf` | `???` |  |
| `calib` |  | stage 0: δ_ref = κ-th percentile of δ_t, as for math (paper §2.3) |
| `calib.n_questions` | `200` | N_cal: questions written by the INITIAL synthesizer (no external data) |
| `calib.questions` | `${paths.out}/${pair}/calib/calib_questions.jsonl` | sampled by `a2d calibrate`; reused when it holds N_cal rows |
| `calib.question_sampling` |  | models.synthesizer_init under the synthesizer-RL prompt |
| `calib.question_sampling.topic_seed` | `1000` | Random(1000) topic rotation over synrl.topics |
| `calib.question_sampling.temperature` | `${synrl.rollout_temperature}` | as in the synthesizer-RL rollouts |
| `calib.question_sampling.top_p` | `1.0` | as in the synthesizer-RL rollouts (verl default) |
| `calib.question_sampling.max_new` | `${synrl.max_response_length}` | L_q, the question budget |
| `calib.question_sampling.prompt_head` | `${harvest.prompt_head}` | max_model_len = max_new + prompt_head |
| `calib.question_sampling.gpu_memory_utilization` | `0.85` |  |
| `calib.question_sampling.oversample` | `0.25` | a round asks for 25% more questions than it still needs |
| `calib.question_sampling.max_rounds` | `3` |  |
| `calib.rolls` | `1` | one student solution per question |
| `calib.max_new` | `4096` | = code opd.max_response_length (L_max) |
| `calib.qwen3_nothink` | `${models.qwen3_nothink}` |  |
| `calib.solve_template` | `lcb` | the LiveCodeBench template (calibration, reward, banks and eval) |
| `calib.solve_system` | `""` | no system prompt (calibration, reward, banks and eval alike) |
| `calib.temperature` | `0.7` |  |
| `calib.top_p` | `0.95` | vLLM, no top_k, EOS kept |
| `calib.prompt_head` | `1024` |  |
| `calib.gpu_memory_utilization` | `0.85` |  |
| `calib.topk` | `16` | K |
| `calib.chunk` | `128` |  |
| `calib.min_tokens` | `10` |  |
| `calib.percentile` | `95` | κ |
| `calib.round_decimals` | `4` |  |
| `calib.teacher_device` | `cuda:1` | 57-62GB of teacher will not share an 80GB card with the student. |
| `calib.expected_delta_ref` | `???` |  |
| `calib.out` | `${paths.out}/${pair}/calib/calib_delta_ref.json` |  |
| `reward` |  | -> reward_config.json -> SynRewardWorker |
| `reward.domain` | `code` |  |
| `reward.solver` | `${models.student}` | the frozen student π_S |
| `reward.teacher` | `${models.teacher}` | the frozen teacher π_T |
| `reward.embedder` | `${models.embedder}` | loaded only in diversity.mode emb |
| `reward.qwen3_nothink` | `${models.qwen3_nothink}` |  |
| `reward.solve_template` | `lcb` | same template as calibration, banks and LCB eval |
| `reward.solve_system` | `""` | empty for every final code pair |
| `reward.delta_ref` | `???` | δ_ref: the calibrated percentile, rounded to 4 decimals (`a2d calibrate`) |
| `reward.topk` | `16` | K |
| `reward.chunk` | `128` |  |
| `reward.use_omega` | `true` |  |
| `reward.len_alpha` | `1.0` | α = 1: per-token mean |
| `reward.len_pen` | `0.0` | λ_len = 0: no length penalty; truncation is handled by trunc_zero |
| `reward.max_new` | `4096` | L_max = opd.max_response_length |
| `reward.rolls` | `2` | two solver rollouts per question |
| `reward.temperature` | `0.7` |  |
| `reward.top_p` | `0.95` |  |
| `reward.sample_top_k` | `0` | generation_config supplies top_k and, for Coder-3B, repetition_penalty 1.05 |
| `reward.trunc_zero` | `true` | Code solutions are bimodal: a few hundred tokens, or a runaway loop that rides the cap. A soft length penalty cannot separate those, so a truncated rollout is dropped from the question's mean (all truncated -> 0), not counted as 0. |
| `reward.gen_bs` | `???` | throughput knob, but batching changes the sampled rollouts: kept per pair |
| `reward.min_question_chars` | `20` |  |
| `reward.min_rollout_tokens` | `10` |  |
| `reward.fix_eos` | `false` | EOS check on tok.eos_token_id only (true: every stop id) |
| `reward.step_counter` | `local` |  |
| `reward.diversity` |  | P_div = max(0, 1 - λ_div * share(q)) (paper Eq. 9-10) |
| `reward.diversity.mode` | `bleu` | bleu: 1 - sentence BLEU over the full scoring batch (paper); emb: embedding cosine per group |
| `reward.diversity.thresh` | `0.1` | τ |
| `reward.diversity.pen` | `1.0` | λ_div |
| `reward.diversity.group` | `8` | emb only: rows per clustering group |
| `reward.diversity.max_len` | `512` | emb only |
| `reward.diversity.embed_bs` | `16` | emb only |
| `reward.diversity.per_group_embed` | `true` | emb only |
| `reward.dump` | `${paths.out}/${pair}/synrl/synrl_dump_${synrl.experiment}.jsonl` |  |
| `synrl.experiment` | `???` |  |
| `synrl.prompts` | `${paths.out}/${pair}/prompts/SYNRL_TOPICS_CODE.parquet` | built by `a2d prompts build` (or by `a2d synrl` when absent) |
| `synrl.topics` | `${paths.repo}/configs/topics/code.yaml` | one topic per prompt, seeded rotation (also used by sweep / harvest) |
| `synrl.prompt_rows` | `4096` | rows of the prompt parquet (>= total_steps x train_batch_size) |
| `synrl.prompt_seed` | `0` | Random(0) topic rotation of the prompt parquet |
| `synrl.total_steps` | `60` |  |
| `synrl.save_freq` | `???` | 5, or 10 for Code_Qwen3-4B_Qwen3-4B-Instruct-2507 |
| `synrl.total_epochs` | `1` |  |
| `synrl.train_batch_size` | `8` | instructions (GRPO groups) per step |
| `synrl.rollout_n` | `8` | g: questions per instruction; 8 x 8 = 64 scored per step |
| `synrl.ppo_mini_batch_size` | `8` |  |
| `synrl.ppo_micro_batch_size_per_gpu` | `1` |  |
| `synrl.lr` | `1.0e-6` |  |
| `synrl.entropy_coeff` | `0.01` |  |
| `synrl.use_kl_loss` | `true` |  |
| `synrl.kl_loss_coef` | `0.005` |  |
| `synrl.kl_loss_type` | `low_var_kl` |  |
| `synrl.max_prompt_length` | `2048` | the topic prompt is a few hundred tokens; nothing is dropped |
| `synrl.max_response_length` | `2048` | L_q, the question budget (real statements are p90 ~700 tokens) |
| `synrl.ppo_max_token_len_per_gpu` | `16384` |  |
| `synrl.max_num_batched_tokens` | `16384` |  |
| `synrl.offload` | `false` |  |
| `synrl.gpu_memory_utilization` | `0.6` |  |
| `synrl.rollout_temperature` | `1.0` |  |
| `synrl.rollout_tp` | `1` |  |
| `synrl.model_dtype` | `fp32` |  |
| `synrl.ref_param_offload` | `true` |  |
| `synrl.use_remove_padding` | `true` |  |
| `synrl.gradient_checkpointing` | `true` |  |
| `synrl.shuffle` | `false` |  |
| `synrl.filter_overlong_prompts` | `true` |  |
| `synrl.truncation` | `error` |  |
| `synrl.return_raw_chat` | `true` |  |
| `synrl.reward_micro_batch_size_per_gpu` | `1` |  |
| `synrl.critic_warmup` | `0` |  |
| `synrl.logger` | `console` |  |
| `synrl.project_name` | `opd_synrl` |  |
| `synrl.test_freq` | `-1` |  |
| `synrl.val_before_train` | `false` |  |
| `synrl.resume_mode` | `auto` |  |
| `synrl.custom_reward` | `{path: "${paths.repo}/ask2distill/synrl/rule_reward.py", name: compute_score}` |  |
| `synrl.layout` | `{actor_gpus: 8, reward_gpus: 8, reward_nnodes: 1, enable_resource_pool: true}` | The real-question OPD arm is launched on its own (`a2d opd --arm vanilla`), not on a third node of this job. |
| `synrl.ckpt_dir` | `${paths.ckpt}/${synrl.experiment}` |  |
| `synrl.extra_overrides` | `[]` |  |
| `synrl.forbidden_overrides` | `[]` |  |
| `select.sweep.steps` | `???` |  |
| `select.sweep.n` | `300` |  |
| `select.sweep.max_new` | `2048` | = the training question budget |
| `select.sweep.temperature` | `1.0` |  |
| `select.sweep.top_p` | `0.95` |  |
| `select.sweep.topic_seed` | `1` | Random(1) topic rotation, the same for every checkpoint |
| `select.sweep.prompt_head` | `2304` | max_model_len = max_new + prompt_head |
| `select.sweep.gpu_memory_utilization` | `0.85` |  |
| `select.sweep.strip_thinking` | `false` | raw .strip()-ed text, as the pick thresholds were set on |
| `select.sweep.out_dir` | `${paths.out}/${pair}/sweep` |  |
| `select.pick` |  | code uses only the two coded gates |
| `select.pick.cos_max` | `0.85` |  |
| `select.pick.min_eff_clusters` | `5` |  |
| `select.pick.eff_cluster_cos` | `0.9` |  |
| `select.pick.reward_frac` | `0.95` |  |
| `select.pick.reward_window` | `2` |  |
| `select.pick.save_freq` | `${synrl.save_freq}` | maps dump steps to checkpoints |
| `select.pick.min_rows` | `50` |  |
| `select.pick.allow_empty` | `false` |  |
| `select.pick.out` | `${paths.out}/${pair}/pick.json` |  |
| `select.yield` | `null` | no saturation-yield comparison for code |
| `select.chosen_step` | `???` |  |
| `harvest.synthesizer` | `${paths.ckpt}/${synrl.experiment}/hf/step_${select.chosen_step}` | not released for the code pairs: retrain to re-harvest |
| `harvest.shards` | `12` |  |
| `harvest.per_shard` | `8000` |  |
| `harvest.seed_formula` | `"7919*i+104729"` | The seed MUST be a function of the shard (the math campaign once lost eight parallel shards to byte-identical output) and must NOT be a timestamp, or a resumed shard replays nothing. topic-rotation seed of shard i |
| `harvest.temperature` | `1.0` |  |
| `harvest.top_p` | `0.95` |  |
| `harvest.max_new` | `2048` |  |
| `harvest.prompt_head` | `2304` |  |
| `harvest.gpu_memory_utilization` | `0.85` |  |
| `harvest.strip_thinking` | `true` |  |
| `harvest.vllm_seed` | `0` |  |
| `harvest.out_dir` | `${paths.out}/${pair}/harvest` |  |
| `filter.domain` | `code` |  |
| `filter.slices` | `???` |  |
| `filter.jaccard` | `0.7` |  |
| `filter.sat_thr` | `0.95` |  |
| `filter.embed_bs` | `16` |  |
| `filter.aggregate` | `true` |  |
| `filter.agg_thr` | `0.95` |  |
| `filter.agg_order` | `???` |  |
| `filter.out_dir` | `${paths.out}/${pair}/filter` |  |
| `filter.pool` | `${paths.out}/${pair}/pool.jsonl` |  |
| `bank.name` | `???` |  |
| `bank.row_id_prefix` | `???` | extra_info.index = <row_id_prefix>_<i> (the released bank's) |
| `bank.n` | `10000` | budget-matched to OpenCodeReasoning-10k |
| `bank.pool` | `${filter.pool}` |  |
| `bank.template` | `lcb_no_system` | solve_prompt(q), no system turn |
| `bank.refilter` | `null` |  |
| `bank.allow_short` | `false` |  |
| `bank.expected_rows` | `null` |  |
| `bank.out` | `${paths.out}/${pair}/banks/${bank.name}.parquet` |  |
| `bank.released` | `${paths.data}/banks/${bank.name}.parquet` |  |
| `opd` |  | synthetic (ours) and real (vanilla) arms share these values |
| `opd.student` | `${models.student}` |  |
| `opd.teacher` | `${models.teacher}` |  |
| `opd.max_prompt_length` | `2048` | 1024 fits DAPO math. A code_contests statement in the LiveCodeBench template runs p90 938 / p99 1444 / max 1887, and filter_overlong_prompts DROPS anything over the cap silently -- at 1024 that is 7% of a real arm and a different 7% of a synthetic one. |
| `opd.max_response_length` | `4096` |  |
| `opd.max_val_response_length` | `4096` | max_model_len 6144 |
| `opd.test_files` | `[train]` | val_files has to point at something loadable; it is never scored, and setting it is what skips the AIME/AMC startup gate. the arm's training parquet |
| `opd.custom_reward` | `{path: "${paths.repo}/ask2distill/synrl/rule_reward.py", name: compute_score}` |  |
| `opd.is_plot` | `false` | The per-step diagnostic figures are written through swanlab to a shared, nearly full network filesystem: the driver blocked in a client sync-IO wait for over an hour at step 1, twice. The plots are diagnostics; the logged scalars are not affected. |
| `opd.esi_save` | `false` | The ESI trigger window is "longest step so far + save time"; one 1232s step stretched it to ~21 minutes and force-saved every step inside it (19 checkpoints, 883GB). Off for code: a resubmit loses at most save_freq steps. |

### `configs/base/tau2.yaml`

| Key | Value | Note |
|---|---|---|
| `domain` | `tau2` |  |
| `smoke` | `false` |  |
| `models.student` | `???` |  |
| `models.teacher` | `???` |  |
| `models.user_sim_train` | `???` |  |
| `models.user_sim_eval` | `???` |  |
| `models.embedder` | `${paths.models}/Qwen3-Embedding-4B` | diversity embedding |
| `models.qwen3_nothink` | `true` |  |
| `models.templates` | `???` | Chat templates are chosen by ROLE (the base model behind it), never by served name. A missing template is a hard error: vLLM's fallback is the model's own template, which silently re-enables thinking for Thinking-2507. |
| `models.hf` | `???` |  |
| `tau2_data.repo_path` | `${paths.third_party}/toolorchestra_tau2` | REPO_PATH (the fork's tau2/utils raises without it) |
| `tau2_data.data_dir` | `${tau2_data.repo_path}/data/tau2` | ToolOrchestra data/tau2 @c1c0a40 (downloaded, gitignored) |
| `tau2_data.db_dir_env` | `A2D_TAU2_DB_DIR` | Not TAU2_DATA: that name collides with the training job's prompt-parquet setting. |
| `tau2_data.db_dir` | `${tau2_data.data_dir}/domains` |  |
| `tau2_data.domain_ctx` | `${paths.data}/tau2/domain_ctx.json` | TAU2_CTX (system prompt + tool schemas) |
| `tau2_data.user_guidelines` | `${tau2_data.data_dir}/user_simulator/simulation_guidelines.md` | fork version, differs from upstream |
| `tau2_data.tokenizer_cache` | `${paths.cache}/hf` | The fork's llm_utils.py loads AutoTokenizer("Qwen/Qwen3-8B") at import. |
| `tau2_data.import_tokenizer` | `{repo: Qwen/Qwen3-8B, revision: b968826d9c46dd6066d109eabc6255188de91218}` |  |
| `tau2_data.toolorchestra` | `{repo: NVlabs/ToolOrchestra, revision: c1c0a4002be128b18accb267aac5ffdc26a28103}` |  |
| `tau2_data.upstream_split` | `{repo: sierra-research/tau2-bench, revision: b7ea9074c1cba482b30687fecdb5c8425f…` | tasks.json + split_tasks.json only |
| `tau2_data.tasks_test` | `${paths.data}/tau2/tasks_test` | airline 20 / retail 40 / telecom 40 |
| `reward_env` |  | synthesizer reward (env service + worker), rendered to TAU2_* |
| `reward_env.TAU2_K` | `6` | student rollouts per card |
| `reward_env.TAU2_KT` | `3` | teacher-gate rollouts |
| `reward_env.TAU2_GROUNDING` | `1` |  |
| `reward_env.TAU2_TEACHER_GATE` | `1` |  |
| `reward_env.TAU2_MAX_TURNS` | `12` | synthesizer-reward episodes (OPD uses opd.agent_env) |
| `reward_env.TAU2_GATE_TURNS` | `10` |  |
| `reward_env.TAU2_PREF_TURNS` | `25` |  |
| `reward_env.TAU2_V7_TURNS` | `30` |  |
| `reward_env.TAU2_PREF_TEACHER_GATE` | `0` |  |
| `reward_env.TAU2_PREF_JUDGE` | `1` | The preference judge must be Instruct-2507 (it reuses the user-simulator client): Thinking-2507 under a short budget always answers NO. |
| `reward_env.TAU2_DIVTHRESH` | `0.1` |  |
| `reward_env.TAU2_DIVW` | `1.0` |  |
| `reward_env.TAU2_SVC_TIMEOUT` | `3600` |  |
| `reward_env.TAU2_STUDENT_MODEL` | `Qwen3-4B` | served names |
| `reward_env.TAU2_USER_MODEL` | `Qwen3-4B-Instruct-2507` |  |
| `reward_env.TAU2_TEACHER_MODEL` | `Qwen3-4B-Thinking-2507` |  |
| `synrl.train_batch_size` | `8` |  |
| `synrl.ppo_mini_batch_size` | `8` |  |
| `synrl.ppo_micro_batch_size_per_gpu` | `1` |  |
| `synrl.rollout_n` | `16` | 16 cards per prompt |
| `synrl.lr` | `1.0e-6` |  |
| `synrl.lr_scheduler_type` | `constant` |  |
| `synrl.use_kl_loss` | `true` |  |
| `synrl.kl_loss_coef` | `0.001` |  |
| `synrl.entropy_coeff` | `0.001` |  |
| `synrl.max_response_length` | `1536` | one card |
| `synrl.ppo_max_token_len_per_gpu` | `prompt_plus_resp` | versions.<v>.max_prompt_length + max_response_length |
| `synrl.max_num_batched_tokens` | `16384` |  |
| `synrl.gpu_memory_utilization` | `0.5` |  |
| `synrl.offload` | `true` | actor params + optimizer |
| `synrl.model_dtype` | `null` | not passed in the final runs (verl default); smoke sets bf16 |
| `synrl.train_max_samples` | `-1` | verl default (smoke caps it) |
| `synrl.ref_param_offload` | `true` |  |
| `synrl.use_remove_padding` | `true` |  |
| `synrl.gradient_checkpointing` | `true` |  |
| `synrl.shuffle` | `false` |  |
| `synrl.filter_overlong_prompts` | `true` |  |
| `synrl.return_raw_chat` | `true` |  |
| `synrl.enable_thinking` | `false` |  |
| `synrl.reward_model_path` | `${models.student}` | placeholder; the worker loads no reward model |
| `synrl.reward_micro_batch_size_per_gpu` | `1` |  |
| `synrl.enable_resource_pool` | `false` | No expandable_segments in the tau2 worker, so colocating it with the synthesizer's vLLM rollout is safe. |
| `synrl.n_gpus_per_node` | `8` | rank batch = 8 x 16 / 8 = 16 cards |
| `synrl.nnodes` | `1` |  |
| `synrl.critic_warmup` | `0` |  |
| `synrl.logger` | `console` |  |
| `synrl.project_name` | `synrl_tau2` |  |
| `synrl.total_steps` | `60` |  |
| `synrl.total_epochs` | `1` |  |
| `synrl.save_freq` | `5` |  |
| `synrl.esi_redundant_time` | `0` |  |
| `synrl.test_freq` | `-1` |  |
| `synrl.val_before_train` | `false` |  |
| `synrl.resume_mode` | `auto` |  |
| `synrl.custom_reward` | `{path: "${paths.repo}/ask2distill/synrl/rule_reward.py", name: compute_score}` |  |
| `synrl.gen_tokenizer` | `${models.student}` |  |
| `synrl.env_service` | `{port: 8710}` |  |
| `synrl.diversity.group` | `16` | the rank batch (16 cards with 8 actor ranks); unique card <= 15/16 |
| `synrl.diversity.scope` | `valid_only` | only parse-valid cards are clustered |
| `synrl.diversity.text` | `scenario_json` | json.dumps(card["scenario"], ensure_ascii=False) |
| `synrl.diversity.thresh` | `${reward_env.TAU2_DIVTHRESH}` |  |
| `synrl.diversity.pen` | `${reward_env.TAU2_DIVW}` |  |
| `synrl.diversity.max_len` | `512` |  |
| `synrl.diversity.embed_bs` | `16` |  |
| `synrl.step_counter` | `global_steps` | When the env service fails, every card of that step scores 0 (svc_down); kept as run. dump rows carry verl's global step |
| `synrl.serve` |  | two serve nodes |
| `synrl.serve.node_maps` | `["Qwen3-4B=2\|Qwen3-4B-Instruct-2507=3\|Qwen3-4B-Thinking-2507=3", "Qwen3-4B=4\…` |  |
| `synrl.serve.gpu_util` | `0.90` |  |
| `synrl.serve.max_len` | `32768` |  |
| `synrl.serve.tp` | `1` |  |
| `synrl.serve.sampling_overrides` | `none` | no --generation-config / sampling flags (generation_config defaults) |
| `synrl.extra_overrides` | `[]` |  |
| `synrl.forbidden_overrides` | `[]` |  |
| `synrl.versions` | `???` |  |
| `select` |  | synthesizer checkpoint per version |
| `select.rule` | `{window: 4, score: valid_frac_x_mean_R_valid}` | Trailing 4-step window of (valid fraction x mean R over valid cards); the steps used are synrl.versions.<v>.chosen_step. |
| `harvest.v6` | `???` |  |
| `harvest.v7` | `???` |  |
| `harvest.complete_rule` | `all_targets_x_per_target` | counted over targets x per-target |
| `harvest.redo_errors` | `[rollout_error, gen_error]` | infra-failed rows are redone |
| `harvest.gen_sampling` | `{temperature: 1.0, max_tokens: 2048}` | top_p / top_k from the server's generation_config |
| `harvest.serve` | `{replicas_per_model: 2, nodes: 2, gpu_util: 0.90, max_len: 32768}` |  |
| `harvest.env` | `{TAU2_GROUNDING: 1, TAU2_TEACHER_GATE: 1, TAU2_KT: 3}` | TAU2_DATA is never set |
| `harvest.out_dir` | `${paths.out}/${pair}/harvest` |  |
| `bank` | `???` |  |
| `opd.student` | `${models.student}` |  |
| `opd.teacher` | `${models.teacher}` |  |
| `opd.chat_template_args` | `"+data.apply_chat_template_kwargs.enable_thinking=False"` |  |
| `opd.max_prompt_length` | `6144` | system policy + tool schemas + greeting + first user turn |
| `opd.max_response_length` | `8192` | ONE budget shared by all turns: policy tokens (mask 1) and observations (mask 0) |
| `opd.max_val_response_length` | `8192` |  |
| `opd.mini_batch_size` | `8` | = train_batch_size |
| `opd.n_responses` | `4` | episodes per task card |
| `opd.shuffle` | `true` | math / code: false |
| `opd.test_files` | `${paths.data}/eval/math/AIME25/test.parquet, ${paths.data}/eval/math/AMC23/test…` | No test file is set for tau2, so the launcher falls back to the math validation files, as in the final runs. Inert: test_freq -1. |
| `opd.custom_reward` | `{path: "${paths.repo}/ask2distill/synrl/rule_reward.py", name: compute_score}` |  |
| `opd.is_plot` | `true` |  |
| `opd.esi_save` | `true` | force-save before the wall clock |
| `opd.total_training_steps` | `200` | the reported arm (the run was later extended to 300 steps) |
| `opd.ckpt_keep` | `3` |  |
| `opd.extra_overrides` | `actor_rollout_ref.rollout.mode=async, actor_rollout_ref.rollout.agent.default_a…` | appended after the shared OPD overrides, in this order |
| `opd.forbidden_overrides` | `[trainer.rollout_data_dir]` | verl's _dump_generations json-dumps a Tensor from the agent-loop batch and crashes the first step. |
| `opd.agent_env` |  | delivered to every AgentLoopWorker via runtime_env.env_vars |
| `opd.agent_env.TAU2_MAX_TURNS` | `20` | at most 20 assistant turns per episode |
| `opd.agent_env.TAU2_STOP_ON_REPEAT` | `1` | Stop-on-repeat: when the student's user-facing reply near-duplicates its previous one, that turn is removed from the sequence and the mask and the episode ends -- a small model's degenerate sign-off loop never reaches the loss. |
| `opd.agent_env.TAU2_REPEAT_SIM` | `0.9` | near-duplicate = difflib ratio >= 0.9 (first 600 chars) |
| `opd.agent_env.TAU2_MAX_TOOL_CHARS` | `2000` | tool output is truncated before it enters the context (mask 0) |
| `opd.agent_env.TAU2_USER_RETRY_S` | `1800` |  |
| `opd.user_sim_serve` |  | One node serves the user simulator; an arm may override `map` (arms.<arm>.user_sim_serve). frozen user simulator (Qwen3-4B-Instruct-2507, temperature 1.0, 512 tokens per turn) |
| `opd.user_sim_serve.map` | `???` |  |
| `opd.user_sim_serve.max_len` | `40960` |  |
| `opd.user_sim_serve.gpu_util` | `0.90` |  |
| `opd.user_sim_serve.replicas` | `1` |  |
| `opd.env_service` | `{port: 8711, workers: 64}` | env-only: tool calls, no reward |
| `opd.arms` | `???` |  |
| `eval.domains` | `[airline, retail, telecom]` |  |
| `eval.tasks_dir` | `${tau2_data.tasks_test}` |  |
| `eval.tag` | `test4` |  |
| `eval.trials` | `4` |  |
| `eval.user_llm` | `gpt-4o-mini` | evaluation user simulator |
| `eval.max_steps` | `200` |  |
| `eval.concurrency` | `12` |  |
| `eval.TAU2_MAX_TOKENS` | `8000` |  |
| `eval.TAU2_ORCHESTRATOR_PROMPT` | `0` | ToolOrchestra's <think>/<message> contract off |
| `eval.TAU2_OPENAI_DIRECT` | `1` | gpt-4o-mini branch |
| `eval.openai_env_file` | `${paths.openai_env_file}` | the user's own key file; never copied |
| `eval.openai_base_url` | `https://api.openai.com/v1` | written explicitly into the temp env file (defeats a relay in ~/.bashrc) |
| `eval.served_name` | `"${pair}-{short_exp}-s{step}"` | ToolOrchestra routes the agent model to its vLLM client only when the name contains "qwen": <pair>-<arm short_name>-s<step>; the pair name carries the student's "Qwen3". rendered by tau2.serve; must contain "qwen" |
| `eval.template_role` | `student` |  |
| `eval.serve` | `{gpus: 1, max_len: 40960, gpu_util: 0.85, sampling_overrides: none}` | one GPU per replica; vLLM defaults otherwise |
| `eval.watchdog_min` | `30` | no new result for 30 min -> restart the pass |
| `eval.num_tasks` | `null` | tasks per domain counted as complete; null = all. The fork ignores --num-tasks without --task-ids, so to RUN fewer tasks point tasks_dir at trimmed files (tests/smoke/) |
| `eval.pass_threshold` | `0.999` | pass^1 = reward >= 0.999 |
| `eval.out_dir` | `${paths.out}/tau2/outputs` |  |

### `configs/base/opd.yaml`

| Key | Value | Note |
|---|---|---|
| `opd.student` | `???` |  |
| `opd.teacher` | `???` |  |
| `opd.adv_estimator` | `token_reward_direct` | advantage = reward x mask; no baseline / GAE / group norm |
| `opd.grpo_outcome_weight` | `1.0` | no effect under token_reward_direct |
| `opd.log_prob_top_k` | `16` | top-16 OPD |
| `opd.top_k_strategy` | `only_stu` | support = the student's top-16 |
| `opd.reward_weight_mode` | `student_p` | reverse KL weighted by the renormalised student probabilities |
| `opd.temperature` | `1.0` | rollout temperature |
| `opd.teacher_temperature` | `1.0` |  |
| `opd.repetition_penalty` | `1.0` |  |
| `opd.n_responses` | `4` |  |
| `opd.mini_batch_size` | `64` | data.train_batch_size = mini x parallel_size |
| `opd.parallel_size` | `1` | ulysses sequence parallel and rollout TP |
| `opd.lr` | `1.0e-6` |  |
| `opd.model_dtype` | `fp32` | fp32 is the paper default and it matters: at lr=1e-6 the relative update is far below bf16's resolution, so with bf16 master weights updates get rounded away. |
| `opd.rollout_gpu_memory_utilization` | `0.8` | 80GB cards |
| `opd.loss_agg_mode` | `token-mean` |  |
| `opd.enable_format_reward` | `false` |  |
| `opd.use_kl` | `false` |  |
| `opd.kl_coef` | `0.005` | only read when use_kl |
| `opd.kl_loss_type` | `low_var_kl` |  |
| `opd.entropy_coeff` | `0` |  |
| `opd.shuffle` | `false` | every epoch replays the same order |
| `opd.filter_overlong_prompts` | `true` | drops prompts over max_prompt_length silently |
| `opd.truncation` | `error` |  |
| `opd.return_raw_chat` | `true` |  |
| `opd.ppo_micro_batch_size_per_gpu` | `1` | use_dynamic_bsz=True |
| `opd.ref_log_prob_micro_batch_size_per_gpu` | `1` |  |
| `opd.use_remove_padding` | `true` |  |
| `opd.offload.actor_param` | `true` |  |
| `opd.offload.actor_optimizer` | `true` |  |
| `opd.offload.activation` | `true` |  |
| `opd.offload.gradient_checkpointing` | `true` |  |
| `opd.offload.forward_prefetch` | `true` |  |
| `opd.offload.ref_param` | `true` |  |
| `opd.offload.reward_model_param` | `true` |  |
| `opd.reward_model_input_tokenizer` | `null` |  |
| `opd.calculate_log_probs` | `true` |  |
| `opd.esi_redundant_time` | `0` | Keep this at 0. verl re-evaluates the expiration check every step and has no "already saved for expiration" flag, so the trigger window is one save PER STEP until the job dies. A run with 600 force-saved on steps 114-120: 7 checkpoints, 6 of them redundant, 120GB of disk. |
| `opd.val_kwargs` | `{do_sample: true, n: 16, temperature: 0.7, top_p: 0.95}` | never used: test_freq -1 |
| `opd.max_val_response_length` | `???` | feeds max_model_len and val_kwargs.max_tokens |
| `opd.test_freq` | `-1` |  |
| `opd.val_before_train` | `false` |  |
| `opd.log_val_generations` | `2` |  |
| `opd.n_gpus_per_node` | `8` |  |
| `opd.nnodes` | `1` | every final arm ran on one node |
| `opd.resume_mode` | `auto` |  |
| `opd.project_name` | `OnPolicyDistillation` |  |
| `opd.swanlab_mode` | `local` |  |
| `opd.logger` | `[console, swanlab]` |  |
| `opd.train_max_samples` | `-1` | verl default (smoke caps it) |
| `opd.val_max_samples` | `-1` | verl default (smoke caps it) |
| `opd.chat_template_args` | `???` |  |
| `opd.max_prompt_length` | `???` |  |
| `opd.max_response_length` | `???` |  |
| `opd.test_files` | `???` |  |
| `opd.custom_reward` | `???` |  |
| `opd.is_plot` | `???` |  |
| `opd.esi_save` | `???` |  |
| `opd.extra_overrides` | `[]` |  |
| `opd.forbidden_overrides` | `[]` |  |
| `opd.arms` | `???` |  |

### `configs/base/eval_math.yaml`

| Key | Value | Note |
|---|---|---|
| `eval.tasks` | `[AIME24, AIME25, AIME26, AMC23, BRUMO25, MATH500]` | avg6 |
| `eval.optional_tasks` | `[HMMT_FEB25, HMMT_FEB26]` | extra; not part of avg6 |
| `eval.curve_tasks` | `[AIME24, AIME25, AMC23]` | tasks of `a2d eval math --curve` |
| `eval.data_dir` | `${paths.data}/eval/math` |  |
| `eval.task_dirs` | `{AIME24: AIME24, AIME25: AIME25, AIME26: AIME26, AMC23: AMC23, BRUMO25: BRUMO25…` | MATH500 -> MATH-500/test.parquet |
| `eval.collect_keys` | `{AIME24: aime24, AIME25: aime25, AIME26: aime26, AMC23: amc23, BRUMO25: brumo25…` | grading_results task_name (lower-cased) |
| `eval.n` | `{default: 16, MATH500: 4}` | avg@16; MATH500 avg@4 |
| `eval.max_tokens` | `31744` |  |
| `eval.temperature` | `0.7` |  |
| `eval.top_p` | `0.95` |  |
| `eval.gpu_memory_utilization` | `0.9` |  |
| `eval.disable_thinking` | `true` | the DeepSeek template ignores it |
| `eval.enforce_eager` | `true` |  |
| `eval.strict_boxed` | `false` | "Answer:" fallback on |
| `eval.len_tokenizer` | `${paths.models}/DeepSeek-R1-Distill-Qwen-1.5B` | avg_output_length statistic only, never a score |
| `eval.stop_ids_rule` | `first_token_of_im_end_and_endoftext` | Stop ids are the FIRST token of encode("<\|im_end\|>") and encode("<\|endoftext\|>"). Under the R1-Distill tokenizer both encode to "<" first, so token 27 is a stop id for the two DeepSeek-R1-Distill pairs -- kept as run (docs/DEVIATIONS.md). |
| `eval.limit` | `null` | --limit k: first k problems per task (smoke); null = all |
| `eval.engine_seed` | `null` | off in the final runs (one generate per rollout_id, no per-request seed) |
| `eval.out_dir` | `${paths.out}/eval/math` | <exp>_step<N>/<task>_t0.7_p0.95_n16-MNT31744.jsonl + grading_results.json |

### `configs/base/eval_code.yaml`

| Key | Value | Note |
|---|---|---|
| `eval.tag` | `"{experiment}_step{step}"` | TAG is always <experiment>_step<N>. LCB's --continue_existing keys its cached generations on the tag, so a tag reused across runs (e.g. the basename "step_150-nothink") silently scores another run's generations. Rendered by run_code_eval, not interpolated. |
| `eval.prompt_style` | `???` | the student's style (`python -m ask2distill.eval.code.prompt_style`) |
| `eval.think` | `false` | non-thinking evaluation protocol |
| `eval.lcb.version` | `v5_v6` | test5 + test6 = 342 problems |
| `eval.lcb.dataset` | `livecodebench/code_generation_lite` |  |
| `eval.lcb.dataset_revision` | `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` | pinned (the patched loader reads the shard jsonl) |
| `eval.lcb.n` | `4` |  |
| `eval.lcb.temperature` | `0.7` | non-thinking sampling (Qwen3 model card) |
| `eval.lcb.top_p` | `0.8` |  |
| `eval.lcb.top_k` | `20` |  |
| `eval.lcb.min_p` | `0` |  |
| `eval.lcb.max_tokens` | `8192` | non-thinking generation budget |
| `eval.lcb.max_model_len` | `16384` | capped at the model's context |
| `eval.lcb.stop` | `"<\|im_end\|>"` |  |
| `eval.lcb.timeout` | `6` |  |
| `eval.lcb.num_process_evaluate` | `24` |  |
| `eval.lcb.tp` | `2` | GPUs of the 2-GPU eval job |
| `eval.lcb.continue_existing` | `true` |  |
| `eval.lcb.cwd` | `${paths.out}/eval/code/_lcb_cwd` | LCB writes output/<TAG>/ under its cwd |
| `eval.evalplus` |  | official protocol: greedy, assistant prefill |
| `eval.evalplus.datasets` | `[mbpp, humaneval]` | final checkpoints; HE+ reported, not in Avg |
| `eval.evalplus.greedy` | `true` |  |
| `eval.evalplus.max_new_tokens` | `1024` |  |
| `eval.evalplus.max_model_len` | `4096` |  |
| `eval.evalplus.parallel` | `24` |  |
| `eval.evalplus.tp` | `2` |  |
| `eval.curve_evalplus_datasets` | `[mbpp]` | curve points skip HE+ |
| `eval.mbpp_lenient` | `true` | MBPP's required function name lives in the prompt's assert, and training on competitive-programming problems (which never impose one) drifts that habit, so strict and lenient scores of one model can differ. Scoring both keeps "cannot solve it" separate from "renamed it". |
| `eval.extra_bench.benches` | `[apps_introductory, apps_interview1k, apps_competition, codecontests]` |  |
| `eval.extra_bench.data_dir` | `${paths.data}/eval/code/extra_bench` |  |
| `eval.extra_bench.n` | `1` |  |
| `eval.extra_bench.temperature` | `0.2` |  |
| `eval.extra_bench.top_p` | `0.95` |  |
| `eval.extra_bench.top_k` | `20` |  |
| `eval.extra_bench.max_new` | `4096` |  |
| `eval.extra_bench.max_model_len_extra` | `4096` | max_model_len = max_new + 4096 |
| `eval.extra_bench.tp` | `2` |  |
| `eval.extra_bench.util` | `0.9` |  |
| `eval.extra_bench.system_prompt` | `"You are a helpful assistant."` |  |
| `eval.extra_bench.nothink_close` | `true` | close a dangling <think> |
| `eval.extra_bench.extraction` | `CodeQwenInstruct` | LCB extract_code style |
| `eval.extra_bench.num_process_evaluate` | `24` |  |
| `eval.extra_bench.timeout` | `6` |  |
| `eval.extra_bench.detail` | `false` | opt-in per-problem output |
| `eval.extra_bench.limit` | `null` | first k problems (smoke); null = all |
| `eval.big_model_rule` | `{weights_gb_per_gpu_gt: 10, util: 0.85, enforce_eager: true, default_util: 0.9}` | vLLM sizes its KV cache for gpu_memory_utilization, then captures CUDA graphs on top; a 32B model at 0.9 overshoots by ~8GB and dies during capture. |
| `eval.avg_items` | `[lcb, mbpp_plus_lenient, apps_introductory, apps_interview1k, apps_competition,…` |  |
| `eval.out_dir` | `${paths.out}/eval/code` | <TAG>/{livecodebench_v5_v6_eval.json, evalplus/, evalplus_mbpp_lenient.log, xb/} |

### `configs/pairs/Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500.yaml`

| Key | Value | Note |
|---|---|---|
| `_base_` | `[../base/opd.yaml, ../base/math.yaml, ../base/eval_math.yaml]` |  |
| `pair` | `Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500` |  |
| `models.student` | `${paths.models}/Qwen3-1.7B` |  |
| `models.teacher` | `${paths.models}/Qwen3-4B-Non-Thinking-RL-Math-Step500` |  |
| `models.synthesizer_init` | `${models.student}` | S_θ is initialised from the student |
| `models.qwen3_nothink` | `true` |  |
| `models.hf` |  | pinned from <model>/.cache/huggingface/download/*.metadata |
| `models.hf.student` | `{repo: Qwen/Qwen3-1.7B, revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}` |  |
| `models.hf.teacher` | `{repo: Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500, revision: 05d82d02780d4a6…` |  |
| `models.hf.embedder` | `{repo: Qwen/Qwen3-Embedding-4B, revision: 5cf2132abc99cad020ac570b19d031efec650…` |  |
| `calib.expected_delta_ref` | `1.9049` |  |
| `reward.delta_ref` | `1.9049` | round(p95, 4) |
| `synrl.experiment` | `"synthesizer_${pair}"` |  |
| `synrl.offload` | `false` |  |
| `synrl.gpu_memory_utilization` | `0.6` |  |
| `select.yield` | `{candidates: [35, 50, 60], n: 5000, seed: 100, bank_n: 64, tie: earliest}` | Yield = saturation emb count of 5000 questions at seed 100. |
| `select.chosen_step` | `50` |  |
| `harvest.shards` | `16` |  |
| `filter.slices` | `[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]` | ONE 80k saturation pass over all 16 shards; no cross-slice step. Numeric shard order, not lexicographic: the bank takes the FIRST N questions in this order, and s10 sorting before s2 would change which questions the arm trains on first. |
| `filter.aggregate` | `false` |  |
| `filter.agg_order` | `null` |  |
| `bank.name` | `${pair}` |  |
| `bank.row_id_prefix` | `q3armg9` | extra_info.index of the released bank |
| `bank.n` | `3200` | 50 steps x 64 |
| `bank.pool` | `${filter.out_dir}/sat.s0_questions.jsonl` |  |
| `bank.allow_short` | `true` | The filter only yielded 2,889 independent questions (dedup-limited), so the bank is short of N; the final run trained on all 2,889. |
| `bank.expected_rows` | `2889` |  |
| `opd.chat_template_args` | `"+data.apply_chat_template_kwargs.enable_thinking=False"` |  |
| `opd.arms.ours` | `{train: "${bank.released}", experiment: "opd_${pair}_ours", total_epochs: 2, sa…` | 2,889 questions is 45 steps per epoch; two epochs replay the same order (shuffle=False) -> 90 steps, saved on a 5-step grid. |
| `opd.arms.vanilla` | `{train: "${paths.data}/real/DAPO-Math-17k.parquet", experiment: "opd_${pair}_va…` |  |

### `configs/pairs/Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B.yaml`

| Key | Value | Note |
|---|---|---|
| `_base_` | `[../base/opd.yaml, ../base/math.yaml, ../base/eval_math.yaml]` |  |
| `pair` | `Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B` |  |
| `models.student` | `${paths.models}/DeepSeek-R1-Distill-Qwen-1.5B` |  |
| `models.teacher` | `${paths.models}/JustRL-DeepSeek-1.5B` |  |
| `models.synthesizer_init` | `${paths.models}/DeepSeek-R1-Distill-Qwen-1.5B-SynNoThink` | Same weights as the student with a chat template that closes <think> immediately: the DeepSeek analogue of enable_thinking=False. |
| `models.qwen3_nothink` | `false` |  |
| `models.syn_nothink` | `true` | built by `a2d models syn-nothink --pair Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B` |
| `models.hf.student` | `{repo: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, revision: ad9f0ae0864d7fbcd1c…` |  |
| `models.hf.teacher` | `{repo: hbx/JustRL-DeepSeek-1.5B, revision: 0637e4096c789c67f9eecbe8355e0bdedded…` |  |
| `models.hf.embedder` | `{repo: Qwen/Qwen3-Embedding-4B, revision: 5cf2132abc99cad020ac570b19d031efec650…` |  |
| `calib.expected_delta_ref` | `0.4411` |  |
| `reward.delta_ref` | `0.4411` | round(p95, 4) |
| `synrl.experiment` | `"synthesizer_${pair}"` |  |
| `synrl.offload` | `false` |  |
| `synrl.gpu_memory_utilization` | `0.6` |  |
| `select.yield` | `{candidates: [10, 40, 60], n: 5000, seed: 100, bank_n: 64, tie: earliest}` | Yield = saturation emb count of 5000 questions at seed 100. |
| `select.chosen_step` | `40` |  |
| `harvest.shards` | `16` |  |
| `filter.slices` | `[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]` |  |
| `filter.aggregate` | `true` |  |
| `filter.agg_order` | `[0, 1, 2, 3]` |  |
| `bank.name` | `${pair}` |  |
| `bank.row_id_prefix` | `b15arm` | extra_info.index of the released bank |
| `bank.n` | `12800` | 200 steps x 64 |
| `bank.pool` | `${filter.pool}` |  |
| `opd.chat_template_args` | `""` | DeepSeek template has no thinking switch |
| `opd.arms.ours` | `{train: "${bank.released}", experiment: "opd_${pair}_ours", total_epochs: 2, sa…` |  |
| `opd.arms.vanilla` | `{train: "${paths.data}/real/DAPO-Math-17k.parquet", experiment: "opd_${pair}_va…` | Vanilla ran on the plain launcher defaults, including 32768 tokens / micro batch 8 where every other arm used 16384 / 4. token-mean is computed per micro-batch, so this changes the loss weighting. |

### `configs/pairs/Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B.yaml`

| Key | Value | Note |
|---|---|---|
| `_base_` | `[../base/opd.yaml, ../base/math.yaml, ../base/eval_math.yaml]` |  |
| `pair` | `Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B` |  |
| `models.student` | `${paths.models}/DeepSeek-R1-Distill-Qwen-7B` |  |
| `models.teacher` | `${paths.models}/Skywork-OR1-Math-7B` |  |
| `models.synthesizer_init` | `${paths.models}/DeepSeek-R1-Distill-Qwen-7B-SynNoThink` |  |
| `models.qwen3_nothink` | `false` |  |
| `models.syn_nothink` | `true` | built by `a2d models syn-nothink --pair Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B` |
| `models.hf.student` | `{repo: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, revision: 916b56a44061fd5cd7d6a…` |  |
| `models.hf.teacher` | `{repo: Skywork/Skywork-OR1-Math-7B, revision: fcef970f1d10682f13c182b280146a194…` |  |
| `models.hf.embedder` | `{repo: Qwen/Qwen3-Embedding-4B, revision: 5cf2132abc99cad020ac570b19d031efec650…` |  |
| `calib.expected_delta_ref` | `0.4867` |  |
| `reward.delta_ref` | `0.4867` | round(p95, 4) |
| `synrl.experiment` | `"synthesizer_${pair}"` | the run restarted from step 0 under its own name |
| `synrl.offload` | `true` | The 7B synthesizer cannot hold params and optimizer resident alongside vLLM. |
| `synrl.gpu_memory_utilization` | `0.5` |  |
| `select.yield` | `null` |  |
| `select.chosen_step` | `55` |  |
| `harvest.shards` | `8` |  |
| `filter.slices` | `[[0, 1, 2, 3], [4, 5, 6, 7]]` |  |
| `filter.aggregate` | `true` |  |
| `filter.agg_order` | `[0, 1]` |  |
| `bank.name` | `${pair}` |  |
| `bank.row_id_prefix` | `b7barm` | extra_info.index of the released bank |
| `bank.n` | `12800` |  |
| `bank.pool` | `${filter.pool}` |  |
| `opd.chat_template_args` | `""` |  |
| `opd.arms.ours` | `{train: "${bank.released}", experiment: "opd_${pair}_ours", total_epochs: 2, sa…` | 12800 questions x 2 epochs = 400 steps. |
| `opd.arms.vanilla` | `{train: "${paths.data}/real/DAPO-Math-17k.parquet", experiment: "opd_${pair}_va…` |  |

### `configs/pairs/Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507.yaml`

| Key | Value | Note |
|---|---|---|
| `_base_` | `[../base/opd.yaml, ../base/code.yaml, ../base/eval_code.yaml]` |  |
| `pair` | `Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507` |  |
| `models.student` | `${paths.models}/Qwen3-4B` | Hybrid Qwen3-4B held in non-thinking mode, NOT the single-mode Instruct-2507: its gap to the teacher is larger, and the gap is what the divergence reward feeds on. The student's template injects an empty <think></think> into the PROMPT, which the 2507 teacher never saw. |
| `models.teacher` | `${paths.models}/Qwen3-30B-A3B-Instruct-2507` |  |
| `models.qwen3_nothink` | `true` | enable_thinking=False everywhere |
| `models.hf.student` | `{repo: Qwen/Qwen3-4B, revision: 1cfa9a7208912126459214e8b04321603b3df60c}` |  |
| `models.hf.teacher` | `{repo: Qwen/Qwen3-30B-A3B-Instruct-2507, revision: 0d7cf23991f47feeb3a57ecb4c9c…` |  |
| `models.hf.embedder` | `{repo: Qwen/Qwen3-Embedding-4B, revision: 5cf2132abc99cad020ac570b19d031efec650…` |  |
| `calib.expected_delta_ref` | `3.9756` |  |
| `reward.delta_ref` | `3.9756` | `a2d calibrate`: p95 of δ_t on 200 initial-synthesizer questions |
| `reward.gen_bs` | `2` | 30B MoE teacher + solver = 64.4GB of weights on every reward rank, so the generation batch is capped; the final run used 2. |
| `synrl.experiment` | `"synthesizer_${pair}"` |  |
| `synrl.save_freq` | `5` |  |
| `select.sweep.steps` | `[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]` |  |
| `select.chosen_step` | `60` |  |
| `filter.slices` | `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]` | 4 slices x 3 shards |
| `filter.agg_order` | `[0, 1, 2, 3]` | ls order; lexicographic == numeric for 4 slices |
| `bank.name` | `${pair}` |  |
| `bank.row_id_prefix` | `code_a_syn10k` | extra_info.index of the released bank |
| `bank.pool` | `${paths.out}/${pair}/pool_clean.jsonl` |  |
| `bank.refilter` | `{kind: comp_strong_salvage_exact_dedup, input: "${filter.pool}", output: "${ban…` | This pair's slice funnel ran under an earlier code_filter; the aggregated pool was re-filtered with the current one before the bank was rebuilt: comp_filter + strong_filter salvage (may rewrite text) + exact-string dedup, 29,675 -> 26,275 rows. |
| `opd.chat_template_args` | `"+data.apply_chat_template_kwargs.enable_thinking=False"` |  |
| `opd.arms.ours` | `{train: "${bank.released}", experiment: "opd_${pair}_ours", total_epochs: 1, sa…` |  |
| `opd.arms.vanilla` | `{train: "${paths.data}/real/OpenCodeReasoning-10k.parquet", experiment: "opd_${…` | same values as the ours arm |
| `eval.prompt_style` | `Qwen3NoThink` | hybrid Qwen3 in non-thinking mode |

### `configs/pairs/Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct.yaml`

| Key | Value | Note |
|---|---|---|
| `_base_` | `[../base/opd.yaml, ../base/code.yaml, ../base/eval_code.yaml]` |  |
| `pair` | `Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct` |  |
| `models.student` | `${paths.models}/Qwen2.5-Coder-3B-Instruct` |  |
| `models.teacher` | `${paths.models}/Qwen2.5-Coder-32B-Instruct` |  |
| `models.qwen3_nothink` | `false` | Qwen2.5-Coder has no thinking switch |
| `models.hf.student` | `{repo: Qwen/Qwen2.5-Coder-3B-Instruct, revision: 488639f1ff808d1d3d0ba301aef8c1…` |  |
| `models.hf.teacher` | `{repo: Qwen/Qwen2.5-Coder-32B-Instruct, revision: 381fc969f78efac66bc87ff7ddead…` |  |
| `models.hf.embedder` | `{repo: Qwen/Qwen3-Embedding-4B, revision: 5cf2132abc99cad020ac570b19d031efec650…` |  |
| `calib.expected_delta_ref` | `1.3629` |  |
| `reward.delta_ref` | `1.3629` | Vocab sizes differ (151936 vs 152064); both heads are cut to the common size before the softmax. The solver's generation_config supplies repetition_penalty 1.05 (sample_top_k 0), so the reward rollouts used it. `a2d calibrate`: p95 of δ_t on 200 initial-synthesizer questions |
| `reward.gen_bs` | `4` |  |
| `synrl.experiment` | `"synthesizer_${pair}"` |  |
| `synrl.save_freq` | `5` |  |
| `select.sweep.steps` | `[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]` |  |
| `select.chosen_step` | `40` |  |
| `filter.slices` | `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]` |  |
| `filter.agg_order` | `[0, 1, 2, 3]` |  |
| `bank.name` | `${pair}` | 9,981 rows survive verl's overlong-prompt filter (155 steps) |
| `bank.row_id_prefix` | `code_b_syn10k` | extra_info.index of the released bank |
| `opd.chat_template_args` | `""` |  |
| `opd.use_kl` | `true` | Same-family pair: the token distributions barely differ (δ_ref 1.3629 vs 3.9756 for the Qwen3-4B <- Qwen3-30B-A3B pair), the OPD gradient is weak and length is the only free direction left; without these the student's entropy and rollout length collapse. Both arms of the pair get the same setting. |
| `opd.kl_coef` | `0.005` |  |
| `opd.entropy_coeff` | `0.005` |  |
| `opd.arms.ours` | `{train: "${bank.released}", experiment: "opd_${pair}_ours", total_epochs: 1, sa…` |  |
| `opd.arms.vanilla` | `{train: "${paths.data}/real/OpenCodeReasoning-10k.parquet", experiment: "opd_${…` | restarted from step 1 with the KL / entropy terms above |
| `eval.prompt_style` | `CodeQwenInstruct` | detected for Qwen2.5-Coder (`python -m ask2distill.eval.code.prompt_style`) |

### `configs/pairs/Code_Qwen3-4B_Qwen3-4B-Instruct-2507.yaml`

| Key | Value | Note |
|---|---|---|
| `_base_` | `[../base/opd.yaml, ../base/code.yaml, ../base/eval_code.yaml]` |  |
| `pair` | `Code_Qwen3-4B_Qwen3-4B-Instruct-2507` |  |
| `models.student` | `${paths.models}/Qwen3-4B` | Same student as Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507, teacher swapped for Qwen's own "4B + more post-training": single-mode, direct-answer, no reasoning to leak. |
| `models.teacher` | `${paths.models}/Qwen3-4B-Instruct-2507` |  |
| `models.qwen3_nothink` | `true` |  |
| `models.hf.student` | `{repo: Qwen/Qwen3-4B, revision: 1cfa9a7208912126459214e8b04321603b3df60c}` |  |
| `models.hf.teacher` | `{repo: Qwen/Qwen3-4B-Instruct-2507, revision: cdbee75f17c01a7cc42f958dc65090717…` |  |
| `models.hf.embedder` | `{repo: Qwen/Qwen3-Embedding-4B, revision: 5cf2132abc99cad020ac570b19d031efec650…` |  |
| `calib.expected_delta_ref` | `3.8221` |  |
| `reward.delta_ref` | `3.8221` | `a2d calibrate`: p95 of δ_t on 200 initial-synthesizer questions |
| `reward.gen_bs` | `8` | 4B + 4B of weights per reward rank |
| `synrl.experiment` | `"synthesizer_${pair}"` |  |
| `synrl.save_freq` | `10` | the cadence chosen for this pair |
| `select.sweep.steps` | `[10, 20, 30, 40, 50, 60]` | swept incrementally as checkpoints landed |
| `select.chosen_step` | `50` |  |
| `filter.slices` | `[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]` | 12 one-shard slices, not 4: the Jaccard pass compares each question against every earlier survivor in its slice, so a slice costs ~n^2 -- 24k questions ran 1h26 on the other two code pairs, 8k run ~10 min. The in-slice Jaccard then covers only 8k questions and the cross-slice embedding step removes 29% (61,263 -> 43,575). |
| `filter.agg_order` | `[0, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9]` | The aggregate listed the slice files with `ls`, i.e. lexicographic order, and the bank takes the first N questions in that order (s0 / s10 / s11 supplied 5,138 / 4,252 / 610). Kept as run. |
| `bank.name` | `${pair}` | 9,992 rows survive verl's overlong-prompt filter (156 steps) |
| `bank.row_id_prefix` | `code_d_syn10k` | extra_info.index of the released bank |
| `opd.chat_template_args` | `"+data.apply_chat_template_kwargs.enable_thinking=False"` |  |
| `opd.arms.ours` | `{train: "${bank.released}", experiment: "opd_${pair}_ours", total_epochs: 1, sa…` |  |
| `opd.arms.vanilla` | `{train: "${paths.data}/real/OpenCodeReasoning-10k.parquet", experiment: "opd_${…` |  |
| `eval.prompt_style` | `Qwen3NoThink` |  |

### `configs/pairs/Agent_Qwen3-4B_Qwen3-4B-Thinking-2507.yaml`

| Key | Value | Note |
|---|---|---|
| `_base_` | `[../base/opd.yaml, ../base/tau2.yaml]` |  |
| `pair` | `Agent_Qwen3-4B_Qwen3-4B-Thinking-2507` |  |
| `models.student` | `${paths.models}/Qwen3-4B` |  |
| `models.teacher` | `${paths.models}/Qwen3-4B-Thinking-2507` | served and used non-thinking |
| `models.user_sim_train` | `${paths.models}/Qwen3-4B-Instruct-2507` | also the v6/v7 preference judge |
| `models.user_sim_eval` | `gpt-4o-mini` |  |
| `models.templates.student` | `${paths.repo}/ask2distill/tau2/templates/Qwen3-4B.nothink.jinja` |  |
| `models.templates.synthesizer` | `${paths.repo}/ask2distill/tau2/templates/Qwen3-4B.nothink.jinja` |  |
| `models.templates.teacher` | `${paths.repo}/ask2distill/tau2/templates/Qwen3-4B-Thinking-2507.nothink.jinja` |  |
| `models.templates.user` | `null` | Instruct-2507 is single-mode: its own template |
| `models.hf.student` | `{repo: Qwen/Qwen3-4B, revision: 1cfa9a7208912126459214e8b04321603b3df60c}` |  |
| `models.hf.teacher` | `{repo: Qwen/Qwen3-4B-Thinking-2507, revision: 768f209d9ea81521153ed38c47d515654…` |  |
| `models.hf.user_sim_train` | `{repo: Qwen/Qwen3-4B-Instruct-2507, revision: cdbee75f17c01a7cc42f958dc65090717…` |  |
| `models.hf.embedder` | `{repo: Qwen/Qwen3-Embedding-4B, revision: 5cf2132abc99cad020ac570b19d031efec650…` |  |
| `synrl.versions` |  | The synthesizer chain: each version starts from the previous version's chosen (merged) checkpoint. Synthesizer weights v4-v7 were not kept; the chain has to be retrained to re-harvest. v4 history: its first two windows ran max_prompt_length 5632 with 8192 batched tokens; the repo uses the steady-state 7168 / 16384 (documented deviation). |
| `synrl.versions.v4` | `{experiment: "synthesizer_${pair}_v4", prompts: "${paths.data}/prompts/GENRL_TA…` |  |
| `synrl.versions.v5` | `{experiment: "synthesizer_${pair}_v5", prompts: "${paths.data}/prompts/GENRL_TA…` |  |
| `synrl.versions.v6` | `{experiment: "synthesizer_${pair}_v6", prompts: "${paths.data}/prompts/GENRL_TA…` |  |
| `synrl.versions.v7` | `{experiment: "synthesizer_${pair}_v7", prompts: "${paths.data}/prompts/GENRL_TA…` | v7 starts from v6 @50 |
| `select.served_names` | `{v6: "${synrl.versions.v6.experiment}_step${synrl.versions.v6.chosen_step}", v7…` | Each merged synthesizer is served under its directory name, <experiment>_step<N>; the harvest addresses it by that name. |
| `harvest.v6` | `{synthesizer: "${paths.ckpt}/${synrl.versions.v6.experiment}/hf/step_${synrl.ve…` |  |
| `harvest.v7` | `{synthesizer: "${paths.ckpt}/${synrl.versions.v7.experiment}/hf/step_${synrl.ve…` |  |
| `bank.parts` | `{pool: "${paths.data}/tau2/pool_v6s50.jsonl", domains: [airline], out: "${pair}…` | Released pools rebuild the released banks exactly (verified). Every card no gate rejected is kept, deduped by task identity; v6 retail cards are dropped. airline cards of the v6 pool retail cards of the v7 pool |
| `bank.mix` | `${pair}` | concatenate the parts, renumber extra_info.index |
| `bank.out_dir` | `${paths.out}/${pair}/banks` |  |
| `bank.released` | `${paths.data}/banks/${bank.mix}.parquet` | 2,302 rows (976 airline + 1,326 retail) |
| `opd.user_sim_serve` | `{map: "Qwen3-4B-Instruct-2507=4"}` | synthetic arm: four simulator replicas |
| `opd.arms.ours` | `{train: "${bank.released}", experiment: "opd_${pair}_ours", short_name: ours, t…` |  |
| `opd.arms.vanilla` | `{train: "${paths.data}/real/tau2-official-train.parquet", experiment: "opd_${pa…` | Trained in the 3-node two-arm job; its first window saved every 50 and kept 8, later windows 20 / 3. |

### `configs/pairs/TEMPLATE_custom.yaml`

| Key | Value | Note |
|---|---|---|
| `_base_` | `[configs/base/opd.yaml, configs/base/math.yaml, configs/base/eval_math.yaml]` |  |
| `pair` | `Math_MyStudent_MyTeacher` | FILL: = the file name, <Domain>_<Student>_<Teacher> |
| `models.student` | `${paths.models}/MyStudent` | Student π_S and teacher π_T must share one tokenizer / vocabulary (README: "Prerequisites"); a padded output head of a different size is fine. FILL: local HF directory of π_S |
| `models.teacher` | `${paths.models}/MyTeacher` | FILL: local HF directory of π_T |
| `models.synthesizer_init` | `${models.student}` | S_θ starts as a copy of the student. A student that always opens a <think> block (DeepSeek-R1-Distill style) needs a no-think copy: set syn_nothink: true, point synthesizer_init at ${models.student}-SynNoThink and run `a2d models syn-nothink --pair <pair>`. |
| `models.syn_nothink` | `false` |  |
| `models.qwen3_nothink` | `false` | true: hybrid Qwen3 student, every chat template gets enable_thinking=False (synthesizer prompts, solve prompts, OPD). false: templates are used as is. FILL |
| `models.hf` |  | FILL: pinned revisions for `a2d models download` |
| `models.hf.student` | `{repo: my-org/MyStudent, revision: 0000000000000000000000000000000000000000}` |  |
| `models.hf.teacher` | `{repo: my-org/MyTeacher, revision: 0000000000000000000000000000000000000000}` |  |
| `models.hf.embedder` | `{repo: Qwen/Qwen3-Embedding-4B, revision: 5cf2132abc99cad020ac570b19d031efec650…` |  |
| `calib.expected_delta_ref` | `null` | optional: a previous calibration, for the printed comparison |
| `reward` |  | The calibration questions are sampled from models.synthesizer_init (N_cal = calib.n_questions = 200); the solve budget calib.max_new must equal opd.max_response_length (L_max). |
| `reward.delta_ref` | `null` | δ_ref: `delta_ref_rounded` of $A2D_OUT/<pair>/calib/calib_delta_ref.json. `a2d synrl` refuses to start while it is null. FILL after `a2d calibrate` |
| `synrl` |  | Knobs that depend on the domain (math values in configs/base/math.yaml, code values in configs/base/code.yaml): len_alpha (α), len_pen (λ_len), trunc_zero (drop truncated rollouts), rolls (student rollouts per question), max_new (L_max), diversity.{mode, thresh (τ), pen (λ_div)}. CODE: gen_bs: 4                                          # questions per generate call on a reward rank (memory) |
| `synrl.experiment` | `"synthesizer_${pair}"` |  |
| `synrl.offload` | `false` | Memory knobs of the trained synthesizer (8 actor GPUs by default, synrl.layout): offload actor parameters + optimizer for a >= 4B synthesizer in fp32, and the vLLM rollout share of each actor GPU. FILL: true for a >= 4B synthesizer on 80GB cards |
| `synrl.gpu_memory_utilization` | `0.6` |  |
| `select` |  | CODE: save_freq: 5 |
| `select.yield` | `null` | optional saturation-yield comparison (select.yield_compare) |
| `select.chosen_step` | `60` | FILL: the step `a2d select pick` reports |
| `harvest` |  | CODE: sweep: {steps: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]} |
| `harvest.shards` | `16` | N = shards x per_shard raw questions (80,000 here) |
| `filter.slices` | `[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]` | one slice over all shards (one dedup pass) |
| `filter.aggregate` | `false` | true with several slices: cross-slice dedup in agg_order |
| `filter.agg_order` | `null` |  |
| `bank` |  | CODE: slices: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]] CODE: agg_order: [0, 1, 2, 3] |
| `bank.name` | `${pair}` |  |
| `bank.row_id_prefix` | `my_pair` | FILL: extra_info.index = <row_id_prefix>_<i> |
| `bank.n` | `3200` | FILL: planned OPD steps x 64 questions per step (one epoch) |
| `bank.pool` | `${filter.out_dir}/sat.s0_questions.jsonl` | one slice: its survivors; several slices: ${filter.pool} |
| `opd` |  | CODE: pool: ${filter.pool} |
| `opd.chat_template_args` | `""` | Thinking control of the OPD rollouts: "+data.apply_chat_template_kwargs.enable_thinking=False" for a hybrid Qwen3 student held in non-thinking mode, "" otherwise. FILL |
| `opd.arms.ours` | `{train: "${bank.out}", experiment: "opd_${pair}_ours", total_epochs: 1, save_fr…` | ours trains on the bank this pair builds; select the evaluated checkpoint on a held-out validation set, not on the test benchmarks. |
| `opd.arms.vanilla` | `{train: "${paths.data}/real/DAPO-Math-17k.parquet", experiment: "opd_${pair}_va…` | optional real-question baseline (`a2d data build math` writes DAPO-Math-17k) |

### `configs/smoke/math_smoke.yaml`

| Key | Value | Note |
|---|---|---|
| `_base_` | `[../pairs/Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500.yaml]` |  |
| `smoke` | `true` |  |
| `calib.n_questions` | `16` | N_cal of the smoke |
| `calib.question_sampling` | `{max_new: 512, gpu_memory_utilization: 0.4}` |  |
| `calib.max_new` | `1024` |  |
| `calib.gpu_memory_utilization` | `0.4` | student vLLM on GPU0 |
| `calib.teacher_device` | `cuda:1` | teacher HF on GPU1 |
| `reward.max_new` | `1024` |  |
| `synrl.experiment` | `"synthesizer_${pair}_smoke"` |  |
| `synrl.layout` | `{actor_gpus: 1, reward_gpus: 1, reward_nnodes: 1, enable_resource_pool: true}` |  |
| `synrl.train_batch_size` | `2` |  |
| `synrl.ppo_mini_batch_size` | `2` |  |
| `synrl.rollout_n` | `4` | 8 questions per step |
| `synrl.max_response_length` | `512` |  |
| `synrl.ppo_max_token_len_per_gpu` | `4096` |  |
| `synrl.max_num_batched_tokens` | `4096` |  |
| `synrl.total_steps` | `3` |  |
| `synrl.save_freq` | `3` |  |
| `synrl.offload` | `true` |  |
| `synrl.gpu_memory_utilization` | `0.35` | actor stays fp32 (1.7B is ~27GB of state) |
| `synrl.extra_overrides` | `["+trainer.max_actor_ckpt_to_keep=1"]` |  |
| `select.sweep` | `{steps: [3], n: 16, max_new: 512}` |  |
| `select.pick` | `{min_rows: 8, allow_empty: true}` | relaxed gates: falls back to the last step and warns |
| `select.yield` | `null` |  |
| `select.chosen_step` | `3` |  |
| `harvest.shards` | `2` |  |
| `harvest.per_shard` | `64` |  |
| `harvest.max_new` | `512` |  |
| `filter.slices` | `[[0, 1]]` |  |
| `bank.name` | `"${pair}_smoke"` | the smoke bank stage writes ${bank.out} |
| `bank.n` | `32` |  |
| `bank.allow_short` | `true` |  |
| `bank.released` | `${paths.data}/banks/${pair}.parquet` | OPD Ours trains on the pair's RELEASED bank, not on the 32-row smoke bank: `released` would otherwise follow bank.name to a file that never exists. |
| `opd.n_gpus_per_node` | `2` |  |
| `opd.mini_batch_size` | `8` |  |
| `opd.n_responses` | `2` |  |
| `opd.max_response_length` | `1024` |  |
| `opd.max_val_response_length` | `1024` |  |
| `opd.rollout_gpu_memory_utilization` | `0.4` |  |
| `opd.esi_save` | `false` |  |
| `opd.train_max_samples` | `64` |  |
| `opd.val_max_samples` | `8` |  |
| `opd.extra_overrides` | `["trainer.total_training_steps=2", "+trainer.max_actor_ckpt_to_keep=1"]` | Ours: 2 steps on the released bank. For the 1-step Vanilla check pass opd.extra_overrides='[trainer.total_training_steps=1]' on the command line. |
| `opd.arms.ours` | `{experiment: "opd_${pair}_smoke_ours", save_freq: 2, ppo_max_token_len_per_gpu:…` |  |
| `opd.arms.vanilla` | `{experiment: "opd_${pair}_smoke_vanilla", save_freq: 1000, ppo_max_token_len_pe…` |  |
| `eval.tasks` | `[AIME24, MATH500]` |  |
| `eval.limit` | `5` |  |
| `eval.n` | `{default: 2, MATH500: 2}` |  |
| `eval.max_tokens` | `2048` |  |

### `configs/smoke/code_smoke.yaml`

| Key | Value | Note |
|---|---|---|
| `_base_` | `[../pairs/Code_Qwen3-4B_Qwen3-4B-Instruct-2507.yaml]` |  |
| `smoke` | `true` |  |
| `models.synthesizer_init` | `${paths.models}/Qwen3-1.7B` | single-GPU actor |
| `calib.n_questions` | `16` | N_cal of the smoke |
| `calib.question_sampling` | `{max_new: 512, gpu_memory_utilization: 0.4}` |  |
| `calib.max_new` | `1024` |  |
| `reward.max_new` | `1024` | solver Qwen3-4B + teacher Instruct-2507 + embedder (bf16) on GPU1 |
| `synrl.experiment` | `"synthesizer_${pair}_smoke"` |  |
| `synrl.layout` | `{actor_gpus: 1, reward_gpus: 1, reward_nnodes: 1, enable_resource_pool: true}` |  |
| `synrl.train_batch_size` | `2` |  |
| `synrl.ppo_mini_batch_size` | `2` |  |
| `synrl.rollout_n` | `4` |  |
| `synrl.max_response_length` | `512` |  |
| `synrl.ppo_max_token_len_per_gpu` | `4096` |  |
| `synrl.max_num_batched_tokens` | `4096` |  |
| `synrl.total_steps` | `3` |  |
| `synrl.save_freq` | `3` |  |
| `synrl.offload` | `true` |  |
| `synrl.gpu_memory_utilization` | `0.35` |  |
| `synrl.extra_overrides` | `["+trainer.max_actor_ckpt_to_keep=1"]` |  |
| `select.sweep` | `{steps: [3], n: 16, max_new: 512}` |  |
| `select.pick` | `{min_rows: 8, allow_empty: true}` |  |
| `select.chosen_step` | `3` |  |
| `harvest.shards` | `2` |  |
| `harvest.per_shard` | `64` |  |
| `harvest.max_new` | `512` |  |
| `filter.slices` | `[[0], [1]]` |  |
| `filter.agg_order` | `[1, 0]` | exercises the explicit ordering |
| `bank.name` | `"${pair}_smoke"` | the smoke bank stage writes ${bank.out} |
| `bank.n` | `32` |  |
| `bank.allow_short` | `true` |  |
| `bank.released` | `${paths.data}/banks/${pair}.parquet` | OPD Ours trains (and validates) on the pair's RELEASED bank, not on the 32-row smoke bank: `released` would otherwise follow bank.name. |
| `opd.model_dtype` | `bf16` | Qwen3-4B student over 2 GPUs, teacher offloaded |
| `opd.n_gpus_per_node` | `2` |  |
| `opd.mini_batch_size` | `8` |  |
| `opd.n_responses` | `2` |  |
| `opd.max_response_length` | `1024` |  |
| `opd.max_val_response_length` | `1024` |  |
| `opd.rollout_gpu_memory_utilization` | `0.4` |  |
| `opd.train_max_samples` | `64` |  |
| `opd.val_max_samples` | `8` | TEST_FILE is the 10k training parquet otherwise |
| `opd.extra_overrides` | `["trainer.total_training_steps=2", "+trainer.max_actor_ckpt_to_keep=1"]` |  |
| `opd.arms.ours` | `{experiment: "opd_${pair}_smoke_ours", save_freq: 2, ppo_max_token_len_per_gpu:…` |  |
| `opd.arms.vanilla` | `{experiment: "opd_${pair}_smoke_vanilla", save_freq: 1000, ppo_max_token_len_pe…` |  |
| `eval.lcb` | `{n: 1, max_tokens: 2048}` | `a2d eval code --smoke` adds LCB --debug (15 problems) |
| `eval.evalplus` | `{datasets: [humaneval]}` | full 164: evalplus asserts complete coverage |
| `eval.extra_bench` | `{tp: 1, limit: 5}` |  |

### `configs/smoke/tau2_smoke.yaml`

| Key | Value | Note |
|---|---|---|
| `_base_` | `[../pairs/Agent_Qwen3-4B_Qwen3-4B-Thinking-2507.yaml]` |  |
| `smoke` | `true` |  |
| `reward_env.TAU2_K` | `2` |  |
| `reward_env.TAU2_V7_TURNS` | `8` |  |
| `reward_env.TAU2_GATE_TURNS` | `6` |  |
| `synrl.train_batch_size` | `2` |  |
| `synrl.ppo_mini_batch_size` | `2` |  |
| `synrl.rollout_n` | `4` | 8 cards per step; diversity group 16 = the whole batch |
| `synrl.n_gpus_per_node` | `1` |  |
| `synrl.total_steps` | `2` |  |
| `synrl.save_freq` | `2` |  |
| `synrl.gpu_memory_utilization` | `0.3` |  |
| `synrl.model_dtype` | `bf16` |  |
| `synrl.train_max_samples` | `4` | first 4 rows of GENRL_TAU2_V7 |
| `synrl.serve.node_maps` | `["Qwen3-4B=1\|Qwen3-4B-Instruct-2507=1\|Qwen3-4B-Thinking-2507=1"]` |  |
| `synrl.serve.gpu_util` | `0.25` |  |
| `synrl.serve.max_len` | `16384` |  |
| `synrl.extra_overrides` | `["+trainer.max_actor_ckpt_to_keep=1", "actor_rollout_ref.rollout.tensor_model_p…` | The final runs left the rollout TP at verl's default (2) on 8 GPUs; the smoke trainer has one GPU, so TP must be 1. |
| `synrl.versions.v7` | `{experiment: "synthesizer_${pair}_v7_smoke", init: "${paths.models}/Qwen3-1.7B"…` |  |
| `harvest.v7` | `{served_name: "${synrl.versions.v7.experiment}", per_target: 2, k: 2, workers: …` |  |
| `harvest.serve` | `{replicas_per_model: 1, nodes: 1, gpu_util: 0.2, max_len: 16384}` |  |
| `opd.student` | `${paths.models}/Qwen3-1.7B` |  |
| `opd.model_dtype` | `bf16` | student and teacher (offloaded) both on GPU1 |
| `opd.n_gpus_per_node` | `1` |  |
| `opd.mini_batch_size` | `4` | 4 prompts x 2 responses = 8 trajectories: the agent loop splits the batch equally over verl's 8 AgentLoopWorkers (rollout.agent.num_workers default; final run: 8 x 4 = 32), so the batch must be a multiple of 8. |
| `opd.n_responses` | `2` |  |
| `opd.max_response_length` | `4096` |  |
| `opd.max_val_response_length` | `4096` |  |
| `opd.train_max_samples` | `16` |  |
| `opd.val_max_samples` | `4` | val files are the math sets, as in the final run |
| `opd.total_training_steps` | `2` |  |
| `opd.ckpt_keep` | `1` |  |
| `opd.agent_env` | `{TAU2_MAX_TURNS: 6}` |  |
| `opd.user_sim_serve` | `{map: "Qwen3-4B-Instruct-2507=1"}` | on GPU0 |
| `opd.arms.ours` | `{experiment: "opd_${pair}_smoke_ours", save_freq: 2, ppo_max_token_len_per_gpu:…` |  |
| `opd.arms.vanilla` | `{experiment: "opd_${pair}_smoke_vanilla", save_freq: 1000, ppo_max_token_len_pe…` |  |
| `eval.openai_base_url` | `local` | Instruct-2507 served locally under the name gpt-4o-mini takes the gpt-4o-mini branch of LLM_CALL.py with no fork change; run_eval writes OPENAI_API_KEY=EMPTY and the local base URL into its temp env file. |
| `eval.trials` | `1` |  |
| `eval.max_steps` | `30` |  |
| `eval.num_tasks` | `2` |  |
| `eval.tasks_dir` | `${paths.out}/tau2/tasks_smoke` | Task files written by tests/smoke/run_tau2.sh: the first num_tasks tasks of each test file with nl_assertions removed. (1) The fork ignores --num-tasks unless --task-ids is given (tau2/run.py get_tasks returns early), so the full 20/40/40 split would run. (2) tau2's evaluator always calls the NL- assertion judge, hard-coded gpt-5 (tau2/config.py), through the OpenAI branch; the local server only … |
| `eval.served_name` | `"Qwen3-smoke-s{step}"` |  |
