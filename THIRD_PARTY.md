# Third-party code

`third_party/` holds full, patched copies of the four external code bases the pipeline runs on.
They are vendored rather than fetched so that every run uses exactly the code that produced the
paper's numbers, including our local modifications. Each tree is the exact checkout the final
runs used, with the few documented exceptions listed under Provenance below.

| Directory | Upstream | Commit | License | How it is loaded |
|---|---|---|---|---|
| `third_party/verl/` | [volcengine/verl](https://github.com/volcengine/verl) via [Thinking-Space/Rethinking-OPD](https://github.com/Thinking-Space/Rethinking-OPD) (formerly thunlp/OPD) | verl `0eb50ec4a33c`, Rethinking-OPD `ac26e38d6f15` | Apache-2.0 | `pip install --no-deps -e third_party/verl` |
| `third_party/LiveCodeBench/` | [LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) | `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` | MIT | not installed; `python -m lcb_runner.runner.main` with `third_party/LiveCodeBench` on `PYTHONPATH` |
| `third_party/evalplus/` | [evalplus/evalplus](https://github.com/evalplus/evalplus) | `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e` (`v0.3.1-44`) | Apache-2.0 | `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_EVALPLUS=0.4.0.dev44 pip install --no-deps -e third_party/evalplus` |
| `third_party/toolorchestra_tau2/` | [NVlabs/ToolOrchestra](https://github.com/NVlabs/ToolOrchestra); its `evaluation/tau2-bench/` is a fork of [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) v0.0.1 | `c1c0a4002be128b18accb267aac5ffdc26a28103` | Apache-2.0 (ToolOrchestra), MIT (tau2-bench) | not installed; `REPO_PATH=third_party/toolorchestra_tau2` and `PYTHONPATH=<repo>:$REPO_PATH:$REPO_PATH/evaluation/tau2-bench`, set by `ask2distill.tau2.spawn` for the tau2 interpreter |

All four installs are done by `env/build_a2d_env.sh`; the tau2 trees need no install. Each
tree keeps its upstream `LICENSE` file (`toolorchestra_tau2/` also keeps
`THIRD_PARTY_NOTICES.md`, and `evaluation/tau2-bench/LICENSE` is sierra's MIT license).

`third_party/patches/` records every modification as a patch file. `third_party/` already
contains the patched code, so the patches are documentation: they say what we changed and let
anyone re-derive the trees from upstream.

| Patch | Against | Applies to |
|---|---|---|
| `verl_vs_rethinking_opd_ac26e38.patch` | Rethinking-OPD@`ac26e38` `verl/` | 5 files (4 modified, 1 new) |
| `livecodebench_28fef95.patch` | LiveCodeBench@`28fef95` | 9 files, the changes the final runs used (`git diff HEAD` of the checkout the final runs used) |
| `livecodebench_a2d.patch` | LiveCodeBench@`28fef95` + `livecodebench_28fef95.patch` | 2 files, added for this repo |
| `evalplus_26d6d00.patch` | evalplus@`26d6d00` | 2 files (`git diff HEAD` of the checkout the final runs used) |
| `toolorchestra_c1c0a40.patch` | ToolOrchestra@`c1c0a40` | 2 files (`git diff HEAD` of the checkout the final runs used) |

To rebuild a tree from upstream: check out the commit, then `git apply` the patches in the order
listed (`patch -p1 -d <checkout>/verl` for the verl patch, whose header explains its paths).

---

## verl (`third_party/verl/`, version `0.7.0.dev0`)

The fork has three layers on top of upstream verl:

- **L0:** volcengine/verl at `0eb50ec4a33cda97e05ed8caab9c7f17a30c05a9` (2025-11-10,
  "[rollout] fix: resolve agent loop config path in multi-node").
- **L1:** Rethinking-OPD (thunlp/OPD). Its `verl/` subtree, unchanged from `c6cd96b15e` to HEAD
  `ac26e38d6f1572eb027597b48a9f4e01f6915ef8`, adds about 3,000 lines across 21 files. These
  provide almost everything the method relies on: top-k OPD (`compute_distillation_reward` with
  the `only_stu`/`only_tch`/`intersection`/`union` strategies), `token_reward_direct` and the 3-D
  per-token loss, the teacher served as a `RewardModelWorker` (`compute_rm_score`, fp32 teacher
  with CPU offload), the `rm_scores` bypass in the naive reward manager, `val_kwargs.max_tokens`
  and `repetition_penalty` in the vLLM rollout, the `ttrl_math` logging reward, and the
  swanlab/matplotlib plots behind `trainer.is_plot`.
- **L2/L3:** our changes, listed below. Change 1 came first; changes 2–4 were made afterwards.

**Our modifications** (line numbers refer to the vendored files; the full diff is
`patches/verl_vs_rethinking_opd_ac26e38.patch`):

| # | File | Change | Status |
|---|---|---|---|
| 1 | `verl/workers/actor/dp_actor.py` (`_forward_micro_batch`, about `:199-286`) | Chunked (1024 rows) and checkpointed `log_softmax` + gather for the target and top-k tokens. Numerically identical; avoids OOM at 16K tokens. | used by every OPD run |
| 2 | `dp_actor.py:864-872` | The 2-D path keeps `entropy`; without it `entropy_coeff != 0` raised `UnboundLocalError`. | used: synthesizer GRPO runs with `entropy_coeff=0.01` |
| 3 | `dp_actor.py:852-861` | The 3-D path keeps `log_prob`; without it `use_kl_loss=True` on the top-k path raised `UnboundLocalError`. | used: the OPD runs of `Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct` with KL 0.005 and entropy 0.005 |
| 4 | `verl/trainer/ppo/ray_trainer.py:843-846, 913-929, 1012-1019, 2290-2305, 2317-2319, 2352-2356` | Token ledger and budget, enabled by `+trainer.log_token_counts` / `+trainer.token_budget=B`. Writes `token_counts.json` into each checkpoint and stops once the cumulative loss tokens reach B. | inert (key absent by default) |

Change 4 comes from later experiments outside the scope of this repository. Its branches are
gated on keys that are absent by default (`+trainer.log_token_counts`, `+trainer.token_budget`),
so the code path the configs exercise is the one that ran. `tests/unit/test_verl_inert.py`
checks that no shipped config sets them.

**Removed:** `*.bak*` backups and `__pycache__`. Nothing else: the docs, examples, recipes and
tests are kept so that the tree is a complete copy (about 12 MB).

**How it is loaded:** `pip install --no-deps -e third_party/verl`. `setup.py` pins
`numpy<2.0.0`, while the env runs numpy 2.2.6 because the CUDA stack vLLM 0.11 pulls in requires
numpy 2.x. `--no-deps` keeps the locked versions, so `pip check` reports this pin; that report
is expected. Synthesizer RL and the tau2 agent loop are plugins in `ask2distill/` (a swapped
`RewardModelWorker`, a registered `tau2_agent` loop) and need no verl edits. Their hydra entry
points take `config_path` from the installed `verl.__file__`.

---

## LiveCodeBench (`third_party/LiveCodeBench/`)

Tracked files at `28fef95` plus the working-tree edits the final runs used. `output/` (8.1 GB of
cached generations) and `.git` are not vendored.

**Our earlier modifications** (`patches/livecodebench_28fef95.patch`, 9 files):

1. `lcb_runner/benchmarks/code_generation.py`: reads the `livecodebench/code_generation_lite`
   jsonl shards directly with `hf_hub_download`, because `datasets>=4` dropped loading scripts.
   The shard-to-version map is copied from the upstream builder, plus a `v5_v6` entry.
2. `lcb_runner/lm_styles.py`: adds the `Qwen3NoThink` and `AutoTemplate` styles, and
   `get_model()`, which registers a local checkpoint ad hoc when given `--model_style`.
3. `lcb_runner/prompts/code_generation.py`: `Qwen3NoThink` (the `CodeQwenInstruct` prompt plus
   the empty `<think>\n\n</think>\n\n` block) and `AutoTemplate` (`_auto_render`, which renders
   with the local model's chat template; `LCB_AUTO_SYS` overrides the system message).
4. `lcb_runner/runner/parser.py`: flags `--model_style --top_k --min_p --max_model_len
   --gpu_memory_utilization --cuda_graph`.
5. `lcb_runner/runner/vllm_runner.py`: passes those flags to vLLM and drops empty stop strings.
6. `lcb_runner/runner/main.py`: sets `LCB_AUTO_MODEL` from `--local_model_path` and uses
   `get_model`.
7. `lcb_runner/prompts/{code_execution,self_repair,test_output_prediction}.py`: the `anthropic`
   import becomes optional, and Qwen styles share one ChatML helper.

**a2d modifications** (`patches/livecodebench_a2d.patch`, 2 files, added for this repository):

- `lcb_runner/prompts/code_generation.py`: the few-shot files (`func.json`, `stdin.json`) are
  opened relative to the module (`__file__`) rather than the working directory, so `lcb_runner`
  imports from any cwd. Upstream had to be run from the LiveCodeBench root.
- `lcb_runner/benchmarks/code_generation.py`: `_load_lcb_jsonl` passes
  `revision=os.environ.get("LCB_DATASET_REVISION", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505")` to
  `hf_hub_download`. Before this change the dataset revision floated with `main`. The default
  is the dataset revision the paper's LCB numbers were computed on, and
  `a2d data build code-eval` prefetches the same revision.

Neither change alters prompts, sampling or grading on the same data.

**Inert for the core pairs:** `AutoTemplate` (every core student and teacher uses ChatML, so
the styles are `Qwen3NoThink` or `CodeQwenInstruct`), and the code-execution, self-repair and
test-output-prediction scenarios.

---

## evalplus (`third_party/evalplus/`, installed as `0.4.0.dev44`)

Tracked files at `26d6d00` plus two working-tree edits (`patches/evalplus_26d6d00.patch`):

1. `evalplus/provider/__init__.py`: `make_model` forwards `**kwargs` to the vLLM decoder.
2. `evalplus/provider/vllm.py`: `max_model_len` (upstream hard-codes 2048),
   `gpu_memory_utilization`, `enforce_eager`, `top_p`, `top_k`, `min_p`, and a `reasoning`
   mode, which drops the assistant prefill and the EOS heuristics and keeps only the text after
   `</think>`.

No edits were made for this repository. The version comes from setuptools-scm, which fails without
`.git`, so the install sets `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_EVALPLUS=0.4.0.dev44` (the version
`git describe` gave for this commit). `env/check_env.py` asserts that the `evalplus.evaluate`
console script exists.

**Inert:** `reasoning` mode. The final code evaluation ran EvalPlus greedily (`--greedy
--max-new-tokens 1024 --max-model-len 4096`). No-think handling needs no patch, because
`make_raw_chat_prompt` renders a complete assistant turn and Qwen3 templates then insert the
empty think block themselves.

---

## ToolOrchestra tau2 fork (`third_party/toolorchestra_tau2/`)

The directory is the `REPO_PATH` root that the fork expects. `tau2/utils/utils.py` builds
`DATA_DIR` from `os.environ["REPO_PATH"]` and raises a `TypeError` when it is unset. Only these
parts of ToolOrchestra are vendored:

- `LLM_CALL.py` (patched), `LICENSE`, `THIRD_PARTY_NOTICES.md`;
- `evaluation/tau2-bench/`: the tracked files, including the patched `tau2/utils/llm_utils.py`.

**Removed** (the rest of ToolOrchestra is 124 GB):

- everything else in the repository;
- inside tau2-bench: the top-level `run.py` (unused, and it contains another user's cluster
  path), `keys/`, `cut_error_message.json`, `done` and the `*.bak-0919` backups;
- the untracked `fetch_tool_models.py` and `wiki/`.

`tau2/run.py` is kept; it is a different module, and the CLI needs it.

`data/` (ToolOrchestra's `data/tau2`: domain DBs, policies, tasks and the user-simulator
guidelines, 35 MB in plain git) is not vendored. `a2d data build tau2` fetches it at `c1c0a40`
into `third_party/toolorchestra_tau2/data/`, which is gitignored. The fork's
`simulation_guidelines.md` differs from sierra upstream, and every stage uses the fork's version.

**Our modifications** (`patches/toolorchestra_c1c0a40.patch`):

1. `LLM_CALL.py`: adds `get_direct_openai_client()`, a plain OpenAI client built from
   `OPENAI_API_KEY`/`OPENAI_BASE_URL`. It loads `TAU2_OPENAI_ENV` into `os.environ` first, pins
   `Accept-Encoding: gzip` (an old brotli breaks httpx), and sets a 120 s timeout.
   `get_llm_response` uses it in place of the NVIDIA OAuth proxy when `TAU2_OPENAI_DIRECT=1`.
2. `evaluation/tau2-bench/tau2/utils/llm_utils.py`: `TAU2_MAX_TOKENS` replaces the hard-coded
   8000-token agent budget (default 8000). ToolOrchestra's
   `<think>/<message>/<tool_call>` system-prompt suffix is appended only when
   `TAU2_ORCHESTRATOR_PROMPT=1`.

The final runs used `TAU2_OPENAI_DIRECT=1`, `TAU2_MAX_TOKENS=8000` and
`TAU2_ORCHESTRATOR_PROMPT=0`; `ask2distill.tau2.env` renders all three explicitly.

**Upstream behaviour kept as is:** importing `LLM_CALL` creates a `keys/` directory in the
current working directory. The `a2d` spawner runs envside processes from run directories, and
`env/check_env.py --tau2` probes from a temporary directory.

**How it is loaded:** the tau2 code runs only in the separate `a2d-tau2` interpreter
(`$A2D_TAU2_PYTHON`, built by `env/build_tau2_env.sh`) with `PYTHONPATH` and `REPO_PATH` set by
`ask2distill.tau2.spawn`. tau2 is not pip-installed.

**Not vendored:** sierra-research/tau2-bench at `b7ea9074c1cba482b30687fecdb5c8425fd6f619`
(v1.0.1, MIT). Only its `tasks.json` and `split_tasks.json` are used, to build the official
train/test split. `a2d data build tau2` downloads both files.

---

## Provenance

Every vendored file is byte-identical to the checkout the final runs used, except the two
LiveCodeBench files changed by `livecodebench_a2d.patch` (`lcb_runner/benchmarks/code_generation.py`
and `lcb_runner/prompts/code_generation.py`). The three `git diff HEAD` patches are the working-tree
changes of those checkouts, and the environment locks in `env/` are the frozen `pip freeze` of the
two environments the final runs used. `tests/unit/test_third_party.py` checks that every patch
matches the vendored tree.

The flash-attn wheel behind the final runs was a local source build of 2.8.1, made with what
is now `env/build_flash_attn.sh`, and has sha256
`039f4d32d34b89d99058d0a0169d41a5613a88e0d2c71eb8dedcc425c69c62c2`. By default,
`env/build_a2d_env.sh` installs the upstream prebuilt 2.8.1 wheel for torch 2.8 / cp312
instead, and that script is the fallback.
