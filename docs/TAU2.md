# tau2 agent pipeline

The agent pair `Agent_Qwen3-4B_Qwen3-4B-Thinking-2507` (written `P` in the commands below) distils a
Qwen3-4B student (non-thinking) from Qwen3-4B-Thinking-2507 (served non-thinking) on multi-turn
tool-use dialogues of tau2-bench's airline and retail domains. The
synthesizer writes **task cards** (a scenario plus the gold actions that solve it); unlike math and
code, its reward is not a divergence but the student's failure rate on cards whose gold actions
execute, are grounded in the database, change it, and pass a set of gates. The OPD stage is the
same top-16 distillation objective, applied to a masked multi-turn trajectory (see
[Agent-specific OPD](#agent-specific-opd)).

## Two environments

tau2-bench and its dependency chain (litellm, loguru, ...) conflict with the training stack, so
the final runs used two conda envs, and so does this repository:

| | a2d env | tau2 env |
|---|---|---|
| Built by | `env/build_a2d_env.sh` | `env/build_tau2_env.sh` (prints the `A2D_TAU2_PYTHON` to export) |
| Runs | `a2d` itself, verl, vLLM, every trainer and server | `ask2distill.tau2.envside.*` only |
| Has | torch, vLLM, verl, pyarrow, omegaconf | tau2 (from `third_party/toolorchestra_tau2` on `PYTHONPATH`), litellm 1.82.6, fastapi; no torch, no pyarrow |

`a2d` always runs in the a2d env. Env-side steps are started as
`$A2D_TAU2_PYTHON -m ask2distill.tau2.envside.<module>` by `ask2distill.tau2.spawn`, which sets:

- `PYTHONPATH=<repo>:$REPO_PATH:$REPO_PATH/evaluation/tau2-bench` (tau2 is not pip-installed);
- `REPO_PATH=third_party/toolorchestra_tau2` (the fork's `tau2/utils/utils.py` raises without it);
- `HF_HOME=$A2D_CACHE/hf` and `HF_HUB_OFFLINE=1`: the fork loads the `Qwen/Qwen3-8B` tokenizer at
  import; `a2d data build tau2` pre-caches it at a pinned revision;
- the complete `TAU2_*` environment of the role, rendered by `ask2distill.tau2.env` from the config
  (several env-side modules read it at import, so no module default is relied on); inherited
  `TAU2_*` variables are dropped first, and the database directory is passed as `A2D_TAU2_DB_DIR`
  (not `TAU2_DATA`, a name that collides with a prompt-file variable).

Data crosses between the two envs as files: the env side writes `rows.json`, and
`a2d tau2 rows2parquet` turns it into a parquet in the a2d env.

## Serving

`a2d tau2 serve --plan {synrl,harvest,opd,eval}` starts one `vllm serve` per replica (one GPU
each, hermes tool parser, auto tool choice), one at a time, each checked for its own served name
before the next starts, and writes an endpoints file of `kind url served_name` lines (kinds `stu`,
`usr`, `tea`, `gen`). It stays up until terminated and takes its servers down with it.

- **Served names are explicit.** The env service, the harvest client and ToolOrchestra's client
  address models by name: `Qwen3-4B` (student), `Qwen3-4B-Instruct-2507` (user simulator and
  preference judge), `Qwen3-4B-Thinking-2507` (teacher), `synthesizer_<pair>_v6_step50` /
  `synthesizer_<pair>_v7_step60` (harvest synthesizers, `select.served_names`). Eval names are
  `<pair>-<short_exp>-s<step>` (e.g. `<pair>-ours-s<N>`) and must contain
  "qwen", as the pair name does: the eval client
  routes a model to vLLM only then (the fork's `LLM_CALL.py` / `llm_utils.py`).
- **Templates are chosen by role, never by served name** (`models.templates`): `student` and
  `synthesizer` use `ask2distill/tau2/templates/Qwen3-4B.nothink.jinja`, `teacher` uses
  `Qwen3-4B-Thinking-2507.nothink.jinja`, `user: null` keeps Instruct-2507's own template. A
  configured template that is missing is an error: vLLM's fallback is the model's own template,
  which silently turns thinking back on. Non-thinking is forced through the template because
  ToolOrchestra's client sends no `chat_template_kwargs`.
- **No sampling flags.** Requests sample with each model's `generation_config` defaults, as the
  final runs did.
- A merged checkpoint that will be served needs its template next to the weights:
  `a2d merge ... --chat-template synthesizer|student` copies `models.templates[ROLE]` there.

## Stages

| Stage | Command (SLURM wrapper) | Notes |
|---|---|---|
| data | `a2d data download`, `a2d data build tau2`, `a2d data build math` | the bank, prompts and card pools; ToolOrchestra `data/tau2` @c1c0a40, the official split of tau2-bench @b7ea907 (test files 20 / 40 / 40, 104 real training rows), the Qwen3-8B tokenizer cache, `domain_ctx.json`; the math test sets, which the OPD launcher loads as (never scored) validation files |
| prompts (optional) | `a2d tau2 build-prompts --version v7` | equivalent to, not identical with, the released `GENRL_TAU2_V*.parquet` |
| synthesizer RL | `launchers/slurm/submit.sh tau2-synrl P --version vX` (3 nodes: train + two serve nodes) | v4 from Qwen3-4B; v5 from v4 @40, v6 from v5 @40, v7 from v6 @50; 60 steps each |
| select + merge | `a2d tau2 select --pair P --version vX`, then `a2d merge --actor-dir ... --target-dir ... --chat-template synthesizer` | trailing 4-step window of valid x R\|valid; the steps actually used are frozen (v4 40, v5 40, v6 50, v7 60; the rule gives 55 for v4) |
| harvest | `launchers/slurm/submit.sh tau2-harvest P --version v6\|v7` | each card's full reward is re-run with k=2; resumable, error rows are redone until complete |
| bank | `a2d tau2 bank --pair P` | released pools -> `<pair>_part-airline`, `<pair>_part-retail` and their mix `<pair>` (the OPD bank), byte-identical to the release |
| OPD | `launchers/slurm/submit.sh tau2-opd P --arm ours\|vanilla` (2 nodes: train + user simulator) | env service in the tau2 env; agent-loop settings reach every worker through Ray's `runtime_env` |
| eval | serve on a compute node, client on the login node (below) | gpt-4o-mini user simulator |
| summarize | `a2d tau2 summarize --pair P --models <served names>` | pass^1 per domain (reward >= 0.999) and the unweighted mean |

## Agent-specific OPD

The tau2 OPD arms use **the same OPD objective as math and code**, applied to a masked multi-turn
trajectory. Nothing in the per-token loss changes: top-16 student support, reverse KL weighted by
the renormalised student probabilities (`top_k_strategy: only_stu`, `reward_weight_mode:
student_p`), the per-token signal used directly as the advantage (`adv_estimator:
token_reward_direct`), the same learning rate (1e-6) and fp32 master weights
(`configs/base/opd.yaml`). For a response position t with student top-16 set A_t,

    l_t^OPD  is the top-16 OPD term with token-level advantage
    Adv_{t,v} = - w_{t,v} * (log p_S(v | x_<t) - log p_T(v | x_<t)),   v in A_t,
    w_{t,v} = p_S(v | x_<t) / sum_{u in A_t} p_S(u | x_<t).

What is specific to the agent domain is the trajectory the objective is applied to, built by
`ask2distill/tau2/trainside/agent_loop.py`:

1. **Multi-turn masked distillation.** An episode interleaves policy turns (the student's replies
   and tool calls), tool outputs from the tau2 environment and simulated-user turns. They form
   one token sequence, but only tokens the student generated get `response_mask = 1`; tool
   outputs and user-simulator text are rendered into the context with mask 0. verl multiplies
   both the OPD advantage and the policy loss by the mask, and the `token-mean` normaliser counts
   only policy tokens:

       L = (1 / sum_t m_t) * sum_t m_t * l_t^OPD,    m_t = 1[token t was generated by the student]

   **Why the mask.** The teacher and the student are both evaluated on every position of the
   sequence, but a position the environment or the user simulator wrote is not an action of the
   student: its "disagreement" is the two models' surprise at JSON from a database or at a
   simulated customer, which the student cannot change and should not be pushed towards. Masking
   keeps the signal on the student's own decisions -- which tool to call, with which arguments,
   what to tell the customer -- exactly as in single-turn OPD, where every response token is the
   student's. Observation tokens still condition both models, so the teacher grades each policy
   turn in the context the student actually created (on-policy at the episode level).
2. **Stop-on-repeat** (`opd.agent_env.TAU2_STOP_ON_REPEAT: 1`). When the student's user-facing
   reply near-duplicates its previous one (`difflib` ratio >= `TAU2_REPEAT_SIM` = 0.9 on the
   first 600 characters), that turn is removed from both the token sequence and the mask and the
   episode ends. **Why.** A small student falls into degenerate loops -- the same sign-off every
   turn while the simulator keeps thanking it and never ends the conversation. Without the stop
   these turns fill the turn and token budget, and every repeat is another set of policy tokens
   the loss trains on; with it, the loop is cut at its first repeat and the repeated turn is
   neither scored nor kept as context. An earlier OPD run without it learned exactly such a
   closing loop.
3. **Budgets and frozen services.** One 8,192-token response budget (`opd.max_response_length`)
   is shared by all turns of an episode -- policy tokens and observations alike -- and an episode
   has at most 20 assistant turns (`TAU2_MAX_TURNS`); whichever runs out first ends it. Tool outputs are truncated to 2,000 characters
   (`TAU2_MAX_TOOL_CHARS`). The user simulator (Qwen3-4B-Instruct-2507, temperature 1.0, at most
   512 tokens per turn) and the tau2 environment are external, frozen services: the simulator is
   a vLLM server on its own node (`opd.user_sim_serve`), the environment the env-only
   `envside.env_service` (`opd.env_service`); neither is ever the policy. The prompt is the
   domain policy and tool schemas, a scripted greeting and the simulator's first message, so the
   first scored token answers a real request.

So the agent arms differ from the math and code arms in the data the objective sees, not in the
objective: `l_t^OPD`, its support, weighting, advantage estimator and optimiser are identical.

## Evaluation: compute node serves, login node asks OpenAI

The user simulator in evaluation is OpenAI's `gpt-4o-mini`. On clusters where only the login
node can reach the OpenAI API, evaluation is split:

```bash
P=Agent_Qwen3-4B_Qwen3-4B-Thinking-2507
# compute node (1 GPU): merge with the student template, then serve
a2d merge --pair $P --arm ours --step <N> --chat-template student
a2d tau2 serve --pair $P --plan eval --arm ours --step <N> --out $A2D_OUT/tau2/eval_endpoints.txt
# login node: the client (A2D_OPENAI_ENV_FILE points at your own key file)
a2d tau2 eval --pair $P --endpoints $A2D_OUT/tau2/eval_endpoints.txt --tag test4
a2d tau2 summarize --pair $P --models $P-ours-s<N>
```

**OpenAI key handling.** The key is read from `$A2D_OPENAI_ENV_FILE` (a file with
`OPENAI_API_KEY=...`) or `$OPENAI_API_KEY`, and is never copied anywhere persistent. `run_eval`
writes a 0600 temporary env file with the key and an explicit
`OPENAI_BASE_URL=https://api.openai.com/v1`, points `TAU2_OPENAI_ENV` at it and deletes it on exit.
The fork's direct client (`TAU2_OPENAI_DIRECT=1`) loads that file into the environment, which
overrides an `OPENAI_BASE_URL` relay inherited from a shell profile (a `~/.bashrc` that points
it at a relay would otherwise win). The tau2 env is locked exactly as frozen (`litellm==1.82.6`,
no `brotli`; brotli 1.0.9 made httpx fail with "Connection error", and the fork pins
`Accept-Encoding: gzip`).

The fork writes `cut_error_message.json` into its working directory; every run gets its own
directory under `$A2D_OUT/tau2/outputs`. tau2's own vLLM client retries forever, so a watchdog
restarts a pass that wrote nothing for 30 minutes, and tasks with fewer than `eval.trials`
results are rerun whole until every task is complete. Summaries under-report until all trials
have finished.

**Local user simulator (smoke only).** `tests/smoke/run_tau2.sh` serves Instruct-2507 under the
name `gpt-4o-mini` and sets `eval.openai_base_url: local`: `run_eval` then writes
`OPENAI_API_KEY=EMPTY` and the local URL into the temp env file, and the fork's gpt-4o-mini branch
of `LLM_CALL.py` is taken unchanged (a served name like "qwen" would instead crash in the fork's
vLLM branch). Numbers obtained this way are not comparable to the paper's.

**The NL-assertion judge is `gpt-5`.** tau2's evaluator calls its natural-language-assertion judge
for every task that lists `nl_assertions`, with the model hard-coded to `gpt-5`
(`tau2/config.py`, temperature 1.0) through the same OpenAI client as the user simulator. In the
official test split that is all 20 airline tasks (none scored on it), 11 of 40 retail tasks (all 11
scored on it: their reward is DB x NL judge) and no telecom task. Your key therefore needs `gpt-5`
access as well as `gpt-4o-mini`; without it the fork retries the call every 60 s forever. The local smoke server answers only to `gpt-4o-mini`, so
`tests/smoke/run_tau2.sh` evaluates a trimmed copy of the first `eval.num_tasks` tasks per domain
with `nl_assertions` removed (`eval.tasks_dir` in `configs/smoke/tau2_smoke.yaml`), refusing any
task whose score would depend on them. The same trimmed `tasks_dir` is also how to run fewer tasks:
the fork ignores `--num-tasks` unless `--task-ids` is given, so `eval.num_tasks` only sets how many
tasks per domain count as complete.

## Arms

| Arm | Training data | Served name |
|---|---|---|
| Ours | the pair's bank, 2,302 cards | `<pair>-ours-s<N>` |
| Vanilla | 104 official train tasks, `real/tau2-official-train.parquet` | `<pair>-vanilla-s<N>` |

`a2d tau2 summarize` reports pass^1 for airline, retail and telecom and their unweighted mean.
Telecom is in neither training set.
