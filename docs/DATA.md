# Data, models and the release

The repository contains no data. Everything a stage reads comes from one of four places:

| Source | Command | Lands in |
|---|---|---|
| **Base models** at pinned commits (HF, or the ModelScope fallback) | `a2d models download --pair P` / `--all` | `$A2D_MODELS/<name>` |
| **The synthetic banks**, one HF dataset per pair: `gasolsun/<pair>` (`data/train-00000-of-00001.parquet`) | `a2d data download` | `$A2D_DATA/banks/<pair>.parquet` |
| **The auxiliary release** (the HF dataset repo `gasolsun/Ask_to_Distill-data`): δ_ref records, tau2 synthesizer-RL prompts and task-card pools | `a2d data download` | `$A2D_DATA` |
| **The evaluation release** (the HF dataset repo `gasolsun/Ask_to_Distill-eval`): every math and code test set in one format, with its prompt | `a2d data eval` | `$A2D_DATA/eval`, `$A2D_CACHE` |
| **Public datasets**, converted at pinned revisions: real-question training sets, tau2 data (and the evaluation sets, rebuilt from their sources) | `a2d data build {math,code-eval,ocr,code-real,tau2}` | `$A2D_DATA`, `$A2D_CACHE` |

The only data file in git is `data/MANIFEST.json`: path, md5, size, row count, license, provenance
and source of every downloaded file, plus (under `reference`) the md5 of every file the builders
make from public data. `a2d data download` checks every file it lands against it, and the
builders report whether their output equals the recorded file.

## Setup order

```bash
source configs/paths.env                      # A2D_DATA, A2D_MODELS, A2D_CKPT, A2D_OUT, A2D_CACHE, ...
a2d models download --all                     # or --pair <pair>; --modelscope where the HF CDN is blocked
a2d models syn-nothink --pair Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B   # DeepSeek synthesizer bases
a2d models syn-nothink --pair Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B
a2d data download                             # the seven banks + the auxiliary release
a2d data eval                                 # math + code test sets, LCB @0fe84c3, EvalPlus
a2d data build math                           # DAPO-Math-17k (and the test sets, from their sources)
a2d data build ocr                            # OpenCodeReasoning @20a1ca1 (~9 GB), input of code-real
a2d data build code-real                      # OpenCodeReasoning-10k, the Vanilla code arm
a2d data build tau2                           # tau2 domain data, official split, tokenizer cache, domain_ctx
```

## Evaluation sets

`a2d data eval` fetches `gasolsun/Ask_to_Distill-eval` at a pinned commit
(`ask2distill.data.eval_data.EVAL_REVISION`; `--revision` overrides it) and writes what `a2d eval
math|code` reads:

| Writes | From |
|---|---|
| `eval/math/{AIME24,AIME25,AIME26,BRUMO25,MATH-500}/test.parquet` | the release (`math/*.jsonl`) |
| `eval/math/AMC23/test.parquet` | `math-ai/amc23@80815d3` (not rehosted: no license declared) |
| `eval/code/extra_bench/{apps_introductory,apps_interview1k,apps_competition,codecontests}.jsonl` | the release (`code/*.jsonl`) |
| `$A2D_CACHE/evalplus/{HumanEvalPlus-v0.1.10,MbppPlus-v0.2.0}.jsonl` | the release (`code/evalplus_native/`, the official files verbatim) |
| LiveCodeBench `test5` / `test6` in `$A2D_CACHE/hf` | `livecodebench/code_generation_lite@0fe84c3` (not rehosted) |

With the locked pandas / pyarrow every file is byte-identical to the copy the final runs evaluated
on; the command checks each against `reference` in `data/MANIFEST.json`. The tau2 test tasks are
in the release too, but the tau2 evaluation also needs its environment, so they come with `a2d
data build tau2`. The HMMT sets (`eval.optional_tasks`) are only built by `a2d data build math`.
Offline: `--from-dir DIR` with `DIR/Ask_to_Distill-eval`, `DIR/AMC23` and `DIR/hf` (an HF cache
holding the two LiveCodeBench files).

## What is downloaded

`a2d data download` fetches, for the selected groups (`--only <group> ...`; all by default):

| Group | Files | From |
|---|---|---|
| `banks` | `banks/<pair>.parquet` for the seven pairs | the dataset `gasolsun/<pair>`, file `data/train-00000-of-00001.parquet` at the revision pinned in the manifest (`source`) |
| `banks` | `banks/Agent_Qwen3-4B_Qwen3-4B-Thinking-2507_part-{airline,retail}.parquet` | built locally from the two tau2 pools (`ask2distill.tau2.bank.build_synth`), byte-identical to the parts of the agent bank |
| `calib` | `calib/delta_ref/<pair>.json`: the δ_ref calibration record of each math and code pair (`percentile`, `delta_ref`, `delta_ref_rounded` = the pair's `reward.delta_ref`, δ_t quantiles) | `gasolsun/Ask_to_Distill-data` |
| `prompts` | `prompts/GENRL_TAU2_V{4,5,6,7}.parquet`, the instruction sets of the tau2 synthesizer-RL versions | `gasolsun/Ask_to_Distill-data` |
| `tau2` | `tau2/pool_v6s50.jsonl`, `tau2/pool_v7s60.jsonl`, the scored task-card pools of the trained tau2 synthesizers | `gasolsun/Ask_to_Distill-data` |

The repo id of the auxiliary release comes from `--repo-id`, else `$A2D_HF_DATA_REPO`, else the
`release.repo_id` of `data/MANIFEST.json`; `--revision` pins a commit and `--endpoint URL` (or
`HF_ENDPOINT`) selects an HF mirror. The dataset card and the `MANIFEST.json` at the root of the
dataset repo are never copied over the committed manifest.

The math and code synthesizer-RL instruction sets are not downloaded: `a2d prompts build` (or
`a2d synrl`, when the file is missing) builds them from the topic lists in `configs/topics/`.
Calibration questions are sampled from the initial synthesizer by `a2d calibrate`.

**Banks are the OPD default.** `opd.arms.ours.train` points at `$A2D_DATA/banks/<pair>.parquet`,
so `a2d opd --pair P --arm ours` trains on the released bank without re-harvesting. To train on a
bank you built, override it: `opd.arms.ours.train=$A2D_OUT/<pair>/banks/<pair>.parquet`. Each bank
keeps the row ids of the released file (`extra_info.index`, prefix `bank.row_id_prefix`, e.g.
`q3armg9_0`).

## What is built from public sources

These files are not redistributed. The builders write them in the exact format and selection the
Vanilla arms and the evaluation use, and print whether each equals the md5 recorded under
`reference` in `data/MANIFEST.json`:

| Command | Writes | Original |
|---|---|---|
| `a2d data build math` | `real/DAPO-Math-17k.parquet`: deduplicated on the prompt text (17,398 rows), the DAPO prompt with its "Remember to put your answer..." tail; `eval/math/<task>/test.parquet` for AIME24/25/26, AMC23, BRUMO25, MATH-500, HMMT Feb 25/26 | [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k); the test-set repos listed in the manifest |
| `a2d data build ocr` then `a2d data build code-real` | `OpenCodeReasoning/` (both splits at the pinned revision), then `real/OpenCodeReasoning-10k.parquet`: 10,000 statements selected with `random.Random(0)` over the sorted shard list, `comp_filter` + `strong_filter`, in the LiveCodeBench solve template (row ids `code_ocr10k_<i>`) | [nvidia/OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) |
| `a2d data build code-eval` | the LiveCodeBench and EvalPlus caches; `eval/code/extra_bench/{apps_introductory,apps_interview1k,apps_competition,codecontests}.jsonl` | `livecodebench/code_generation_lite`, EvalPlus, `codeparrot/apps`, `deepmind/code_contests` |
| `a2d data build tau2` | the tau2 domain data, `tau2/tasks_test/{airline,retail,telecom}.json` (the official test split), `real/tau2-official-train.parquet` (the official training split, airline 30 / retail 74), `tau2/domain_ctx.json`, the tokenizer cache | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench), NVlabs/ToolOrchestra |

The DAPO and tau2 files rebuild byte-identically to the files of the released runs (parquet bytes
can differ with the upstream revision or the writer version while the content is equal; the
builders say so). `code-real` runs the selection recipe with the current `code_filter`; the file
the released Vanilla code arms trained on was selected under an earlier version of that filter,
so the two share most but not all questions (the builder reports the difference).

## Licenses

| Data | License |
|---|---|
| Synthetic question banks (`gasolsun/<pair>`), tau2 card pools and the agent bank | CC-BY-NC-4.0 (questions written by the trained synthesizers; tau2-bench-derived content keeps its MIT notice) |
| δ_ref records | Apache-2.0, as this repository |
| tau2 prompt parquets, `domain_ctx.json`, `tasks_test`, `tau2-official-train` | MIT (derived from sierra-research/tau2-bench data) |
| DAPO-Math-17k | Apache-2.0 (`BytedTsinghua-SIA/DAPO-Math-17k`) |
| OpenCodeReasoning and `OpenCodeReasoning-10k` | CC-BY-4.0 (`nvidia/OpenCodeReasoning`) |
| Math test sets | the terms of each upstream dataset card (AIME-2024, math-ai/aime25, math-ai/amc23, MathArena sets, HuggingFaceH4/MATH-500) |
| APPS / CodeContests | MIT (`codeparrot/apps`) / CC-BY-4.0 (`deepmind/code_contests`) |
| LiveCodeBench, EvalPlus data | fetched at run time under their own terms |

The per-file license is in `data/MANIFEST.json` and on the dataset cards.

## Offline use and clusters where the HF file CDN is blocked

On some clusters the HF API answers but the LFS file CDN is firewalled, so `huggingface_hub`,
`hf_xet` and `hf-mirror` all fail. Three ways around it:

- **Models:** `a2d models download --pair P --modelscope` fetches the files from the ModelScope
  REST API (resumable ranged downloads) and verifies each one against the pinned HF revision
  through the HF API: LFS files by their sha256, small files by their git blob id. A ModelScope
  file that differs is rejected (documentation files are skipped with a warning). The
  `.cache/huggingface/download/*.metadata` files are written as HF would, so the directory is
  indistinguishable from an HF download. A model that is not mirrored on ModelScope -- the
  teachers `Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500` and `hbx/JustRL-DeepSeek-1.5B` were not, when this was written -- has to come from one of the next
  two options.
- **Any HF download through a mirror:** `--endpoint URL` (models and data) or `HF_ENDPOINT`.
- **Download elsewhere, then copy:** on a machine that can reach the hub, fetch the files, rsync
  them over and point the commands at the copies:

  | Command | What the local copy holds |
  |---|---|
  | `a2d data download --from-dir DIR` | a snapshot of `gasolsun/Ask_to_Distill-data` (the files of `MANIFEST.json` without a `source`) |
  | `a2d data download --banks-dir DIR` | the seven banks, each as `DIR/<pair>.parquet`, `DIR/banks/<pair>.parquet` or `DIR/<pair>/data/train-00000-of-00001.parquet` (a snapshot of `gasolsun/<pair>`) |
  | `a2d data build math --from-dir DIR` | `DIR/<raw name>/` per source dataset (`DAPO-Math-17k`, `AIME-2024`, `AIME25`, `AMC23`, `AIME26`, `MATH-500`, `HMMT_FEB25`, `HMMT_FEB26`, `BRUMO25`), i.e. a copy of `$A2D_CACHE/raw` |
  | `a2d data build code-eval --from-dir DIR` | `DIR/hf` (a copy of `$A2D_CACHE/hf`) and `DIR/evalplus`; the build then runs with `HF_HUB_OFFLINE=1` |
  | `a2d data build ocr --from-dir DIR` | `DIR/split_0/*.parquet`, `DIR/split_1/*.parquet` at revision `20a1ca1` |
  | `a2d data build tau2 --from-dir DIR` | `DIR/ToolOrchestra` (checkout with `data/tau2`), `DIR/tau2-bench` (checkout with `data/tau2/domains`), `DIR/hf` (HF cache with the Qwen3-8B tokenizer) |

  `--from-dir` and `--banks-dir` together make `a2d data download` work without network.

Model weights can be rsynced into `$A2D_MODELS` the same way; `a2d models download` then sees the
pinned commit in the download metadata and skips them.

## Pins

| What | Revision |
|---|---|
| Base models | full commit SHAs in `models.hf` of each pair config |
| Synthetic banks | the dataset commit of each `gasolsun/<pair>` in `data/MANIFEST.json` (`source.revision`) |
| OpenCodeReasoning | `nvidia/OpenCodeReasoning@20a1ca19c0d050fe9057fc08339d6b370ec1c67a` |
| LiveCodeBench data | `livecodebench/code_generation_lite@0fe84c3912ea0c4d4a78037083943e8f0c4dd505` (same default as `LCB_DATASET_REVISION` in the vendored loader) |
| EvalPlus data | MBPP+ v0.2.0, HumanEval+ v0.1.10 (evalplus defaults at `26d6d00`) |
| tau2 domain data | `NVlabs/ToolOrchestra@c1c0a4002be128b18accb267aac5ffdc26a28103` `data/tau2` (the fork's `simulation_guidelines.md`, md5-checked) |
| tau2 official split | `sierra-research/tau2-bench@b7ea9074c1cba482b30687fecdb5c8425fd6f619` `tasks.json` + `split_tasks.json` (md5-checked) |
| tau2 import-time tokenizer | `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` |

The math and APPS / CodeContests downloads of the released runs recorded no revision;
`a2d data build math` and `code-eval` fetch the hub's current files and report when a rebuilt
file differs from the recorded md5.

## Checkpoints

The Ours OPD student of each pair is published as the HF model `gasolsun/<pair>`.
`a2d eval math|code --model gasolsun/<pair>` downloads it to `$A2D_MODELS/gasolsun__<pair>` and
evaluates it (README, "Evaluation"). The code and tau2 synthesizers were not kept; re-harvesting their
banks needs a retrained synthesizer (docs/PIPELINE.md).
