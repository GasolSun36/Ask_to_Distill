"""Config system: every pair loads complete, values equal the final runs, guards fire.

Expected values are the settings of the runs behind the reported numbers. The
last checks compare the configs with the released delta_ref records and with the
download metadata of the models; they skip when the release or a model is
absent.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import ConfigAttributeError

from ask2distill import config as C
from ask2distill.paths import REPO_ROOT, get_paths

from conftest import (
    AGENT_QWEN3_4B, ALL_PAIRS, CODE_CODER_32B, CODE_PAIRS, CODE_QWEN3_30B, CODE_QWEN3_4B,
    MATH_PAIRS, MATH_QWEN3_1P7B, MATH_R1_1P5B, MATH_R1_7B,
)

PAIRS = ALL_PAIRS
# The real A2D_* locations, resolved at import, before the autouse fixture
# below points them at placeholder paths.
_PATHS = get_paths()
SMOKES = ["configs/smoke/math_smoke.yaml", "configs/smoke/code_smoke.yaml",
          "configs/smoke/tau2_smoke.yaml"]


@pytest.fixture(autouse=True)
def _repo_paths(monkeypatch):
    """Fixed, absolute A2D_* locations so expected paths are predictable."""
    for var, sub in (("A2D_DATA", "d"), ("A2D_MODELS", "m"), ("A2D_CKPT", "c"),
                     ("A2D_OUT", "o"), ("A2D_CACHE", "x")):
        monkeypatch.setenv(var, f"/a2d/{sub}")


def load(pair, *overrides):
    return C.load(pair, None, list(overrides))


# ------------------------------------------------------------------------ completeness

def test_pair_list_is_the_core_scope():
    assert C.list_pairs() == sorted(PAIRS)


@pytest.mark.parametrize("pair", PAIRS)
def test_pair_loads_complete_and_faithful(pair):
    cfg = load(pair)
    assert C.missing_keys(cfg) == []
    assert C.absent_keys(cfg) == []
    assert cfg.pair == pair and cfg.smoke is False
    # a pair config reproduces the final runs, so no guard may warn
    assert C.validate(cfg) == []
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


@pytest.mark.parametrize("path", SMOKES)
def test_smoke_configs_load(path):
    cfg = C.load(None, path)
    assert cfg.smoke is True
    assert C.missing_keys(cfg) == []
    # ppo_mini_batch_size == train_batch_size, one save at the last step
    assert cfg.synrl.ppo_mini_batch_size == cfg.synrl.train_batch_size
    assert cfg.synrl.save_freq == cfg.synrl.total_steps
    assert cfg.opd.train_max_samples > 0 and cfg.opd.val_max_samples > 0


def test_math_code_smokes_validate_without_warnings():
    for path in SMOKES[:2]:
        assert C.validate(C.load(None, path)) == []


@pytest.mark.parametrize("pair", PAIRS)
def test_resolved_paths_are_absolute(pair, monkeypatch):
    monkeypatch.setenv("A2D_DATA", "relative/data")          # resolved against the repo root
    cfg = load(pair)
    assert cfg.paths.data == str(REPO_ROOT / "relative" / "data")
    raw = OmegaConf.to_container(cfg, resolve=False)
    res = OmegaConf.to_container(cfg, resolve=True)

    checked = []

    def walk(r, v, key):
        if isinstance(r, dict):
            for k in r:
                walk(r[k], v[k], f"{key}.{k}")
        elif isinstance(r, list):
            for i, (a, b) in enumerate(zip(r, v, strict=False)):
                walk(a, b, f"{key}[{i}]")
        elif isinstance(r, str) and "${paths." in r and v is not None:   # optional paths may be unset
            path_part = v.split("=", 1)[-1]          # hydra override strings carry key=path
            assert Path(path_part).is_absolute(), (key, v)
            checked.append(key)

    walk(raw, res, "")
    assert checked


# --------------------------------------------------------------------- expected values

MOD, DAT, CKP, OUT = "/a2d/m", "/a2d/d", "/a2d/c", "/a2d/o"
EXPECTED = [
    # ---- Math_Qwen3-1.7B_Qwen3-4B-Non-Thinking-RL-Math-Step500
    (MATH_QWEN3_1P7B, "models.student", f"{MOD}/Qwen3-1.7B"),
    (MATH_QWEN3_1P7B, "models.teacher", f"{MOD}/Qwen3-4B-Non-Thinking-RL-Math-Step500"),
    (MATH_QWEN3_1P7B, "models.synthesizer_init", f"{MOD}/Qwen3-1.7B"),
    (MATH_QWEN3_1P7B, "models.qwen3_nothink", True),
    (MATH_QWEN3_1P7B, "models.hf.teacher.repo", "Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500"),
    (MATH_QWEN3_1P7B, "models.hf.teacher.revision", "05d82d02780d4a6f8295b2909dbbd89e8a8b5aaa"),
    (MATH_QWEN3_1P7B, "calib.n_questions", 200),
    (MATH_QWEN3_1P7B, "calib.questions", f"{OUT}/{MATH_QWEN3_1P7B}/calib/calib_questions.jsonl"),
    (MATH_QWEN3_1P7B, "calib.question_sampling", {"topic_seed": 1000, "temperature": 1.0, "top_p": 1.0,
                                                  "max_new": 4096, "prompt_head": 1536,
                                                  "gpu_memory_utilization": 0.85, "oversample": 0.25,
                                                  "max_rounds": 3}),
    (MATH_QWEN3_1P7B, "calib.rolls", 1), (MATH_QWEN3_1P7B, "calib.max_new", 7168), (MATH_QWEN3_1P7B, "calib.temperature", 0.7),
    (MATH_QWEN3_1P7B, "calib.top_p", 0.95), (MATH_QWEN3_1P7B, "calib.prompt_head", 1024),
    (MATH_QWEN3_1P7B, "calib.gpu_memory_utilization", 0.85), (MATH_QWEN3_1P7B, "calib.topk", 16), (MATH_QWEN3_1P7B, "calib.chunk", 128),
    (MATH_QWEN3_1P7B, "calib.min_tokens", 10), (MATH_QWEN3_1P7B, "calib.percentile", 95), (MATH_QWEN3_1P7B, "calib.round_decimals", 4),
    (MATH_QWEN3_1P7B, "calib.teacher_device", None), (MATH_QWEN3_1P7B, "calib.expected_delta_ref", 1.9049),
    (MATH_QWEN3_1P7B, "reward.domain", "math"), (MATH_QWEN3_1P7B, "reward.solve_template", "math_boxed"),
    (MATH_QWEN3_1P7B, "reward.solve_system", ""), (MATH_QWEN3_1P7B, "reward.delta_ref", 1.9049), (MATH_QWEN3_1P7B, "reward.topk", 16),
    (MATH_QWEN3_1P7B, "reward.chunk", 128), (MATH_QWEN3_1P7B, "reward.use_omega", True), (MATH_QWEN3_1P7B, "reward.len_alpha", 0.5),
    (MATH_QWEN3_1P7B, "reward.len_pen", 0.3), (MATH_QWEN3_1P7B, "reward.max_new", 7168), (MATH_QWEN3_1P7B, "reward.rolls", 1),
    (MATH_QWEN3_1P7B, "reward.temperature", 0.7), (MATH_QWEN3_1P7B, "reward.top_p", 0.95), (MATH_QWEN3_1P7B, "reward.sample_top_k", 0),
    (MATH_QWEN3_1P7B, "reward.trunc_zero", False), (MATH_QWEN3_1P7B, "reward.gen_bs", 0),
    (MATH_QWEN3_1P7B, "reward.min_question_chars", 20), (MATH_QWEN3_1P7B, "reward.min_rollout_tokens", 10),
    (MATH_QWEN3_1P7B, "reward.fix_eos", False), (MATH_QWEN3_1P7B, "reward.step_counter", "local"),
    (MATH_QWEN3_1P7B, "reward.diversity", {"mode": "bleu", "thresh": 0.1, "pen": 1.0, "group": 8, "max_len": 512,
                                "embed_bs": 16, "per_group_embed": True}),
    (MATH_QWEN3_1P7B, "reward.dump", f"{OUT}/{MATH_QWEN3_1P7B}/synrl/synrl_dump_synthesizer_{MATH_QWEN3_1P7B}.jsonl"),
    (MATH_QWEN3_1P7B, "synrl.experiment", f"synthesizer_{MATH_QWEN3_1P7B}"), (MATH_QWEN3_1P7B, "synrl.prompts", f"{OUT}/{MATH_QWEN3_1P7B}/prompts/SYNRL_TOPICS_MATH.parquet"),
    (MATH_QWEN3_1P7B, "synrl.prompt_rows", 4096), (MATH_QWEN3_1P7B, "synrl.prompt_seed", 0),
    (MATH_QWEN3_1P7B, "synrl.total_steps", 60), (MATH_QWEN3_1P7B, "synrl.save_freq", 5), (MATH_QWEN3_1P7B, "synrl.train_batch_size", 8),
    (MATH_QWEN3_1P7B, "synrl.rollout_n", 8), (MATH_QWEN3_1P7B, "synrl.ppo_mini_batch_size", 8),
    (MATH_QWEN3_1P7B, "synrl.ppo_micro_batch_size_per_gpu", 1), (MATH_QWEN3_1P7B, "synrl.lr", 1.0e-6),
    (MATH_QWEN3_1P7B, "synrl.entropy_coeff", 0.01), (MATH_QWEN3_1P7B, "synrl.kl_loss_coef", 0.005),
    (MATH_QWEN3_1P7B, "synrl.max_prompt_length", 1024), (MATH_QWEN3_1P7B, "synrl.max_response_length", 4096),
    (MATH_QWEN3_1P7B, "synrl.ppo_max_token_len_per_gpu", 16384), (MATH_QWEN3_1P7B, "synrl.max_num_batched_tokens", 16384),
    (MATH_QWEN3_1P7B, "synrl.offload", False), (MATH_QWEN3_1P7B, "synrl.gpu_memory_utilization", 0.6),
    (MATH_QWEN3_1P7B, "synrl.rollout_temperature", 1.0), (MATH_QWEN3_1P7B, "synrl.model_dtype", "fp32"), (MATH_QWEN3_1P7B, "synrl.shuffle", False),
    (MATH_QWEN3_1P7B, "synrl.layout", {"actor_gpus": 8, "reward_gpus": 8, "reward_nnodes": 1, "enable_resource_pool": True}),
    (MATH_QWEN3_1P7B, "select.sweep.steps", list(range(5, 61, 5))), (MATH_QWEN3_1P7B, "select.sweep.n", 300),
    (MATH_QWEN3_1P7B, "select.sweep.max_new", 4096), (MATH_QWEN3_1P7B, "select.sweep.temperature", 1.0),
    (MATH_QWEN3_1P7B, "select.sweep.top_p", 0.95), (MATH_QWEN3_1P7B, "select.sweep.topic_seed", 1),
    (MATH_QWEN3_1P7B, "select.sweep.prompt_head", 1536), (MATH_QWEN3_1P7B, "select.sweep.gpu_memory_utilization", 0.85),
    (MATH_QWEN3_1P7B, "select.sweep.strip_thinking", False),
    (MATH_QWEN3_1P7B, "select.pick.cos_max", 0.85), (MATH_QWEN3_1P7B, "select.pick.min_eff_clusters", 5),
    (MATH_QWEN3_1P7B, "select.pick.eff_cluster_cos", 0.9), (MATH_QWEN3_1P7B, "select.pick.reward_frac", 0.95),
    (MATH_QWEN3_1P7B, "select.pick.reward_window", 2), (MATH_QWEN3_1P7B, "select.pick.save_freq", 5), (MATH_QWEN3_1P7B, "select.pick.min_rows", 50),
    (MATH_QWEN3_1P7B, "select.yield", {"candidates": [35, 50, 60], "n": 5000, "seed": 100, "bank_n": 64, "tie": "earliest"}),
    (MATH_QWEN3_1P7B, "select.chosen_step", 50),
    (MATH_QWEN3_1P7B, "harvest.synthesizer", f"{CKP}/synthesizer_{MATH_QWEN3_1P7B}/hf/step_50"), (MATH_QWEN3_1P7B, "harvest.shards", 16),
    (MATH_QWEN3_1P7B, "harvest.per_shard", 5000), (MATH_QWEN3_1P7B, "harvest.seed_formula", "index"),
    (MATH_QWEN3_1P7B, "harvest.temperature", 1.0), (MATH_QWEN3_1P7B, "harvest.top_p", 0.95), (MATH_QWEN3_1P7B, "harvest.max_new", 4096),
    (MATH_QWEN3_1P7B, "harvest.prompt_head", 1536), (MATH_QWEN3_1P7B, "harvest.gpu_memory_utilization", 0.85),
    (MATH_QWEN3_1P7B, "harvest.strip_thinking", True), (MATH_QWEN3_1P7B, "harvest.vllm_seed", 0),
    (MATH_QWEN3_1P7B, "filter.domain", "math"), (MATH_QWEN3_1P7B, "filter.slices", [list(range(16))]), (MATH_QWEN3_1P7B, "filter.jaccard", 0.7),
    (MATH_QWEN3_1P7B, "filter.sat_thr", 0.95), (MATH_QWEN3_1P7B, "filter.embed_bs", 16), (MATH_QWEN3_1P7B, "filter.aggregate", False),
    (MATH_QWEN3_1P7B, "filter.agg_order", None),
    (MATH_QWEN3_1P7B, "bank.name", MATH_QWEN3_1P7B), (MATH_QWEN3_1P7B, "bank.row_id_prefix", "q3armg9"), (MATH_QWEN3_1P7B, "bank.n", 3200), (MATH_QWEN3_1P7B, "bank.template", "dapo_prefix_only"),
    (MATH_QWEN3_1P7B, "bank.pool", f"{OUT}/{MATH_QWEN3_1P7B}/filter/sat.s0_questions.jsonl"), (MATH_QWEN3_1P7B, "bank.refilter", None),
    (MATH_QWEN3_1P7B, "bank.allow_short", True), (MATH_QWEN3_1P7B, "bank.expected_rows", 2889),
    (MATH_QWEN3_1P7B, "bank.out", f"{OUT}/{MATH_QWEN3_1P7B}/banks/{MATH_QWEN3_1P7B}.parquet"),
    (MATH_QWEN3_1P7B, "bank.released", f"{DAT}/banks/{MATH_QWEN3_1P7B}.parquet"),
    (MATH_QWEN3_1P7B, "opd.chat_template_args", "+data.apply_chat_template_kwargs.enable_thinking=False"),
    (MATH_QWEN3_1P7B, "opd.max_prompt_length", 1024), (MATH_QWEN3_1P7B, "opd.max_response_length", 7168),
    (MATH_QWEN3_1P7B, "opd.max_val_response_length", 7168),
    (MATH_QWEN3_1P7B, "opd.test_files", [f"{DAT}/eval/math/AIME25/test.parquet", f"{DAT}/eval/math/AMC23/test.parquet",
                              f"{DAT}/eval/math/AIME24/test.parquet"]),
    (MATH_QWEN3_1P7B, "opd.custom_reward.name", "reward_func"), (MATH_QWEN3_1P7B, "opd.is_plot", True), (MATH_QWEN3_1P7B, "opd.esi_save", True),
    (MATH_QWEN3_1P7B, "opd.train_max_samples", -1), (MATH_QWEN3_1P7B, "opd.val_max_samples", -1), (MATH_QWEN3_1P7B, "opd.shuffle", False),
    (MATH_QWEN3_1P7B, "opd.arms.ours", {"train": f"{DAT}/banks/{MATH_QWEN3_1P7B}.parquet",
                                          "experiment": f"opd_{MATH_QWEN3_1P7B}_ours",
                             "total_epochs": 2, "save_freq": 5, "ppo_max_token_len_per_gpu": 16384,
                             "reward_micro_bs": 4}),
    (MATH_QWEN3_1P7B, "opd.arms.vanilla", {"train": f"{DAT}/real/DAPO-Math-17k.parquet",
                                             "experiment": f"opd_{MATH_QWEN3_1P7B}_vanilla",
                                "total_epochs": 2, "save_freq": 20, "ppo_max_token_len_per_gpu": 16384,
                                "reward_micro_bs": 4}),
    (MATH_QWEN3_1P7B, "eval.tasks", ["AIME24", "AIME25", "AIME26", "AMC23", "BRUMO25", "MATH500"]),
    (MATH_QWEN3_1P7B, "eval.curve_tasks", ["AIME24", "AIME25", "AMC23"]), (MATH_QWEN3_1P7B, "eval.task_dirs.MATH500", "MATH-500"),
    (MATH_QWEN3_1P7B, "eval.n", {"default": 16, "MATH500": 4}), (MATH_QWEN3_1P7B, "eval.max_tokens", 31744),
    (MATH_QWEN3_1P7B, "eval.temperature", 0.7), (MATH_QWEN3_1P7B, "eval.top_p", 0.95), (MATH_QWEN3_1P7B, "eval.gpu_memory_utilization", 0.9),
    (MATH_QWEN3_1P7B, "eval.disable_thinking", True), (MATH_QWEN3_1P7B, "eval.enforce_eager", True), (MATH_QWEN3_1P7B, "eval.strict_boxed", False),
    (MATH_QWEN3_1P7B, "eval.stop_ids_rule", "first_token_of_im_end_and_endoftext"), (MATH_QWEN3_1P7B, "eval.engine_seed", None),
    # ---- OPD method constants, checked on one pair
    (MATH_QWEN3_1P7B, "opd.adv_estimator", "token_reward_direct"), (MATH_QWEN3_1P7B, "opd.grpo_outcome_weight", 1.0),
    (MATH_QWEN3_1P7B, "opd.log_prob_top_k", 16), (MATH_QWEN3_1P7B, "opd.top_k_strategy", "only_stu"),
    (MATH_QWEN3_1P7B, "opd.reward_weight_mode", "student_p"), (MATH_QWEN3_1P7B, "opd.temperature", 1.0),
    (MATH_QWEN3_1P7B, "opd.teacher_temperature", 1.0), (MATH_QWEN3_1P7B, "opd.repetition_penalty", 1.0), (MATH_QWEN3_1P7B, "opd.n_responses", 4),
    (MATH_QWEN3_1P7B, "opd.mini_batch_size", 64), (MATH_QWEN3_1P7B, "opd.lr", 1.0e-6), (MATH_QWEN3_1P7B, "opd.model_dtype", "fp32"),
    (MATH_QWEN3_1P7B, "opd.rollout_gpu_memory_utilization", 0.8), (MATH_QWEN3_1P7B, "opd.loss_agg_mode", "token-mean"),
    (MATH_QWEN3_1P7B, "opd.enable_format_reward", False), (MATH_QWEN3_1P7B, "opd.use_kl", False), (MATH_QWEN3_1P7B, "opd.entropy_coeff", 0),
    (MATH_QWEN3_1P7B, "opd.filter_overlong_prompts", True), (MATH_QWEN3_1P7B, "opd.truncation", "error"),
    (MATH_QWEN3_1P7B, "opd.esi_redundant_time", 0), (MATH_QWEN3_1P7B, "opd.test_freq", -1), (MATH_QWEN3_1P7B, "opd.val_before_train", False),
    (MATH_QWEN3_1P7B, "opd.log_val_generations", 2), (MATH_QWEN3_1P7B, "opd.n_gpus_per_node", 8), (MATH_QWEN3_1P7B, "opd.nnodes", 1),
    (MATH_QWEN3_1P7B, "opd.resume_mode", "auto"), (MATH_QWEN3_1P7B, "opd.calculate_log_probs", True),
    (MATH_QWEN3_1P7B, "opd.val_kwargs", {"do_sample": True, "n": 16, "temperature": 0.7, "top_p": 0.95}),
    # ---- Math_DeepSeek-R1-Distill-Qwen-1.5B_JustRL-DeepSeek-1.5B / Math_DeepSeek-R1-Distill-Qwen-7B_Skywork-OR1-Math-7B
    (MATH_R1_1P5B, "models.student", f"{MOD}/DeepSeek-R1-Distill-Qwen-1.5B"),
    (MATH_R1_1P5B, "models.teacher", f"{MOD}/JustRL-DeepSeek-1.5B"),
    (MATH_R1_1P5B, "models.synthesizer_init", f"{MOD}/DeepSeek-R1-Distill-Qwen-1.5B-SynNoThink"),
    (MATH_R1_1P5B, "models.qwen3_nothink", False), (MATH_R1_1P5B, "reward.delta_ref", 0.4411),
    (MATH_R1_1P5B, "synrl.experiment", f"synthesizer_{MATH_R1_1P5B}"), (MATH_R1_1P5B, "synrl.offload", False),
    (MATH_R1_1P5B, "synrl.gpu_memory_utilization", 0.6),
    (MATH_R1_1P5B, "select.yield.candidates", [10, 40, 60]), (MATH_R1_1P5B, "select.chosen_step", 40),
    (MATH_R1_1P5B, "harvest.shards", 16), (MATH_R1_1P5B, "harvest.per_shard", 5000), (MATH_R1_1P5B, "harvest.seed_formula", "index"),
    (MATH_R1_1P5B, "filter.slices", [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]),
    (MATH_R1_1P5B, "filter.agg_order", [0, 1, 2, 3]), (MATH_R1_1P5B, "bank.name", MATH_R1_1P5B), (MATH_R1_1P5B, "bank.row_id_prefix", "b15arm"), (MATH_R1_1P5B, "bank.n", 12800),
    (MATH_R1_1P5B, "opd.chat_template_args", ""), (MATH_R1_1P5B, "opd.max_val_response_length", 7168),
    (MATH_R1_1P5B, "opd.arms.ours.experiment", f"opd_{MATH_R1_1P5B}_ours"), (MATH_R1_1P5B, "opd.arms.ours.total_epochs", 2),
    (MATH_R1_1P5B, "opd.arms.ours.save_freq", 10), (MATH_R1_1P5B, "opd.arms.ours.ppo_max_token_len_per_gpu", 16384),
    (MATH_R1_1P5B, "opd.arms.ours.reward_micro_bs", 4),
    (MATH_R1_1P5B, "opd.arms.vanilla.experiment", f"opd_{MATH_R1_1P5B}_vanilla"),
    (MATH_R1_1P5B, "opd.arms.vanilla.total_epochs", 2), (MATH_R1_1P5B, "opd.arms.vanilla.save_freq", 20),
    (MATH_R1_1P5B, "opd.arms.vanilla.ppo_max_token_len_per_gpu", 32768), (MATH_R1_1P5B, "opd.arms.vanilla.reward_micro_bs", 8),
    (MATH_R1_7B, "models.student", f"{MOD}/DeepSeek-R1-Distill-Qwen-7B"),
    (MATH_R1_7B, "models.teacher", f"{MOD}/Skywork-OR1-Math-7B"),
    (MATH_R1_7B, "models.synthesizer_init", f"{MOD}/DeepSeek-R1-Distill-Qwen-7B-SynNoThink"),
    (MATH_R1_7B, "reward.delta_ref", 0.4867), (MATH_R1_7B, "synrl.experiment", f"synthesizer_{MATH_R1_7B}"), (MATH_R1_7B, "synrl.offload", True),
    (MATH_R1_7B, "synrl.gpu_memory_utilization", 0.5), (MATH_R1_7B, "select.yield", None), (MATH_R1_7B, "select.chosen_step", 55),
    (MATH_R1_7B, "harvest.shards", 8), (MATH_R1_7B, "filter.slices", [[0, 1, 2, 3], [4, 5, 6, 7]]),
    (MATH_R1_7B, "filter.agg_order", [0, 1]), (MATH_R1_7B, "bank.name", MATH_R1_7B), (MATH_R1_7B, "bank.row_id_prefix", "b7barm"), (MATH_R1_7B, "bank.n", 12800),
    (MATH_R1_7B, "opd.arms.ours.experiment", f"opd_{MATH_R1_7B}_ours"), (MATH_R1_7B, "opd.arms.ours.save_freq", 20),
    (MATH_R1_7B, "opd.arms.ours.total_epochs", 2), (MATH_R1_7B, "opd.arms.vanilla.experiment", f"opd_{MATH_R1_7B}_vanilla"),
    (MATH_R1_7B, "opd.arms.vanilla.total_epochs", 1), (MATH_R1_7B, "opd.arms.vanilla.save_freq", 20),
    (MATH_R1_7B, "opd.arms.vanilla.ppo_max_token_len_per_gpu", 16384), (MATH_R1_7B, "opd.arms.vanilla.reward_micro_bs", 4),
    # ---- Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507
    (CODE_QWEN3_30B, "models.student", f"{MOD}/Qwen3-4B"), (CODE_QWEN3_30B, "models.teacher", f"{MOD}/Qwen3-30B-A3B-Instruct-2507"),
    (CODE_QWEN3_30B, "models.synthesizer_init", f"{MOD}/Qwen3-4B"), (CODE_QWEN3_30B, "models.qwen3_nothink", True),
    (CODE_QWEN3_30B, "calib.questions", f"{OUT}/{CODE_QWEN3_30B}/calib/calib_questions.jsonl"),
    (CODE_QWEN3_30B, "calib.n_questions", 200), (CODE_QWEN3_30B, "calib.question_sampling.max_new", 2048),
    (CODE_QWEN3_30B, "calib.question_sampling.prompt_head", 2304), (CODE_QWEN3_30B, "calib.max_new", 4096),
    (CODE_QWEN3_30B, "calib.solve_template", "lcb"), (CODE_QWEN3_30B, "calib.teacher_device", "cuda:1"),
    (CODE_QWEN3_30B, "calib.expected_delta_ref", 3.9756),
    (CODE_QWEN3_30B, "reward.domain", "code"), (CODE_QWEN3_30B, "reward.solve_template", "lcb"), (CODE_QWEN3_30B, "reward.delta_ref", 3.9756),
    (CODE_QWEN3_30B, "reward.len_alpha", 1.0), (CODE_QWEN3_30B, "reward.len_pen", 0.0), (CODE_QWEN3_30B, "reward.max_new", 4096),
    (CODE_QWEN3_30B, "reward.rolls", 2), (CODE_QWEN3_30B, "reward.trunc_zero", True), (CODE_QWEN3_30B, "reward.sample_top_k", 0),
    (CODE_QWEN3_30B, "reward.gen_bs", 2), (CODE_QWEN3_30B, "reward.diversity.mode", "bleu"),
    (CODE_QWEN3_30B, "reward.diversity.thresh", 0.1), (CODE_QWEN3_30B, "reward.diversity.pen", 1.0),
    (CODE_QWEN3_30B, "synrl.experiment", f"synthesizer_{CODE_QWEN3_30B}"),
    (CODE_QWEN3_30B, "synrl.prompts", f"{OUT}/{CODE_QWEN3_30B}/prompts/SYNRL_TOPICS_CODE.parquet"),
    (CODE_QWEN3_30B, "synrl.save_freq", 5),
    (CODE_QWEN3_30B, "synrl.max_prompt_length", 2048), (CODE_QWEN3_30B, "synrl.max_response_length", 2048),
    (CODE_QWEN3_30B, "synrl.offload", False), (CODE_QWEN3_30B, "synrl.gpu_memory_utilization", 0.6),
    (CODE_QWEN3_30B, "select.sweep.max_new", 2048), (CODE_QWEN3_30B, "select.sweep.prompt_head", 2304),
    (CODE_QWEN3_30B, "select.yield", None), (CODE_QWEN3_30B, "select.chosen_step", 60),
    (CODE_QWEN3_30B, "harvest.shards", 12), (CODE_QWEN3_30B, "harvest.per_shard", 8000),
    (CODE_QWEN3_30B, "harvest.seed_formula", "7919*i+104729"), (CODE_QWEN3_30B, "harvest.max_new", 2048),
    (CODE_QWEN3_30B, "harvest.prompt_head", 2304),
    (CODE_QWEN3_30B, "filter.slices", [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]),
    (CODE_QWEN3_30B, "filter.aggregate", True), (CODE_QWEN3_30B, "filter.agg_thr", 0.95),
    (CODE_QWEN3_30B, "filter.agg_order", [0, 1, 2, 3]),
    (CODE_QWEN3_30B, "bank.name", CODE_QWEN3_30B), (CODE_QWEN3_30B, "bank.row_id_prefix", "code_a_syn10k"), (CODE_QWEN3_30B, "bank.n", 10000), (CODE_QWEN3_30B, "bank.template", "lcb_no_system"),
    (CODE_QWEN3_30B, "bank.pool", f"{OUT}/{CODE_QWEN3_30B}/pool_clean.jsonl"),
    (CODE_QWEN3_30B, "bank.refilter", {"kind": "comp_strong_salvage_exact_dedup", "input": f"{OUT}/{CODE_QWEN3_30B}/pool.jsonl",
                                         "output": f"{OUT}/{CODE_QWEN3_30B}/pool_clean.jsonl"}),
    (CODE_QWEN3_30B, "opd.max_prompt_length", 2048), (CODE_QWEN3_30B, "opd.max_response_length", 4096),
    (CODE_QWEN3_30B, "opd.max_val_response_length", 4096), (CODE_QWEN3_30B, "opd.test_files", ["train"]),
    (CODE_QWEN3_30B, "opd.custom_reward.name", "compute_score"), (CODE_QWEN3_30B, "opd.use_kl", False),
    (CODE_QWEN3_30B, "opd.entropy_coeff", 0), (CODE_QWEN3_30B, "opd.is_plot", False), (CODE_QWEN3_30B, "opd.esi_save", False),
    (CODE_QWEN3_30B, "opd.arms.ours", {"train": f"{DAT}/banks/{CODE_QWEN3_30B}.parquet",
                                         "experiment": f"opd_{CODE_QWEN3_30B}_ours",
                                 "total_epochs": 1, "save_freq": 5, "ppo_max_token_len_per_gpu": 16384,
                                 "reward_micro_bs": 4}),
    (CODE_QWEN3_30B, "opd.arms.vanilla", {"train": f"{DAT}/real/OpenCodeReasoning-10k.parquet",
                                            "experiment": f"opd_{CODE_QWEN3_30B}_vanilla",
                                    "total_epochs": 1, "save_freq": 5, "ppo_max_token_len_per_gpu": 16384,
                                    "reward_micro_bs": 4}),
    (CODE_QWEN3_30B, "eval.prompt_style", "Qwen3NoThink"), (CODE_QWEN3_30B, "eval.lcb.version", "v5_v6"),
    (CODE_QWEN3_30B, "eval.lcb.n", 4), (CODE_QWEN3_30B, "eval.lcb.temperature", 0.7), (CODE_QWEN3_30B, "eval.lcb.top_p", 0.8),
    (CODE_QWEN3_30B, "eval.lcb.top_k", 20), (CODE_QWEN3_30B, "eval.lcb.max_tokens", 8192),
    (CODE_QWEN3_30B, "eval.lcb.max_model_len", 16384),
    (CODE_QWEN3_30B, "eval.lcb.stop", "<|im_end|>"), (CODE_QWEN3_30B, "eval.lcb.tp", 2),
    (CODE_QWEN3_30B, "eval.lcb.dataset_revision", "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"),
    (CODE_QWEN3_30B, "eval.evalplus.max_new_tokens", 1024), (CODE_QWEN3_30B, "eval.evalplus.max_model_len", 4096),
    (CODE_QWEN3_30B, "eval.extra_bench.temperature", 0.2), (CODE_QWEN3_30B, "eval.extra_bench.top_k", 20),
    (CODE_QWEN3_30B, "eval.extra_bench.max_new", 4096),
    (CODE_QWEN3_30B, "eval.extra_bench.system_prompt", "You are a helpful assistant."),
    (CODE_QWEN3_30B, "eval.avg_items", ["lcb", "mbpp_plus_lenient", "apps_introductory", "apps_interview1k",
                                  "apps_competition", "codecontests"]),
    # ---- Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct / Code_Qwen3-4B_Qwen3-4B-Instruct-2507
    (CODE_CODER_32B, "models.student", f"{MOD}/Qwen2.5-Coder-3B-Instruct"),
    (CODE_CODER_32B, "models.teacher", f"{MOD}/Qwen2.5-Coder-32B-Instruct"), (CODE_CODER_32B, "models.qwen3_nothink", False),
    (CODE_CODER_32B, "reward.delta_ref", 1.3629),
    (CODE_CODER_32B, "reward.gen_bs", 4), (CODE_CODER_32B, "synrl.experiment", f"synthesizer_{CODE_CODER_32B}"), (CODE_CODER_32B, "synrl.save_freq", 5),
    (CODE_CODER_32B, "select.chosen_step", 40), (CODE_CODER_32B, "bank.name", CODE_CODER_32B), (CODE_CODER_32B, "bank.row_id_prefix", "code_b_syn10k"),
    (CODE_CODER_32B, "bank.pool", f"{OUT}/{CODE_CODER_32B}/pool.jsonl"),
    (CODE_CODER_32B, "opd.chat_template_args", ""), (CODE_CODER_32B, "opd.use_kl", True), (CODE_CODER_32B, "opd.kl_coef", 0.005),
    (CODE_CODER_32B, "opd.entropy_coeff", 0.005), (CODE_CODER_32B, "opd.arms.ours.experiment", f"opd_{CODE_CODER_32B}_ours"),
    (CODE_CODER_32B, "opd.arms.vanilla.experiment", f"opd_{CODE_CODER_32B}_vanilla"), (CODE_CODER_32B, "opd.arms.vanilla.save_freq", 5),
    (CODE_CODER_32B, "eval.prompt_style", "CodeQwenInstruct"), (CODE_QWEN3_4B, "models.student", f"{MOD}/Qwen3-4B"), (CODE_QWEN3_4B, "models.teacher", f"{MOD}/Qwen3-4B-Instruct-2507"),
    (CODE_QWEN3_4B, "reward.delta_ref", 3.8221),
    (CODE_QWEN3_4B, "reward.gen_bs", 8), (CODE_QWEN3_4B, "synrl.experiment", f"synthesizer_{CODE_QWEN3_4B}"),
    (CODE_QWEN3_4B, "synrl.save_freq", 10), (CODE_QWEN3_4B, "select.pick.save_freq", 10),
    (CODE_QWEN3_4B, "select.sweep.steps", [10, 20, 30, 40, 50, 60]), (CODE_QWEN3_4B, "select.chosen_step", 50),
    (CODE_QWEN3_4B, "filter.slices", [[i] for i in range(12)]),
    (CODE_QWEN3_4B, "filter.agg_order", [0, 10, 11, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (CODE_QWEN3_4B, "bank.name", CODE_QWEN3_4B), (CODE_QWEN3_4B, "bank.row_id_prefix", "code_d_syn10k"), (CODE_QWEN3_4B, "opd.use_kl", False),
    (CODE_QWEN3_4B, "opd.arms.ours.experiment", f"opd_{CODE_QWEN3_4B}_ours"), (CODE_QWEN3_4B, "opd.arms.ours.save_freq", 5),
    (CODE_QWEN3_4B, "opd.arms.vanilla.experiment", f"opd_{CODE_QWEN3_4B}_vanilla"), (CODE_QWEN3_4B, "opd.arms.vanilla.save_freq", 20),
    (CODE_QWEN3_4B, "eval.prompt_style", "Qwen3NoThink"), # ---- Agent_Qwen3-4B_Qwen3-4B-Thinking-2507
    (AGENT_QWEN3_4B, "models.student", f"{MOD}/Qwen3-4B"), (AGENT_QWEN3_4B, "models.teacher", f"{MOD}/Qwen3-4B-Thinking-2507"),
    (AGENT_QWEN3_4B, "models.user_sim_train", f"{MOD}/Qwen3-4B-Instruct-2507"), (AGENT_QWEN3_4B, "models.user_sim_eval", "gpt-4o-mini"),
    (AGENT_QWEN3_4B, "models.templates.user", None),
    (AGENT_QWEN3_4B, "tau2_data.db_dir_env", "A2D_TAU2_DB_DIR"),
    (AGENT_QWEN3_4B, "tau2_data.domain_ctx", f"{DAT}/tau2/domain_ctx.json"),
    (AGENT_QWEN3_4B, "tau2_data.tokenizer_cache", "/a2d/x/hf"),
    (AGENT_QWEN3_4B, "reward_env", {"TAU2_K": 6, "TAU2_KT": 3, "TAU2_GROUNDING": 1, "TAU2_TEACHER_GATE": 1,
                            "TAU2_MAX_TURNS": 12, "TAU2_GATE_TURNS": 10, "TAU2_PREF_TURNS": 25,
                            "TAU2_V7_TURNS": 30, "TAU2_PREF_TEACHER_GATE": 0, "TAU2_PREF_JUDGE": 1,
                            "TAU2_DIVTHRESH": 0.1, "TAU2_DIVW": 1.0, "TAU2_SVC_TIMEOUT": 3600,
                            "TAU2_STUDENT_MODEL": "Qwen3-4B", "TAU2_USER_MODEL": "Qwen3-4B-Instruct-2507",
                            "TAU2_TEACHER_MODEL": "Qwen3-4B-Thinking-2507"}),
    (AGENT_QWEN3_4B, "synrl.train_batch_size", 8), (AGENT_QWEN3_4B, "synrl.ppo_mini_batch_size", 8), (AGENT_QWEN3_4B, "synrl.rollout_n", 16),
    (AGENT_QWEN3_4B, "synrl.lr", 1.0e-6), (AGENT_QWEN3_4B, "synrl.lr_scheduler_type", "constant"),
    (AGENT_QWEN3_4B, "synrl.kl_loss_coef", 0.001),
    (AGENT_QWEN3_4B, "synrl.entropy_coeff", 0.001), (AGENT_QWEN3_4B, "synrl.max_response_length", 1536),
    (AGENT_QWEN3_4B, "synrl.max_num_batched_tokens", 16384), (AGENT_QWEN3_4B, "synrl.gpu_memory_utilization", 0.5),
    (AGENT_QWEN3_4B, "synrl.offload", True), (AGENT_QWEN3_4B, "synrl.enable_resource_pool", False), (AGENT_QWEN3_4B, "synrl.total_steps", 60),
    (AGENT_QWEN3_4B, "synrl.save_freq", 5), (AGENT_QWEN3_4B, "synrl.shuffle", False), (AGENT_QWEN3_4B, "synrl.enable_thinking", False),
    (AGENT_QWEN3_4B, "synrl.diversity.group", 16), (AGENT_QWEN3_4B, "synrl.diversity.scope", "valid_only"),
    (AGENT_QWEN3_4B, "synrl.step_counter", "global_steps"),
    (AGENT_QWEN3_4B, "synrl.serve.node_maps", ["Qwen3-4B=2|Qwen3-4B-Instruct-2507=3|Qwen3-4B-Thinking-2507=3",
                                       "Qwen3-4B=4|Qwen3-4B-Instruct-2507=2|Qwen3-4B-Thinking-2507=2"]),
    (AGENT_QWEN3_4B, "synrl.versions.v4.max_prompt_length", 7168), (AGENT_QWEN3_4B, "synrl.versions.v4.svc_workers", 32),
    (AGENT_QWEN3_4B, "synrl.versions.v4.chosen_step", 40), (AGENT_QWEN3_4B, "synrl.versions.v5.max_prompt_length", 8192),
    (AGENT_QWEN3_4B, "synrl.versions.v5.svc_workers", 32), (AGENT_QWEN3_4B, "synrl.versions.v5.chosen_step", 40),
    (AGENT_QWEN3_4B, "synrl.versions.v5.init", f"{CKP}/synthesizer_{AGENT_QWEN3_4B}_v4/hf/step_40"),
    (AGENT_QWEN3_4B, "synrl.versions.v6.max_prompt_length", 8192), (AGENT_QWEN3_4B, "synrl.versions.v6.svc_workers", 64),
    (AGENT_QWEN3_4B, "synrl.versions.v6.chosen_step", 50), (AGENT_QWEN3_4B, "synrl.versions.v7.max_prompt_length", 10240),
    (AGENT_QWEN3_4B, "synrl.versions.v7.svc_workers", 64), (AGENT_QWEN3_4B, "synrl.versions.v7.chosen_step", 60),
    (AGENT_QWEN3_4B, "synrl.versions.v7.init", f"{CKP}/synthesizer_{AGENT_QWEN3_4B}_v6/hf/step_50"),
    (AGENT_QWEN3_4B, "select.rule", {"window": 4, "score": "valid_frac_x_mean_R_valid"}),
    (AGENT_QWEN3_4B, "select.served_names", {"v6": f"synthesizer_{AGENT_QWEN3_4B}_v6_step50",
                                             "v7": f"synthesizer_{AGENT_QWEN3_4B}_v7_step60"}),
    (AGENT_QWEN3_4B, "synrl.versions.v6.dump", f"{OUT}/{AGENT_QWEN3_4B}/synrl/dump_synthesizer_{AGENT_QWEN3_4B}_v6.jsonl"),
    (AGENT_QWEN3_4B, "harvest.v6.served_name", f"synthesizer_{AGENT_QWEN3_4B}_v6_step50"),
    (AGENT_QWEN3_4B, "harvest.v7.synthesizer", f"{CKP}/synthesizer_{AGENT_QWEN3_4B}_v7/hf/step_60"),
    (AGENT_QWEN3_4B, "harvest.v7.pool", f"{OUT}/{AGENT_QWEN3_4B}/harvest/pool_v7s60.jsonl"),
    (AGENT_QWEN3_4B, "harvest.v6.per_target", 222), (AGENT_QWEN3_4B, "harvest.v6.targets", 18),
    (AGENT_QWEN3_4B, "harvest.v6.seed0", 1500000), (AGENT_QWEN3_4B, "harvest.v7.per_target", 400),
    (AGENT_QWEN3_4B, "harvest.v7.targets", 4), (AGENT_QWEN3_4B, "harvest.v7.seed0", 2500000),
    (AGENT_QWEN3_4B, "harvest.v7.k", 2), (AGENT_QWEN3_4B, "harvest.v7.workers", 96),
    (AGENT_QWEN3_4B, "harvest.gen_sampling", {"temperature": 1.0, "max_tokens": 2048}),
    (AGENT_QWEN3_4B, "bank.mix", AGENT_QWEN3_4B), (AGENT_QWEN3_4B, "bank.out_dir", f"{OUT}/{AGENT_QWEN3_4B}/banks"),
    (AGENT_QWEN3_4B, "bank.parts.0.out", f"{AGENT_QWEN3_4B}_part-airline"),
    (AGENT_QWEN3_4B, "bank.parts.1.out", f"{AGENT_QWEN3_4B}_part-retail"),
    (AGENT_QWEN3_4B, "bank.released", f"{DAT}/banks/{AGENT_QWEN3_4B}.parquet"),
    (AGENT_QWEN3_4B, "opd.max_prompt_length", 6144), (AGENT_QWEN3_4B, "opd.max_response_length", 8192),
    (AGENT_QWEN3_4B, "opd.max_val_response_length", 8192), (AGENT_QWEN3_4B, "opd.mini_batch_size", 8), (AGENT_QWEN3_4B, "opd.n_responses", 4),
    (AGENT_QWEN3_4B, "opd.shuffle", True), (AGENT_QWEN3_4B, "opd.esi_save", True), (AGENT_QWEN3_4B, "opd.is_plot", True),
    (AGENT_QWEN3_4B, "opd.test_files", [f"{DAT}/eval/math/AIME25/test.parquet", f"{DAT}/eval/math/AMC23/test.parquet",
                                f"{DAT}/eval/math/AIME24/test.parquet"]),
    (AGENT_QWEN3_4B, "opd.forbidden_overrides", ["trainer.rollout_data_dir"]), (AGENT_QWEN3_4B, "opd.ckpt_keep", 3),
    (AGENT_QWEN3_4B, "opd.agent_env", {"TAU2_MAX_TURNS": 20, "TAU2_STOP_ON_REPEAT": 1, "TAU2_REPEAT_SIM": 0.9,
                               "TAU2_MAX_TOOL_CHARS": 2000, "TAU2_USER_RETRY_S": 1800}),
    (AGENT_QWEN3_4B, "opd.env_service", {"port": 8711, "workers": 64}),
    (AGENT_QWEN3_4B, "opd.arms.ours.experiment", f"opd_{AGENT_QWEN3_4B}_ours"), (AGENT_QWEN3_4B, "opd.arms.ours.total_steps", 200),
    (AGENT_QWEN3_4B, "opd.arms.ours.save_freq", 20), (AGENT_QWEN3_4B, "opd.arms.ours.reward_micro_bs", 8),
    (AGENT_QWEN3_4B, "opd.arms.ours.ppo_max_token_len_per_gpu", 16384), (AGENT_QWEN3_4B, "opd.arms.ours.total_epochs", 1000),
    (AGENT_QWEN3_4B, "opd.arms.vanilla.experiment", f"opd_{AGENT_QWEN3_4B}_vanilla"),
    (AGENT_QWEN3_4B, "opd.arms.ours.short_name", "ours"), (AGENT_QWEN3_4B, "opd.arms.vanilla.short_name", "vanilla"),
    (AGENT_QWEN3_4B, "opd.arms.vanilla.train", f"{DAT}/real/tau2-official-train.parquet"),
    (AGENT_QWEN3_4B, "opd.arms.vanilla.user_sim_serve.map", "Qwen3-4B-Instruct-2507=2"),
    (AGENT_QWEN3_4B, "opd.user_sim_serve.map", "Qwen3-4B-Instruct-2507=4"),
    (AGENT_QWEN3_4B, "eval.trials", 4), (AGENT_QWEN3_4B, "eval.user_llm", "gpt-4o-mini"), (AGENT_QWEN3_4B, "eval.max_steps", 200),
    (AGENT_QWEN3_4B, "eval.concurrency", 12), (AGENT_QWEN3_4B, "eval.TAU2_MAX_TOKENS", 8000),
    (AGENT_QWEN3_4B, "eval.TAU2_ORCHESTRATOR_PROMPT", 0),
    (AGENT_QWEN3_4B, "eval.TAU2_OPENAI_DIRECT", 1), (AGENT_QWEN3_4B, "eval.pass_threshold", 0.999), (AGENT_QWEN3_4B, "eval.watchdog_min", 30),
    (AGENT_QWEN3_4B, "eval.served_name", f"{AGENT_QWEN3_4B}-{{short_exp}}-s{{step}}"),
]


@pytest.fixture(scope="module")
def loaded():
    import os
    old = {k: os.environ.get(k) for k in ("A2D_DATA", "A2D_MODELS", "A2D_CKPT", "A2D_OUT", "A2D_CACHE")}
    for var, sub in (("A2D_DATA", "d"), ("A2D_MODELS", "m"), ("A2D_CKPT", "c"),
                     ("A2D_OUT", "o"), ("A2D_CACHE", "x")):
        os.environ[var] = f"/a2d/{sub}"
    try:
        yield {p: load(p) for p in PAIRS}
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.mark.parametrize("pair,key,value", EXPECTED, ids=[f"{p}:{k}" for p, k, _ in EXPECTED])
def test_expected_value(loaded, pair, key, value):
    got = OmegaConf.select(loaded[pair], key, throw_on_missing=True)
    if hasattr(got, "_metadata"):
        got = OmegaConf.to_container(got, resolve=True)
    assert got == value


def test_tau2_extra_overrides_in_original_order(loaded):
    assert [x.split("=")[0] for x in loaded[AGENT_QWEN3_4B].opd.extra_overrides] == [
        "actor_rollout_ref.rollout.mode", "actor_rollout_ref.rollout.agent.default_agent_loop",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path", "actor_rollout_ref.rollout.multi_turn.format",
        "trainer.total_training_steps", "+trainer.max_actor_ckpt_to_keep"]
    assert loaded[AGENT_QWEN3_4B].opd.extra_overrides[-2:] == ["trainer.total_training_steps=200",
                                                        "+trainer.max_actor_ckpt_to_keep=3"]


@pytest.mark.parametrize("pair", MATH_PAIRS + CODE_PAIRS)
def test_solve_budget_matches_opd(loaded, pair):
    cfg = loaded[pair]
    assert cfg.reward.max_new == cfg.opd.max_response_length == cfg.calib.max_new


# ------------------------------------------------------------------------------ guards

def test_require_raises_on_missing_and_deleted_keys():
    cfg = load(MATH_QWEN3_1P7B)
    C.require(cfg, "reward")
    OmegaConf.update(cfg, "reward.delta_ref", "???")
    with pytest.raises(C.ConfigError, match="reward.delta_ref"):
        C.require(cfg, "reward")
    cfg = load(MATH_QWEN3_1P7B)
    with open_dict(cfg):
        del cfg.reward["len_pen"]
    with pytest.raises(C.ConfigError, match=r"reward.len_pen \(deleted\)"):
        C.require(cfg, "reward")
    with pytest.raises(C.ConfigError, match="reward.len_pen"):
        C.validate(cfg)
    with pytest.raises(C.ConfigError, match="no section"):
        C.require(cfg, "nonexistent")


def test_base_value_left_missing_is_refused(tmp_path):
    f = tmp_path / "half.yaml"
    f.write_text(f"_base_: [{C.CONFIG_DIR}/base/opd.yaml, {C.CONFIG_DIR}/base/math.yaml, "
                 f"{C.CONFIG_DIR}/base/eval_math.yaml]\npair: half\n")
    with pytest.raises(C.ConfigError, match="reward.delta_ref"):
        C.load(None, f)


def test_resource_pool_guard_raises():
    with pytest.raises(C.ConfigError, match="enable_resource_pool"):
        load(MATH_QWEN3_1P7B, "synrl.layout.enable_resource_pool=false")


def test_diversity_layout_guards():
    # bleu clusters the gathered full batch: any even split over the reward ranks is fine
    assert C.validate(load(CODE_QWEN3_30B, "synrl.layout.reward_gpus=4")) == []
    with pytest.raises(C.ConfigError, match="split"):
        load(CODE_QWEN3_30B, "synrl.layout.reward_gpus=6")
    # emb clusters groups of the rank batch
    cfg = load(CODE_QWEN3_30B, "reward.diversity.mode=emb", "synrl.layout.reward_gpus=4")
    assert any(C.EMB_GROUP in w for w in C.validate(cfg))
    assert C.validate(load(CODE_QWEN3_30B, "reward.diversity.mode=emb")) == []
    with pytest.raises(C.ConfigError, match="diversity.mode"):
        load(CODE_QWEN3_30B, "reward.diversity.mode=cosine")
    cfg = load(AGENT_QWEN3_4B, "synrl.diversity.group=8")
    assert any(C.NON_FAITHFUL in w for w in C.validate(cfg))


def test_smoke_only_keys_rejected_in_pair_configs():
    with pytest.raises(C.ConfigError, match="allow_empty"):
        load(MATH_QWEN3_1P7B, "select.pick.allow_empty=true")
    with pytest.raises(C.ConfigError, match="expected_rows"):
        load(MATH_R1_1P5B, "bank.allow_short=true")
    load(MATH_R1_1P5B, "bank.allow_short=true", "bank.expected_rows=100", "smoke=true")


def test_forbidden_override_rejected():
    with pytest.raises(C.ConfigError, match="forbidden"):
        load(AGENT_QWEN3_4B, "opd.extra_overrides=[trainer.rollout_data_dir=/tmp/x]")


def test_other_guards():
    with pytest.raises(C.ConfigError, match="4 decimals"):
        load(MATH_QWEN3_1P7B, "reward.delta_ref=1.90494")
    with pytest.raises(C.ConfigError, match="agg_order"):
        load(CODE_QWEN3_4B, "filter.agg_order=[0,1]")
    with pytest.raises(C.ConfigError, match="seed_formula"):
        load(CODE_CODER_32B, "harvest.seed_formula=7919*i")
    with pytest.raises(C.ConfigError, match="qwen"):
        load(AGENT_QWEN3_4B, "eval.served_name=agent-{step}")
    assert any("opd.train_max_samples" in w
               for w in C.validate(load(MATH_QWEN3_1P7B, "opd.train_max_samples=64")))


def test_overrides_strict_keys_and_add():
    with pytest.raises(C.ConfigError, match="no key"):
        load(MATH_QWEN3_1P7B, "reward.no_such_knob=1")
    cfg = load(MATH_QWEN3_1P7B, "+reward.extra_knob=1", "synrl.total_steps=3")
    assert cfg.reward.extra_knob == 1 and cfg.synrl.total_steps == 3
    with pytest.raises(C.ConfigError, match="key=value"):
        load(MATH_QWEN3_1P7B, "synrl.total_steps")
    with pytest.raises(ConfigAttributeError):
        _ = cfg.reward.not_there            # struct mode: unknown keys raise


def test_unknown_pair_and_conflict():
    with pytest.raises(C.ConfigError, match="unknown pair"):
        load("q4")
    with pytest.raises(C.ConfigError, match="need --pair"):
        C.load()
    with pytest.raises(C.ConfigError, match="conflicts"):
        C.load(MATH_R1_1P5B, "configs/smoke/math_smoke.yaml")


def test_interpolation_follows_overrides():
    cfg = load(MATH_QWEN3_1P7B, "synrl.save_freq=10", "synrl.experiment=synrl_x")
    assert cfg.select.pick.save_freq == 10
    assert cfg.reward.dump.endswith(f"/{MATH_QWEN3_1P7B}/synrl/synrl_dump_synrl_x.jsonl")


def test_save_resolved_and_section_json(tmp_path):
    cfg = load(CODE_CODER_32B)
    out = C.save_resolved(cfg, tmp_path / "run")
    again = OmegaConf.load(out)
    assert again.reward.delta_ref == 1.3629 and "${" not in out.read_text()
    j = C.section_json(cfg, "reward", tmp_path / "reward_config.json")
    data = json.loads(j.read_text())
    assert data["delta_ref"] == 1.3629 and data["solver"] == "/a2d/m/Qwen2.5-Coder-3B-Instruct"
    assert data["diversity"]["group"] == 8


def test_resolved_file_reloads_identically(tmp_path):
    cfg = load(AGENT_QWEN3_4B)
    out = C.save_resolved(cfg, tmp_path)
    again = C.load(None, out)
    assert OmegaConf.to_container(again, resolve=True) == OmegaConf.to_container(cfg, resolve=True)


# ------------------------------------------------- against the release and the models

@pytest.mark.parametrize("pair", sorted(set(MATH_PAIRS) | set(CODE_PAIRS)))
def test_delta_ref_is_the_released_calibration_record(loaded, pair):
    rec = _PATHS.data / "calib" / "delta_ref" / f"{pair}.json"
    if not rec.is_file():
        pytest.skip(f"{rec} absent (a2d data download)")
    d = json.loads(rec.read_text())
    assert d["percentile"] == loaded[pair].calib.percentile
    assert loaded[pair].reward.delta_ref == d["delta_ref_rounded"] == round(d["delta_ref"], 4) \
        == loaded[pair].calib.expected_delta_ref


@pytest.mark.parametrize("pair", PAIRS)
def test_hf_revisions_match_downloaded_models(loaded, pair):
    cfg = loaded[pair]
    for role, spec in cfg.models.hf.items():
        local = Path(str(cfg.models[role])).name
        meta = sorted((_PATHS.models / local / ".cache" / "huggingface" / "download").glob("*.metadata"))
        if not meta:
            pytest.skip(f"{local}: no HF download metadata under {_PATHS.models}")
        assert meta[0].read_text().splitlines()[0].strip() == spec.revision, (pair, role)
