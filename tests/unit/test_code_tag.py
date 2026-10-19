"""Code eval: unique tags, resolved settings, command lines, prompt style, lenient MBPP.

The tag is always ``<experiment>_step<N>``: LCB's ``--continue_existing`` keys
cached generations on it, so a basename-style tag (``step_150-nothink``) would
silently score another run's generations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ask2distill.eval.code import mbpp_lenient as ML
from ask2distill.eval.code import run_code_eval as RC
from ask2distill.eval.code.prompt_style import StyleInfo, detect

from conftest import CODE_CODER_32B, CODE_PAIRS, CODE_QWEN3_30B, CODE_QWEN3_4B

EXP_OURS = f"opd_{CODE_QWEN3_30B}_ours"
GIB = 1 << 30


# ------------------------------------------------------------------------------ tags

@pytest.mark.parametrize("exp,step", [(EXP_OURS, 150), (f"opd_{CODE_CODER_32B}_vanilla", 110),
                                      ("opd_DeepSeek-R1-Distill-Qwen-1.5B_x_fp32", 5)])
def test_tag_is_experiment_step(exp, step):
    assert RC.render_tag("{experiment}_step{step}", exp, step) == f"{exp}_step{step}"


@pytest.mark.parametrize("tag", ["step_150-nothink", "step_150", "Qwen3-4B-nothink", "Qwen3-4B",
                                 f"{EXP_OURS}_step150-nothink", "a/b_step3", "_step3", ""])
def test_basename_style_tags_refused(tag):
    with pytest.raises(RC.TagError):
        RC.validate_tag(tag)


def test_tag_must_name_this_checkpoint():
    with pytest.raises(RC.TagError):
        RC.validate_tag(f"{EXP_OURS}_step130", EXP_OURS, 150)
    with pytest.raises(RC.TagError):
        RC.validate_tag(f"opd_{CODE_QWEN3_30B}_vanilla_step150", EXP_OURS, 150)
    with pytest.raises(RC.TagError):
        RC.render_tag("step_{step}-nothink", EXP_OURS, 150)


@pytest.fixture(scope="module", params=CODE_PAIRS)
def code_cfg(request):
    from ask2distill.config import load
    return load(request.param)


def test_config_tags_render(code_cfg):
    for arm in ("ours", "vanilla"):
        exp = RC.experiment_of(code_cfg, arm)
        assert RC.render_tag(str(code_cfg.eval.tag), exp, 130) == f"{exp}_step130"


# -------------------------------------------------------------------------- planning

@pytest.fixture(scope="module")
def cfg_a():
    from ask2distill.config import load
    return load(CODE_QWEN3_30B)


def test_small_model_plan(cfg_a):
    info = StyleInfo("Qwen3NoThink", 40960, True, 8 * GIB)          # 4B fp32-merged in bf16
    p = RC.make_plan(cfg_a, Path("/m"), f"{EXP_OURS}_step150", info)
    assert (p.lcb_util, p.lcb_eager, p.ep_util, p.ep_eager, p.xb_util) == (0.9, False, 0.9, False, 0.9)
    assert (p.lcb_max_model_len, p.lcb_max_tokens) == (16384, 8192)
    assert p.ep_datasets == ("mbpp", "humaneval") and p.run_xb and not p.debug


def test_big_model_rule(cfg_a):
    # 32B bf16 ~ 61 GiB over TP 2 -> 30 GiB per GPU > 10: util 0.85, eager
    info = StyleInfo("CodeQwenInstruct", 32768, False, 61 * GIB)
    p = RC.make_plan(cfg_a, Path("/m"), "x_step1", info)
    assert (p.lcb_util, p.lcb_eager, p.ep_util, p.ep_eager, p.xb_util) == (0.85, True, 0.85, True, 0.85)
    # exactly at the threshold (21 GiB / 2 = 10, integer division) stays on the default
    p = RC.make_plan(cfg_a, Path("/m"), "x_step1", StyleInfo("Qwen3NoThink", 40960, True, 21 * GIB + 5))
    assert (p.lcb_util, p.lcb_eager) == (0.9, False)


def test_lcb_lengths_capped_by_context(cfg_a):
    assert RC.lcb_lengths(cfg_a, 262144) == (16384, 8192)
    assert RC.lcb_lengths(cfg_a, 32768) == (16384, 8192)
    assert RC.lcb_lengths(cfg_a, 8192) == (8192, 4096)


def test_curve_and_smoke_plans(cfg_a):
    info = StyleInfo("Qwen3NoThink", 40960, True, 8 * GIB)
    curve = RC.make_plan(cfg_a, Path("/m"), "x_step1", info, curve=True)
    assert curve.ep_datasets == ("mbpp",) and not curve.run_xb
    smoke = RC.make_plan(cfg_a, Path("/m"), "x_step1", info, smoke=True)
    assert smoke.ep_datasets == ("humaneval",) and smoke.debug


def test_lcb_command_matches_eval_code_sh(cfg_a):
    info = StyleInfo("Qwen3NoThink", 40960, True, 8 * GIB)
    p = RC.make_plan(cfg_a, Path("/ckpt/hf/step_150"), f"{EXP_OURS}_step150", info)
    cmd = RC.lcb_cmd(cfg_a, p)
    assert cmd[:3] == [sys.executable, "-m", "lcb_runner.runner.main"]
    # the final LCB command: non-thinking, TP 2, 4B weights (CUDA graphs on)
    assert cmd[3:] == [
        "--model", f"{EXP_OURS}_step150", "--model_style", "Qwen3NoThink",
        "--local_model_path", "/ckpt/hf/step_150", "--scenario", "codegeneration",
        "--release_version", "v5_v6", "--n", "4", "--temperature", "0.7", "--top_p", "0.8",
        "--top_k", "20", "--min_p", "0", "--max_tokens", "8192", "--max_model_len", "16384",
        "--tensor_parallel_size", "2", "--gpu_memory_utilization", "0.9", "--cuda_graph",
        "--stop", "<|im_end|>", "--num_process_evaluate", "24", "--timeout", "6",
        "--continue_existing", "--evaluate"]
    big = RC.make_plan(cfg_a, Path("/m"), "x_step1", StyleInfo("CodeQwenInstruct", 32768, False, 61 * GIB),
                       smoke=True)
    big_cmd = RC.lcb_cmd(cfg_a, big)
    assert "--cuda_graph" not in big_cmd and big_cmd[-1] == "--debug"
    assert RC.lcb_result(cfg_a, "t_step1").name == "Scenario.codegeneration_4_0.7_eval.json"


def test_evalplus_and_xb_commands(cfg_a):
    info = StyleInfo("Qwen3NoThink", 40960, True, 8 * GIB)
    p = RC.make_plan(cfg_a, Path("/m"), "t_step1", info)
    ep = RC.evalplus_cmd(cfg_a, p, "mbpp")
    assert ep[1:] == ["--model", "/m", "--dataset", "mbpp", "--backend", "vllm", "--tp", "2",
                      "--root", str(p.out / "evalplus"), "--parallel", "24",
                      "--gpu-memory-utilization", "0.9", "--enforce-eager", "False", "--greedy",
                      "--max-new-tokens", "1024", "--max-model-len", "4096"]
    xb = RC.xb_cmd(cfg_a, p, "codecontests", None)
    a = __import__("ask2distill.eval.code.extra_bench", fromlist=["x"]).build_parser().parse_args(xb[3:])
    assert (a.n, a.temperature, a.top_p, a.top_k, a.max_new, a.max_model_len_extra, a.tp, a.util,
            a.system_prompt, a.nothink, a.extraction, a.limit, a.detail) == \
        (1, 0.2, 0.95, 20, 4096, 4096, 2, 0.9, "You are a helpful assistant.", True,
         "CodeQwenInstruct", 0, False)
    assert a.bench_file.endswith("extra_bench/codecontests.jsonl")
    assert a.out == str(p.out / "xb" / "codecontests.json")


def test_lenient_source_selection(tmp_path):
    for n in ["m_vllm_temp_0.0.raw.jsonl", "lenient.jsonl", "m_vllm_temp_0.0.jsonl",
              "m_vllm_temp_0.0.eval_results.json"]:
        (tmp_path / n).write_text("")
    assert RC.lenient_source(tmp_path).name == "m_vllm_temp_0.0.jsonl"
    assert RC.lenient_source(tmp_path / "missing") is None


# ----------------------------------------------------------------------- prompt style

STYLE_MODELS = {
    "Qwen3-4B": "Qwen3NoThink",                    # Code_Qwen3-4B_* students (hybrid)
    "Qwen2.5-Coder-3B-Instruct": "CodeQwenInstruct",   # Code_Qwen2.5-Coder-3B-Instruct_* student
    "Qwen3-4B-Instruct-2507": "CodeQwenInstruct",      # single-mode: no empty think block
}


@pytest.mark.parametrize("model,style", sorted(STYLE_MODELS.items()))
def test_prompt_style(models_dir, model, style):
    path = models_dir / model
    if not (path / "tokenizer_config.json").is_file():
        pytest.skip(f"{model} not downloaded")
    info = detect(str(path), think=False)
    assert info.style == style
    assert info.hybrid == (model == "Qwen3-4B")
    assert info.weight_bytes > 0


def test_pair_configs_name_the_detected_style(models_dir):
    from ask2distill.config import load
    students = {CODE_QWEN3_30B: "Qwen3-4B", CODE_CODER_32B: "Qwen2.5-Coder-3B-Instruct", CODE_QWEN3_4B: "Qwen3-4B"}
    for pair, model in students.items():
        path = models_dir / model
        if not (path / "tokenizer_config.json").is_file():
            pytest.skip(f"{model} not downloaded")
        assert detect(str(path), think=False).style == str(load(pair).eval.prompt_style), pair


# ----------------------------------------------------------------------- lenient MBPP

LENIENT_CASES = [
    ("def to_uppercase(s):\n    return s.upper()\n", "is_upper"),
    ("def is_upper(s):\n    return s.upper()\n", "is_upper"),
    ("def helper(x):\n    return x\n\ndef f(s):\n    return helper(s)\n", "g"),
    ("def fact(n):\n    return 1 if n < 2 else n * fact(n - 1)\n", "factorial"),
    ("def f(x):\n    return f_inner(x)\n", "g"),
    ("not python (", "g"),
    ("import math\ndef area(r):\n    return math.pi * r * r\n", "circle_area"),
    ("def a(x):\n    return x\n", None),
    ("class K:\n    def m(self):\n        pass\n", "k"),
    ("def go(x):\n    go_on = 1\n    return go(x) + go_on\n", "run"),
]


def test_lenient_repair_cases():
    assert ML.repair(*LENIENT_CASES[0]) == ("def is_upper(s):\n    return s.upper()\n", True)
    assert ML.repair(*LENIENT_CASES[1])[1] is False
    assert ML.repair(*LENIENT_CASES[2])[1] is False                     # two functions: ambiguous
    assert ML.repair(*LENIENT_CASES[3]) == (
        "def factorial(n):\n    return 1 if n < 2 else n * factorial(n - 1)\n", True)
    assert ML.repair(*LENIENT_CASES[9])[0] == "def run(x):\n    go_on = 1\n    return run(x) + go_on\n"


def test_lenient_rescore_file(tmp_path):
    src = tmp_path / "s.jsonl"
    src.write_text("".join(json.dumps({"task_id": f"Mbpp/{i}", "solution": s}) + "\n"
                           for i, (s, _) in enumerate(LENIENT_CASES)))
    entries = {f"Mbpp/{i}": e for i, (_, e) in enumerate(LENIENT_CASES) if e}
    n, fixed = ML.rescore(str(src), str(tmp_path / "lenient.jsonl"), entries)
    assert (n, fixed) == (len(LENIENT_CASES), sum(ML.repair(s, e)[1] for s, e in LENIENT_CASES))
