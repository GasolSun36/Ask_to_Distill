"""Math eval generation: stop ids per tokenizer, file naming, call structure.

Stop ids are the first token of ``encode("<|im_end|>")`` and
``encode("<|endoftext|>")``, as in the evaluation the reported numbers came
from. Under the R1-Distill tokenizer neither string is a token, so both start
with ``<`` (27): a quirk the numbers of the two DeepSeek-R1-Distill math pairs
depend on, pinned here. Tokenizers are read from ``$A2D_MODELS``.

The generation loop is checked against a fake vLLM: one ``generate`` call per
rollout index over all problems, no per-request seed, rows in rollout-major
order, file written once and skipped when present.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from ask2distill.common.prompts_math import eval_prompt
from ask2distill.eval.math import generate as GEN
from ask2distill.eval.math.grade import parse_hyperparameters_from_filename, task_name_of

QWEN_STOPS = [151645, 151643]
R1_STOPS = [27, 27]


@pytest.mark.parametrize("model,expected", [
    ("Qwen3-1.7B", QWEN_STOPS),                              # Math_Qwen3-1.7B_* student
    ("Qwen3-4B-Non-Thinking-RL-Math-Step500", QWEN_STOPS),   # Math_Qwen3-1.7B_* teacher
    ("DeepSeek-R1-Distill-Qwen-1.5B", R1_STOPS),             # Math_DeepSeek-R1-Distill-Qwen-1.5B_* student
    ("DeepSeek-R1-Distill-Qwen-7B", R1_STOPS),               # Math_DeepSeek-R1-Distill-Qwen-7B_* student
])
def test_stop_ids_per_tokenizer(models_dir, model, expected):
    transformers = pytest.importorskip("transformers")
    path = models_dir / model
    if not (path / "tokenizer_config.json").is_file():
        pytest.skip(f"{model} not downloaded")
    tok = transformers.AutoTokenizer.from_pretrained(str(path))
    assert GEN.stop_token_ids(tok) == expected


def test_stop_ids_tolerate_broken_tokenizers():
    class Boom:
        def encode(self, *a, **k):
            raise ValueError("no")

    class Empty:
        def encode(self, *a, **k):
            return []

    assert GEN.stop_token_ids(Boom()) == []
    assert GEN.stop_token_ids(Empty()) == []
    assert GEN.stop_token_ids(object()) == []


def test_output_name_roundtrip():
    name = GEN.output_name("HMMT_FEB25", 0.7, 0.95, 16, 31744)
    assert name == "hmmt_feb25_t0.7_p0.95_n16-MNT31744.jsonl"
    assert GEN.output_name("MATH500", 0.7, 0.95, 4, 31744) == "math500_t0.7_p0.95_n4-MNT31744.jsonl"
    assert parse_hyperparameters_from_filename(name) == {
        "temperature": "0.7", "top_p": "0.95", "n": "16", "max_tokens": "31744"}
    from pathlib import Path
    assert task_name_of(Path(name)) == "hmmt_feb25"


def test_format_prompt_falls_back_without_enable_thinking():
    class NoKw:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kw):
            if kw:
                raise TypeError("unexpected kwarg")
            return "plain:" + msgs[0]["content"]

    class Kw:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
            return f"think={enable_thinking}:" + msgs[0]["content"]

    assert GEN.format_prompt(NoKw(), "q", False) == "plain:q"
    assert GEN.format_prompt(Kw(), "q", False) == "think=False:q"


# ------------------------------------------------------------------ fake vLLM engine

class _Out:
    def __init__(self, text):
        self.outputs = [types.SimpleNamespace(text=text)]


class _FakeTok:
    def encode(self, s, add_special_tokens=False):
        return [7, 8] if s == "<|im_end|>" else [9]

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
        return f"<u>{msgs[0]['content']}</u>"


class _FakeLLM:
    instances: list = []

    def __init__(self, **kw):
        self.kw = kw
        self.calls = []
        _FakeLLM.instances.append(self)

    def get_tokenizer(self):
        return _FakeTok()

    def generate(self, prompts, sampling, use_tqdm=False):
        self.calls.append((list(prompts), sampling))
        k = len(self.calls) - 1
        return [_Out(f"r{k}-{i}") for i in range(len(prompts))]


class _FakeSampling:
    def __init__(self, **kw):
        self.kw = kw


@pytest.fixture
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.LLM = _FakeLLM
    mod.SamplingParams = _FakeSampling
    monkeypatch.setitem(sys.modules, "vllm", mod)
    _FakeLLM.instances.clear()
    return _FakeLLM


def test_run_task_call_structure(tmp_path, fake_vllm, monkeypatch):
    pd = pytest.importorskip("pandas")
    data = tmp_path / "test.parquet"
    pd.DataFrame({
        "prompt": [[{"role": "user", "content": f" problem {i} "}] for i in range(3)],
        "reward_model": [{"ground_truth": f" {i} ", "style": "rule"} for i in range(3)],
    }).to_parquet(data)
    kw = dict(model="M", task="AIME24", data=data, n=4, out_dir=tmp_path / "out", temperature=0.7,
              top_p=0.95, max_tokens=31744, gpu_memory_utilization=0.9, enforce_eager=True,
              enable_thinking=False)
    out = GEN.run_task(**kw)
    assert out.name == "aime24_t0.7_p0.95_n4-MNT31744.jsonl"
    (llm,) = fake_vllm.instances
    assert llm.kw == {"model": "M", "trust_remote_code": True, "gpu_memory_utilization": 0.9,
                      "tensor_parallel_size": 1, "enforce_eager": True}           # no engine seed
    assert len(llm.calls) == 4                                                  # one call per rollout
    for prompts, sampling in llm.calls:
        assert prompts == [f"<u>{eval_prompt(f'problem {i}')}</u>" for i in range(3)]
        assert sampling.kw == {"temperature": 0.7, "top_p": 0.95, "max_tokens": 31744,
                               "stop_token_ids": [7, 9]}                        # no seed
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["seed"], r["example_id"]) for r in rows] == [(k, i) for k in range(4) for i in range(3)]
    assert rows[0] == {"example_id": 0, "prompt": eval_prompt("problem 0"), "answer": "0",
                       "seed": 0, "response": "r0-0"}
    # an existing file is a finished task
    GEN.run_task(**kw)
    assert len(fake_vllm.instances) == 1
    # engine seed and limit
    GEN.run_task(**{**kw, "out_dir": tmp_path / "o2", "engine_seed": 3, "limit": 2})
    assert fake_vllm.instances[-1].kw["seed"] == 3
    assert len(fake_vllm.instances[-1].calls[0][0]) == 2


def test_crash_writes_nothing(tmp_path, fake_vllm, monkeypatch):
    pd = pytest.importorskip("pandas")
    data = tmp_path / "test.parquet"
    pd.DataFrame({"problem": ["p"], "answer": ["1"]}).to_parquet(data)

    def boom(self, *a, **k):
        if self.calls:
            raise RuntimeError("cuda died")
        self.calls.append(a)
        return [_Out("x")]

    monkeypatch.setattr(_FakeLLM, "generate", boom)
    rc = GEN.main(["--model", "M", "--task", "AMC23", "--data", str(data), "--n", "2",
                   "--out-dir", str(tmp_path / "o"), "--max-tokens", "8", "--temperature", "0.7",
                   "--top-p", "0.95", "--gpu-memory-utilization", "0.9", "--enforce-eager",
                   "--disable-thinking"])
    assert rc == 1
    assert not any((tmp_path / "o").glob("*.jsonl"))
