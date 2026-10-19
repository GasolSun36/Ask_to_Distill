"""Sampling a question synthesizer under the conditioning it was trained under.

Shared by the checkpoint sweep (``select.sweep``), the harvest
(``harvest.harvest``) and the math yield comparison. G was trained on
topic-conditioned prompts, so it is always sampled with exactly that
conditioning -- the same topic file (``synrl.topics``), the same seeded topic
rotation (``common.topics.rotation``) and the same system/user pair
(``prompts_math`` / ``prompts_code.synthesizer_messages``) -- or it is queried on
a distribution it was never optimised under.

Prompts are rendered with ``enable_thinking=False`` and a leading BOS string is
removed (vLLM adds its own). vLLM samples one completion per prompt with
``top_p = 0.95`` (GRPO rollouts used 1.0; the difference is preserved), all
prompts in a single ``generate`` call. The engine seed stays at vLLM's default
0, so the per-shard seed only moves the topic assignment -- which is why it must
differ between shards.

What this module does NOT do is clean the output: the sweep keeps the raw
``.strip()``-ed text while the harvest strips thinking blocks, and each writes
its own records.
"""
from __future__ import annotations

import gc
import os
import random
from typing import Any, Sequence

from ask2distill.common import prompts_code, prompts_math

DOMAINS = ("math", "code")


def export_stage_env(stage: str) -> None:
    """Export ``runtime.base_env(stage)`` into this process (vLLM / HF read it at import)."""
    from ask2distill import runtime
    for k, v in runtime.base_env(stage=stage).items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def load_topics(cfg: Any) -> list[str]:
    """The synthesizer's topic list, ``synrl.topics`` of ``cfg``."""
    from ask2distill.common.topics import load_topics as _load
    return _load(str(cfg.synrl.topics))


def synthesizer_messages(domain: str, topic: str) -> list[dict]:
    """The synthesizer's chat messages for one ``topic`` of ``domain``."""
    if domain == "code":
        return prompts_code.synthesizer_messages(topic)
    if domain == "math":
        return prompts_math.synthesizer_messages(topic)
    raise ValueError(f"domain must be one of {DOMAINS}, got {domain!r}")


def draw_topics(n: int, rng: random.Random, topics: Sequence[str]) -> list[str]:
    """``n`` topics of the seeded rotation (``common.topics.rotation``)."""
    from ask2distill.common.topics import rotation
    return rotation(topics, n, rng)


def draw_messages(n: int, rng: random.Random, domain: str, topics: Sequence[str]) -> list[list[dict]]:
    """``n`` chat-message lists, one rotated topic each."""
    return [synthesizer_messages(domain, t) for t in draw_topics(n, rng, topics)]


def render(tok: Any, messages: list[dict]) -> str:
    """Chat-template one prompt with thinking disabled and the BOS string removed."""
    s = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    b = tok.bos_token or ""
    return s[len(b):] if b and s.startswith(b) else s


def build_prompts(tok: Any, n: int, rng: random.Random, domain: str,
                  topics: Sequence[str]) -> tuple[list[str], list[str]]:
    """``(prompts, topics)``: rendered prompts for ``n`` synthesizer samples and the topic of each."""
    drawn = draw_topics(n, rng, topics)
    return [render(tok, synthesizer_messages(domain, t)) for t in drawn], drawn


def generate(model: str | os.PathLike, prompts: list[str], *, max_new: int, prompt_head: int,
             temperature: float, top_p: float, gpu_memory_utilization: float,
             seed: int | None = None) -> list[str]:
    """One vLLM engine, one ``generate`` call; the first completion's text per prompt.

    ``max_model_len = max_new + prompt_head``; the topic prompts are a few
    hundred tokens, well inside the configured headroom (1536 math, 2304 code).
    ``seed=None`` leaves vLLM's engine seed at its default (0 under the V1
    engine).
    """
    import torch
    from vllm import LLM, SamplingParams

    kw = {} if seed is None else {"seed": seed}
    llm = LLM(model=str(model), max_model_len=max_new + prompt_head,
              gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=True, **kw)
    outs = llm.generate(prompts, SamplingParams(n=1, temperature=temperature, top_p=top_p,
                                                max_tokens=max_new))
    texts = [o.outputs[0].text for o in outs]
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return texts


def load_tokenizer(model: str | os.PathLike) -> Any:
    """The synthesizer's own tokenizer (it carries the chat template the synthesizer was trained with)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
