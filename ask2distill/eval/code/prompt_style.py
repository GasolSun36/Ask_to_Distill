"""Pick the LiveCodeBench prompt style from a model's own chat template.

Only hybrid Qwen3 (Qwen3-4B/8B/..., the ones with the /think /no_think switch)
render an empty ``<think></think>`` block for non-thinking mode, and
LiveCodeBench's ``Qwen3NoThink`` style reproduces that. Single-mode models --
Qwen3-*-Instruct-2507, Qwen2.5-Coder, ... -- must NOT get that block: they were
never trained with it. Deciding this from the tokenizer rather than from the
directory name keeps the launcher honest for new checkpoints. Getting it wrong
is silent: ``Qwen3NoThink`` on a single-mode model injects a block it has never
seen, and ``CodeQwenInstruct`` on a hybrid Qwen3 with thinking off lets it start
reasoning inside a budget sized for plain code.

``detect`` also returns the context length and the checkpoint footprint, which
the code eval uses to cap ``max_model_len`` and to back off
``gpu_memory_utilization`` for big models.

Command line (the original's interface): prints
``<LMStyle> <max_position_embeddings> <is_hybrid> <weight_bytes>``::

    python -m ask2distill.eval.code.prompt_style <model_dir> <think 0|1>
"""
from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class StyleInfo:
    style: str            # LMStyle name: Qwen3NoThink | CodeQwenInstruct | AutoTemplate
    ctx: int              # config.json max_position_embeddings
    hybrid: bool          # template renders differently with enable_thinking on/off
    weight_bytes: int     # total size of the safetensors weights

    def line(self) -> str:
        return f"{self.style} {self.ctx} {int(self.hybrid)} {self.weight_bytes}"


def weight_bytes(model_path: str) -> int:
    """Checkpoint footprint: the safetensors index total, else the sum of the shard sizes."""
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            return json.load(f)["metadata"]["total_size"]
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(model_path, "*.safetensors")))


def detect(model_path: str, think: bool) -> StyleInfo:
    """Style, context length, hybrid flag and weight bytes of ``model_path``."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]

    def render(enable_thinking: bool) -> str:
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=enable_thinking)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    hybrid = render(True) != render(False)
    rendered = render(think)
    # LiveCodeBench's Qwen styles hand-write ChatML. A model from another family
    # (Llama's <|start_header_id|> turns, say) must not be wrapped in it, so anything
    # non-ChatML renders through its own template instead.
    if not rendered.startswith("<|im_start|>system"):
        style = "AutoTemplate"
    elif rendered.rstrip().endswith("</think>"):
        style = "Qwen3NoThink"
    else:
        style = "CodeQwenInstruct"
    with open(os.path.join(model_path, "config.json")) as f:
        ctx = json.load(f)["max_position_embeddings"]
    return StyleInfo(style, int(ctx), hybrid, int(weight_bytes(model_path)))


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python -m ask2distill.eval.code.prompt_style <model_dir> <think 0|1>",
              file=sys.stderr)
        return 2
    print(detect(argv[0], argv[1] == "1").line())
    return 0


if __name__ == "__main__":
    sys.exit(main())
