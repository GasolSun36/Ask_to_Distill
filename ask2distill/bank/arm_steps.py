"""How many optimiser steps an OPD arm actually has.

``rows // batch`` overcounts: verl drops every prompt longer than
``max_prompt_length`` (``data.filter_overlong_prompts=True``) before batching,
and the tail that does not fill a batch is dropped too. The bank of
Code_Qwen2.5-Coder-3B-Instruct_Qwen2.5-Coder-32B-Instruct has 10000 rows but only 9981 fit in 2048 tokens, so its last step is 155 -- and a
resubmission loop told to wait for 156 resubmits the finished arm forever.

The count renders each prompt with the student's chat template and
``add_generation_prompt=True`` and no template arguments, exactly as the
step counter used for the final runs did (it reported 155 for that bank and 156 for the bank of
Code_Qwen3-4B_Qwen3-4B-Instruct-2507). verl itself also passes ``apply_chat_template_kwargs``
(``enable_thinking=False`` for Qwen3 arms, a few tokens longer); pass
``chat_template_kwargs`` to count that way.

The answer is cached next to the parquet, keyed by the arguments and the
parquet's mtime, because it costs a full tokenisation pass.

    python -m ask2distill.bank.arm_steps <parquet> <tokenizer> [max_prompt_len] [batch]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Mapping


def count_steps(parquet: str | os.PathLike, tokenizer: str | os.PathLike, max_prompt_len: int,
                batch: int, chat_template_kwargs: Mapping | None = None) -> dict:
    """``{"rows", "kept", "steps"}`` for ``parquet`` under ``max_prompt_len`` / ``batch``."""
    pq, tokpath = os.fspath(parquet), os.fspath(tokenizer)
    extra = "" if not chat_template_kwargs else "|" + json.dumps(dict(chat_template_kwargs),
                                                                    sort_keys=True)
    key = hashlib.md5(f"{pq}|{tokpath}|{max_prompt_len}|{batch}|{os.path.getmtime(pq)}{extra}"
                      .encode()).hexdigest()[:16]
    cache = os.path.join(os.path.dirname(pq), f".arm_steps_{key}.json")
    if os.path.exists(cache):
        with open(cache) as fh:
            return json.load(fh)
    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokpath, trust_remote_code=True)
    df = pd.read_parquet(pq)
    kw = dict(chat_template_kwargs or {})
    kept = 0
    for p in df["prompt"]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in p]
        if len(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       **kw)) <= max_prompt_len:
            kept += 1
    record = {"rows": len(df), "kept": kept, "steps": kept // batch}
    try:
        with open(cache, "w") as fh:
            json.dump(record, fh)
    except OSError:
        pass
    return record


def main(argv: list[str] | None = None) -> int:
    """Print the step count (command-line form of the original helper)."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print(__doc__.split("\n\n")[-2], file=sys.stderr)
        return 2
    cap = int(argv[2]) if len(argv) > 2 else 2048
    bs = int(argv[3]) if len(argv) > 3 else 64
    print(count_steps(argv[0], argv[1], cap, bs)["steps"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
