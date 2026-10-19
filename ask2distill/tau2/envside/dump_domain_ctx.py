"""Dump each domain's agent system prompt and tool schemas to JSON (``domain_ctx.json``).

The reward, the prompt builders and the OPD agent loop all need the system
prompt and the tool schemas of each domain. They run in the a2d env, which has
torch but not tau2's dependency chain (loguru, litellm, ...); handing the
context over as data keeps the two environments from having to merge.

    python -m ask2distill.tau2.envside.dump_domain_ctx --out data/tau2/domain_ctx.json

The output path is an argument and REPO_PATH is required.
"""
from __future__ import annotations

import argparse
import json
import os

from ask2distill.tau2.envside import repo

DOMAINS = ("retail", "telecom", "airline")


def dump(out: str) -> dict:
    """Write ``{domain: {"system": str, "tools": [openai schema]}}`` to ``out``."""
    repo.setup()
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT
    from tau2.registry import registry
    ctx = {}
    for dom in DOMAINS:
        env = registry.get_env_constructor(dom)()
        ctx[dom] = {
            "system": SYSTEM_PROMPT.format(agent_instruction=AGENT_INSTRUCTION,
                                           domain_policy=env.get_policy()),
            "tools": [t.openai_schema for t in env.get_tools()],
        }
        print(f"{dom}: policy {len(ctx[dom]['system'])} chars, {len(ctx[dom]['tools'])} tools")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(ctx, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, out)
    print("DUMP_DONE", out)
    return ctx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ask2distill.tau2.envside.dump_domain_ctx")
    ap.add_argument("--out", required=True)
    dump(ap.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
