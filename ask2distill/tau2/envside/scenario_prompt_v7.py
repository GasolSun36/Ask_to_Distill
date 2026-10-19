"""v7 synthesizer prompts (retail only): one call, 1-3 requests, conditional branches.

Airline keeps the v6 cards (its plateau already matches the real-question arm);
this module only produces retail. Targets "v7:1" / "v7:2" / "v7:3" fix the number
of requests in the call so the mix stays controlled (real retail: 43% need >=2
writes). The prompt lists up to 3 CALL PLANS enumerated by retail_v7 (each solves
to exactly one list of writes); the synthesizer copies one into "plan" and voices it.
"""
import json
import random

from ask2distill.tau2.envside import retail_v7 as R7
from ask2distill.tau2.envside import scenario_prompt as SP

TARGETS = {"retail": ["v7:1", "v7:2", "v7:2", "v7:3"], "airline": []}

TAIL = """

THIS CALL HAS {n} REQUEST(S) and they are resolved by code from the plan you pick.
Pick ONE plan from CALL PLANS below:
- copy its "plan" object verbatim as "plan" in your output;
- the customer makes EVERY request in "in words", in their own voice, in
  reason_for_call (one call, several things). State conditions as "if ...,
  otherwise ..." / "only if ..." with the exact dollar amounts, and every option,
  item, reason, address field and payment method mentioned;
- known_info: how to identify you -- your email, or your full name AND zip code;{hide}
- never mention a new item id, and never write the plan's key names
  (kind, parts, cond, max_extra, hide_order_ids, persona, item_ids, payment_method_id);
- task_instructions: anything about how you behave (a persona may be added by the grader);
- "actions": may be empty -- the grader rebuilds gold from the plan.
Output ONLY this JSON:
{{"scenario": {{"reason_for_call": "...", "known_info": "...", "unknown_info": null,
  "task_instructions": "..."}},
  "plan": {{...}},
  "actions": []}}

CALL PLANS (the outcome depends on the live order data; voice the conditions, do not resolve them):
{plans}"""

HIDE = ("\n- this customer does NOT remember order ids: never say one; refer to each order by the"
        " items in it (\"the order with the laptop\"). Put \"order ids\" in unknown_info.")


def build(domain, seed, target):
    assert domain == "retail" and target.startswith("v7:")
    n = int(target.split(":")[1])
    rng = random.Random(seed)
    db = SP._load_db("retail")
    for _ in range(60):
        rec = SP._retail(db, rng)
        if rec is None:
            continue
        cands = [(s, w) for s, w in R7.enumerate_specs(rec, rng, k=8) if len(s["parts"]) == n]
        if cands:
            break
    else:
        raise RuntimeError(f"retail/{target}: no solvable plan")
    hide = rng.random() < 0.4
    plans = []
    for s, w in cands[:3]:
        s = {**s, "hide_order_ids": hide}
        # no "writes": with the resolved outcome in view the synthesizer voiced only the
        # branch that fires ("return the sunglasses") and dropped the condition --
        # 99 of 128 condition cards at v7 steps 1-3. Gold is rebuilt from the plan anyway.
        plans.append({"plan": s, "in words": R7.verbalize(s)})
    prompt = SP.INSTRUCTION.format(domain=domain, tools=SP.tool_list(domain), rules=SP.RULES.get(domain, "- (none)"),
                                   records=json.dumps(rec, ensure_ascii=False, indent=1))
    head, rest = prompt.split("\nRequirements:")
    req = rest.split("\nOutput ONLY this JSON")[0]
    return (head + "\nRequirements:" + req.rstrip()
            + TAIL.format(n=n, hide=HIDE if hide else "", plans=json.dumps(plans, ensure_ascii=False, indent=1))), rec

