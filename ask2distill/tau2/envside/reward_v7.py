"""v7 scoring: compound retail calls (retail_v7) on top of v6/v5.

For a "v7:<n>" target the card carries "plan" (a compound spec copied from the
prompt). Gates before any rollout, mirroring the v6 preference gates:
  plan_unsolvable      retail_v7.solve fails (a part does not resolve, parts share an order, ...)
  plan_parts           the plan has a different number of requests than the target
  plan_dump            plan key names pasted into the customer's text (graded, <= 0.1)
  plan_leak            a new item id, or an order id when hide_order_ids is set
  plan_unstated:<x>    a request / option / condition / amount never stated (graded, <= 0.1)
  plan_judge_no        the Instruct-2507 judge says the text does not state the plan
Gold = canonical lookups + the solver's writes; the persona is appended to
task_instructions by code. Scored as a v5 multi-write card with the teacher gate
off (well-posedness is proven by the solver + the checks above) and 30 turns.
Other targets go to v6.
"""
import json
import os
import re

from ask2distill.tau2.envside import retail_v7 as R7
from ask2distill.tau2.envside import reward_v5 as R5
from ask2distill.tau2.envside import reward_v6 as R6

TURNS = int(os.environ.get("TAU2_V7_TURNS", 30))
DUMP = re.compile(r"\b(kind|parts|cond|max_extra|hide_order_ids|persona|item_ids|payment_method_id|order_id|"
                  r"new_item_ids|compound)\b\s*[:=\"]|price_leq|\"available\"|\"pending\"", re.I)
LEAK_ITEM = re.compile(r"\b\d{10}\b")


def judge_ok(spec, said, client, model):
    items = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(R7.verbalize(spec)))
    q = ("A customer will read the text below to a service agent.\n\n<customer_text>\n" + said +
         "\n</customer_text>\n\nThe call must contain these requests:\n" + items +
         "\n\nDoes the customer text make EVERY request above, with the same meaning -- same items, options, "
         "conditions (if/otherwise), dollar limits, reasons and payment methods -- and nothing that contradicts one? "
         "Answer with exactly one word: YES or NO.")
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": q}],
                                           max_tokens=4, temperature=0.0)
        return (r.choices[0].message.content or "").strip().upper().startswith("YES")
    except Exception:
        return None


def _ground_extra(spec, said):
    """The user id when the customer identified themselves by full name + zip (the
    agent's find_user_id_by_name_zip lookup; the grounding closure only follows emails)."""
    u = R7._db()["users"].get(spec.get("user_id"), {})
    nm = u.get("name") or {}
    zp = str((u.get("address") or {}).get("zip", ""))
    t = said.lower()
    if nm and nm.get("first_name", "").lower() in t and nm.get("last_name", "").lower() in t and zp and zp in t:
        return spec["user_id"]
    return ""


def score(cand, stu, stu_model, usr, usr_model, k=2, teacher_client=None, teacher_model=None):
    tgt = cand.get("target") or ""
    if not tgt.startswith("v7:"):
        return R6.score(cand, stu, stu_model, usr, usr_model, k=k, teacher_client=teacher_client, teacher_model=teacher_model)
    dom = cand["domain"]
    base = {"domain": dom, "family": "v7"}
    spec = cand.get("plan")
    if not isinstance(spec, dict) or not isinstance(spec.get("parts"), list) or len(spec["parts"]) != int(tgt.split(":")[1]):
        return {**base, "R": 0.0, "rejected": "plan_parts"}
    try:
        writes = R7.solve(spec)
        R7.verbalize(spec)
    except R7.PS.SpecError as e:
        return {**base, "R": 0.0, "rejected": f"plan_unsolvable:{str(e)[:40]}"}
    except Exception as e:
        return {**base, "R": 0.0, "rejected": f"plan_unsolvable:malformed {type(e).__name__}"}
    sc = cand.get("scenario") or {}
    said = " ".join(str(sc.get(f) or "") for f in ("reason_for_call", "known_info", "unknown_info", "task_instructions"))
    if DUMP.search(said):
        n = len({m.group(0).lower() for m in DUMP.finditer(said)})
        return {**base, "R": round(0.1 / (1 + n), 4), "rejected": "plan_dump"}
    new_ids = {i for w in writes for i in (w["arguments"].get("new_item_ids") or [])}
    if any(i in said for i in new_ids):
        return {**base, "R": 0.0, "rejected": "plan_leak"}
    miss, tot = R7.unstated(spec, said)
    if miss:
        return {**base, "R": round(0.1 * (tot - len(miss)) / max(1, tot), 4), "rejected": f"plan_unstated:{miss[0]}"}
    if R6.JUDGE and usr is not None:
        ok = judge_ok(spec, said, usr, usr_model)
        if ok is None:
            return {**base, "R": 0.0, "rejected": "rollout_error:judge"}
        if not ok:
            return {**base, "R": 0.0, "rejected": "plan_judge_no"}
    acts = R7.canonical_actions(spec, writes)
    names = [w["name"] for w in writes]
    t5 = names[0] if len(names) == 1 else "multi:" + "+".join(names)
    res = R5.score({**cand, "scenario": R7.with_persona(sc, spec), "actions": acts, "target": t5},
                   stu, stu_model, usr, usr_model, k=k, teacher_client=None, teacher_model=teacher_model,
                   student_turns=TURNS, gate_turns=TURNS, ground_extra=_ground_extra(spec, said))
    return {**res, "family": "v7", "n_writes": len(writes)}
