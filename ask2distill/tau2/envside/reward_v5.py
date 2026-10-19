"""v5 scoring on top of the v4 reward (reward_base): multi-write targets and refusals.

  single / multi:  every target write must appear in gold (off_target otherwise);
                   calculate/think calls are dropped before gold runs -- they change
                   nothing, and v4 lost 78 of 400 flight cards to a malformed
                   calculate expression inside otherwise valid gold.
  refuse:<write>:  the card carries forbidden_action. Valid only if
                     (1) forbidden_action is a <write> the policy rules out for that
                         record, checked in code (scenario_prompt_v5.cancel_forbidden /
                         change_forbidden) -- never on the synthesizer's word;
                     (2) gold (lookups only) + forbidden_action executes, is grounded
                         in the card, and changes the DB: the tool would comply, so
                         only the agent's policy knowledge stands between the
                         customer and the write;
                   and the target state is the UNCHANGED database. A student fails
                   by writing anything; the teacher gate asks that some teacher
                   episode leaves the DB untouched.
Both keep v4's student-first teacher gate.

The v4 base is `reward_base`.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor

from ask2distill.tau2.envside import reward_base as TR
from ask2distill.tau2.envside import scenario_prompt as SP
from ask2distill.tau2.envside import scenario_prompt_v5 as V

DROP = {"calculate", "think"}
REFUSE_PERSONA = ("You really want this done and give your reason. If the agent says it cannot be done, "
                  "push back once and ask them to make an exception; if they still decline, accept it and end the call.")
# A refusal card must not tell the simulator the answer: v5 probe cards had the
# customer open with "I'm being told it's not possible" or state that basic
# economy cannot be modified, which the simulator repeats to the agent.
LEAK = re.compile(r"not (possible|allowed|permitted|eligible)|cannot be (modified|changed|cancell?ed)|"
                  r"can't be (modified|changed|cancell?ed)|against (the )?polic|polic(y|ies)|not aware|refus|"
                  r"\bdeclin|ineligible|non-refundable", re.I)


def _names(actions):
    return [a.get("name") for a in actions if isinstance(a, dict)]


def _policy_forbids(domain, fa, write):
    if domain != "airline" or not isinstance(fa, dict) or fa.get("name") != write:
        return None
    rid = (fa.get("arguments") or {}).get("reservation_id")
    r = SP._load_db("airline")["reservations"].get(rid)
    if r is None:
        return None
    db = SP._load_db("airline")
    return V.cancel_forbidden(db, r) if write == "cancel_reservation" else (
        V.change_forbidden(db, r) if write == "update_reservation_flights" else None)


def score(cand, stu, stu_model, usr, usr_model, k=2, teacher_client=None, teacher_model=None,
          student_turns=None, gate_turns=None, ground_extra=""):
    """student_turns / gate_turns override TAU2_MAX_TURNS / TAU2_GATE_TURNS for this
    card (v6 preference cards need ~10 turns of lookups + searches + confirmation)."""
    dom, tgt = cand["domain"], cand.get("target") or ""
    acts = [a for a in (cand.get("actions") or []) if isinstance(a, dict) and a.get("name") not in DROP]
    # ground_extra: values the caller has already verified the customer states in
    # other words (v6: "May 18th" for date=2024-05-18) -- grounding only, the
    # simulator never sees it
    text = json.dumps(cand.get("scenario", {}), ensure_ascii=False) + " " + ground_extra
    base = {"domain": dom, "family": tgt.split(":")[0] if ":" in tgt else "single"}
    if tgt.startswith("refuse:"):
        write = tgt.split(":", 1)[1]
        fa = cand.get("forbidden_action")
        # task_instructions steer the SIMULATOR. v5 step 2/4: 63 of 64 refusal cards
        # wrote them as the agent's solution ("check the policy and tell the customer
        # cancellation is not allowed") and the leak gate zeroed every group -- no
        # gradient. The field is overwritten with a fixed customer persona instead,
        # and the leak check covers only what the customer says.
        cand = {**cand, "scenario": {**(cand.get("scenario") or {}), "task_instructions": REFUSE_PERSONA}}
        sc = cand["scenario"]
        text = json.dumps(sc, ensure_ascii=False)          # grounding sees what the simulator sees
        if LEAK.search(" ".join(str(sc.get(k) or "") for k in ("reason_for_call", "known_info", "unknown_info"))):
            return {**base, "R": 0.0, "rejected": "refuse_leak"}
        if any(not TR.READ.match(n or "") for n in _names(acts)):
            return {**base, "R": 0.0, "rejected": "refuse_gold_writes"}
        why = _policy_forbids(dom, fa, write)
        if not why:
            return {**base, "R": 0.0, "rejected": "refuse_not_forbidden"}
        gh, rej = TR.gold(dom, acts + [fa], card_text=text)
        if gh is None:
            return {**base, "R": 0.0, "rejected": f"refuse_{rej}"}
        target_hash = TR.base_hash(dom)
    else:
        writes = tgt.split(":", 1)[1].split("+") if tgt.startswith("multi:") else [tgt]
        miss = [w for w in writes if w not in _names(acts)]
        if miss:
            return {**base, "R": 0.0, "rejected": f"off_target:{','.join(miss)}"}
        target_hash, rej = TR.gold(dom, acts, card_text=text)
        if target_hash is None:
            return {**base, "R": 0.0, "rejected": rej}

    def many(n, client, model, cap=None):
        with ThreadPoolExecutor(n) as ex:
            return list(ex.map(lambda _: TR.rollout(dom, cand["scenario"], client, model, usr, usr_model,
                                                    max_turns=cap), range(n)))
    try:
        outs = many(k, stu, stu_model, cap=student_turns)
    except Exception as e:
        return {**base, "R": 0.0, "rejected": f"rollout_error:{str(e)[:60]}"}
    fails = sum(1 for h, _ in outs if h != target_hash)
    t_ok = None
    if TR.TEACHER_GATE and teacher_client is not None:
        if fails < k:
            t_ok = -1
        else:
            try:
                touts = many(TR.K_T, teacher_client, teacher_model, cap=gate_turns or TR.GATE_TURNS)
            except Exception as e:
                return {**base, "R": 0.0, "rejected": f"rollout_error:{str(e)[:60]}"}
            t_ok = int(any(h == target_hash for h, _ in touts))
            if not t_ok:
                return {**base, "R": 0.0, "rejected": "teacher_unsolvable"}
    return {**base, "R": fails / k, "rejected": None, "k": k, "student_fail": fails,
            "mean_turns": sum(t for _, t in outs) / len(outs), "teacher_ok": t_ok}
