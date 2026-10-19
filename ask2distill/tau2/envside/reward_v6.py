"""v6 scoring: preference cards on top of the v5 gates.

For a "pref:<write>" target the card carries "preference" (a spec copied from the
prompt). Before any rollout:
  pref_unsolvable   the spec does not solve to exactly one write (pref_solver);
  pref_gold_mismatch the card's <write> call differs from the solver's answer;
  pref_leak         the customer's text names the answer (a flight number of the
                    new itinerary, a new item id);
  pref_unstated:<f> a preference in the spec is never stated by the customer --
                    the agent could not know it, the same under-specification the
                    teacher gate catches, but caught for free.
Then the card is scored exactly as a v5 single-write card (gold executes, is
grounded, changes the DB; student k rollouts; student-first teacher gate).
Other targets go straight to v5.
"""
import json
import os
import re

from ask2distill.tau2.envside import pref_solver as PS
from ask2distill.tau2.envside import reward_v5 as R5
from ask2distill.tau2.envside import scenario_prompt as SP

PREF_TURNS = int(os.environ.get("TAU2_PREF_TURNS", 25))
# Teacher gate off for preference cards (TAU2_PREF_TEACHER_GATE=1 restores it). At 20
# turns the teacher still failed 10/16 bookings and 13/16 exchanges: the gate was
# filtering hard cards, not ill-posed ones -- well-posedness is already proven by
# pref_solver (unique answer) + the pref_unstated check. The official train split is
# not filtered by teacher success either.
PREF_TEACHER_GATE = os.environ.get("TAU2_PREF_TEACHER_GATE", "0") == "1"
NUM = {0: "zero|no", 1: "one|a single|1", 2: "two|2", 3: "three|3", 4: "four|4"}


def _canon(x):
    return json.dumps(x, sort_keys=True)


def _hour_rx(hhmm):
    h = int(hhmm[:2]); h12 = h % 12 or 12
    ap = "am" if h < 12 else "pm"
    return rf"(\b{h}(:00)?\b|\b{h12}(:00)?\s*{ap}|\b{h12}\s*o'?clock|noon)" if h == 12 else \
           rf"(\b{h:02d}:00\b|\b{h}:00\b|\b{h12}(:00)?\s*{ap}\b|\b{h12}\s*o'?clock)"


RANK_RX = {"cheapest": r"cheap|lowest|least expensive|lowest price|lowest fare|budget",
           "earliest": r"earliest|first (available|flight|one)|as early",
           "latest": r"latest|last (available|flight|one)|as late"}


def _pay_rx(user, pid):
    pms = user.get("payment_methods") or {}
    pm = pms.get(pid) or {}
    alts = [re.escape(pid)]
    # "my credit card" is unambiguous when the customer has only one
    if pm.get("source") == "credit_card" and sum(v.get("source") == "credit_card" for v in pms.values()) == 1:
        alts.append("credit card")
    if pm.get("last_four"): alts.append(re.escape(pm["last_four"]))
    if pm.get("source") == "paypal": alts.append("paypal")
    if pm.get("source") == "gift_card": alts.append("gift card")
    if pm.get("source") == "certificate": alts.append("certificate")
    return "|".join(alts)


def unstated_all(spec, text, domain):
    """(missing preference names, number of checks)."""
    return unstated(spec, text, domain, _all=True)


def unstated(spec, text, domain, _all=False):
    """Name of the first preference the customer never states, or None."""
    t = text.lower()
    def need(name, rx):
        return None if re.search(rx, t, re.I) else name
    k = spec["kind"]
    if k in ("flights_pref", "book_pref"):
        day = int((spec.get("new_date") or spec.get("date"))[-2:])
        checks = [("date", rf"\b{day}(st|nd|rd|th)?\b|05-{day:02d}|5/{day}\b"),
                  ("rank", RANK_RX[spec["rank"]])]
        if spec.get("depart_after"): checks.append(("depart_after", _hour_rx(spec["depart_after"])))
        if spec.get("depart_before"): checks.append(("depart_before", _hour_rx(spec["depart_before"])))
        if spec["stops"] in ("direct", "prefer_direct"): checks.append(("stops", r"direct|non-?stop"))
        db = SP._load_db("airline")
        if k == "flights_pref":
            r = db["reservations"].get(spec["reservation_id"], {})
            u = db["users"].get(r.get("user_id"), {})
            checks.append(("payment", _pay_rx(u, spec["payment_id"])))
        else:
            u = db["users"].get(spec["user_id"], {})
            checks += [("cabin", rf"\b{spec['cabin']}\b" if spec["cabin"] == "business" else r"(?<!basic )\beconomy\b"),
                       ("bags", rf"\b({NUM.get(spec['total_bags'], spec['total_bags'])})\b[^.;]{{0,20}}(bag|luggage|suitcase)|no (checked )?bag"),
                       ("insurance", r"insurance"),
                       ("card", _pay_rx(u, spec["payment_rule"]["card"]))]
            if spec["payment_rule"]["type"] == "certificate_then_card": checks.append(("certificate", r"certificate"))
            if spec["payment_rule"]["type"] == "gift_then_card": checks.append(("gift", r"gift card"))
            for n in spec["passengers"]:
                checks.append((f"passenger:{n}", re.escape(n.split()[0])))
        miss = [n for n, rx in checks if need(n, rx)]
        if _all:
            return miss, len(checks)
        return miss[0] if miss else None
    if k == "variant_pref":
        miss, tot = [], 0
        for opt, v in list((spec.get("want") or {}).items()) + list((spec.get("fallback") or {}).items()):
            v = str(v).lower()
            # boolean options are voiced by name ("a water-resistant one"), not "yes"
            if v in ("none", "no"):       # "no backlight", "without backlight"
                ok = re.search(rf"\b(no|without|none)\b[^.;]{{0,15}}{re.escape(opt.lower().split()[0])}|{re.escape(opt.lower().split()[0])}[^.;]{{0,15}}\bnone\b", t)
            else:
                probe = opt.lower().split()[0] if v in ("yes", "true", "false") else v
                ok = probe in t
            tot += 1
            if not ok: miss.append(f"option:{opt}={v}")
        if _all:
            return miss, tot
        return miss[0] if miss else None
    return ([], 0) if _all else None


# Spec keys pasted into the card ("depart_before: 12:00, stops: prefer_direct") pass
# the keyword check without the customer ever saying it (v6 step 12).
DUMP = re.compile(r"depart_(after|before)|prefer_direct|payment_id|payment_rule|total_bags|"
                  r"\bstops\s*:|\brank\s*:|\bkind\s*:|flights_pref|book_pref|variant_pref|gift_then_card|"
                  r"certificate_then_card|\bwant\s*:|\bfallback\s*:", re.I)
JUDGE = os.environ.get("TAU2_PREF_JUDGE", "1") == "1"


def verbalize(spec):
    """The spec as the plain-English checklist the judge compares against."""
    k = spec["kind"]
    if k == "variant_pref":
        w = ", ".join(f"{a}: {b}" for a, b in (spec.get("want") or {}).items())
        out = [f"wants the {spec.get('item_name') or 'item'} changed to a variant with {w}"]
        if spec.get("fallback"):
            f = ", ".join(f"{a}: {b}" for a, b in spec["fallback"].items())
            out.append(f"if that is not available, a variant with {f} instead")
        return out
    date = spec.get("new_date") or spec.get("date")
    out = [("move the " + spec["direction"] + f" flight(s) of reservation {spec['reservation_id']} to {date}")
           if k == "flights_pref" else f"book a one-way trip {spec['origin']} -> {spec['destination']} on {date}"]
    if spec.get("depart_after"): out.append(f"departure at or after {spec['depart_after']}")
    if spec.get("depart_before"): out.append(f"departure before {spec['depart_before']}")
    out.append({"direct": "direct flights only", "prefer_direct": "direct preferred, one stop acceptable if no direct fits",
                "any": "direct or one-stop both fine"}[spec["stops"]])
    out.append({"cheapest": "among the flights that fit, the cheapest", "earliest": "among the flights that fit, the earliest departure that day",
                "latest": "among the flights that fit, the latest departure that day"}[spec["rank"]])
    if k == "book_pref":
        out += [f"cabin: {spec['cabin']}", f"passengers: {', '.join(spec['passengers'])}",
                f"{spec['total_bags']} checked bags in total", f"insurance: {spec['insurance']}",
                {"card": "pay with one credit card", "certificate_then_card": "pay with a travel certificate first, the rest by credit card",
                 "gift_then_card": "pay with gift card(s) first, the rest by credit card"}[spec["payment_rule"]["type"]]]
    return out


def judge_ok(spec, said, client, model):
    items = "\n".join(f"{i + 1}. {x}" for i, x in enumerate(verbalize(spec)))
    q = ("A customer will read the text below to a service agent.\n\n<customer_text>\n" + said +
         "\n</customer_text>\n\nRequired preferences:\n" + items +
         "\n\nDoes the customer text state EVERY required preference correctly, with the same meaning, "
         "and nothing that contradicts one (e.g. 'latest date' is NOT 'latest departure that day')? "
         "Answer with exactly one word: YES or NO.")
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": q}],
                                           max_tokens=4, temperature=0.0)
        return (r.choices[0].message.content or "").strip().upper().startswith("YES")
    except Exception:
        return None


def _ground_extra(spec):
    """ISO date / airport codes the customer voices as "May 18th" / "Miami": steps
    31-40 lost 30 flight cards to ungrounded:search_direct_flight:date=2024-05-xx
    although pref_unstated + the judge had verified the date was stated."""
    k = spec.get("kind")
    if k == "flights_pref":
        return spec["new_date"]
    if k == "book_pref":
        return f"{spec['date']} {spec['origin']} {spec['destination']}"
    return ""


def canonical_actions(spec, g):
    k = spec["kind"]
    if k == "flights_pref":
        r = SP._load_db("airline")["reservations"][spec["reservation_id"]]
        outb, ret = PS._directions(r)
        grp = outb if spec["direction"] == "outbound" else ret
        o, d = grp[0]["origin"], grp[-1]["destination"]
        acts = [{"name": "get_user_details", "arguments": {"user_id": r["user_id"]}},
                {"name": "get_reservation_details", "arguments": {"reservation_id": r["reservation_id"]}}]
        date = spec["new_date"]
    elif k == "book_pref":
        o, d, date = spec["origin"], spec["destination"], spec["date"]
        acts = [{"name": "get_user_details", "arguments": {"user_id": spec["user_id"]}}]
    else:
        o_ = SP._load_db("retail")["orders"][spec["order_id"]]
        pid = next(i["product_id"] for i in o_["items"] if i["item_id"] == spec["item_id"])
        return [{"name": "get_user_details", "arguments": {"user_id": o_["user_id"]}},
                {"name": "get_order_details", "arguments": {"order_id": spec["order_id"]}},
                {"name": "get_product_details", "arguments": {"product_id": pid}}, g]
    acts.append({"name": "search_direct_flight", "arguments": {"origin": o, "destination": d, "date": date}})
    if spec["stops"] != "direct":
        acts.append({"name": "search_onestop_flight", "arguments": {"origin": o, "destination": d, "date": date}})
    return acts + [g]


def score(cand, stu, stu_model, usr, usr_model, k=2, teacher_client=None, teacher_model=None):
    tgt = cand.get("target") or ""
    if not tgt.startswith("pref:"):
        return R5.score(cand, stu, stu_model, usr, usr_model, k=k, teacher_client=teacher_client, teacher_model=teacher_model)
    dom, write = cand["domain"], tgt.split(":", 1)[1]
    base = {"domain": dom, "family": "pref"}
    spec = cand.get("preference")
    try:
        g = PS.solve(spec)
    except PS.SpecError as e:
        return {**base, "R": 0.0, "rejected": f"pref_unsolvable:{str(e)[:40]}"}
    if g["name"] != write:
        return {**base, "R": 0.0, "rejected": "pref_wrong_kind"}
    # Gold is built here, not taken from the card: the canonical lookups an agent
    # needs + the solver's write. Probe cards lost most of their yield to gold
    # lists that skipped get_reservation_details (unchanged legs "ungrounded") or
    # mis-copied the write -- bookkeeping, not scenario quality. Grounding still
    # runs on this gold, so the card must still say who the customer is.
    acts = canonical_actions(spec, g)
    sc = cand.get("scenario") or {}
    said = " ".join(str(sc.get(f) or "") for f in ("reason_for_call", "known_info", "unknown_info", "task_instructions"))
    if write in ("update_reservation_flights", "book_reservation"):
        new = {x["flight_number"] for x in g["arguments"]["flights"]}
        if any(fn.lower() in said.lower() for fn in new):
            return {**base, "R": 0.0, "rejected": "pref_leak"}
    elif any(i in said for i in g["arguments"]["new_item_ids"]):
        return {**base, "R": 0.0, "rejected": "pref_leak"}
    if DUMP.search(said):
        # Graded, not flat 0: at v6 step 20 every airline pref card in a group dumped,
        # so all 16 scored 0 and GRPO had no gradient to leave the habit. Fewer
        # pasted keys score slightly higher (max 0.1, far below a valid card), and
        # the group is ordered again. Still counted as rejected for yield stats.
        n = len({m.group(0).lower() for m in DUMP.finditer(said)})
        return {**base, "R": round(0.1 / (1 + n), 4), "R_raw": 0.0, "rejected": "pref_dump"}
    miss, tot = unstated_all(spec, said, dom)
    if miss:
        # Graded like pref_dump: v6 steps 22 and 26 had every booking card miss at
        # least one of its ~8 preferences, a flat 0 left the group without gradient.
        # Stating more of them scores a little more (max 0.1 << a valid card).
        return {**base, "R": round(0.1 * (tot - len(miss)) / max(1, tot), 4), "rejected": f"pref_unstated:{miss[0]}"}
    # Judge = the user-simulator model (Instruct-2507, non-thinking). The teacher
    # (Thinking-2507) reasons at length even under the no-think template and never
    # reached YES/NO within a short budget; calibration: Instruct-2507 said YES to
    # correct cards, NO to a dropped fallback and to "latest date" for "latest departure".
    if JUDGE and usr is not None:
        ok = judge_ok(spec, said, usr, usr_model)
        if ok is None:
            return {**base, "R": 0.0, "rejected": "rollout_error:judge"}
        if not ok:
            return {**base, "R": 0.0, "rejected": "pref_judge_no"}
    # v6 step 1-3: valid preference cards took 9.7 student turns on average (6.3 for
    # other cards) against a 10-turn teacher gate, and 47 of 64 were "teacher
    # unsolvable" -- the gate was measuring the cap, and RL would have pushed the
    # synthesizer back to easy preferences.
    res = R5.score({**cand, "actions": acts, "target": write}, stu, stu_model, usr, usr_model, k=k,
                   teacher_client=teacher_client if PREF_TEACHER_GATE else None, teacher_model=teacher_model,
                   student_turns=PREF_TURNS, gate_turns=PREF_TURNS, ground_extra=_ground_extra(spec))
    return {**res, "family": "pref"}
