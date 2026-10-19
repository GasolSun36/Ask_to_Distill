"""v6 synthesizer prompts: the customer states PREFERENCES, the agent derives the answer.

v5 closed the task-type gap (refusals now match the real-question arm) but not
the difficulty gap: its flight changes named the flight number, so the student
never searched, filtered or ranked. Real tasks read "not before 11am, direct
preferred, the cheapest; certificate first, rest on the card ending 7447".

For the four targets where the answer is a choice -- flight change, booking,
exchange, item modification -- the prompt now lists specs precomputed by
pref_solver (each solves to exactly one write). The synthesizer picks one, copies
it into "preference", voices it as the customer, and must not name the answer.
All other targets are v5's.

`_book_specs` sorts its route set when `scenario_prompt.DETERMINISTIC` is on.
"""
import json
import random

from ask2distill.tau2.envside import pref_solver as PS
from ask2distill.tau2.envside import scenario_prompt as SP
from ask2distill.tau2.envside import scenario_prompt_v5 as V

PREF = {"retail": ["pref:exchange_delivered_order_items", "pref:modify_pending_order_items"],
        "airline": ["pref:update_reservation_flights", "pref:book_reservation"]}
REPLACED = {"exchange_delivered_order_items", "modify_pending_order_items",
            "update_reservation_flights", "book_reservation"}
TARGETS = {d: [t for t in V.TARGETS[d] if t not in REPLACED] + PREF[d] for d in ("retail", "airline")}

WINDOWS = [(None, None), ("11:00", None), (None, "12:00"), ("14:00", None), ("08:00", "16:00")]
RANKS = ["cheapest", "earliest", "latest"]

PREF_TAIL = """

REQUIRED: the customer's goal is resolved by `{write}`, but the customer states
PREFERENCES, not the answer. Pick ONE spec from PREFERENCE SPECS below:
- copy it verbatim as "preference" in your output;
- the customer states every preference in the spec in their own words (dates,
  time window, direct/stops, cheapest/earliest/latest, cabin, passengers, number
  of bags, insurance, how to pay, wanted options and the fallback);
- NEVER mention a flight number or an item id -- the agent must search and choose;
- write like a real customer: known_info holds ONLY how to identify you (user id /
  email / order or reservation id); every preference goes in reason_for_call in
  plain English. NEVER write the spec's key names or values verbatim (depart_after,
  depart_before, stops, prefer_direct, rank, payment_rule, total_bags, payment_id,
  certificate_then_card, gift_then_card, want, fallback) -- a card that does is rejected;
- "actions": the lookups/searches an agent needs, then the `gold write` given
  for that spec (the grader rebuilds gold from the preference, so the spec matters most).
Output ONLY this JSON:
{{"scenario": {{"reason_for_call": "...", "known_info": "...", "unknown_info": null,
  "task_instructions": "..."}},
  "preference": {{...}},
  "actions": [{{"name": "tool_name", "arguments": {{}}}}]}}

PREFERENCE SPECS (each has exactly one correct answer):
{specs}"""


def _ok(spec):
    try:
        return PS.solve(spec)
    except PS.SpecError:
        return None


def _pick(cands, rng, k=3):
    """Prefer harder specs (a time window, prefer_direct, a fallback), keep variety."""
    rng.shuffle(cands)
    cands.sort(key=lambda sg: -sum([bool(sg[0].get("depart_after") or sg[0].get("depart_before")),
                                    sg[0].get("stops") == "prefer_direct", bool(sg[0].get("fallback")),
                                    sg[0].get("rank") != "cheapest"]))
    return cands[:k]


def _flight_specs(db, rec, rng):
    u = rec["user"]; pays = V._card_payments(u)
    out = []
    for r in rec["reservations"]:
        if r.get("cabin") == "basic_economy" or r.get("status") == "cancelled" or not V._not_flown(db, r):
            continue
        outb, ret = PS._directions(r)
        for direction, grp in (("outbound", outb), ("return", ret)):
            if not grp:
                continue
            day = int(grp[0]["date"][-2:])
            for dd in rng.sample([-2, -1, 1, 2, 3], 3):
                nd = day + dd
                if not 16 <= nd <= 30:
                    continue
                for stops in ("prefer_direct", "direct", "any"):
                    for (a, b) in rng.sample(WINDOWS, 3):
                        rank = rng.choice(RANKS)
                        spec = {"kind": "flights_pref", "reservation_id": r["reservation_id"], "direction": direction,
                                "new_date": f"2024-05-{nd:02d}", "stops": stops, "depart_after": a,
                                "depart_before": b, "rank": rank, "payment_id": pays[0] if pays else None}
                        if spec["payment_id"] and (g := _ok(spec)):
                            out.append((spec, g))
                if len(out) >= 12:
                    return out
    return out


def _book_specs(db, rec, rng):
    u = rec["user"]; pm = u.get("payment_methods") or {}
    cards = [k for k, v in pm.items() if v.get("source") == "credit_card"]
    if not cards:
        return []
    certs = [k for k, v in pm.items() if v.get("source") == "certificate"]
    gifts = [k for k, v in pm.items() if v.get("source") == "gift_card"]
    names = [f"{u['name']['first_name']} {u['name']['last_name']}"] + \
            [f"{p['first_name']} {p['last_name']}" for p in (u.get("saved_passengers") or [])[:1]]
    routes = {(f["origin"], f["destination"]) for f in db["flights"].values()}
    routes = sorted(routes) if SP.DETERMINISTIC else list(routes)   # set order is hash-dependent
    out = []
    for _ in range(40):
        o, d = rng.choice(routes)
        spec = {"kind": "book_pref", "user_id": u["user_id"], "origin": o, "destination": d,
                "date": f"2024-05-{rng.randint(16, 30):02d}", "cabin": rng.choice(["economy", "business", "economy"]),
                "stops": rng.choice(["prefer_direct", "direct", "any"]),
                "depart_after": None, "depart_before": None, "rank": rng.choice(RANKS),
                "passengers": names[:rng.choice([1, len(names)])], "total_bags": rng.choice([0, 1, 2, 3, 4]),
                "insurance": rng.choice(["no", "yes"]),
                "payment_rule": rng.choice(
                    ([{"type": "certificate_then_card", "certificate": rng.choice(certs), "card": cards[0]}] if certs else []) +
                    ([{"type": "gift_then_card", "gift_cards": gifts[:3], "card": cards[0]}] if gifts else []) +
                    [{"type": "card", "card": rng.choice(cards)}])}
        spec["depart_after"], spec["depart_before"] = rng.choice(WINDOWS)
        if (g := _ok(spec)):
            out.append((spec, g))
        if len(out) >= 6:
            break
    return out


def _variant_specs(db, rec, rng, action):
    u = rec["user"]
    # a gift card may not cover the price difference ("Insufficient gift card balance")
    pms = [m["id"] for m in (u.get("payment_methods") or {}).values()
           if m.get("source") != "gift_card" or (m.get("balance") or 0) >= 1000] or \
          [m["id"] for m in (u.get("payment_methods") or {}).values()]
    status = "delivered" if action.startswith("exchange") else "pending"
    out = []
    for o in rec["orders"]:
        if o.get("status") != status:
            continue
        for it in o.get("items", []):
            prod = db["products"].get(it["product_id"], {})
            var = prod.get("variants", {})
            cur = it.get("options") or {}
            for vid, v in var.items():
                if vid == it["item_id"] or not v.get("available"):
                    continue
                diff = {k: val for k, val in v["options"].items() if cur.get(k) != val}
                if not diff:
                    continue
                spec = {"kind": "variant_pref", "order_id": o["order_id"], "item_id": it["item_id"],
                        "item_name": it.get("name"), "want": diff, "fallback": None, "action": action,
                        "payment_method_id": rng.choice(pms)}
                # a fallback: first ask for a combination that is unavailable
                un = [x for x in var.values() if not x.get("available")]
                if un and rng.random() < 0.6:
                    w = {k: val for k, val in rng.choice(un)["options"].items() if cur.get(k) != val}
                    if w:
                        spec = {**spec, "want": w, "fallback": diff}
                if (g := _ok(spec)):
                    out.append((spec, g))
        if len(out) >= 8:
            break
    return out


def build(domain, seed, target):
    if not target.startswith("pref:"):
        return V.build(domain, seed, target)
    write = target.split(":", 1)[1]
    rng = random.Random(seed)
    db = SP._load_db(domain)
    for _ in range(40):
        rec = SP.SAMPLERS[domain](db, rng)
        if rec is None:
            continue
        if write == "update_reservation_flights":
            c = _flight_specs(db, rec, rng)
        elif write == "book_reservation":
            c = _book_specs(db, rec, rng)
        else:
            c = _variant_specs(db, rec, rng, write)
        if c:
            break
    else:
        raise RuntimeError(f"{domain}/{target}: no solvable spec")
    specs = [{"preference": s, "gold write": g} for s, g in _pick(c, rng)]
    prompt = SP.INSTRUCTION.format(domain=domain, tools=SP.tool_list(domain), rules=SP.RULES.get(domain, "- (none)"),
                                   records=json.dumps(rec, ensure_ascii=False, indent=1))
    if domain == "airline":
        prompt = prompt.replace("- known_info MUST let the agent identify the customer: give the email, or the full\n  name AND zip code, or the order / reservation id",
                                "- known_info MUST contain the customer's user id (e.g. \"user id: mia_li_3668\"); airline agents cannot look a customer up by email or name")
    # the v4 body asks the customer to state every value; preferences replace values here
    head, rest = prompt.split("\nRequirements:")
    req = rest.split("\nOutput ONLY this JSON")[0]
    return head + "\nRequirements:" + req.rstrip() + PREF_TAIL.format(write=write, specs=json.dumps(specs, ensure_ascii=False, indent=1)), rec

