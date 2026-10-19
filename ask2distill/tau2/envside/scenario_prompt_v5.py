"""v5 synthesizer prompts: close the coverage gaps the first OPD comparison exposed.

Against the official test split the synthetic arm matched the real-question arm
on every task type the pool covered, and lost on exactly the types it could not
produce:
  * refusals (policy says no; DB must stay unchanged) -- impossible under v4,
    whose reward rejects db_unchanged;
  * multi-write tasks -- v4 fixes ONE target write per card;
  * bookings -- 0 of 400 book_reservation cards valid (payment given as a dict,
    amount not equal to the price the tool computes);
  * flight changes -- 11 of 400 valid (the customer said "the cheapest one" /
    "a different flight", which pins down no single gold, and the synthesizer
    padded gold with calculate("min(prices.economy)")).

Targets are strings:  "<write>"            single write (v4 semantics)
                      "multi:<w1>+<w2>"    both writes in gold
                      "refuse:<write>"     customer asks for <write>, policy forbids it
Everything v4 had is kept; this module only adds families and replaces the
skeletons for booking and flight changes with fully precomputed valid options.

Notes:
  * the DB directory has its own variable (A2D_TAU2_DB_DIR, see scenario_prompt);
  * `_alternatives` seeds with the built-in `hash()` of a string, which is
      per-process (hash randomisation); build_prompts pins PYTHONHASHSEED.
"""
import json
import random

from ask2distill.tau2.envside import scenario_prompt as SP

NOW = "2024-05-15T15:00:00"
H24_AGO = "2024-05-14T15:00:00"

TARGETS = {
    "retail": SP.TARGETS["retail"] + [
        "multi:cancel_pending_order+return_delivered_order_items",
        "multi:modify_pending_order_address+exchange_delivered_order_items",
        "multi:modify_user_address+modify_pending_order_address",
    ],
    "airline": SP.TARGETS["airline"] + [
        "refuse:cancel_reservation",
        "refuse:update_reservation_flights",
        "multi:update_reservation_baggages+update_reservation_passengers",
    ],
}

REFUSE_INSTRUCTION_TAIL = """

THIS SCENARIO IS A REFUSAL. The customer asks for `{write}`, but the airline
policy FORBIDS it for this reservation ({why}). The correct resolution is for the
agent to identify the customer, look the reservation up, check the rule, and
decline -- the database must NOT change.
- "actions": ONLY the lookups the agent needs to verify the rule (no write action).
- "forbidden_action": the exact `{write}` call the customer is asking for, with
  real ids from the records (it would execute if an agent ignored the policy).
- Make the customer persistent and plausible: they give a reason, and push back
  once when told no. Do not have the customer mention the policy.
Output ONLY this JSON:
{{"scenario": {{"reason_for_call": "...", "known_info": "...", "unknown_info": null,
  "task_instructions": "..."}},
  "actions": [{{"name": "lookup_tool", "arguments": {{}}}}],
  "forbidden_action": {{"name": "{write}", "arguments": {{}}}}}}"""


# ---- airline helpers --------------------------------------------------------------
def _seg_status(db, seg):
    return (db["flights"].get(seg["flight_number"], {}).get("dates", {}).get(seg["date"]) or {}).get("status")


def _not_flown(db, r):
    return all(seg["date"] >= NOW[:10] and _seg_status(db, seg) in ("available", "on time", "delayed")
               for seg in r.get("flights", []))


def cancel_forbidden(db, r):
    """Why cancelling r violates policy, or None if it is allowed (or not a clean refusal)."""
    if r.get("status") == "cancelled" or not _not_flown(db, r):
        return None                                   # flown -> transfer, not a clean refusal
    if r.get("created_at", "") >= H24_AGO:
        return None                                   # booked within 24h -> allowed
    if r.get("cabin") == "business":
        return None
    if any(_seg_status(db, s) == "cancelled" for s in r.get("flights", [])):
        return None
    if r.get("insurance") == "yes":
        return None                                   # coverage depends on the reason; skip
    return (f"booked {r.get('created_at')} (more than 24h ago), cabin {r.get('cabin')}, "
            "no travel insurance, no flight cancelled by the airline")


def change_forbidden(db, r):
    if r.get("cabin") != "basic_economy" or r.get("status") == "cancelled" or not _not_flown(db, r):
        return None
    return "basic economy flights cannot be modified"


def _alternatives(db, r, seg, n_pax):
    """Available flights that could replace segment `seg` (same origin/destination)."""
    out = []
    for f in db["flights"].values():
        if (f["origin"], f["destination"]) != (seg["origin"], seg["destination"]):
            continue
        for dt, d in f["dates"].items():
            if d.get("status") != "available" or dt < NOW[:10]:
                continue
            if (f["flight_number"], dt) == (seg["flight_number"], seg["date"]):
                continue
            if (d.get("available_seats") or {}).get(r["cabin"], 0) < n_pax:
                continue
            out.append({"flight_number": f["flight_number"], "date": dt,
                        "price_in_cabin": (d.get("prices") or {}).get(r["cabin"])})
    random.Random(hash(seg["flight_number"] + seg["date"])).shuffle(out)
    return out[:4]


def _card_payments(u):
    """Credit cards first: a gift card may not cover a fare difference."""
    pm = (u.get("payment_methods") or {}).items()
    return ([k for k, v in pm if v.get("source") == "credit_card"]
            + [k for k, v in pm if v.get("source") == "gift_card"])


def _airline_skeleton(db, rec, target):
    u = rec["user"]
    res = rec["reservations"]
    if target == "update_reservation_flights":
        for r in res:
            if r.get("cabin") == "basic_economy" or not _not_flown(db, r) or r.get("status") == "cancelled":
                continue
            n = len(r.get("passengers", []))
            for i, seg in enumerate(r.get("flights", [])):
                alts = _alternatives(db, r, seg, n)
                if alts and _card_payments(u):
                    keep = [{"flight_number": s["flight_number"], "date": s["date"]} for s in r["flights"]]
                    options = []
                    for a in alts[:3]:
                        new = [dict(x) for x in keep]
                        new[i] = {"flight_number": a["flight_number"], "date": a["date"]}
                        options.append({"reservation_id": r["reservation_id"], "cabin": r["cabin"],
                                        "flights": new, "payment_id": _card_payments(u)[0]})
                    return {"current flights": keep,
                            f"change segment #{i} ({seg['origin']}->{seg['destination']} {seg['flight_number']} {seg['date']})": [
                                {"new flight": f"{a['flight_number']} on {a['date']}", "price": a["price_in_cabin"]} for a in alts[:3]],
                            "update_reservation_flights arguments: use ONE of these EXACTLY": options,
                            "rule": ("gold = get_user_details, get_reservation_details, ONE search for that segment's "
                                     "route and new date, then update_reservation_flights with one option above. "
                                     "The customer states the new flight number and date and the payment method in known_info.")}
        return None
    if target == "book_reservation":
        pays = {k: v for k, v in (u.get("payment_methods") or {}).items()
                if v.get("source") in ("credit_card", "gift_card")}
        if not pays:
            return None
        me = {"first_name": u["name"]["first_name"], "last_name": u["name"]["last_name"], "dob": u.get("dob")}
        pax_opts = [[me]] + [[me, p] for p in (u.get("saved_passengers") or [])[:1]]
        flights = [f for f in db["flights"].values()]
        rng = random.Random(u["user_id"])
        rng.shuffle(flights)
        combos = []
        for f in flights:
            for dt, d in f["dates"].items():
                if d.get("status") != "available" or dt < NOW[:10]:
                    continue
                for cabin in ("economy", "business", "basic_economy"):
                    for pax in pax_opts:
                        if (d.get("available_seats") or {}).get(cabin, 0) < len(pax):
                            continue
                        ins = rng.choice(["no", "yes"])
                        total = d["prices"][cabin] * len(pax) + (30 * len(pax) if ins == "yes" else 0)
                        pid = next((k for k, v in pays.items()
                                    if v["source"] == "credit_card" or v.get("amount", 0) >= total), None)
                        if not pid:
                            continue
                        combos.append({"user_id": u["user_id"], "origin": f["origin"], "destination": f["destination"],
                                       "flight_type": "one_way", "cabin": cabin,
                                       "flights": [{"flight_number": f["flight_number"], "date": dt}],
                                       "passengers": pax, "payment_methods": [{"payment_id": pid, "amount": total}],
                                       "total_baggages": 0, "nonfree_baggages": 0, "insurance": ins})
                        break
                    if len(combos) >= 3:
                        break
                if len(combos) >= 3:
                    break
            if len(combos) >= 3:
                break
        if not combos:
            return None
        return {"book_reservation arguments: use ONE of these exactly": combos,
                "rule": "payment_methods is a LIST of {payment_id, amount}; COPY the amount exactly as shown -- it already includes the insurance fee, do not recompute it; "
                        "the customer states origin, destination, date, flight number, cabin, every passenger's "
                        "name and date of birth, insurance yes/no, and which payment method"}
    if target == "refuse:cancel_reservation":
        for r in res:
            why = cancel_forbidden(db, r)
            if why:
                return {"reservation_id": r["reservation_id"], "why forbidden": why,
                        "forbidden_action": {"name": "cancel_reservation",
                                             "arguments": {"reservation_id": r["reservation_id"]}},
                        "customer's reason": "change of plans (NOT health/weather, NOT an airline cancellation)"}
        return None
    if target == "refuse:update_reservation_flights":
        for r in res:
            why = change_forbidden(db, r)
            if not why:
                continue
            n = len(r.get("passengers", []))
            for i, seg in enumerate(r.get("flights", [])):
                alts = _alternatives(db, r, seg, n)
                if alts and _card_payments(u):
                    new = [{"flight_number": s["flight_number"], "date": s["date"]} for s in r["flights"]]
                    new[i] = {"flight_number": alts[0]["flight_number"], "date": alts[0]["date"]}
                    return {"reservation_id": r["reservation_id"], "why forbidden": why,
                            "forbidden_action": {"name": "update_reservation_flights",
                                                 "arguments": {"reservation_id": r["reservation_id"],
                                                               "cabin": r["cabin"], "flights": new,
                                                               "payment_id": _card_payments(u)[0]}},
                            "the customer asks for": f"{alts[0]['flight_number']} on {alts[0]['date']} instead of {seg['flight_number']}"}
        return None
    return SP.skeleton("airline", rec, target)


def _airline_sampler(db, rng, target):
    """v4's sampler, but for refusals keep drawing until a reservation qualifies."""
    for _ in range(300):
        rec = SP._airline(db, rng)
        if rec is None:
            continue
        sk = _airline_skeleton(db, rec, target)
        if sk is not None:
            return rec, sk
    return None, None


def skeleton(domain, db, rec, target):
    if target.startswith("multi:"):
        a, b = target.split(":", 1)[1].split("+")
        ska = skeleton(domain, db, rec, a)
        skb = skeleton(domain, db, rec, b)
        if ska is None or skb is None:
            return None
        return {f"first write `{a}`": ska, f"second write `{b}`": skb,
                "rule": "both writes appear in the action list; if both touch orders, use DIFFERENT orders"}
    if domain == "airline":
        return _airline_skeleton(db, rec, target)
    return SP.skeleton(domain, rec, target)


def build(domain, seed, target):
    rng = random.Random(seed)
    db = SP._load_db(domain)
    rec = sk = None
    for _ in range(40):
        if domain == "airline":
            rec, sk = _airline_sampler(db, rng, target)
        else:
            rec = SP.SAMPLERS[domain](db, rng)
            sk = skeleton(domain, db, rec, target) if rec else None
        if sk is not None:
            break
    if sk is None:
        raise RuntimeError(f"{domain}/{target}: records never support this target")
    prompt = SP.INSTRUCTION.format(domain=domain, tools=SP.tool_list(domain), rules=SP.RULES.get(domain, "- (none)"),
                                   records=json.dumps(rec, ensure_ascii=False, indent=1))
    if domain == "airline":
        # airline has no email lookup; the policy makes the customer give the user id
        prompt = prompt.replace("- known_info MUST let the agent identify the customer: give the email, or the full\n  name AND zip code, or the order / reservation id",
                                "- known_info MUST contain the customer's user id (e.g. \"user id: mia_li_3668\"); airline agents cannot look a customer up by email or name")
    prompt = prompt.replace("\n- Make it take several dependent steps",
                            "\n- Never put `calculate` in the action list; it changes nothing.\n- Make it take several dependent steps")
    if target.startswith("refuse:"):
        write = target.split(":", 1)[1]
        # the v4 body demands a DB-changing goal; a refusal is the opposite
        head = prompt.split("\nRequirements:")[0]
        tail_req = prompt.split("\nRequirements:")[1].split("\nOutput ONLY this JSON")[0]
        tail_req = tail_req.replace("- The customer's goal must require changing the database: at least one action from\n  the tool list above that is not a lookup.\n", "")
        prompt = (head + "\nRequirements:" + tail_req.rstrip()
                  + REFUSE_INSTRUCTION_TAIL.format(write=write, why=sk["why forbidden"])
                  + "\n\nVALID SLOTS (use exactly these ids):\n" + json.dumps(sk, ensure_ascii=False, indent=1))
        return prompt, rec
    writes = target.split(":", 1)[1].split("+") if target.startswith("multi:") else [target]
    req = " AND ".join(f"`{w}`" for w in writes)
    prompt += (f"\n\nREQUIRED: the customer's goal must be resolved by calling {req} "
               f"(every one must appear in the action list).")
    for w in writes:
        if w in SP.TARGET_HINTS and w not in ("update_reservation_flights",):
            prompt += " " + SP.TARGET_HINTS[w]
    prompt += ("\n\nVALID SLOTS (use exactly these ids; anything else fails):\n"
               + json.dumps(sk, ensure_ascii=False, indent=1))
    return prompt, rec

