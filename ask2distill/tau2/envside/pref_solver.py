"""v6: preference specs -> the unique gold write, computed from the environment.

v5 made flight changes valid by writing the flight number into the card; the
student then never had to search and choose. Real tau2 tasks state preferences
("not before 11am, direct preferred, the cheapest") and payment rules
("certificate first, rest on the card ending 7447") and leave the search, the
filtering and the arithmetic to the agent.

A spec is structured, the synthesizer copies it verbatim into its output, and the
reward re-solves it here: gold is accepted only if it equals the solver's
answer and the answer is unique. Itineraries come from the environment's own
search tools, so the solver sees exactly what an agent would.

Spec kinds:
  flights_pref   {"kind", "reservation_id", "direction": "outbound"|"return", "new_date",
                  "stops": "direct"|"any"|"prefer_direct", "depart_after": "HH:MM"|None,
                  "depart_before": "HH:MM"|None, "rank": "cheapest"|"earliest"|"latest",
                  "payment_id"}
  book_pref      {"kind", "user_id", "origin", "destination", "date", "cabin", "stops",
                  "depart_after", "depart_before", "rank", "passengers": [names],
                  "total_bags", "insurance": "yes"|"no",
                  "payment_rule": {"type": "card", "card": pid}
                                | {"type": "certificate_then_card", "certificate": pid, "card": pid}
                                | {"type": "gift_then_card", "gift_cards": [pid..], "card": pid}}
  variant_pref   {"kind", "order_id", "item_id", "want": {opt: val}, "fallback": {opt: val}|None,
                  "action": "exchange_delivered_order_items"|"modify_pending_order_items",
                  "payment_method_id"}

REPO_PATH is required; the fork is put on sys.path by `repo.setup`.
"""
import json

from ask2distill.tau2.envside import repo

repo.setup()
from tau2.registry import registry  # noqa: E402  (needs the fork on sys.path)

FREE_BAGS = {"regular": {"basic_economy": 0, "economy": 1, "business": 2},
             "silver": {"basic_economy": 1, "economy": 2, "business": 3},
             "gold": {"basic_economy": 2, "economy": 3, "business": 4}}


class SpecError(Exception):
    pass


_ENV = {}


def _env(domain):
    """One cached environment per domain: the solver only calls read tools
    (search / get_*), so it never mutates the DB and can be shared."""
    if domain not in _ENV:
        _ENV[domain] = registry.get_env_constructor(domain)()
    return _ENV[domain]


def _call(env, name, **kw):
    out = env.make_tool_call(tool_name=name, requestor="assistant", **kw)
    return json.loads(out) if isinstance(out, str) and out[:1] in "[{" else out


def _as_dict(x):
    return x if isinstance(x, dict) else json.loads(json.dumps(x, default=lambda o: o.__dict__))


def itineraries(env, origin, destination, date, cabin, n_pax, stops):
    """Bookable itineraries as the search tools return them:
    [{"legs": [{flight_number, date}], "price": per-passenger total, "depart": "HH:MM:SS", "n_stops"}]."""
    out = []
    if stops in ("direct", "any", "prefer_direct"):
        for f in _call(env, "search_direct_flight", origin=origin, destination=destination, date=date) or []:
            f = _as_dict(f)
            if (f.get("available_seats") or {}).get(cabin, 0) < n_pax:
                continue
            out.append({"legs": [{"flight_number": f["flight_number"], "date": date}],
                        "price": f["prices"][cabin], "depart": f["scheduled_departure_time_est"], "n_stops": 0})
    if stops in ("any", "prefer_direct"):
        for pair in _call(env, "search_onestop_flight", origin=origin, destination=destination, date=date) or []:
            a, b = (_as_dict(x) for x in pair)
            if min((a.get("available_seats") or {}).get(cabin, 0), (b.get("available_seats") or {}).get(cabin, 0)) < n_pax:
                continue
            out.append({"legs": [{"flight_number": a["flight_number"], "date": a.get("date", date)},
                                 {"flight_number": b["flight_number"], "date": b.get("date", date)}],
                        "price": a["prices"][cabin] + b["prices"][cabin],
                        "depart": a["scheduled_departure_time_est"], "n_stops": 1})
    return out


def choose(its, stops, depart_after, depart_before, rank):
    """Apply the customer's preferences; return the unique best itinerary or raise."""
    c = [i for i in its if (not depart_after or i["depart"][:5] >= depart_after)
         and (not depart_before or i["depart"][:5] < depart_before)]
    if stops == "prefer_direct" and any(i["n_stops"] == 0 for i in c):
        c = [i for i in c if i["n_stops"] == 0]
    if not c:
        raise SpecError("no itinerary satisfies the preferences")
    key = {"cheapest": lambda i: i["price"], "earliest": lambda i: i["depart"],
           "latest": lambda i: tuple(-ord(ch) for ch in i["depart"])}[rank]
    c.sort(key=key)
    if len(c) > 1 and key(c[0]) == key(c[1]):
        raise SpecError(f"tie on {rank}")
    return c[0]


# ---- flights_pref ------------------------------------------------------------------
def _directions(res):
    """Split a reservation's legs into outbound (origin -> destination) and return."""
    legs = res["flights"]
    out, i = [], 0
    while i < len(legs):
        out.append(legs[i]); i += 1
        if out[-1]["destination"] == res["destination"]:
            break
    return out, legs[i:]


def solve_flights(spec):
    env = _env("airline")
    res = _as_dict(_call(env, "get_reservation_details", reservation_id=spec["reservation_id"]))
    if res.get("cabin") == "basic_economy":
        raise SpecError("basic economy cannot change flights")
    outb, ret = _directions(res)
    grp = outb if spec["direction"] == "outbound" else ret
    if not grp:
        raise SpecError("reservation has no such direction")
    o, d = grp[0]["origin"], grp[-1]["destination"]
    its = itineraries(env, o, d, spec["new_date"], res["cabin"], len(res["passengers"]), spec["stops"])
    best = choose(its, spec["stops"], spec.get("depart_after"), spec.get("depart_before"), spec["rank"])
    keep = lambda legs: [{"flight_number": x["flight_number"], "date": x["date"]} for x in legs]
    new = (best["legs"] + keep(ret)) if spec["direction"] == "outbound" else (keep(outb) + best["legs"])
    if new == keep(res["flights"]):
        raise SpecError("preferences select the current flights")
    return {"name": "update_reservation_flights",
            "arguments": {"reservation_id": res["reservation_id"], "cabin": res["cabin"],
                          "flights": new, "payment_id": spec["payment_id"]}}


# ---- book_pref ---------------------------------------------------------------------
def solve_book(spec):
    env = _env("airline")
    u = _as_dict(_call(env, "get_user_details", user_id=spec["user_id"]))
    people = {f"{u['name']['first_name']} {u['name']['last_name']}":
              {"first_name": u["name"]["first_name"], "last_name": u["name"]["last_name"], "dob": u["dob"]}}
    for p in u.get("saved_passengers") or []:
        people[f"{p['first_name']} {p['last_name']}"] = dict(p)
    try:
        pax = [people[n] for n in spec["passengers"]]
    except KeyError as e:
        raise SpecError(f"unknown passenger {e}")
    cabin = spec["cabin"]
    its = itineraries(env, spec["origin"], spec["destination"], spec["date"], cabin, len(pax), spec["stops"])
    best = choose(its, spec["stops"], spec.get("depart_after"), spec.get("depart_before"), spec["rank"])
    free = FREE_BAGS[u["membership"]][cabin] * len(pax)
    nonfree = max(0, spec["total_bags"] - free)
    total = best["price"] * len(pax) + (30 * len(pax) if spec["insurance"] == "yes" else 0) + 50 * nonfree
    pm, rule = u["payment_methods"], spec["payment_rule"]
    if rule["type"] == "card":
        pays = [{"payment_id": rule["card"], "amount": total}]
    elif rule["type"] == "certificate_then_card":
        c = min(int(pm[rule["certificate"]]["amount"]), total)
        pays = [{"payment_id": rule["certificate"], "amount": c}] + (
            [{"payment_id": rule["card"], "amount": total - c}] if total > c else [])
    elif rule["type"] == "gift_then_card":
        pays, left = [], total
        for g in rule["gift_cards"][:3]:
            a = min(int(pm[g]["amount"]), left)
            if a > 0:
                pays.append({"payment_id": g, "amount": a}); left -= a
        if left > 0:
            pays.append({"payment_id": rule["card"], "amount": left})
    else:
        raise SpecError(f"payment rule {rule['type']}")
    return {"name": "book_reservation",
            "arguments": {"user_id": u["user_id"], "origin": spec["origin"], "destination": spec["destination"],
                          "flight_type": "one_way", "cabin": cabin, "flights": best["legs"], "passengers": pax,
                          "payment_methods": pays, "total_baggages": spec["total_bags"],
                          "nonfree_baggages": nonfree, "insurance": spec["insurance"]}}


# ---- variant_pref ------------------------------------------------------------------
def solve_variant(spec):
    env = _env("retail")
    o = _as_dict(_call(env, "get_order_details", order_id=spec["order_id"]))
    it = next((i for i in o["items"] if i["item_id"] == spec["item_id"]), None)
    if it is None:
        raise SpecError("item not in order")
    prod = _as_dict(_call(env, "get_product_details", product_id=it["product_id"]))
    avail = {vid: v for vid, v in prod["variants"].items() if v.get("available") and vid != it["item_id"]}
    def match(want):
        base = dict(it.get("options") or {}); base.update(want or {})
        return [vid for vid, v in avail.items() if all(v["options"].get(k) == val for k, val in base.items())]
    hit = match(spec["want"])
    if not hit and spec.get("fallback"):
        hit = match(spec["fallback"])
    if len(hit) != 1:
        raise SpecError(f"{len(hit)} variants match")
    return {"name": spec["action"],
            "arguments": {"order_id": o["order_id"], "item_ids": [it["item_id"]], "new_item_ids": hit,
                          "payment_method_id": spec["payment_method_id"]}}


SOLVERS = {"flights_pref": solve_flights, "book_pref": solve_book, "variant_pref": solve_variant}


def solve(spec):
    if not isinstance(spec, dict) or spec.get("kind") not in SOLVERS:
        raise SpecError("unknown spec kind")
    try:
        return SOLVERS[spec["kind"]](spec)
    except SpecError:
        raise
    except Exception as e:                    # bad ids, missing keys, tool errors
        raise SpecError(f"{type(e).__name__}: {str(e)[:60]}")
