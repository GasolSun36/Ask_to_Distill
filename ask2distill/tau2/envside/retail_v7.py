"""v7 retail: one call, several requests, conditional branches, persona -- solved by code.

Why: v6's retail training cards were simpler than real retail tasks on every
shape statistic we could count (conditionals per card 0.41 vs 1.24, >=2 writes 24% vs 43%, multi-request 0.21 vs 0.53, persona 0% vs
32%). Real tasks were used only for those aggregate counts, never as training data.

A v7 card is a "compound" spec: 1-3 parts on DIFFERENT orders. A part is
  {"kind": "variant", ...}   exchange / modify to a variant by wanted options (pref_solver)
  {"kind": "return", "order_id", "item_ids", "payment_method_id"}
  {"kind": "cancel", "order_id", "reason"}
  {"kind": "address", "order_id", "address"}          modify_pending_order_address
  {"kind": "cond", "cond": {...}, "then": part, "else": part | None}
with conditions decided from the DB by code:
  {"type": "available", "variant": <variant part>}     then = that exchange, else = return the item
  {"type": "price_leq", "variant": <variant part>, "max_extra": X}   then = exchange, else = nothing
  {"type": "pending", "order_id"}                      then = cancel, else = return
plus "persona" (appended to task_instructions by code) and "hide_order_ids"
(the customer names items, not order ids; the agent has to find the orders).
solve(compound) -> list of gold writes; SpecError if a part does not resolve.
"""
import copy
import json
import random
import re

from ask2distill.tau2.envside import pref_solver as PS
from ask2distill.tau2.envside import scenario_prompt as SP

PERSONAS = [None, None, None,
            "You are terse and impatient: answer only what the agent asks, in as few words as possible.",
            "You do not volunteer information; wait until the agent asks for each detail.",
            "You are polite but in a hurry, and you want everything handled in this one call.",
            "You are a little confused at first and need the agent to confirm each change before it is made."]
REASONS = {"no longer needed": "you no longer need it", "ordered by mistake": "you ordered it by mistake"}


class SpecError(PS.SpecError):
    pass


def _db():
    return SP._load_db("retail")


def _price(pid, vid):
    return _db()["products"][pid]["variants"][vid]["price"]


# ---- solving ----------------------------------------------------------------------
def _solve_part(p):
    """-> gold write dict, or None (a branch that does nothing)."""
    if p is None:
        return None
    k = p["kind"]
    db = _db()
    if k == "variant":
        return PS.solve_variant({**p, "kind": "variant_pref"})
    if k == "return":
        o = db["orders"][p["order_id"]]
        if o["status"] != "delivered":
            raise SpecError("return on a non-delivered order")
        return {"name": "return_delivered_order_items",
                "arguments": {"order_id": p["order_id"], "item_ids": list(p["item_ids"]),
                              "payment_method_id": p["payment_method_id"]}}
    if k == "cancel":
        if db["orders"][p["order_id"]]["status"] != "pending":
            raise SpecError("cancel on a non-pending order")
        return {"name": "cancel_pending_order", "arguments": {"order_id": p["order_id"], "reason": p["reason"]}}
    if k == "address":
        if db["orders"][p["order_id"]]["status"] != "pending":
            raise SpecError("address change on a non-pending order")
        return {"name": "modify_pending_order_address", "arguments": {"order_id": p["order_id"], **p["address"]}}
    if k == "cond":
        c = p["cond"]
        if c["type"] == "available":
            try:
                g = _solve_part(c["variant"])
                return g
            except PS.SpecError:
                return _solve_part(p["else"])
        if c["type"] == "price_leq":
            g = _solve_part(c["variant"])
            o = db["orders"][c["variant"]["order_id"]]
            it = next(i for i in o["items"] if i["item_id"] == c["variant"]["item_id"])
            extra = _price(it["product_id"], g["arguments"]["new_item_ids"][0]) - it["price"]
            if abs(extra - c["max_extra"]) < 1.0:
                raise SpecError("price too close to the threshold")
            return g if extra <= c["max_extra"] else _solve_part(p["else"])
        if c["type"] == "pending":
            return _solve_part(p["then"] if db["orders"][c["order_id"]]["status"] == "pending" else p["else"])
    raise SpecError(f"unknown part {k}")


def _orders_of(p):
    if p is None:
        return set()
    if p["kind"] == "cond":
        c = p["cond"]
        s = {c["order_id"]} if c["type"] == "pending" else {c["variant"]["order_id"]}
        return s | _orders_of(p.get("then")) | _orders_of(p.get("else"))
    return {p["order_id"]}


def solve(spec):
    """Any malformed plan (the synthesizer can edit it) is a SpecError, never a crash:
    v7 step 1-3 logged 36 score_error rows from KeyError('variant' / 'order_id')."""
    try:
        return _solve(spec)
    except PS.SpecError:
        raise
    except Exception as e:
        raise SpecError(f"malformed plan: {type(e).__name__}: {str(e)[:40]}")


def _solve(spec):
    if not isinstance(spec, dict) or spec.get("kind") != "compound" or not spec.get("parts"):
        raise SpecError("not a compound spec")
    seen = set()
    for p in spec["parts"]:
        o = _orders_of(p)
        if o & seen:
            raise SpecError("two parts touch the same order")
        seen |= o
    try:
        writes = [g for g in (_solve_part(p) for p in spec["parts"]) if g]
    except PS.SpecError:
        raise
    except Exception as e:
        raise SpecError(f"{type(e).__name__}: {str(e)[:60]}")
    if not writes:
        raise SpecError("no part writes anything")
    return writes


# ---- canonical gold ---------------------------------------------------------------
def canonical_actions(spec, writes):
    db = _db()
    uid = spec["user_id"]
    acts = [{"name": "get_user_details", "arguments": {"user_id": uid}}]
    for oid in sorted({o for p in spec["parts"] for o in _orders_of(p)}):
        acts.append({"name": "get_order_details", "arguments": {"order_id": oid}})
        for it in db["orders"][oid]["items"]:
            acts.append({"name": "get_product_details", "arguments": {"product_id": it["product_id"]}})
    seen, out = set(), []
    for a in acts:                         # drop duplicate product lookups
        k = json.dumps(a, sort_keys=True)
        if k not in seen:
            seen.add(k); out.append(a)
    return out + writes


# ---- words ------------------------------------------------------------------------
def _item(oid, iid):
    return next(i for i in _db()["orders"][oid]["items"] if i["item_id"] == iid)


def _order_ref(oid, spec):
    o = _db()["orders"][oid]
    names = ", ".join(sorted({i["name"] for i in o["items"]}))
    return f"the order with the {names}" if spec.get("hide_order_ids") else f"order {oid}"


def _pm_words(pmid):
    u = next(u for u in _db()["users"].values() if pmid in (u.get("payment_methods") or {}))
    pm = u["payment_methods"][pmid]
    if pm.get("source") == "credit_card":
        return f"the credit card ending {pm.get('last_four')}"
    return {"paypal": "PayPal", "gift_card": "the gift card"}.get(pm.get("source"), pmid)


def _v_words(v, spec):
    it = _item(v["order_id"], v["item_id"])
    w = ", ".join(f"{a}: {b}" for a, b in v["want"].items())
    s = f"{'exchange' if v['action'].startswith('exchange') else 'change'} the {it['name']} in {_order_ref(v['order_id'], spec)} for one with {w}"
    if v.get("fallback"):
        s += " (if that is not available: " + ", ".join(f"{a}: {b}" for a, b in v["fallback"].items()) + ")"
    return s + f", settling any difference with {_pm_words(v['payment_method_id'])}"


def verbalize_part(p, spec):
    if p is None:
        return "do nothing"
    k = p["kind"]
    if k == "variant":
        return _v_words(p, spec)
    if k == "return":
        names = ", ".join(_item(p["order_id"], i)["name"] for i in p["item_ids"])
        return f"return the {names} from {_order_ref(p['order_id'], spec)}, refund to {_pm_words(p['payment_method_id'])}"
    if k == "cancel":
        return f"cancel {_order_ref(p['order_id'], spec)} because {REASONS[p['reason']]}"
    if k == "address":
        a = p["address"]
        return f"ship {_order_ref(p['order_id'], spec)} to {a['address1']}, {a['address2']}, {a['city']}, {a['state']} {a['zip']}, {a['country']}"
    c = p["cond"]
    if c["type"] == "available":
        return f"IF possible {_v_words(c['variant'], spec)}; OTHERWISE {verbalize_part(p['else'], spec)}"
    if c["type"] == "price_leq":
        return f"{_v_words(c['variant'], spec)} ONLY IF it costs at most ${c['max_extra']} more; OTHERWISE {verbalize_part(p['else'], spec)}"
    return (f"IF {_order_ref(c['order_id'], spec)} has not shipped yet, {verbalize_part(p['then'], spec)}; "
            f"IF it was already delivered, {verbalize_part(p['else'], spec)}")


def verbalize(spec):
    return [verbalize_part(p, spec) for p in spec["parts"]]


# ---- enumeration ------------------------------------------------------------------
def _variant_parts(o, u, rng, action):
    db = _db()
    pms = [m["id"] for m in (u.get("payment_methods") or {}).values()
           if m.get("source") != "gift_card" or (m.get("balance") or 0) >= 1000] or list(u.get("payment_methods") or {})
    out = []
    for it in o["items"]:
        var = db["products"].get(it["product_id"], {}).get("variants", {})
        cur = it.get("options") or {}
        for vid, v in var.items():
            if vid == it["item_id"]:
                continue
            diff = {k: val for k, val in v["options"].items() if cur.get(k) != val}
            if not diff:
                continue
            out.append({"kind": "variant", "order_id": o["order_id"], "item_id": it["item_id"],
                        "item_name": it["name"], "want": diff, "fallback": None, "action": action,
                        "payment_method_id": rng.choice(pms), "_available": bool(v.get("available"))})
    rng.shuffle(out)
    return out


def _clean(p):
    p = {k: v for k, v in p.items() if not k.startswith("_")}
    return p


def _parts_for_order(o, u, rng):
    """Candidate parts (incl. conditionals) for one order."""
    pms = [m["id"] for m in (u.get("payment_methods") or {}).values()]
    # refunds must go back to the order's original payment method (tool check)
    orig = [(ph.get("payment_method_id")) for ph in o.get("payment_history", []) if ph.get("transaction_type") == "payment"][:1] \
        or [ph.get("payment_method_id") for ph in o.get("payment_history", [])][:1] or pms[:1]
    st = o["status"]
    items = o["items"]
    cands = []
    if st == "delivered":
        ret = {"kind": "return", "order_id": o["order_id"],
               "item_ids": [i["item_id"] for i in rng.sample(items, rng.randint(1, min(2, len(items))))],
               "payment_method_id": orig[0]}
        cands.append(ret)
        vs = _variant_parts(o, u, rng, "exchange_delivered_order_items")
        av = [v for v in vs if v["_available"]]
        un = [v for v in vs if not v["_available"]]
        if av:
            cands.append(_clean(av[0]))
            ret1 = {**ret, "item_ids": [av[0]["item_id"]]}
            cands.append({"kind": "cond", "cond": {"type": "available", "variant": _clean(av[0])}, "then": None, "else": ret1})
            db = _db()
            it = _item(o["order_id"], av[0]["item_id"])
            extra = db["products"][it["product_id"]]["variants"][PS.solve_variant({**_clean(av[0]), "kind": "variant_pref"})["arguments"]["new_item_ids"][0]]["price"] - it["price"] \
                if _ok_variant(av[0]) else None
            if extra is not None:
                x = max(5, int(round(extra + rng.choice([-1, 1]) * rng.randint(10, 60), -1)))
                cands.append({"kind": "cond", "cond": {"type": "price_leq", "variant": _clean(av[0]), "max_extra": x},
                              "then": None, "else": None})
        if un:
            ret1 = {**ret, "item_ids": [un[0]["item_id"]]}
            cands.append({"kind": "cond", "cond": {"type": "available", "variant": _clean(un[0])}, "then": None, "else": ret1})
    if st == "pending":
        reason = rng.choice(list(REASONS))
        cands.append({"kind": "cancel", "order_id": o["order_id"], "reason": reason})
        vs = [v for v in _variant_parts(o, u, rng, "modify_pending_order_items") if v["_available"]]
        if vs:
            cands.append(_clean(vs[0]))
        a = u.get("address") or {}
        if a:
            new = {**a, "address1": f"{rng.randint(100, 999)} {rng.choice(['Oak', 'Maple', 'Cedar', 'Pine', 'Elm'])} {rng.choice(['Street', 'Avenue', 'Lane'])}",
                   "address2": f"Suite {rng.randint(100, 999)}"}
            cands.append({"kind": "address", "order_id": o["order_id"], "address": new})
    if st in ("pending", "delivered") and items:
        # Always a real branch (cancel vs return): "cancel if pending, else do nothing"
        # on a pending order was voiced as a plain cancel -- same gold -- and the v7
        # probe rejected 21 of 55 cards for the missing "if". Both branches now act,
        # so the agent must read the status to get it right.
        cands.append({"kind": "cond", "cond": {"type": "pending", "order_id": o["order_id"]},
                      "then": {"kind": "cancel", "order_id": o["order_id"], "reason": rng.choice(list(REASONS))},
                      "else": {"kind": "return", "order_id": o["order_id"], "item_ids": [items[0]["item_id"]],
                               "payment_method_id": orig[0]}})
    return cands


def _ok_variant(v):
    try:
        PS.solve_variant({**_clean(v), "kind": "variant_pref"})
        return True
    except PS.SpecError:
        return False


def enumerate_specs(rec, rng, k=3):
    u = rec["user"]
    db = _db()
    orders = [db["orders"][o] for o in (u.get("orders") or []) if o in db["orders"]
              and db["orders"][o]["status"] in ("pending", "delivered")]
    if not orders:
        return []
    per = {o["order_id"]: _parts_for_order(o, u, rng) for o in orders}
    out, tries = [], 0
    while len(out) < k and tries < 60:
        tries += 1
        n = rng.choices([1, 2, 3], weights=[35, 45, 20])[0]
        n = min(n, len(orders))
        chosen = rng.sample(orders, n)
        parts = []
        for o in chosen:
            if not per[o["order_id"]]:
                break
            parts.append(copy.deepcopy(rng.choice(per[o["order_id"]])))
        else:
            spec = {"kind": "compound", "user_id": u["user_id"], "parts": parts,
                    "hide_order_ids": rng.random() < 0.4, "persona": rng.choice(PERSONAS)}
            try:
                w = solve(spec)
            except PS.SpecError:
                continue
            if any(json.dumps(spec["parts"], sort_keys=True) == json.dumps(s["parts"], sort_keys=True) for s, _ in out):
                continue
            out.append((spec, w))
    return out


# ---- checks -----------------------------------------------------------------------
COND_RX = re.compile(r"\bif\b|\botherwise\b|\bunless\b|\bonly if\b|\bin case\b|\bas long as\b", re.I)


def unstated(spec, text, _all=True):
    """(missing things, number of checks) for a compound card."""
    t = text.lower()
    db = _db()
    miss, tot = [], 0
    def need(name, ok):
        nonlocal tot
        tot += 1
        if not ok:
            miss.append(name)
    def part(p):
        if p is None:
            return
        k = p["kind"]
        if k == "variant":
            need(f"item:{p['item_name']}", p["item_name"].lower().split()[-1] in t)
            for a, v in list(p["want"].items()) + list((p.get("fallback") or {}).items()):
                v = str(v).lower()
                need(f"option:{a}={v}", (a.lower().split()[0] in t) if v in ("yes", "no", "none", "true", "false") else v in t)
        elif k == "return":
            for i in p["item_ids"]:
                nm = _item(p["order_id"], i)["name"].lower()
                need(f"return:{nm}", nm.split()[-1] in t)
            need("return:refund", re.search(r"refund|money back|back to|return", t) is not None)
        elif k == "cancel":
            need("cancel", re.search(r"cancel", t) is not None)
            need("cancel:reason", re.search(r"no longer need|don'?t need|not need|mistake|by accident|wrong", t) is not None)
        elif k == "address":
            need("address:zip", str(p["address"]["zip"]) in t)
            need("address:street", p["address"]["address1"].lower().split()[0] in t)
        elif k == "cond":
            need("condition", COND_RX.search(t) is not None)
            c = p["cond"]
            if c["type"] in ("available", "price_leq"):
                part(c["variant"])
            if c["type"] == "price_leq":
                need("condition:amount", re.search(rf"\$?\s?{c['max_extra']}\b", t) is not None)
            part(p.get("then")); part(p.get("else"))
    for p in spec["parts"]:
        part(p)
    if spec.get("hide_order_ids"):
        leaked = [o for p in spec["parts"] for o in _orders_of(p) if o.lower() in t]
        need("hide_order_ids", not leaked)
    return (miss, tot) if _all else (miss[0] if miss else None)


def with_persona(scenario, spec):
    sc = dict(scenario or {})
    if spec.get("persona"):
        ti = str(sc.get("task_instructions") or "").strip()
        sc["task_instructions"] = (ti + " " if ti else "") + spec["persona"]
    return sc
