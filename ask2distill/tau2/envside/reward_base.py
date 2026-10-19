"""Score a generated scenario: does the student fail to reach the state its gold actions produce?

    R = [gold actions executable] x [gold changes the DB] x (fraction of student
                                                             rollouts ending elsewhere)

Each factor is an environment observable, so nothing here needs a human label.
The first two are the gate that replaces ω_t: a fabricated id makes the action
list throw, and a read-only scenario leaves the hash untouched, so neither can be
rewarded. ω_t itself is nearly constant in this setting and gates nothing.

The rollout is written out here rather than driven through tau2's orchestrator
because the reward needs the environment instance the student mutated, to read
its hash afterwards; the orchestrator owns its env and does not hand it back.
Protocol is kept identical to the benchmark: scripted greeting, the user speaks
first, the agent may call tools or reply, the episode ends on ###STOP### or a
turn cap.

The student and the user simulator are separate OpenAI-compatible endpoints. The
user simulator must never be the model being trained -- a simulator that drifts
with the policy stops being a fixed measuring stick.

REPO_PATH and TAU2_CTX are required; the user guidelines default to the
fork's own file.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from ask2distill.tau2.envside import repo, settings

REPO = repo.setup()

from tau2.registry import registry  # noqa: E402  (needs the fork on sys.path)

CTX = json.load(open(settings.require("TAU2_CTX")))
# The fork's guidelines differ from upstream tau2-bench; every stage used the fork's.
GUIDE = open(os.environ.get(
    "TAU2_USER_GUIDELINES", f"{REPO}/data/tau2/user_simulator/simulation_guidelines.md")).read()
READ = re.compile(r"^(get_|find_|search_|list_)|^calculate$")
GREETING = "Hi! How can I help you today?"
MAX_TURNS = int(os.environ.get("TAU2_MAX_TURNS", 12))
GATE_TURNS = int(os.environ.get("TAU2_GATE_TURNS", 10))   # turn cap for teacher-gate episodes
STOP_TOKENS = ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###")

_BASE_HASH = {}


def base_hash(domain):
    if domain not in _BASE_HASH:
        _BASE_HASH[domain] = registry.get_env_constructor(domain)().get_db_hash()
    return _BASE_HASH[domain]


GROUNDING = os.environ.get("TAU2_GROUNDING", "0") == "1"
# Free-text arguments the customer phrases in their own words; everything else
# is an identifier or a value the agent must be told or must look up.
FREE_TEXT_ARGS = {"reason", "summary", "expression"}


# Grounding gate (TAU2_GROUNDING=1). By step 36 the synthesizer had learned to
# leave every identifier out of known_info -- "cannot recall the reservation
# ID", no name, no email -- while gold went straight to
# get_user_details(user_id=...). The simulator may not invent what it was not
# told, so the student can never authenticate and every such card scores 1.0
# without being hard. A value is grounded if it appears in the card text or in
# the output of an EARLIER gold action (a user_id returned by
# find_user_id_by_email is fair game); the check runs while gold executes.
# Short values (<4 chars) and free-text fields are exempt.


def _leaves(v):
    """Scalar leaves of a possibly nested argument (dict keys are schema, not values)."""
    if isinstance(v, dict):
        for x in v.values():
            yield from _leaves(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _leaves(x)
    else:
        yield v


def gold(domain, actions, card_text=""):
    """Execute the gold action list. Returns (hash, reason_rejected_or_None)."""
    if not actions:
        return None, "no_actions"
    if not any(not READ.match(a.get("name", "")) for a in actions):
        return None, "read_only"          # hash could not move, so R is undefined
    env = registry.get_env_constructor(domain)()
    seen = card_text.lower()

    def grounded(args):
        for k, v in (args or {}).items():
            if k in FREE_TEXT_ARGS:
                continue
            for x in _leaves(v):
                xs = str(x).strip().lower()
                if len(xs) >= 4 and xs not in seen:
                    return f"{k}={xs[:24]}"
        return None

    # Order-independent grounding for lookups. v4 step 1 rejected every item
    # modification because gold called get_product_details before
    # get_order_details, whose output holds the product_id -- a real agent can
    # look things up in any order. Reads whose arguments are grounded run first,
    # repeatedly, each output extending what is known; a card that never states
    # who the customer is still fails, since its first lookup has no source.
    if GROUNDING:
        # What an agent could look up from identifiers the customer states:
        # order / reservation / user ids and emails in the card text, expanded
        # through the lookups that take them, to a fixed point. Gold's own lookup
        # order is not a constraint on the agent.
        probe = registry.get_env_constructor(domain)()
        LOOKUPS = [(re.compile(r"#w\d{7}"), "get_order_details", "order_id"),
                   (re.compile(r"\b[a-z]+_[a-z]+_\d{4}\b"), "get_user_details", "user_id"),
                   (re.compile(r"[\w.]+@example\.com"), "find_user_id_by_email", "email"),
                   (re.compile(r"\b[0-9a-z]{6}\b"), "get_reservation_details", "reservation_id"),
                   (re.compile(r"\b\d{10}\b"), "get_product_details", "product_id")]
        tried = set()
        for _ in range(4):
            grew = False
            for rx, tool, arg in LOOKUPS:
                for m in set(rx.findall(seen)):
                    if (tool, m) in tried:
                        continue
                    tried.add((tool, m))
                    val = m.upper() if arg == "reservation_id" else ("#W" + m[2:] if arg == "order_id" else m)
                    try:
                        out = probe.make_tool_call(tool_name=tool, requestor="assistant", **{arg: val})
                        seen += " " + (out if isinstance(out, str) else json.dumps(out, default=str)).lower()
                        grew = True
                    except Exception:
                        pass
            if not grew:
                break
        pending = [a for a in actions if READ.match(a.get("name", ""))]
        progress = True
        while pending and progress:
            progress = False
            for a in list(pending):
                if grounded(a.get("arguments")) is None:
                    try:
                        out = registry.get_env_constructor(domain)().make_tool_call(
                            tool_name=a["name"], requestor=a.get("requestor", "assistant"),
                            **(a.get("arguments") or {}))
                        seen += " " + (out if isinstance(out, str) else json.dumps(out, default=str)).lower()
                    except Exception:
                        pass
                    pending.remove(a); progress = True
    for a in actions:
        args = a.get("arguments") or {}
        if GROUNDING:
            miss = grounded(args)
            if miss:
                return None, f"ungrounded:{a.get('name')}:{miss}"
        try:
            out = env.make_tool_call(tool_name=a["name"], requestor=a.get("requestor", "assistant"),
                                     **args)
            seen += " " + (out if isinstance(out, str) else json.dumps(out, default=str)).lower()
        except Exception as e:
            return None, f"action_failed:{a.get('name')}:{str(e)[:60]}"
    h = env.get_db_hash()
    if h == base_hash(domain):
        return None, "db_unchanged"       # actions ran but wrote nothing observable
    return h, None


def _chat(client, model, messages, max_tokens, tools=None):
    kw = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=1.0)
    if tools:
        kw["tools"] = tools
    return client.chat.completions.create(**kw)


def rollout(domain, scenario, student_client, student_model, user_client, user_model, max_turns=None):
    """One agent episode. Returns (final_db_hash, n_turns)."""
    max_turns = max_turns or MAX_TURNS
    env = registry.get_env_constructor(domain)()
    system, tools = CTX[domain]["system"], CTX[domain]["tools"]
    sysu = f"{GUIDE}\n\n<scenario>\n{json.dumps(scenario, ensure_ascii=False)}\n</scenario>"
    dialogue = [{"role": "assistant", "content": GREETING}]

    def user_reply():
        flipped = [{"role": "user" if m["role"] == "assistant" else "assistant",
                    "content": m.get("content") or ""} for m in dialogue]
        r = _chat(user_client, user_model,
                  [{"role": "system", "content": sysu}] + flipped, 512)
        return (r.choices[0].message.content or "").strip()

    dialogue.append({"role": "user", "content": user_reply()})
    msgs = [{"role": "system", "content": system}] + dialogue
    for turn in range(max_turns):
        r = _chat(student_client, student_model, msgs, 2048, tools=tools)
        m = r.choices[0].message
        if m.tool_calls:
            msgs.append({"role": "assistant", "content": m.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in m.tool_calls]})
            for tc in m.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    res = env.use_tool(tc.function.name, **args)
                    out = res if isinstance(res, str) else json.dumps(res, default=str)
                except Exception as e:
                    out = f"Error: {e}"
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": out[:2000]})
            continue
        content = m.content or ""
        msgs.append({"role": "assistant", "content": content})
        dialogue.append({"role": "assistant", "content": content})
        reply = user_reply()
        if any(s in reply for s in STOP_TOKENS):
            break
        dialogue.append({"role": "user", "content": reply})
        msgs.append({"role": "user", "content": reply})
    return env.get_db_hash(), turn + 1


TEACHER_GATE = os.environ.get("TAU2_TEACHER_GATE", "0") == "1"
K_T = int(os.environ.get("TAU2_KT", 3))


def score(cand, student_client, student_model, user_client, user_model, k=4,
          teacher_client=None, teacher_model=None):
    """cand = {domain, scenario, actions} -> a record with R and why, if rejected.

    With TAU2_TEACHER_GATE=1 the card must also be solvable FROM THE CARD: the
    teacher gets up to K_T episodes and at least one must end at the gold state.
    Gold still defines correctness; the teacher is only an existence proof. This
    closes the under-specification exploit the grounding gate cannot see -- by
    step 50 of an early synthesizer two thirds of valid cards said "add additional baggage" while gold
    wrote total_baggages=5 with a payment method the customer never named, so no
    agent could match the hash and the student "failed" without the task being
    hard. If the teacher cannot reach gold either, the card did not state the
    task. The reward becomes regret: teacher can, student cannot.
    """
    dom = cand["domain"]
    tgt = cand.get("target")
    if tgt and not any(isinstance(a, dict) and a.get("name") == tgt for a in cand.get("actions") or []):
        return {"R": 0.0, "rejected": f"off_target:{tgt}", "domain": dom}
    gh, why = gold(dom, cand.get("actions"),
                   card_text=json.dumps(cand.get("scenario", {}), ensure_ascii=False))
    if gh is None:
        return {"R": 0.0, "rejected": why, **{k2: cand[k2] for k2 in ("domain",) if k2 in cand}}
    # Rollouts of one card run concurrently (each builds its own environment, so
    # threads do not share state). Serially, a card that fails the teacher gate
    # cost K_T full episodes back to back -- 5 x ~70s -- and with 128 cards over
    # 24 service threads a step took 31 minutes. Concurrency trades the early
    # stop on the first teacher success for a ~K_T-fold cut in wall time.
    def many(n, client, model, cap=None):
        with ThreadPoolExecutor(n) as ex:
            return list(ex.map(lambda _: rollout(dom, cand["scenario"], client, model,
                                                 user_client, user_model, max_turns=cap), range(n)))
    # Student first. Solvability only needs ONE agent to reach gold, and the
    # student's k episodes are computed anyway: if any of them lands on gold the
    # card is solvable and the teacher is not consulted at all. Only cards the
    # student fails k/k go to the teacher for the unsolvable-vs-hard ruling. In
    # the previous order every gated card paid K_T teacher episodes up front and
    # 79% of them were then rejected -- 20 minutes a step.
    try:
        outs = many(k, student_client, student_model)
    except Exception as e:
        return {"R": 0.0, "rejected": f"rollout_error:{str(e)[:60]}", "domain": dom}
    fails = sum(1 for h, _ in outs if h != gh)
    turns = [nt for _, nt in outs]
    t_ok = None
    if TEACHER_GATE and teacher_client is not None:
        if fails < k:
            t_ok = -1                                   # student proved it solvable
        else:
            try:
                touts = many(K_T, teacher_client, teacher_model, cap=GATE_TURNS)
            except Exception as e:
                return {"R": 0.0, "rejected": f"rollout_error:{str(e)[:60]}", "domain": dom}
            t_ok = int(any(h == gh for h, _ in touts))
            if not t_ok:
                return {"R": 0.0, "rejected": "teacher_unsolvable", "domain": dom}
    return {"R": fails / k, "rejected": None, "domain": dom, "k": k,
            "student_fail": fails, "mean_turns": sum(turns) / len(turns),
            "teacher_ok": t_ok}
