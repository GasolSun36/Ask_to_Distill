"""HTTP service that owns the tau2 environments and scores candidate scenarios.

Exists to keep two dependency sets apart. The reward has to run inside verl,
which runs in the a2d env, and tau2 needs loguru/litellm/pydantic-argparse that
the a2d env does not have. Installing them there would touch the environment
the math OPD runs depend on, so instead the environment lives behind HTTP and
the verl-side worker only needs a POST.

It also owns the vLLM endpoints for the student, the user simulator and the
teacher, so the reward worker stays free of rollout logic entirely.

Routes:

* ``GET /health``;
* ``POST /score`` -- synthesizer reward for a batch of parsed cards
  (``TAU2_REWARD_V`` selects the v4 / v5 / v6 / v7 scorer);
* ``POST /env/new``, ``/env/call``, ``/env/close`` -- one fresh environment per
  OPD episode, driven by the agent loop. ``/env/close`` reuses the ``/env/new``
  body model, so its ``domain`` field carries the session id.

Endpoints come from ``TAU2_{STUDENT,USER,TEACHER}_BASE`` (comma-separated base
URLs, one per replica) and the served names from ``TAU2_*_MODEL``; the OPD
service needs none of them (it only serves ``/env/*``)::

    python -m ask2distill.tau2.envside.env_service --port 8710 --workers 64

Started as a module with REPO_PATH required; the reward version may also be
given as ``--reward-v``; the app is built by ``create_app``, not at import.
"""
# No `from __future__ import annotations`: FastAPI resolves the request models of
# the route functions from their annotations, and the models are local to create_app.
import argparse
import json
import os
import random
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

REWARD_VERSIONS = ("4", "5", "6", "7")


class Pool:
    """Round-robin-by-chance over several replicas of one model.

    Each TAU2_*_BASE is a comma-separated list of base URLs. The reward only
    ever touches ``client.chat.completions.create``, so exposing ``chat`` from a
    random member is enough to spread load without changing the reward code.
    Before this, only the first replica per model was ever used.
    """

    def __init__(self, bases: str) -> None:
        from openai import OpenAI
        self.clients = [OpenAI(api_key="EMPTY", base_url=b, timeout=900)
                        for b in bases.split(",") if b.strip()]
        assert self.clients, f"no endpoints in {bases!r}"

    @property
    def chat(self):
        return random.choice(self.clients).chat


def _pool(var: str) -> Pool | None:
    return Pool(os.environ[var]) if os.environ.get(var) else None


def scorer_module(reward_v: str) -> ModuleType:
    """The scoring module for a reward version (v5+ wrap the earlier ones)."""
    if reward_v not in REWARD_VERSIONS:
        raise ValueError(f"TAU2_REWARD_V must be one of {REWARD_VERSIONS}, got {reward_v!r}")
    from ask2distill.tau2.envside import reward_base
    if reward_v == "7":
        from ask2distill.tau2.envside import reward_v7 as mod
    elif reward_v == "6":
        from ask2distill.tau2.envside import reward_v6 as mod
    elif reward_v == "5":
        from ask2distill.tau2.envside import reward_v5 as mod
    else:
        mod = reward_base
    return mod


def create_app(reward_v: str, workers: int):
    """The FastAPI app with the scorer for ``reward_v`` and a ``workers``-thread pool."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    from ask2distill.tau2.envside import reward_base as TR
    scorer = scorer_module(reward_v)
    from tau2.registry import registry as _REG

    STU = _pool("TAU2_STUDENT_BASE")
    USR = _pool("TAU2_USER_BASE")
    STU_MODEL = os.environ.get("TAU2_STUDENT_MODEL", "Qwen3-4B")
    USR_MODEL = os.environ.get("TAU2_USER_MODEL", "Qwen3-4B-Instruct-2507")
    # Teacher endpoint, only for the solvability gate (TAU2_TEACHER_GATE=1).
    TEA = _pool("TAU2_TEACHER_BASE")
    TEA_MODEL = os.environ.get("TAU2_TEACHER_MODEL", "Qwen3-4B-Thinking-2507")
    POOL = ThreadPoolExecutor(workers)

    app = FastAPI()

    class Batch(BaseModel):
        cands: list          # [{domain, scenario, actions} | {domain, parse_error}]
        k: int = 8

    @app.get("/health")
    def health():
        return {"ok": True, "student": STU_MODEL, "user": USR_MODEL,
                "replicas": {"stu": len(STU.clients) if STU else 0, "usr": len(USR.clients) if USR else 0,
                             "tea": len(TEA.clients) if TEA else 0},
                "teacher": TEA_MODEL if TEA else None, "teacher_gate": TR.TEACHER_GATE,
                "reward_v": reward_v}

    @app.post("/score")
    def score(b: Batch):
        def one(c):
            # A malformed generation is a zero, not an error: the synthesizer has to
            # learn to emit parseable JSON, and crashing the batch would stall training.
            if not isinstance(c, dict) or not isinstance(c.get("scenario"), dict) \
                    or not isinstance(c.get("actions"), list):
                return {"R": 0.0, "rejected": "wrong_shape"}
            try:
                return scorer.score(c, STU, STU_MODEL, USR, USR_MODEL, k=b.k,
                                    teacher_client=TEA, teacher_model=TEA_MODEL)
            except Exception as e:
                return {"R": 0.0, "rejected": f"score_error:{str(e)[:80]}"}
        return {"results": list(POOL.map(one, b.cands))}

    # ---- environment sessions, for the OPD agent loop ---------------------------
    # The agent loop runs inside verl (a2d env), which cannot import tau2; it drives
    # a per-episode environment here instead. One session = one fresh environment,
    # because tools write to a database that episodes must not share.
    sessions: dict = {}
    lock = threading.Lock()

    class EnvNew(BaseModel):
        domain: str

    class EnvCall(BaseModel):
        sid: str
        name: str
        args: dict = {}

    @app.post("/env/new")
    def env_new(b: EnvNew):
        sid = uuid.uuid4().hex
        env = _REG.get_env_constructor(b.domain)()
        with lock:
            sessions[sid] = env
        return {"sid": sid}

    @app.post("/env/call")
    def env_call(b: EnvCall):
        env = sessions.get(b.sid)
        if env is None:
            return {"error": "unknown session"}
        try:
            out = env.use_tool(b.name, **(b.args or {}))
            return {"out": out if isinstance(out, str) else json.dumps(out, default=str)}
        except Exception as e:
            return {"out": f"Error: {e}"}

    @app.post("/env/close")
    def env_close(b: EnvNew):          # reuses the model; `domain` carries the sid
        with lock:
            sessions.pop(b.domain, None)
        return {"ok": True}

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ask2distill.tau2.envside.env_service")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True,
                    help="scoring threads (32 for v4/v5, 64 for v6/v7 and OPD in the final runs)")
    ap.add_argument("--reward-v", default=None, help="overrides $TAU2_REWARD_V")
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args(argv)
    reward_v = a.reward_v or os.environ.get("TAU2_REWARD_V", "4")
    import uvicorn
    app = create_app(str(reward_v), a.workers)
    print(f"[tau2-svc] reward_v={reward_v} student={os.environ.get('TAU2_STUDENT_BASE')} "
          f"user={os.environ.get('TAU2_USER_BASE')} teacher={os.environ.get('TAU2_TEACHER_BASE')} "
          f"port={a.port} workers={a.workers}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
