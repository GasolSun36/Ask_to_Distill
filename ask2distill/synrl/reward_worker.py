"""Reward worker for synthesizer RL (math and code): the frozen student π_S and teacher π_T.

verl's actor here is the question synthesizer S_θ. Scoring one of its questions
is a two-stage pipeline -- the student must first write a solution, then the
teacher is teacher-forced on that exact prefix -- which does not fit a plain
reward model. It does fit a worker: ``reward_model.enable_resource_pool`` puts
this worker in its own pool on its own GPUs, so student and teacher live apart
from the model being trained. Both stay frozen; only S_θ is updated.

Reward of one question q (paper Eq. 6-11; ``common.divergence`` for δ_t, ω_t)::

    s_t      = min(δ_t / (δ_ref + 1e-8), 1) * ω_t
    r_roll   = sum_t s_t / L**α * (1 - λ_len * min(L / L_max, 1))
    R_base   = mean of r_roll over the rollouts that were scored (see below)
    R(q)     = R_base * max(0, 1 - λ_div * share(q))

with share(q) the size of q's cluster over the scoring batch
(``reward.diversity.mode``):

* ``bleu`` (default; paper §2.3): 1 - sentence BLEU between question texts,
  average-linkage clustering at τ over the FULL batch of one synthesizer step
  (all ``train_batch_size x rollout_n`` questions, gathered across the reward
  ranks; ``synrl.diversity``). No embedding model is loaded.
* ``emb``: cosine distance of Qwen3-Embedding-4B embeddings, clustered per
  ``diversity.group`` consecutive rows of the rank batch (see below).

Configured values: math α 0.5, λ_len 0.3, L_max 7168, 1 rollout; code α 1.0,
λ_len 0, L_max 4096, 2 rollouts, truncated rollouts dropped; τ 0.1, λ_div 1.0.

R(q) is returned as verl's token-level ``rm_scores`` with the scalar on the
last valid response token: GRPO sums token_level_rewards over the sequence, so
a single position carries the whole outcome reward.

Details that are deliberate (docs/DEVIATIONS.md):

* A question shorter than ``min_question_chars`` (20) after stripping a
  thinking block is not solved and scores 0.
* A rollout shorter than ``min_rollout_tokens`` (10) gives ``(0.0, {})`` and is
  left out of the mean; under ``trunc_zero`` so is a truncated rollout. A
  question none of whose rollouts survive scores 0.
* Truncation is "no ``tok.eos_token_id`` in the generated ids". Qwen3 / Coder
  ``generation_config`` also stops on ``<|endoftext|>`` (151643), so a rollout
  ending there counts as truncated (and is dropped under ``trunc_zero``).
  ``fix_eos: true`` checks every stop id instead; it is off for reproduction.
* HF ``generate`` gets only temperature / top_p (plus top_k when
  ``sample_top_k > 0``), never ``repetition_penalty``: the model's
  ``generation_config`` supplies the rest (Qwen3 / Coder top_k 20, R1-Distill
  the HF default 50, Coder-3B ``repetition_penalty`` 1.05).
* The diversity share covers ALL rows, including ones too short to solve.
  In ``emb`` mode it is computed per ``diversity.group`` consecutive rows of
  the rank batch, each group embedded in its own call.
* ``self.step`` is a per-process counter that restarts at 1 in every job
  window; ``select.pick`` undoes that with ``save_freq``.
* Dump and metrics rows keep the original format, including ``n_docs`` (the
  FRAMES document count, always 0 here).

Configuration comes ONLY from the JSON file named by ``$A2D_REWARD_CONFIG``
(the resolved ``reward`` section, written by ``synrl.launch``; see
``synrl.reward_config``); every key is required, so no code default can leak
into a run.
"""
from __future__ import annotations

import os

# Set BEFORE the first CUDA allocation in this process. The reward worker runs in
# its own processes (reward_model.enable_resource_pool gives it a separate pool)
# and holds no vLLM engine, so it can use expandable segments -- while the
# synthesizer's rollout, which does hold one, must not: vLLM's CuMemAllocator
# asserts against exactly this setting. Setting it job-wide killed that rollout.
#
# It is needed because scoring alternates variable-length generation KV with two
# variable-length [L, vocab] logit tensors: Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507 died at 72.70GB allocated with
# a further 6.09GB reserved-but-unallocated on an 80GB card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402

import torch  # noqa: E402
from verl import DataProto  # noqa: E402
from verl.single_controller.base import Worker  # noqa: E402
from verl.single_controller.base.decorator import Dispatch, register  # noqa: E402
from verl.utils.device import get_device_id  # noqa: E402

from ask2distill.common import divergence  # noqa: E402
from ask2distill.common import embed as emb  # noqa: E402
from ask2distill.common.text import jaccard, norm_text, strip_question  # noqa: E402
from ask2distill.synrl import diversity as bleu_div  # noqa: E402
from ask2distill.synrl.reward_config import check_reward_config, load_reward_config  # noqa: E402
from ask2distill.synrl.solve import render_solve_prompt, solve_template  # noqa: E402

# Jaccard threshold of the logged sem_uniq monitor (not a filter).
SEM_UNIQ_JACCARD = 0.6


class Scorer:
    """Student + teacher (+ embedder in ``emb`` mode), and everything that turns questions into rewards.

    Pure torch / transformers; the verl wrapper is ``SynRewardWorker``. Tests
    replace ``tok``, ``generate``, ``scored_logits`` and ``embedder`` with stubs.
    """

    def __init__(self, cfg: Mapping[str, Any], device: int | str) -> None:
        check_reward_config(cfg)
        self.cfg = dict(cfg)
        self.device = device
        self.solver_path = str(cfg["solver"])
        self.teacher_path = str(cfg["teacher"])
        self.embedder_path = str(cfg["embedder"])
        self.qwen3_nothink = bool(cfg["qwen3_nothink"])
        self.template = solve_template(cfg["solve_template"])
        self.solve_system = str(cfg["solve_system"])
        self.delta_ref = float(cfg["delta_ref"])
        self.topk = int(cfg["topk"])
        self.chunk = int(cfg["chunk"])
        self.use_omega = bool(cfg["use_omega"])
        self.len_alpha = float(cfg["len_alpha"])
        self.len_pen = float(cfg["len_pen"])
        self.max_new = int(cfg["max_new"])
        self.rolls = int(cfg["rolls"])
        self.temperature = float(cfg["temperature"])
        self.top_p = float(cfg["top_p"])
        self.sample_top_k = int(cfg["sample_top_k"])
        self.trunc_zero = bool(cfg["trunc_zero"])
        # 0 = one generate() call for the whole rank batch.
        self.gen_bs = int(cfg["gen_bs"])
        self.min_question_chars = int(cfg["min_question_chars"])
        self.min_rollout_tokens = int(cfg["min_rollout_tokens"])
        self.fix_eos = bool(cfg["fix_eos"])
        div = cfg["diversity"]
        self.div_mode = str(div["mode"])
        self.div_thresh = float(div["thresh"])
        self.div_pen = float(div["pen"])
        self.div_group = int(div["group"])
        self.div_max_len = int(div["max_len"])
        self.tok: Any = None
        self.solver: Any = None
        self.teacher: Any = None
        self.embedder: emb.SupportsEmbed | None = None
        # Mean pairwise similarity of the last scored batch (cosine in emb mode,
        # BLEU in bleu mode), logged as synrm/mean_sim; -1.0 until it has run once.
        self.mean_sim = -1.0

    # ------------------------------------------------------------------ models

    def load(self) -> None:
        """Load the tokenizer, student and teacher (bf16, sdpa) and, in ``emb`` mode, the embedder."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(self.solver_path, trust_remote_code=True)
        # decoder-only batch generation needs LEFT padding, otherwise the pads sit
        # between the prompt and the first generated token and every sample after
        # the longest one decodes from garbage
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.solver = self._frozen(AutoModelForCausalLM.from_pretrained(
            self.solver_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"))
        self.teacher = self._frozen(AutoModelForCausalLM.from_pretrained(
            self.teacher_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"))
        if self.div_mode == "emb":
            # "cuda" is the rank's current device
            self.embedder = emb.Embedder(self.embedder_path, device="cuda", max_len=self.div_max_len)

    def _frozen(self, model):
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def solve_prompt(self, question: str) -> str:
        """Chat-templated solve prompt, shared by student and teacher (tokenizer BOS stripped)."""
        return render_solve_prompt(self.tok, question, template=self.template,
                                   system=self.solve_system, qwen3_nothink=self.qwen3_nothink,
                                   strip_deepseek_bos=False)

    # -------------------------------------------------------------- generation

    def sampling_kwargs(self) -> dict:
        """Sampling arguments passed to ``generate`` besides temperature and top_p.

        top_k is only passed when asked for, so the default call is unchanged;
        HF's spelling of "no top-k" is 0. repetition_penalty is never passed:
        generate() fills every unpassed field from the solver's
        generation_config, and the final runs relied on that.
        """
        return {} if self.sample_top_k == 0 else {"top_k": max(self.sample_top_k, 0)}

    def stop_ids(self) -> set[int]:
        """Ids that end a rollout: ``tok.eos_token_id`` only, unless ``fix_eos``."""
        ids = {self.tok.eos_token_id}
        if self.fix_eos:
            e = getattr(getattr(self.solver, "generation_config", None), "eos_token_id", None)
            ids |= set(e if isinstance(e, (list, tuple)) else ([e] if e is not None else []))
        return ids

    def split_rollout(self, ids: list[int], stop: set[int]) -> tuple[list[int], bool]:
        """``(ids up to the first stop id, truncated?)`` for one generated sequence."""
        # No stop id anywhere means the rollout was still going when the budget
        # ran out -- that is the truncation flag, and it has to be read BEFORE
        # trimming, because trimming is what would hide it.
        cut = next((k for k, t in enumerate(ids) if t in stop), None)
        if cut is None:
            return ids, True
        return ids[:cut], False

    @torch.no_grad()
    def generate(self, questions: Sequence[str]) -> list[list[tuple[list[int], bool]]]:
        """``rolls`` solver rollouts per question: ``[[(ids, truncated), ...], ...]``.

        One padded batch instead of one call per question: the serial version
        spent ~58s per question decoding up to 2048 tokens on a 1.7B model
        (~35 tok/s), i.e. 7.8 min for a rank's 8 questions. Generation is >90% of
        reward time; the two scoring forwards are ~5%.
        """
        # A 30B/32B teacher plus the solver is ~65GB of weights on every reward
        # rank, so the generation KV cache is the only term left to control.
        # Chunking decouples it from the number of rollouts: without this, 8
        # questions x 2 rollouts x 4096 tokens put Code_Qwen3-4B_Qwen3-30B-A3B-Instruct-2507 at 78.7GB of 80.
        # Batch composition changes the sampled rollouts, so gen_bs is part of
        # the reproduced configuration.
        if self.gen_bs and len(questions) > self.gen_bs:
            out: list[list[tuple[list[int], bool]]] = []
            for i in range(0, len(questions), self.gen_bs):
                out += self.generate(questions[i:i + self.gen_bs])
            return out
        prompts = [self.solve_prompt(q) for q in questions]
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       add_special_tokens=True).to(self.device)
        gen = self.solver.generate(
            **enc, do_sample=True, temperature=self.temperature, top_p=self.top_p,
            max_new_tokens=self.max_new, num_return_sequences=self.rolls,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id, **self.sampling_kwargs())
        plen = enc["input_ids"].shape[1]
        stop = self.stop_ids()
        flat = [self.split_rollout(gen[i, plen:].tolist(), stop) for i in range(gen.shape[0])]
        # generate() lays n rollouts out contiguously per prompt
        return [flat[i * self.rolls:(i + 1) * self.rolls] for i in range(len(questions))]

    # ----------------------------------------------------------------- scoring

    @torch.no_grad()
    def scored_logits(self, ids: torch.Tensor, plen: int) -> tuple[torch.Tensor, torch.Tensor]:
        """S and T logits for the positions the score reads, and nothing else.

        The scoring loop only ever touches positions plen-1 .. L-2 -- it is
        predicting the response tokens. Asking for the full [L, vocab] head and
        then slicing throws most of it away (measured at L=10240: 71.2GB peak
        against 66.6GB restricted, on a card whose 64.4GB is already the two
        sets of weights). Returns tensors indexed from 0 == absolute plen-1.
        """
        keep = int(ids.shape[1] - plen + 1)
        ls = self.solver(ids, logits_to_keep=keep).logits[0]
        lt = self.teacher(ids, logits_to_keep=keep).logits[0]
        return ls, lt

    def score_one(self, question: str, rids: list[int], trunc: bool = False) -> tuple[float, dict]:
        """``(r_roll, diag)`` for one rollout; ``(0.0, {})`` when it is not scored.

        δ_t / ω_t on a solution that was generated elsewhere: student and teacher are both
        teacher-forced on prompt + rollout. ``diag`` keys: delta_mean, omega_mean, L,
        R_raw, len_frac (part of the dump format).
        """
        # Code solutions are bimodal: a few hundred tokens, or a runaway loop that
        # rides the cap, with nothing in between. A soft length penalty cannot
        # separate those; δ_t on a looping suffix is meaningless, so under
        # trunc_zero a truncated rollout is not scored at all.
        if len(rids) < self.min_rollout_tokens or (trunc and self.trunc_zero):
            return 0.0, {}
        pids = self.tok(self.solve_prompt(question), add_special_tokens=True,
                        return_tensors="pt")["input_ids"].to(self.device)
        plen = pids.shape[1]
        ids = torch.cat([pids, torch.tensor([rids], device=self.device)], dim=1)
        ls, lt = self.scored_logits(ids, plen)
        n = int(ids.shape[1] - plen)          # response positions; row 0 == absolute plen-1
        delta, omega = divergence.per_token_chunked(ls, lt, self.topk, self.chunk, start=0, stop=n)
        del ls, lt
        return divergence.score_rollout(delta, omega, delta_ref=self.delta_ref, use_omega=self.use_omega,
                                        alpha=self.len_alpha, len_pen=self.len_pen,
                                        max_new=self.max_new)

    def score_question(self, question: str, rollouts: Sequence[tuple[list[int], bool]]
                       ) -> tuple[float, dict, list[tuple[float, dict]]]:
        """``(R_base, diag, per_rollout)``: the mean over the rollouts that were scored.

        ``diag`` averages every numeric per-rollout key (so ``L`` becomes a
        float); a question with no scored rollout gives ``(0.0, {})``.
        """
        got = [self.score_one(question, ids, tr) for ids, tr in rollouts]
        ok = [(rr, dd) for rr, dd in got if dd]
        if not ok:
            return 0.0, {}, got
        r = sum(rr for rr, _ in ok) / len(ok)
        d = {k: sum(dd[k] for _, dd in ok) / len(ok) for k in ok[0][1]
             if isinstance(ok[0][1][k], (int, float))}
        return r, d, got

    def score(self, questions: Sequence[str],
              rollouts: Sequence[Sequence[tuple[list[int], bool]]]) -> list[dict]:
        """Per question ``{"R": R_base, "diag": ..., "rollouts": [(r_roll, diag), ...]}``.

        No diversity term (see ``diversity``); used by the equivalence checks.
        """
        out = []
        for q, rs in zip(questions, rollouts):
            r, d, got = self.score_question(q, rs)
            out.append({"R": r, "diag": d, "rollouts": got})
        return out

    def diversity(self, questions: Sequence[str], batch: Sequence[str] | None = None,
                  offset: int = 0) -> list[float]:
        """Per-question cluster share of the rank batch ``questions`` (updates ``mean_sim``).

        ``bleu`` mode clusters ``batch`` -- the full scoring batch, of which
        ``questions`` is the slice starting at ``offset`` (``batch=None``: the
        rank batch is the whole batch) -- and returns the slice's shares.
        ``emb`` mode clusters consecutive groups of ``diversity.group`` rows of
        ``questions``; each group is embedded in its own call, followed by a
        second embed call for the logged mean cosine.
        """
        if self.div_mode == "bleu":
            full = list(questions) if batch is None else list(batch)
            share, self.mean_sim = bleu_div.bleu_cluster_share(full, self.div_thresh)
            return share[offset:offset + len(questions)]
        qs = list(questions)
        g = self.div_group or len(qs)
        share: list[float] = []
        cos: list[float] = []
        for i in range(0, len(qs), g):
            sub = qs[i:i + g]
            share += emb.cluster_share(sub, self.div_thresh, None, self.embedder)
            try:
                value = emb.mean_cos(sub, None, self.embedder)
                if len(sub) > 1:
                    cos.append(value)
            except Exception:
                pass                          # logging only
        self.mean_sim = sum(cos) / len(cos) if cos else -1.0
        return share


# ------------------------------------------------------------------ rank batch

def sem_uniq(questions: Sequence[str]) -> float:
    """Fraction of questions not within Jaccard 0.6 of an earlier distinct one.

    Logged every step. Prefix-exact matching stayed high in a run whose semantic
    diversity was falling, because the synthesizer kept rewording one question;
    entropy missed it for the same reason.
    """
    norms: list[set] = []
    for q in questions:
        nt = norm_text(q)
        if not any(jaccard(nt, s) > SEM_UNIQ_JACCARD for s in norms):
            norms.append(nt)
    return len(norms) / max(len(questions), 1)


def compute_rewards(scorer: Scorer, questions: Sequence[str], resp_lens: Sequence[int],
                    scores: torch.Tensor, batch: Sequence[str] | None = None,
                    offset: int = 0) -> tuple[list[dict], float]:
    """Fill ``scores`` (``[B, response_length]``, zeros) for one rank batch.

    ``questions`` are the decoded, thinking-stripped synthesizer outputs and
    ``resp_lens`` their valid response lengths; ``batch`` / ``offset`` are the
    full scoring batch and this rank's position in it (``bleu`` diversity).
    Returns the per-row diagnostics (``{}`` for a row that was not scored)
    and the sem_uniq monitor.
    """
    B = len(questions)
    valid = [i for i, q in enumerate(questions) if len(q) >= scorer.min_question_chars]
    sols: dict[int, list[tuple[list[int], bool]]] = {}
    if valid:
        sols = dict(zip(valid, scorer.generate([questions[i] for i in valid])))
        # Generation and scoring peak at different shapes; returning the KV
        # blocks here stops the two peaks from having to coexist.
        torch.cuda.empty_cache()
    diag: list[dict] = []
    for i in range(B):
        if i in sols:
            r, d, _ = scorer.score_question(questions[i], sols[i])
        else:
            r, d = 0.0, {}
        if resp_lens[i] > 0:
            scores[i, resp_lens[i] - 1] = r          # GRPO sums over the sequence
        diag.append(d)
    uniq = sem_uniq(questions)
    if scorer.div_pen > 0:
        # Entropy bonuses and KL anchoring only push against the gradient; making
        # collapse itself score badly is what closes the shortcut. Multiplicative,
        # so it is scale-free: ~1 for a unique question, ~0 when the whole group
        # is one question reworded.
        share = scorer.diversity(questions, batch, offset)
        for i in range(B):
            if resp_lens[i] > 0:
                scores[i, resp_lens[i] - 1] *= max(0.0, 1.0 - scorer.div_pen * share[i])
            if diag[i]:
                diag[i]["share"] = share[i]
    return diag, uniq


def dump_rows(step: int, rank: int, questions: Sequence[str], resp_lens: Sequence[int],
              scores: torch.Tensor, diag: Sequence[dict]) -> list[dict]:
    """Rows of ``<dump>.r<rank>``; ``R`` is the final (diversity-penalised) score."""
    return [{"step": step, "rank": rank,
             "R": float(scores[i, max(resp_lens[i] - 1, 0)]),
             "n_docs": 0,
             "problem": questions[i],
             **(diag[i] or {})} for i in range(len(questions))]


def batch_metrics(diag: Sequence[dict], sem_uniq_value: float, mean_sim: float) -> dict:
    """``synrm/*`` metrics over the scored rows; empty when no row was scored."""
    ok = [d for d in diag if d and "delta_mean" in d]
    if not ok:
        return {}
    n = len(ok)
    return {"synrm/delta_mean": sum(d["delta_mean"] for d in ok) / n,
            "synrm/omega_mean": sum(d["omega_mean"] for d in ok) / n,
            "synrm/sol_len": sum(d["L"] for d in ok) / n,
            "synrm/R_raw": sum(d["R_raw"] for d in ok) / n,
            "synrm/len_frac": sum(d["len_frac"] for d in ok) / n,
            "synrm/parsed_frac": n / len(diag),
            "synrm/sem_uniq": sem_uniq_value,
            "synrm/mean_sim": mean_sim,
            "synrm/cluster_share": sum(d.get("share", 0) for d in ok) / n,
            "synrm/n_clusters": 1.0 / max(sum(d.get("share", 1) for d in ok) / n, 1e-6)}


def append_jsonl(path: str | os.PathLike, rows: Sequence[dict]) -> None:
    """Append rows to a dump file (``ensure_ascii=False``, one JSON object per line)."""
    with open(path, "a") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------- verl wrapper

class SynRewardWorker(Worker):
    """verl reward-model worker that scores generated questions (see the module docstring)."""

    def __init__(self, config) -> None:
        Worker.__init__(self)
        self.config = config
        self.reward_cfg = load_reward_config()
        # Dump what S_theta actually writes alongside its reward. Without this the only
        # visible signal is a scalar, and the two failure modes that matter --
        # collapsing onto one question, or drifting into degenerate text that
        # happens to score well -- are both invisible in the scalar.
        self.dump_path = str(self.reward_cfg["dump"] or "")
        self.step = 0
        self.scorer: Scorer | None = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> None:
        self.scorer = Scorer(self.reward_cfg, get_device_id())
        self.scorer.load()
        s = self.scorer
        if s.div_mode == "bleu":
            # The BLEU share is computed over the whole step's batch: the ranks
            # exchange their question texts over a CPU group (synrl.diversity).
            bleu_div.init_text_group(self.rank, self.world_size)
        print(f"[synrm] alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')} "
              f"gen_bs={s.gen_bs} rolls={s.rolls}", flush=True)
        print(f"[synrm] solver={s.solver_path} teacher={s.teacher_path} "
              f"delta_ref={s.delta_ref} div_mode={s.div_mode} thr={s.div_thresh} pen={s.div_pen}",
              flush=True)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto, **kwargs) -> DataProto:
        assert self.scorer is not None, "init_model() has not run"
        resp = data.batch["responses"]
        attn = data.batch["attention_mask"]
        rlen = resp.shape[1]
        resp_mask = attn[:, -rlen:]
        lens = [int(resp_mask[i].sum()) for i in range(resp.shape[0])]
        # A leading thinking block is dropped: the DeepSeek synthesizer bases use a
        # patched chat template that closes <think> immediately, but a stray block
        # would otherwise be scored and dumped as if it were the question.
        qs = [strip_question(self.scorer.tok.decode(resp[i][:lens[i]], skip_special_tokens=True))
              for i in range(resp.shape[0])]
        batch, offset = None, 0
        if self.scorer.div_mode == "bleu" and self.scorer.div_pen > 0:
            # Before any scoring, so no rank waits on another's generation.
            batch, offset = bleu_div.gather_batch(qs)
        scores = torch.zeros_like(resp, dtype=torch.float32)
        diag, uniq = compute_rewards(self.scorer, qs, lens, scores, batch, offset)
        if self.dump_path:
            self.step += 1
            try:
                append_jsonl(f"{self.dump_path}.r{self.rank}",
                             dump_rows(self.step, self.rank, qs, lens, scores, diag))
            except Exception as e:
                print(f"[synrm] dump failed: {e}", flush=True)
        metrics = batch_metrics(diag, uniq, self.scorer.mean_sim)
        if metrics and self.dump_path:
            try:
                append_jsonl(f"{self.dump_path}.metrics.r{self.rank}", [{"step": self.step, **metrics}])
            except Exception:
                pass
        return DataProto.from_dict(tensors={"rm_scores": scores}, meta_info={"metrics": metrics})

