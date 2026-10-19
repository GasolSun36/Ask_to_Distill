"""The ``a2d`` command: one entry point for every stage of the pipeline.

Usage::

    a2d <command> [<subcommand>...] (--pair P | --config FILE) [options] [key=value ...]

Every command accepts the same common arguments:

* ``--pair P`` and/or ``--config FILE``: which configuration to load (see
  ``ask2distill.config``). Without either, ``$A2D_CONFIG`` is used if set --
  that is how ``launchers/slurm/submit.sh`` hands a resolved config to a job,
  because sbatch ``--export`` cannot carry commas.
* ``key=value`` / ``+key=value``: config overrides, applied last.
* ``--dry-run``: print the resolved config, the stage environment and a rough
  per-GPU memory estimate for training stages, then let the stage print the
  exact command or hydra overrides it would run, without running anything.

Commands are looked up in ``REGISTRY`` and imported lazily (``module:function``),
so ``a2d --help`` does not pull in torch, vLLM or verl. Each target is called as
``fn(cfg, args)`` with the loaded config (``None`` for commands that do not need
one) and the parsed arguments.

tau2 env-side commands run in the separate tau2 interpreter: ``a2d`` spawns
``$A2D_TAU2_PYTHON -m ask2distill.tau2.envside.<module>`` through
``ask2distill.tau2.spawn`` and forwards the remaining arguments unchanged.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

ArgSpec = tuple[tuple[str, ...], dict[str, Any]]


def _a(*flags: str, **kw: Any) -> ArgSpec:
    return flags, kw


@dataclass(frozen=True)
class Command:
    """One ``a2d`` command: where it dispatches and what it accepts."""

    target: str
    help: str
    args: tuple[ArgSpec, ...] = ()
    needs_config: bool = True
    stage: str | None = None          # runtime.base_env stage; also selects the --dry-run memory estimate
    passthrough: bool = False         # unknown arguments are forwarded instead of rejected
    defaults: dict[str, Any] = field(default_factory=dict)


ARM = _a("--arm", choices=["ours", "vanilla"], required=True, help="OPD arm")
STEP = _a("--step", type=int, required=True, help="checkpoint step")
# a2d eval math|code: an OPD checkpoint (--arm + --step) or a ready model (--model)
EVAL_ARM = _a("--arm", choices=["ours", "vanilla"], help="OPD arm (with --step)")
EVAL_STEP = _a("--step", type=int, help="OPD checkpoint step (with --arm)")
EVAL_MODEL = _a("--model", help="evaluate this model instead of an OPD checkpoint: a local HF "
                                "directory or an HF repo id, e.g. gasolsun/<pair> (downloaded "
                                "to $A2D_MODELS); --pair defaults to the repo's name")
FROM_DIR = _a("--from-dir", help="read inputs from a local copy instead of the network")
WALL = _a("--wall-clock-end", type=int, default=None,
          help="unix time the job is killed; enables the save-before-kill path")
VERSION = _a("--version", choices=["v4", "v5", "v6", "v7"], help="tau2 synthesizer version")
ENDPOINTS = _a("--endpoints", help="endpoints file written by `a2d tau2 serve`")


def _envside(module: str, role: str, help: str, **kw: Any) -> Command:
    return Command("ask2distill.cli:run_envside", help, passthrough=True, stage="tau2_env",
                   defaults={"envside_module": module, "role": role}, **kw)


REGISTRY: dict[tuple[str, ...], Command] = {
    # ---- setup ---------------------------------------------------------------
    ("models", "download"): Command(
        "ask2distill.data.models:download_main", "download the pinned base models",
        (_a("--all", action="store_true", help="every model of every pair"),
         _a("--modelscope", action="store_true", help="ModelScope REST fallback (HF CDN blocked)"),
         _a("--endpoint", help="HF endpoint URL")),
        needs_config=False),
    ("models", "syn-nothink"): Command(
        "ask2distill.data.models:syn_nothink_main",
        "build the <student>-SynNoThink synthesizer base (the DeepSeek-R1-Distill math pairs)"),
    ("data", "download"): Command(
        "ask2distill.data.download:main",
        "fetch the release repo and the gasolsun/<pair> banks into $A2D_DATA and md5-verify them",
        (_a("--repo-id", help="HF dataset repo (default: $A2D_HF_DATA_REPO, else gasolsun/Ask_to_Distill-data)"),
         _a("--revision"), _a("--endpoint"),
         _a("--from-dir", help="read the release-repo files from a local snapshot of it"),
         _a("--banks-dir", help="read the banks from local copies instead of gasolsun/<pair>"),
         _a("--only", nargs="+", help="release groups")),
        needs_config=False),
    ("data", "eval"): Command(
        "ask2distill.data.eval_data:main",
        "every math / code evaluation set, from gasolsun/Ask_to_Distill-eval at a pinned revision",
        (_a("--revision", help="release commit (default: the pinned one)"),
         _a("--from-dir", help="local snapshots: DIR/Ask_to_Distill-eval, DIR/AMC23, DIR/hf")),
        needs_config=False),
    ("data", "build", "math"): Command(
        "ask2distill.data.build_math_data:main", "DAPO-Math-17k (Vanilla math arm) and the 8 math test sets",
        (FROM_DIR,), needs_config=False),
    ("data", "build", "code-eval"): Command(
        "ask2distill.data.build_code_eval_data:main", "LCB cache, APPS / CodeContests, EvalPlus",
        (FROM_DIR,), needs_config=False),
    ("data", "build", "ocr"): Command(
        "ask2distill.data.build_public:main", "OpenCodeReasoning at a pinned revision (~9 GB; input of `data build code-real`)",
        (FROM_DIR,), needs_config=False, defaults={"target": "ocr"}),
    ("data", "build", "code-real"): Command(
        "ask2distill.data.build_public:main", "OpenCodeReasoning-10k, the Vanilla code arm (after `data build ocr`)",
        (), needs_config=False, defaults={"target": "code-real"}),
    ("data", "build", "tau2"): Command(
        "ask2distill.data.tau2_data:main", "tau2 data, official split (incl. the Vanilla agent arm), domain context",
        (FROM_DIR,), needs_config=False),
    # ---- math / code pipeline --------------------------------------------------
    ("calibrate",): Command(
        "ask2distill.calib.calibrate:main",
        "stage 0: delta_ref = kappa-th percentile of delta_t on initial-synthesizer questions",
        stage="calib"),
    ("synrl",): Command(
        "ask2distill.synrl.launch:main", "stage 1: synthesizer RL (GRPO against R(q))",
        (WALL,), stage="synrl"),
    ("select", "sweep"): Command(
        "ask2distill.select.sweep:main", "stage 2a: merge and sample every synthesizer checkpoint",
        stage="select"),
    ("select", "pick"): Command(
        "ask2distill.select.pick:main", "stage 2b: diversity gate -> reward gate -> lowest cosine",
        stage="select"),
    ("select", "yield"): Command(
        "ask2distill.select.yield_compare:main", "stage 2c (math): saturation yield of frozen candidates",
        stage="select"),
    ("harvest",): Command(
        "ask2distill.harvest.harvest:main", "stage 3a: sample the chosen synthesizer",
        (_a("--shard", type=int, help="one shard (default: all)"),), stage="harvest"),
    ("filter",): Command(
        "ask2distill.filter.saturation:main", "stage 3b: per-slice format / Jaccard / quality / dedup",
        (_a("--slice", type=int, help="one slice (default: all)"),), stage="filter"),
    ("aggregate",): Command(
        "ask2distill.filter.aggregate:main", "stage 3c: cross-slice embedding dedup in filter.agg_order",
        stage="filter"),
    ("bank",): Command(
        "ask2distill.bank.build_bank:main", "stage 3d: pool -> OPD parquet (a pair with bank.refilter re-filters first)",
        (_a("--allow-short", action="store_true", help="smoke: accept a pool shorter than bank.n"),),
        stage="bank"),
    ("prompts", "build"): Command(
        "ask2distill.bank.build_synrl_prompts:main", "build the synthesizer-RL prompt parquet (synrl.prompts)",
        stage="bank"),
    ("opd",): Command(
        "ask2distill.opd.launch:main", "stage 4: on-policy distillation on a bank",
        (ARM, WALL, _a("--user-endpoints", help="tau2: user-simulator endpoints file")),
        stage="opd"),
    ("merge",): Command(
        "ask2distill.merge:main", "FSDP checkpoint -> HF directory",
        (_a("--arm", choices=["ours", "vanilla"]), _a("--step", type=int),
         _a("--actor-dir"), _a("--target-dir"),
         _a("--chat-template", help="copy models.templates[ROLE] next to the weights"),
         _a("--force", action="store_true", help="allow running on a login node")),
        needs_config=False, stage="merge"),
    ("eval", "math"): Command(
        "ask2distill.eval.math.run:main", "math eval: merge -> generate -> grade",
        (EVAL_ARM, EVAL_STEP, EVAL_MODEL, _a("--tasks", nargs="+"), _a("--limit", type=int)),
        stage="eval_math"),
    ("eval", "code"): Command(
        "ask2distill.eval.code.run_code_eval:main", "code eval: LCB + EvalPlus + extra benches",
        (EVAL_ARM, EVAL_STEP, EVAL_MODEL, _a("--curve", action="store_true", help="curve point: MBPP only"),
         _a("--smoke", action="store_true", help="LCB --debug and HumanEval+ only"),
         _a("--limit", type=int)), stage="eval_code"),
    ("collect", "math"): Command(
        "ask2distill.eval.collect:main", "math avg6 table",
        (_a("--runs", nargs="+"), _a("--out")), needs_config=False, defaults={"domain": "math"}),
    ("collect", "code"): Command(
        "ask2distill.eval.collect:main", "code Avg table",
        (_a("--runs", nargs="+"), _a("--out")), needs_config=False, defaults={"domain": "code"}),
    ("collect", "tau2"): Command(
        "ask2distill.eval.collect:main", "tau2 pass^1 table",
        (_a("--runs", nargs="+"), _a("--out")), needs_config=False, defaults={"domain": "tau2"}),
    # ---- tau2 (a2d env) ---------------------------------------------------------
    ("tau2", "build-prompts"): Command(
        "ask2distill.tau2.bank:build_prompts_main",
        "synthesizer prompt rows (env side) -> parquet", (VERSION,), stage="tau2_env"),
    ("tau2", "rows2parquet"): Command(
        "ask2distill.tau2.bank:rows2parquet_main", "rows.json -> verl parquet",
        (_a("--rows", required=True), _a("--out", required=True)), needs_config=False),
    ("tau2", "serve"): Command(
        "ask2distill.tau2.serve:main", "start a vLLM OpenAI-server fleet and write an endpoints file",
        (_a("--plan", choices=["synrl", "harvest", "opd", "eval"], required=True),
         _a("--gpus"), _a("--model"), _a("--served-name"), VERSION, _a("--arm"), _a("--step", type=int),
         _a("--out")), stage="tau2_serve"),
    ("tau2", "synrl"): Command(
        "ask2distill.tau2.trainside.synrl_launch:main", "tau2 synthesizer RL for one version",
        (VERSION, ENDPOINTS, WALL), stage="tau2_synrl"),
    ("tau2", "select"): Command(
        "ask2distill.tau2.select_synthesizer:main", "trailing-window synthesizer pick (frozen picks kept)",
        (VERSION,), stage="tau2_env"),
    ("tau2", "bank"): Command(
        "ask2distill.tau2.bank:main", "card pools -> OPD parquets (bank.parts, then the bank.mix bank)", stage="tau2_env"),
    ("tau2", "opd"): Command(
        "ask2distill.tau2.trainside.opd_launch:main", "tau2 OPD (agent loop)",
        (ARM, _a("--user-endpoints"), WALL), stage="opd"),
    ("tau2", "summarize"): Command(
        "ask2distill.tau2.summarize:main", "pass^1 per domain and the mean",
        (_a("--models", nargs="+"), _a("--outputs-dir"), _a("--out")), stage="tau2_env"),
    # ---- tau2 (env side, spawned under $A2D_TAU2_PYTHON) ----------------------------
    ("tau2", "env-service"): _envside("env_service", "synrl", "the tau2 environment / reward HTTP service"),
    ("tau2", "harvest"): _envside("harvest", "harvest", "card harvest with per-card reward"),
    ("tau2", "official-split"): _envside("official_split", "synrl", "official train/test split files"),
    ("tau2", "domain-ctx"): _envside("dump_domain_ctx", "synrl", "system prompt + tool schemas -> json"),
    ("tau2", "eval"): _envside("run_eval", "eval", "tau2 evaluation client (login node, OpenAI key)"),
}

# Training stages print a memory estimate on --dry-run.
_MEM_STAGES = {"synrl", "opd", "tau2_synrl"}

# Targets that do not read ``args.dry_run`` themselves (downloads, builders, table
# writers): on --dry-run the CLI prints the call instead of making it, so that
# `a2d data build ocr --dry-run` does not fetch 9 GB or rewrite files in $A2D_DATA.
_NO_DRY_RUN_TARGETS = {
    "ask2distill.data.download:main",
    "ask2distill.data.eval_data:main",
    "ask2distill.data.build_math_data:main",
    "ask2distill.data.build_code_eval_data:main",
    "ask2distill.data.build_public:main",
    "ask2distill.data.tau2_data:main",
    "ask2distill.eval.collect:main",
    "ask2distill.tau2.summarize:main",
    "ask2distill.tau2.select_synthesizer:main",
    "ask2distill.tau2.bank:rows2parquet_main",
}

# Commands of the math/code pipeline; tau2 has its own ``a2d tau2 ...`` stages.
_MATH_CODE_ONLY = {("calibrate",), ("synrl",), ("select", "sweep"), ("select", "pick"),
                   ("select", "yield"), ("harvest",), ("filter",), ("aggregate",), ("bank",),
                   ("prompts", "build"), ("eval", "math"), ("eval", "code")}


# --------------------------------------------------------------------------- parsing

def _match(argv: Sequence[str]) -> tuple[tuple[str, ...], list[str]]:
    words: list[str] = []
    for tok in argv:
        if tok.startswith("-") or "=" in tok:
            break
        words.append(tok)
    for n in range(len(words), 0, -1):
        if tuple(words[:n]) in REGISTRY:
            return tuple(words[:n]), list(argv[n:])
    return (), list(argv)


def _usage() -> str:
    width = max(len(" ".join(k)) for k in REGISTRY)
    lines = ["usage: a2d <command> [--pair P | --config FILE] [options] [key=value ...]", "",
             "commands:"]
    lines += [f"  {' '.join(k):<{width}}  {c.help}" for k, c in REGISTRY.items()]
    lines += ["", "Run `a2d <command> --help` for a command's options."]
    return "\n".join(lines)


def build_parser(path: tuple[str, ...], cmd: Command) -> argparse.ArgumentParser:
    """argparse parser for one command: common arguments plus its own."""
    p = argparse.ArgumentParser(prog="a2d " + " ".join(path), description=cmd.help)
    p.add_argument("--pair", help="pair name (configs/pairs/<pair>.yaml)")
    p.add_argument("--config", help="extra config file merged on top (e.g. configs/smoke/math_smoke.yaml)")
    p.add_argument("--dry-run", action="store_true",
                   help="print config, environment and command; run nothing")
    for flags, kw in cmd.args:
        p.add_argument(*flags, **kw)
    p.set_defaults(**cmd.defaults)
    return p


def parse(argv: Sequence[str]) -> tuple[tuple[str, ...], Command, argparse.Namespace]:
    """Resolve the command and parse its arguments (overrides land in ``args.overrides``)."""
    path, rest = _match(argv)
    if not path:
        if argv and argv[0] in ("-h", "--help"):
            print(_usage())
            raise SystemExit(0)
        raise SystemExit(_usage() if not argv
                         else f"a2d: unknown command {' '.join(argv[:3])!r}\n\n{_usage()}")
    cmd = REGISTRY[path]
    parser = build_parser(path, cmd)
    args, unknown = parser.parse_known_args(rest)
    overrides, passthrough = [], []
    for tok in unknown:
        if "=" in tok and not tok.startswith("-"):
            overrides.append(tok)
        elif cmd.passthrough:
            passthrough.append(tok)
        else:
            parser.error(f"unrecognized argument {tok!r}")
    args.overrides = overrides
    args.passthrough = passthrough
    args.command = path
    return path, cmd, args


def load_config(cmd: Command, args: argparse.Namespace):
    """Load the config named by ``--pair`` / ``--config`` / ``$A2D_CONFIG``."""
    from ask2distill.config import ConfigError, load
    if args.pair is None and args.config is None and getattr(args, "model", None):
        from ask2distill.eval.model_source import pair_of
        args.pair = pair_of(args.model)        # gasolsun/<pair> names its own pair
    config = args.config or (None if args.pair else os.environ.get("A2D_CONFIG") or None)
    if args.pair is None and config is None:
        if cmd.needs_config:
            raise SystemExit(f"a2d {' '.join(args.command)}: needs --pair or --config "
                             "(or $A2D_CONFIG)")
        if args.overrides:
            raise SystemExit("config overrides given but no config to apply them to")
        return None
    try:
        return load(args.pair, config, args.overrides)
    except ConfigError as e:
        raise SystemExit(f"config error:\n{e}") from None


def resolve_target(target: str) -> Callable:
    """Import ``module:function`` lazily."""
    module, _, func = target.partition(":")
    return getattr(importlib.import_module(module), func)


def _print_dry_run(cfg, cmd: Command) -> None:
    from omegaconf import OmegaConf

    from ask2distill import runtime
    if cfg is not None:
        print("# ---- resolved config ----")
        print(OmegaConf.to_yaml(cfg, resolve=True), end="")
    if cmd.stage:
        print(f"# ---- environment (stage {cmd.stage}) ----")
        for k, v in sorted(runtime.base_env(stage=cmd.stage).items()):
            print(f"{k}={'<unset>' if v is None else v}")
    if cfg is not None and cmd.stage in _MEM_STAGES:
        stage = "synrl" if cmd.stage == "tau2_synrl" else cmd.stage
        print("# ---- memory pre-flight (rough, per GPU) ----")
        for k, v in runtime.mem_preflight(cfg, stage).items():
            print(f"{k}: {v}")
    print("# ---- command ----", flush=True)


# ------------------------------------------------------------------- tau2 env side

def run_envside(cfg, args: argparse.Namespace) -> int:
    """Spawn ``ask2distill.tau2.envside.<module>`` under ``$A2D_TAU2_PYTHON``.

    The env-side modules never import omegaconf, torch or pyarrow; they get
    their settings as the rendered ``TAU2_*`` environment plus the forwarded
    command-line arguments.
    """
    from ask2distill.paths import require_tau2_python
    from ask2distill.tau2 import spawn
    env = spawn.env(cfg, args.role)
    if args.dry_run:
        py = require_tau2_python()
        for k in sorted(env):
            if k.startswith(("TAU2_", "A2D_", "REPO_PATH", "PYTHONPATH", "HF_")):
                print(f"{k}={env[k]}")
        print(" ".join([str(py), "-m", f"ask2distill.tau2.envside.{args.envside_module}",
                        *args.passthrough]), flush=True)
        return 0
    return spawn.run(args.envside_module, list(args.passthrough), env).wait()


# ----------------------------------------------------------------------------- main

def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``a2d`` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)
    path, cmd, args = parse(argv)
    cfg = load_config(cmd, args)
    target = cmd.target
    domain = cfg.get("domain") if cfg is not None else None
    if path == ("opd",) and domain == "tau2":
        target = REGISTRY[("tau2", "opd")].target
    if path in _MATH_CODE_ONLY and domain == "tau2":
        raise SystemExit(f"a2d {' '.join(path)}: a math/code stage; the tau2 pipeline runs "
                         "through `a2d tau2 ...` (see docs/TAU2.md)")
    if path[0] == "tau2" and cmd.needs_config and domain not in (None, "tau2"):
        raise SystemExit(f"a2d {' '.join(path)}: a tau2 stage, but the config's domain is {domain!r}")
    if args.dry_run:
        _print_dry_run(cfg, cmd)
        if target in _NO_DRY_RUN_TARGETS:
            shown = {k: v for k, v in sorted(vars(args).items())
                     if k not in ("dry_run", "command", "overrides", "passthrough") and v is not None}
            print(f"{target}({', '.join(f'{k}={v!r}' for k, v in shown.items())})", flush=True)
            return 0
    fn = resolve_target(target)
    rc = fn(cfg, args)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
