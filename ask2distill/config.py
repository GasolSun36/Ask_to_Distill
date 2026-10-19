"""Load, compose, validate and record the per-pair configuration.

A pair file (``configs/pairs/<pair>.yaml``) lists its bases under ``_base_``;
``load`` merges, lowest priority first:

1. a ``paths`` node built from the ``A2D_*`` environment (see ``paths``),
2. the bases, recursively and in listed order,
3. the pair file,
4. an optional ``--config`` file (e.g. ``configs/smoke/math_smoke.yaml``, which
   may itself name the pair file as its base),
5. command-line overrides: ``key=value`` for an existing key, ``+key=value`` to
   add one (the same convention hydra uses).

The result is in struct mode, so reading a key that does not exist raises.
Base files mark every per-pair value ``???``; ``validate`` refuses a config
with any value still missing, which is the "no silent defaults" rule: a code
default that differs from the final runs (worker lambda 0.5, MAXNEW 1024,
δ_ref 0.5, dedup 0.93, ...) can never leak in through an unset key.

verl's own hydra config is not wrapped here. The launchers render it from
these sections into the original override strings.
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ask2distill.common.io import atomic_write_text
from ask2distill.paths import REPO_ROOT, get_paths

log = logging.getLogger(__name__)

CONFIG_DIR = REPO_ROOT / "configs"
PAIRS_DIR = CONFIG_DIR / "pairs"

DOMAINS = ("math", "code", "tau2")
SECTIONS = {
    "math": ("models", "calib", "reward", "synrl", "select", "harvest", "filter", "bank",
             "opd", "eval"),
    "code": ("models", "calib", "reward", "synrl", "select", "harvest", "filter", "bank",
             "opd", "eval"),
    "tau2": ("models", "tau2_data", "reward_env", "synrl", "select", "harvest", "bank",
             "opd", "eval"),
}
# The base files of each domain double as its schema: every key they define
# must still be present after composition (``require`` / ``validate``).
DOMAIN_BASES = {
    "math": ("opd.yaml", "math.yaml", "eval_math.yaml"),
    "code": ("opd.yaml", "code.yaml", "eval_code.yaml"),
    "tau2": ("opd.yaml", "tau2.yaml"),
}
SEED_FORMULAS = ("index", "7919*i+104729")
# Keys that only a smoke config may switch on. They relax a gate the final runs
# enforced, so a pair config that sets them would not reproduce those runs.
SMOKE_ONLY_TRUE = ("select.pick.allow_empty",)
NON_FAITHFUL = "NON-FAITHFUL: diversity grouping differs from the final runs"
EMB_GROUP = "diversity.mode emb: the clustering group is not one reward-rank batch"
_SENTINEL = object()


class ConfigError(Exception):
    """The configuration is incomplete, inconsistent or cannot be read."""


# ----------------------------------------------------------------------------- loading

def list_pairs() -> list[str]:
    """Names of the shipped pair configs (``Domain_Student_Teacher``; ``TEMPLATE_*`` excluded)."""
    return sorted(p.stem for p in PAIRS_DIR.glob("*.yaml") if not p.stem.startswith("TEMPLATE"))


def domain_pair(domain: str) -> str:
    """The one shipped pair of ``domain``; raises if there is none or more than one."""
    found = [p for p in list_pairs()
             if OmegaConf.select(_load_file(PAIRS_DIR / f"{p}.yaml"), "domain") == domain]
    if len(found) != 1:
        raise ConfigError(f"expected exactly one {domain} pair, found {found or 'none'}")
    return found[0]


def _locate(path: str | Path, rel_to: Path | None = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        cands = [p]
    else:
        cands = ([rel_to / p] if rel_to else []) + [Path.cwd() / p, REPO_ROOT / p]
    for c in cands:
        if c.is_file():
            return c.absolute()
    raise ConfigError(f"config file not found: {path} (tried {', '.join(map(str, cands))})")


def _load_file(path: Path, _stack: tuple[Path, ...] = ()) -> DictConfig:
    """One YAML file with its ``_base_`` chain expanded (bases first)."""
    if path in _stack:
        raise ConfigError("cyclic _base_: " + " -> ".join(map(str, _stack + (path,))))
    try:
        raw = OmegaConf.load(path)
    except (OSError, yaml.YAMLError, OmegaConfBaseException) as e:
        raise ConfigError(f"cannot read {path}: {e}") from e
    if not isinstance(raw, DictConfig):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    bases = raw.pop("_base_", None) or []
    if isinstance(bases, str):
        bases = [bases]
    merged = OmegaConf.create()
    for b in bases:
        merged = OmegaConf.merge(merged, _load_file(_locate(b, path.parent), _stack + (path,)))
    return OmegaConf.merge(merged, raw)


def _parse_override(item: str) -> tuple[str, Any, bool]:
    if "=" not in item:
        raise ConfigError(f"override {item!r} is not key=value")
    key, val = item.split("=", 1)
    add = key.startswith("+")
    key = key.lstrip("+").strip()
    if not key:
        raise ConfigError(f"override {item!r} has an empty key")
    try:
        value = yaml.safe_load(val) if val.strip() else ""
    except yaml.YAMLError as e:
        raise ConfigError(f"override {item!r}: cannot parse value: {e}") from e
    return key, value, add


def apply_overrides(cfg: DictConfig, overrides: Iterable[str]) -> DictConfig:
    """Apply ``key=value`` / ``+key=value`` overrides in order."""
    for item in overrides:
        key, value, add = _parse_override(item)
        exists = OmegaConf.select(cfg, key, default=_SENTINEL, throw_on_missing=False) is not _SENTINEL
        if not exists and not add:
            raise ConfigError(f"override {item!r}: no key {key!r} in the config "
                              f"(use +{key}=... to add a new key)")
        try:
            OmegaConf.update(cfg, key, value, merge=True, force_add=add)
        except OmegaConfBaseException as e:
            raise ConfigError(f"override {item!r}: {e}") from e
    return cfg


def load(pair: str | None = None, config: str | Path | None = None,
         overrides: Sequence[str] = ()) -> DictConfig:
    """Compose and validate the config for ``pair`` (see the module docstring).

    At least one of ``pair`` and ``config`` is required. Raises ``ConfigError``
    when the result is incomplete or violates a guard. Guard warnings (e.g.
    NON-FAITHFUL for tau2) are logged; call ``validate`` again to get them as a list.
    """
    if pair is None and config is None:
        raise ConfigError("need --pair or --config (pairs: " + ", ".join(list_pairs()) + ")")
    cfg = OmegaConf.create({"paths": get_paths().as_dict()})
    if pair is not None:
        pfile = PAIRS_DIR / f"{pair}.yaml"
        if not pfile.is_file():
            raise ConfigError(f"unknown pair {pair!r}; available: {', '.join(list_pairs())}")
        cfg = OmegaConf.merge(cfg, _load_file(pfile))
    if config is not None:
        cfg = OmegaConf.merge(cfg, _load_file(_locate(config)))
    if pair is not None and cfg.get("pair") not in (None, pair):
        raise ConfigError(f"--pair {pair} conflicts with pair={cfg.get('pair')!r} in {config}")
    OmegaConf.set_struct(cfg, True)
    apply_overrides(cfg, overrides)
    validate(cfg)
    return cfg


# -------------------------------------------------------------------------- validation

def missing_keys(node: Any, prefix: str = "") -> list[str]:
    """Dotted keys under ``node`` that are ``???`` or whose interpolation fails."""
    out: list[str] = []
    if isinstance(node, DictConfig):
        keys: Iterable[Any] = node.keys()
    elif isinstance(node, ListConfig):
        keys = range(len(node))
    else:
        return out
    for k in keys:
        name = f"{prefix}.{k}" if prefix else str(k)
        if OmegaConf.is_missing(node, k):
            out.append(name)
            continue
        try:
            child = node[k]
        except OmegaConfBaseException as e:
            out.append(f"{name} ({type(e).__name__}: {str(e).splitlines()[0]})")
            continue
        if isinstance(child, (DictConfig, ListConfig)):
            out.extend(missing_keys(child, name))
    return out


@functools.lru_cache(maxsize=None)
def _schema(domain: str) -> DictConfig:
    merged = OmegaConf.create()
    for name in DOMAIN_BASES[domain]:
        merged = OmegaConf.merge(merged, _load_file(CONFIG_DIR / "base" / name))
    return merged


def _leaf_paths(node: DictConfig, prefix: str = "") -> list[str]:
    out = []
    for k in node.keys():
        name = f"{prefix}.{k}" if prefix else str(k)
        child = node._get_node(k)
        if isinstance(child, DictConfig) and not child._is_missing() and not child._is_none() \
                and len(child) > 0:
            out.extend(_leaf_paths(child, name))
        else:
            out.append(name)
    return out


def absent_keys(cfg: DictConfig, section: str | None = None) -> list[str]:
    """Keys the domain's base files define but ``cfg`` no longer has."""
    domain = OmegaConf.select(cfg, "domain", default=None)
    if domain not in DOMAIN_BASES:
        return []
    schema = _schema(domain)
    if section is not None:
        node = OmegaConf.select(schema, section, default=None, throw_on_missing=False)
        if not isinstance(node, DictConfig):
            return []
        leaves = _leaf_paths(node, section)
    else:
        leaves = _leaf_paths(schema)
    return [k for k in leaves
            if OmegaConf.select(cfg, k, default=_SENTINEL, throw_on_missing=False,
                                throw_on_resolution_failure=False) is _SENTINEL]


def require(cfg: DictConfig, section: str) -> None:
    """Raise ``ConfigError`` naming every missing (``???``, unresolvable or
    deleted) key under ``section``."""
    node = OmegaConf.select(cfg, section, default=_SENTINEL, throw_on_missing=False)
    if node is _SENTINEL:
        raise ConfigError(f"config has no section {section!r}")
    parent_key, _, last = section.rpartition(".")
    parent = OmegaConf.select(cfg, parent_key) if parent_key else cfg
    if OmegaConf.is_missing(parent, last):
        raise ConfigError(f"section {section!r} is missing (???)")
    missing = missing_keys(node, section) if isinstance(node, (DictConfig, ListConfig)) else []
    missing += [f"{k} (deleted)" for k in absent_keys(cfg, section)]
    if missing:
        raise ConfigError(f"{len(missing)} missing value(s) under {section!r}:\n  "
                          + "\n  ".join(missing))


def _get(cfg: DictConfig, key: str, default: Any = None) -> Any:
    return OmegaConf.select(cfg, key, default=default)


def _override_key(item: str) -> str:
    return item.split("=", 1)[0].lstrip("+~").strip()


def validate(cfg: DictConfig) -> list[str]:
    """Check completeness and cross-key guards; return warnings, raise on errors."""
    errors: list[str] = []
    warnings: list[str] = []
    domain = _get(cfg, "domain")
    if domain not in DOMAINS:
        raise ConfigError(f"domain must be one of {DOMAINS}, got {domain!r}")
    for key in ("pair", "smoke"):
        if key not in cfg:
            errors.append(f"missing top-level key {key!r}")
    absent = [s for s in SECTIONS[domain] if s not in cfg]
    if absent:
        errors.append(f"missing section(s) for domain {domain}: {', '.join(absent)}")
    missing = missing_keys(cfg) + [f"{k} (deleted)" for k in absent_keys(cfg)]
    if missing:
        errors.append(f"{len(missing)} missing value(s):\n  " + "\n  ".join(missing))
    if errors:
        raise ConfigError("\n".join(errors))

    smoke = bool(cfg.smoke)
    if domain in ("math", "code"):
        _guard_math_code(cfg, errors, warnings)
    else:
        _guard_tau2(cfg, errors, warnings)

    for section in ("opd", "synrl"):
        forbidden = set(_get(cfg, f"{section}.forbidden_overrides", []) or [])
        extra = [_override_key(x) for x in (_get(cfg, f"{section}.extra_overrides", []) or [])]
        bad = sorted(forbidden.intersection(extra))
        if bad:
            errors.append(f"{section}.extra_overrides sets forbidden key(s) {bad} "
                          f"(see {section}.forbidden_overrides)")

    if not smoke:
        for key in SMOKE_ONLY_TRUE:
            if _get(cfg, key) is True:
                errors.append(f"{key}=true is only allowed in a smoke config (smoke: true)")
        if _get(cfg, "bank.allow_short") is True and _get(cfg, "bank.expected_rows") is None:
            errors.append("bank.allow_short=true outside a smoke config needs bank.expected_rows "
                          "(the documented short yield it excuses)")
        for key in ("opd.train_max_samples", "opd.val_max_samples"):
            v = _get(cfg, key)
            if v is not None and v != -1:
                warnings.append(f"{key}={v}: the final runs did not cap samples")

    if errors:
        raise ConfigError("\n".join(errors))
    for w in warnings:
        log.warning("[config %s] %s", cfg.pair, w)
    return warnings


def _guard_math_code(cfg: DictConfig, errors: list[str], warnings: list[str]) -> None:
    layout = cfg.synrl.layout
    if layout.enable_resource_pool is not True:
        # The reward worker sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments at
        # import; colocated with the synthesizer's vLLM rollout that setting breaks
        # vLLM's CuMemAllocator, so the reward worker must have its own pool.
        errors.append("synrl.layout.enable_resource_pool must be true for math/code")
    delta_ref = cfg.reward.delta_ref
    if delta_ref is None:
        # A new pair before calibration; `a2d synrl` refuses to start without it.
        warnings.append("reward.delta_ref is not set: run `a2d calibrate` and copy its delta_ref_rounded")
    elif round(float(delta_ref), 4) != float(delta_ref):
        errors.append(f"reward.delta_ref={delta_ref} must be rounded to 4 decimals (calib.round_decimals)")
    expected = _get(cfg, "calib.expected_delta_ref")
    if expected is not None and delta_ref is not None and float(expected) != float(delta_ref) \
            and not cfg.smoke:
        warnings.append(f"reward.delta_ref={delta_ref} differs from calib.expected_delta_ref={expected}")
    if cfg.harvest.seed_formula not in SEED_FORMULAS:
        errors.append(f"harvest.seed_formula must be one of {SEED_FORMULAS}, "
                      f"got {cfg.harvest.seed_formula!r}")
    if cfg.filter.aggregate:
        order = list(cfg.filter.agg_order or [])
        if sorted(order) != list(range(len(cfg.filter.slices))):
            errors.append(f"filter.agg_order {order} is not a permutation of the "
                          f"{len(cfg.filter.slices)} slice indices")
    # verl splits the step's questions evenly over the reward ranks and pads an
    # uneven batch with duplicated rows, which the diversity share would count.
    n_batch = int(cfg.synrl.train_batch_size) * int(cfg.synrl.rollout_n)
    n_ranks = int(layout.reward_gpus) * int(layout.reward_nnodes)
    if n_batch % n_ranks:
        errors.append(f"synrl.train_batch_size x synrl.rollout_n = {n_batch} questions do not split "
                      f"evenly over {n_ranks} reward ranks")
    div = cfg.reward.diversity
    if div.mode not in ("bleu", "emb"):
        errors.append(f"reward.diversity.mode={div.mode!r} must be 'bleu' or 'emb'")
    elif div.mode == "emb" and div.group != n_batch // n_ranks:
        # emb mode clusters consecutive groups of the rank batch; balance_batch
        # (verl's ray_trainer.py) makes each rank batch a length-balanced shard.
        warnings.append(f"{EMB_GROUP} (diversity.group={div.group}, rank batch={n_batch // n_ranks})")
    if cfg.reward.max_new != cfg.opd.max_response_length:
        warnings.append(f"reward.max_new={cfg.reward.max_new} != opd.max_response_length="
                        f"{cfg.opd.max_response_length}: δ_t is then measured under a "
                        "different solve budget than the OPD run uses")


def _guard_tau2(cfg: DictConfig, errors: list[str], warnings: list[str]) -> None:
    g = cfg.synrl
    rank_batch = g.train_batch_size * g.rollout_n / g.n_gpus_per_node
    if g.n_gpus_per_node != 8 or g.diversity.group != 16 or rank_batch != 16:
        warnings.append(f"{NON_FAITHFUL} (actor ranks={g.n_gpus_per_node}, rank batch="
                        f"{rank_batch:g}, diversity.group={g.diversity.group}; final: 8/16/16)")
    name = str(cfg.eval.served_name)
    # ToolOrchestra routes the agent model by name (third_party/toolorchestra_tau2 LLM_CALL.py, llm_utils.py)
    if "qwen" not in name.lower():
        errors.append(f"eval.served_name {name!r} must contain 'qwen'")


# --------------------------------------------------------------------------- recording

def to_container(cfg: DictConfig, section: str | None = None) -> Any:
    """Plain resolved Python object for the whole config or one section."""
    node = cfg if section is None else OmegaConf.select(cfg, section, throw_on_missing=True)
    if node is None:
        raise ConfigError(f"config has no section {section!r}")
    if not isinstance(node, (DictConfig, ListConfig)):
        return node
    return OmegaConf.to_container(node, resolve=True, throw_on_missing=True)


def save_resolved(cfg: DictConfig, run_dir: Path) -> Path:
    """Write ``<run_dir>/resolved_config.yaml`` with every interpolation resolved."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "resolved_config.yaml"
    atomic_write_text(out, OmegaConf.to_yaml(cfg, resolve=True))
    return out


def section_json(cfg: DictConfig, section: str, path: Path) -> Path:
    """Write one resolved section as JSON (e.g. ``reward`` -> reward_config.json)."""
    path = Path(path)
    atomic_write_text(path, json.dumps(to_container(cfg, section), indent=1, sort_keys=True) + "\n")
    return path
