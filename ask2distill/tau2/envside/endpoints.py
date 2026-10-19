"""The endpoints file shared by ``a2d tau2 serve`` and its consumers.

One line per ready server: ``<kind> <base_url> <served_name>``. Kinds:

* ``stu`` -- the student (or, for evaluation, the checkpoint under test);
* ``usr`` -- the user simulator (also the v6/v7 preference judge);
* ``tea`` -- the teacher (solvability gate);
* ``gen`` -- a merged synthesizer (harvest).

Per-node files are concatenated by the SLURM launchers (``a2d_collect_endpoints``),
so a kind may appear many times; consumers spread load over all of its URLs.
This module is standard-library only and is used from both interpreters.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

KINDS = ("stu", "usr", "tea", "gen")


@dataclass(frozen=True)
class Endpoint:
    """One OpenAI-compatible server."""

    kind: str
    url: str
    served_name: str

    @property
    def host(self) -> str:
        return urlparse(self.url).hostname or ""

    @property
    def port(self) -> int:
        return int(urlparse(self.url).port or 80)

    def line(self) -> str:
        return f"{self.kind} {self.url} {self.served_name}"


def parse(text: str) -> list[Endpoint]:
    """Endpoints in file order; blank lines and ``#`` comments are skipped."""
    out = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"endpoints line {n}: expected '<kind> <url> <served_name>', got {raw!r}")
        kind, url, name = parts
        if kind not in KINDS:
            raise ValueError(f"endpoints line {n}: unknown kind {kind!r} (expected one of {KINDS})")
        out.append(Endpoint(kind, url, name))
    return out


def read(path: str | Path) -> list[Endpoint]:
    """Endpoints from a file."""
    return parse(Path(path).read_text())


def format_lines(eps: list[Endpoint]) -> str:
    """File contents for ``eps``."""
    return "".join(e.line() + "\n" for e in eps)


def by_kind(eps: list[Endpoint], kind: str) -> list[Endpoint]:
    """Endpoints of one kind, in file order."""
    return [e for e in eps if e.kind == kind]


def urls(eps: list[Endpoint], kind: str) -> list[str]:
    """Base URLs of one kind, in file order."""
    return [e.url for e in by_kind(eps, kind)]


def served_name(eps: list[Endpoint], kind: str) -> str:
    """The single served name of ``kind``; raises if absent or ambiguous."""
    names = list(OrderedDict.fromkeys(e.served_name for e in by_kind(eps, kind)))
    if len(names) != 1:
        raise ValueError(f"endpoints of kind {kind!r} must carry exactly one served name, got {names}")
    return names[0]


def model_config(eps: list[Endpoint], config_path: str) -> dict:
    """ToolOrchestra's ``--model_config_path`` JSON: served name -> replicas.

    tau2 addresses a model by the name it was served under and looks its
    replicas up here, so keys are served names.
    """
    cfg: dict = OrderedDict()
    for e in eps:
        cfg.setdefault(e.served_name, []).append({"ip_addr": e.host, "port": str(e.port)})
    cfg["vllm_model_config_path"] = config_path
    return dict(cfg)
