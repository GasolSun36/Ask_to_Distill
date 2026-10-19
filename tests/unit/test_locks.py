"""The environment locks and build scripts in ``env/``.

Both locks must install anywhere: no local-wheel lines and no editable lines.
The a2d lock is the frozen ``pip freeze`` minus exactly the three packages the build script
installs itself; the tau2 lock is the frozen freeze verbatim.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(os.path.abspath(__file__)).parents[2]
ENV = REPO / "env"
A2D_LOCK = ENV / "requirements-a2d.lock.txt"
TAU2_LOCK = ENV / "requirements-tau2.lock.txt"
PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[A-Za-z0-9][A-Za-z0-9.+!_-]*$")

# Versions measured in the env that produced the final runs.
MEASURED = {
    "torch": "2.8.0",
    "vllm": "0.11.0",
    "transformers": "4.56.2",
    "ray": "2.57.0",
    "tensordict": "0.10.0",
    "numpy": "2.2.6",
    "flashinfer-python": "0.3.1",
    "datasets": "5.0.1",
    "hydra-core": "1.3.5",
    "omegaconf": "2.3.1",
    "pandas": "3.0.5",
    "pyarrow": "25.0.1",
    "triton": "3.4.0",
    "xformers": "0.0.32.post1",
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
}
# Installed by hand after the original build script ran; the lock must carry them.
HAND_INSTALLED = ["nltk", "scikit-learn", "pebble", "openai", "swanlab", "math-verify",
                  "latex2sympy2-extended", "fastapi", "uvicorn"]
# Installed explicitly by build_a2d_env.sh, so they must not appear in the lock.
EXPLICIT = ["flash-attn", "evalplus", "verl"]


def _check_env() -> ModuleType:
    spec = importlib.util.spec_from_file_location("a2d_check_env", ENV / "check_env.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _entries(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


@pytest.mark.parametrize("lock", [A2D_LOCK, TAU2_LOCK], ids=["a2d", "tau2"])
def test_lock_is_portable(lock: Path) -> None:
    text = lock.read_text()
    assert "file://" not in text
    for line in text.splitlines():
        assert not line.lstrip().startswith("-e"), line
        assert not line.lstrip().startswith("--"), line
    for entry in _entries(lock):
        assert PIN.match(entry), f"not an exact pin: {entry!r}"


@pytest.mark.parametrize("lock", [A2D_LOCK, TAU2_LOCK], ids=["a2d", "tau2"])
def test_lock_has_no_duplicates(lock: Path) -> None:
    names = [_check_env().normalize(e.split("==")[0]) for e in _entries(lock)]
    assert len(names) == len(set(names))


def test_a2d_lock_pins_measured_versions() -> None:
    pins = _check_env().parse_lock(A2D_LOCK)
    for name, version in MEASURED.items():
        assert pins.get(name) == version, f"{name}: {pins.get(name)} != {version}"
    for name in HAND_INSTALLED:
        assert name in pins, f"{name} missing from the a2d lock"
    for name in EXPLICIT:
        assert name not in pins, f"{name} must be installed by build_a2d_env.sh, not the lock"


def test_tau2_lock_pins() -> None:
    pins = _check_env().parse_lock(TAU2_LOCK)
    # 1.82.7 and 1.82.8 were compromised releases.
    assert pins["litellm"] == "1.82.6"
    assert "brotli" not in pins
    for name in ("torch", "pyarrow", "omegaconf", "tau2", "ask2distill"):
        assert name not in pins, name


def test_parse_lock_rejects_loose_specifiers(tmp_path: Path) -> None:
    ce = _check_env()
    good = tmp_path / "good.txt"
    good.write_text("# comment\nFoo_Bar==1.0  # trailing\n\nbaz[extra]==2\n")
    assert ce.parse_lock(good) == {"foo-bar": "1.0", "baz": "2"}
    bad = tmp_path / "bad.txt"
    bad.write_text("foo>=1.0\n")
    with pytest.raises(ValueError):
        ce.parse_lock(bad)


@pytest.mark.parametrize("script", ["build_a2d_env.sh", "build_tau2_env.sh", "build_flash_attn.sh"])
def test_build_scripts_parse(script: str) -> None:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    subprocess.run([bash, "-n", str(ENV / script)], check=True)


def test_build_a2d_env_installs_the_explicit_packages() -> None:
    s = (ENV / "build_a2d_env.sh").read_text()
    assert 'pip install --no-deps -r "$LOCK"' in s
    assert "flash_attn-2.8.1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" in s
    assert "A2D_FLASH_ATTN_WHEEL" in s
    assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_EVALPLUS=$EVALPLUS_VERSION" in s
    assert "EVALPLUS_VERSION=0.4.0.dev44" in s
    assert '-e "$REPO_DIR/third_party/verl"' in s
    assert "evalplus.evaluate" in s
    assert "cuda-nvcc=12.8 cuda-cudart-dev=12.8 libcurand-dev=10.3.9" in s
    # REPO_DIR is the repository root, not the script's own directory.
    assert 'REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)' in s


def test_build_tau2_env_does_not_install_tau2_or_ask2distill() -> None:
    s = (ENV / "build_tau2_env.sh").read_text()
    assert 'pip install --no-deps -r "$LOCK"' in s
    assert "pip install -e" not in s
    assert "A2D_TAU2_PYTHON" in s
