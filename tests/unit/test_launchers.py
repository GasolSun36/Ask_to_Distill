"""launchers/: shell syntax, common.sh helpers, and submit.sh / resubmit.sh against fake SLURM.

Nothing is submitted: ``sbatch``, ``squeue`` and ``sacct`` are shims, conda is
a no-op function, and ``a2d`` is a stub on PATH.
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ask2distill.paths import REPO_ROOT

from conftest import CODE_QWEN3_30B, CODE_QWEN3_4B, MATH_QWEN3_1P7B

LAUNCHERS = REPO_ROOT / "launchers"
SLURM = LAUNCHERS / "slurm"
SCRIPTS = sorted(list(SLURM.glob("*.sh")) + list(SLURM.glob("*.sbatch")) + list(LAUNCHERS.glob("local/*.sh")))


def _bash(snippet: str, env: dict | None = None) -> subprocess.CompletedProcess:
    full = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), **(env or {})}
    return subprocess.run(["bash", "-c", f'source "{SLURM}/common.sh"\n{snippet}'], env=full,
                          capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_shell_syntax(script):
    r = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_no_account_or_partition_in_headers(script):
    text = script.read_text()
    assert not re.search(r"^#SBATCH\s+--(account|partition)", text, re.M), \
        "account / partition come from cluster.env through submit.sh"


def test_cluster_env_example_has_only_placeholders():
    text = (SLURM / "cluster.env.example").read_text()
    for var in ("ACCOUNT", "PARTITION_BIG", "PARTITION_SMALL", "CONDA_SH"):
        value = re.search(rf"^{var}=(.*)$", text, re.M).group(1)
        assert "<" in value and ">" in value, var


@pytest.mark.parametrize("text,secs", [("1-02:03:04", 93784), ("00:08:09", 489), ("03:04", 184),
                                       ("59", 59), ("2-00:00:00", 172800)])
def test_rem_secs(text, secs):
    r = _bash(f'a2d_rem_secs "{text}"')
    assert r.returncode == 0 and int(r.stdout) == secs


def test_arg_value_and_stage():
    r = _bash('a2d_arg_value --arm opd --arm ours; a2d_arg_value --version tau2 synrl --version=v7; '
              'a2d_arg_value --step opd --arm x || echo none; '
              'a2d_stage_of eval code --arm x; a2d_stage_of tau2 synrl; a2d_stage_of harvest --shard 1')
    assert r.stdout.split() == ["ours", "v7", "none", "eval_code", "tau2_synrl", "harvest"]


# --------------------------------------------------------------- fake SLURM

def _exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def fake_cluster(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _exe(bindir / "sbatch", '#!/bin/bash\nprintf "%s\\0" "$@" >> "$A2D_TEST_SBATCH"\necho 4242\n')
    _exe(bindir / "squeue", "#!/bin/bash\nexit 0\n")
    _exe(bindir / "sacct", "#!/bin/bash\nexit 0\n")
    _exe(bindir / "a2d", "#!/bin/bash\nexit 0\n")
    conda_sh = tmp_path / "conda.sh"
    conda_sh.write_text("conda() { return 0; }\n")
    cluster_env = tmp_path / "cluster.env"
    cluster_env.write_text(f"ACCOUNT=acct\nPARTITION_BIG=big\nPARTITION_SMALL=small\nTIME=4:00:00\n"
                           f"CONDA_SH={conda_sh}\nA2D_CONDA_ENV=a2d\nCPUS_PER_TASK=64\n"
                           f"SMALL_CPUS_PER_TASK=16\nSMALL_MEM=200G\n")
    env = {"PATH": f"{bindir}:{Path(sys.executable).parent}:/usr/bin:/bin",
           "HOME": str(tmp_path), "PYTHONPATH": str(REPO_ROOT),
           "A2D_CLUSTER_ENV": str(cluster_env), "A2D_TEST_SBATCH": str(tmp_path / "sbatch.args"),
           "CONDA_PREFIX": str(tmp_path)}
    for name in ("data", "models", "ckpt", "out", "cache"):
        env[f"A2D_{name.upper()}"] = str(tmp_path / name)
    return env


def _submit(fake_cluster: dict, *args: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    r = subprocess.run(["bash", str(SLURM / "submit.sh"), *args], env=fake_cluster,
                       capture_output=True, text=True, timeout=120)
    log = Path(fake_cluster["A2D_TEST_SBATCH"])
    argv = log.read_bytes().decode().split("\0")[:-1] if log.exists() else []
    return r, argv


def test_submit_small_stage(fake_cluster):
    r, argv = _submit(fake_cluster, "harvest", CODE_QWEN3_4B, "--shard", "3", "harvest.per_shard=7")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "4242"
    exports = [a for a in argv if a.startswith("--export=")]
    assert len(exports) == 1
    config = exports[0].removeprefix("--export=ALL,A2D_CONFIG=")
    assert "," not in config and Path(config).is_file()
    assert yaml.safe_load(Path(config).read_text())["harvest"]["per_shard"] == 7
    assert "--partition=small" in argv and "--gpus-per-node=1" in argv and "--account=acct" in argv
    assert f"--job-name=a2d-harvest-{CODE_QWEN3_4B}" in argv
    i = next(i for i, a in enumerate(argv) if a.endswith("gpu_small.sbatch"))
    assert argv[i + 1:] == ["harvest", "--shard", "3"]


def test_submit_opd_and_code_calibration(fake_cluster):
    r, argv = _submit(fake_cluster, "opd", MATH_QWEN3_1P7B, "--arm", "ours")
    assert r.returncode == 0, r.stderr
    i = next(i for i, a in enumerate(argv) if a.endswith("opd_1node.sbatch"))
    assert argv[i + 1:] == ["opd", "--arm", "ours"]
    assert "--partition=big" in argv and f"--job-name=a2d-opd-{MATH_QWEN3_1P7B}-ours" in argv
    Path(fake_cluster["A2D_TEST_SBATCH"]).unlink()
    r, argv = _submit(fake_cluster, "calibrate", CODE_QWEN3_30B)
    assert r.returncode == 0, r.stderr
    assert "--gpus-per-node=2" in argv                # code: teacher on cuda:1


def test_submit_smoke_config_and_bad_override(fake_cluster):
    r, argv = _submit(fake_cluster, "opd", str(REPO_ROOT / "configs/smoke/math_smoke.yaml"), "--arm", "ours")
    assert r.returncode == 0, r.stderr
    config = next(a for a in argv if a.startswith("--export=")).split("A2D_CONFIG=")[1]
    assert yaml.safe_load(Path(config).read_text())["smoke"] is True
    Path(fake_cluster["A2D_TEST_SBATCH"]).unlink()
    r, argv = _submit(fake_cluster, "opd", MATH_QWEN3_1P7B, "--arm", "ours", "opd.no_such_key=1")
    assert r.returncode != 0 and argv == []


def test_submit_refuses_placeholder_cluster_env(fake_cluster, tmp_path):
    bad = tmp_path / "bad.env"
    bad.write_text((SLURM / "cluster.env.example").read_text())
    r, argv = _submit({**fake_cluster, "A2D_CLUSTER_ENV": str(bad)}, "opd", MATH_QWEN3_1P7B, "--arm", "ours")
    assert r.returncode != 0 and "placeholder" in r.stderr and argv == []


def test_resubmit_stops_at_target(fake_cluster):
    ck = Path(fake_cluster["A2D_CKPT"]) / f"opd_{MATH_QWEN3_1P7B}_ours"
    ck.mkdir(parents=True)
    (ck / "latest_checkpointed_iteration.txt").write_text("90\n")
    r = subprocess.run(["bash", str(SLURM / "resubmit.sh"), "--exp", f"opd_{MATH_QWEN3_1P7B}_ours", "--target", "90",
                        "--poll", "1", "--", "opd", MATH_QWEN3_1P7B, "--arm", "ours"],
                       env=fake_cluster, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "RESUBMIT_DONE" in r.stdout
    assert not Path(fake_cluster["A2D_TEST_SBATCH"]).exists()


def test_resubmit_gives_up_after_fast_failures(fake_cluster, tmp_path):
    bindir = Path(fake_cluster["PATH"].split(":")[0])
    _exe(bindir / "sacct", "#!/bin/bash\necho 30\n")       # every job "ran" 30 s
    r = subprocess.run(["bash", str(SLURM / "resubmit.sh"), "--exp", "nothing_yet", "--target", "10",
                        "--poll", "0", "--max-fast", "2", "--", "opd", MATH_QWEN3_1P7B, "--arm", "ours"],
                       env=fake_cluster, capture_output=True, text=True, timeout=300)
    assert r.returncode == 1 and "consecutive jobs shorter" in r.stdout
    jobs = Path(fake_cluster["A2D_TEST_SBATCH"]).read_bytes().decode().count("opd_1node.sbatch")
    assert jobs == 2                                          # stops after --max-fast fast jobs


def test_local_run_stage_logs_and_keeps_exit_code(fake_cluster, tmp_path):
    bindir = Path(fake_cluster["PATH"].split(":")[0])
    _exe(bindir / "a2d", '#!/bin/bash\necho "stage output $*"\nexit 7\n')
    r = subprocess.run(["bash", str(LAUNCHERS / "local" / "run_stage.sh"), "select", "pick", "--pair", MATH_QWEN3_1P7B],
                       env=fake_cluster, capture_output=True, text=True, timeout=60)
    assert r.returncode == 7
    logs = list((Path(fake_cluster["A2D_OUT"]) / "logs").glob("*_select_pick.log"))
    assert len(logs) == 1 and f"stage output select pick --pair {MATH_QWEN3_1P7B}" in logs[0].read_text()
