"""tau2 serving plans: served names, templates by role, replica maps, vLLM flags.

* eval served names contain "qwen" (ToolOrchestra routes only those to vLLM)
  and render as ``<pair>-<arm>-s<step>``;
* templates are resolved by ROLE, never by served name; a configured template
  that is missing is an error, ``user: null`` serves the model's own template;
* the replica maps of the final runs (synrl node maps, harvest x2, OPD user
  simulator per arm) and no sampling flags on the command line;
* servers sharing a GPU run eager.
"""
from __future__ import annotations

import hashlib

import pytest

from ask2distill.paths import REPO_ROOT

from conftest import AGENT_QWEN3_4B

TEMPLATES = REPO_ROOT / "ask2distill" / "tau2" / "templates"


def md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture
def a2d_dirs(tmp_path, monkeypatch):
    dirs = {k: tmp_path / k for k in ("data", "models", "ckpt", "out", "cache")}
    for name, var in (("data", "A2D_DATA"), ("models", "A2D_MODELS"), ("ckpt", "A2D_CKPT"),
                      ("out", "A2D_OUT"), ("cache", "A2D_CACHE")):
        monkeypatch.setenv(var, str(dirs[name]))
    return dirs


@pytest.fixture
def cfg(a2d_dirs):
    from ask2distill import config
    return config.load(AGENT_QWEN3_4B)


def _by_kind(specs):
    return {s.kind: s for s in specs}


def test_templates_by_role(cfg):
    from ask2distill.tau2 import serve
    assert serve.template_for(cfg, "student") == str(TEMPLATES / "Qwen3-4B.nothink.jinja")
    assert serve.template_for(cfg, "synthesizer") == str(TEMPLATES / "Qwen3-4B.nothink.jinja")
    assert serve.template_for(cfg, "teacher") == str(TEMPLATES / "Qwen3-4B-Thinking-2507.nothink.jinja")
    assert serve.template_for(cfg, "user") is None
    # byte-identical to the templates the final runs served with
    assert md5(TEMPLATES / "Qwen3-4B.nothink.jinja") == "e8c77c46538f0051dd7b0442da379cbb"
    assert md5(TEMPLATES / "Qwen3-4B-Thinking-2507.nothink.jinja") == "6c10af56c34d858bfb06966532174863"
    assert (TEMPLATES / "Qwen3-4B.nothink.jinja").read_text().startswith("{%- set enable_thinking = false %}")
    assert "<think>\\n\\n</think>" in (TEMPLATES / "Qwen3-4B-Thinking-2507.nothink.jinja").read_text()


def test_missing_template_is_an_error(cfg, tmp_path):
    from omegaconf import OmegaConf

    from ask2distill.tau2 import serve
    OmegaConf.update(cfg, "models.templates.teacher", str(tmp_path / "nope.jinja"))
    with pytest.raises(serve.ServeError, match="enables thinking"):
        serve.template_for(cfg, "teacher")
    with pytest.raises(serve.ServeError, match="no role"):
        serve.template_for(cfg, "judge")


def test_synrl_node_maps(cfg, a2d_dirs):
    from ask2distill.tau2 import serve
    n0 = _by_kind(serve.plan_servers(cfg, "synrl", node_index=0))
    n1 = _by_kind(serve.plan_servers(cfg, "synrl", node_index=1))
    assert [n0[k].replicas for k in ("stu", "usr", "tea")] == [2, 3, 3]
    assert [n1[k].replicas for k in ("stu", "usr", "tea")] == [4, 2, 2]
    assert (n0["stu"].served_name, n0["usr"].served_name, n0["tea"].served_name) == \
        ("Qwen3-4B", "Qwen3-4B-Instruct-2507", "Qwen3-4B-Thinking-2507")
    assert n0["stu"].model_path == str(a2d_dirs["models"] / "Qwen3-4B")
    assert n0["usr"].template is None and n0["tea"].template.endswith("Thinking-2507.nothink.jinja")
    assert {s.gpu_util for s in n0.values()} == {0.9} and {s.max_len for s in n0.values()} == {32768}
    with pytest.raises(serve.ServeError, match="out of range"):
        serve.plan_servers(cfg, "synrl", node_index=2)


def test_harvest_plan(cfg, a2d_dirs):
    from ask2distill.tau2 import serve
    specs = serve.plan_servers(cfg, "harvest", version="v7")
    assert [s.kind for s in specs] == ["gen", "stu", "usr", "tea"]
    gen = specs[0]
    assert gen.served_name == f"synthesizer_{AGENT_QWEN3_4B}_v7_step60"
    assert gen.model_path == str(a2d_dirs["ckpt"] / f"synthesizer_{AGENT_QWEN3_4B}_v7" / "hf" / "step_60")
    assert gen.template == str(TEMPLATES / "Qwen3-4B.nothink.jinja")
    assert {s.replicas for s in specs} == {2}
    assert serve.plan_servers(cfg, "harvest", version="v6")[0].served_name == \
        f"synthesizer_{AGENT_QWEN3_4B}_v6_step50"


def test_opd_user_simulator_per_arm(cfg):
    from ask2distill.tau2 import serve
    ours = serve.plan_servers(cfg, "opd", arm="ours")
    vanilla = serve.plan_servers(cfg, "opd", arm="vanilla")
    assert [(s.kind, s.served_name, s.replicas, s.max_len) for s in ours] == \
        [("usr", "Qwen3-4B-Instruct-2507", 4, 40960)]
    assert [s.replicas for s in vanilla] == [2]


def test_eval_names(cfg, a2d_dirs):
    from ask2distill.tau2 import serve
    assert serve.eval_served_name(cfg, "ours", 150) == f"{AGENT_QWEN3_4B}-ours-s150"
    assert serve.eval_served_name(cfg, "vanilla", 110) == f"{AGENT_QWEN3_4B}-vanilla-s110"
    (spec,) = serve.plan_servers(cfg, "eval", arm="ours", step=150)
    assert (spec.kind, spec.served_name, spec.replicas, spec.max_len) == \
        ("stu", f"{AGENT_QWEN3_4B}-ours-s150", 1, 40960)
    assert spec.model_path == str(a2d_dirs["ckpt"] / f"opd_{AGENT_QWEN3_4B}_ours" / "hf" / "step_150")
    assert spec.template == str(TEMPLATES / "Qwen3-4B.nothink.jinja")      # by role, not by name
    with pytest.raises(serve.ServeError, match="qwen"):
        serve.plan_servers(cfg, "eval", model="/m", served_name="my-model-s1")
    # a local stand-in for the OpenAI user simulator: own template, kind usr, no qwen check
    (usr,) = serve.plan_servers(cfg, "eval", model="/m", served_name="gpt-4o-mini")
    assert (usr.kind, usr.template) == ("usr", None)


def test_vllm_command_has_no_sampling_flags(cfg):
    from ask2distill.tau2 import serve
    spec = _by_kind(serve.plan_servers(cfg, "synrl"))["tea"]
    cmd = serve.vllm_command(spec, 20030, eager=False)
    assert cmd[:4] == ["vllm", "serve", spec.model_path, "--served-model-name"]
    assert "--tool-call-parser" in cmd and cmd[cmd.index("--tool-call-parser") + 1] == "hermes"
    assert "--enable-auto-tool-choice" in cmd and "--enforce-eager" not in cmd
    assert cmd[cmd.index("--chat-template") + 1] == spec.template
    for flag in ("--generation-config", "--override-generation-config", "--temperature", "--top-p"):
        assert flag not in cmd
    user = _by_kind(serve.plan_servers(cfg, "synrl"))["usr"]
    assert "--chat-template" not in serve.vllm_command(user, 20031, eager=True)
    assert "--enforce-eager" in serve.vllm_command(user, 20031, eager=True)


def test_gpu_assignment_and_sharing():
    from ask2distill.tau2 import serve
    specs = [serve.ServerSpec("stu", "/a", "Qwen3-4B", None, 2), serve.ServerSpec("usr", "/b", "U", None, 1)]
    slots = serve.assign_gpus(specs, ["0", "1", "2", "3"])
    assert [g for _, _, g in slots] == [["0"], ["1"], ["2"]]
    shared = serve.assign_gpus(specs, ["0"])
    assert [g for _, _, g in shared] == [["0"], ["0"], ["0"]]


def test_dry_run_start_writes_nothing(cfg, tmp_path, capsys):
    from ask2distill.tau2 import serve
    specs = serve.plan_servers(cfg, "synrl")
    out = tmp_path / "endpoints.txt"
    assert serve.start(specs, ["0", "1", "2", "3", "4", "5", "6", "7"], out, dry_run=True) == []
    text = capsys.readouterr().out
    assert not out.exists()
    assert text.count("vllm serve") == 8 and "--enforce-eager" not in text
    assert "ALL_READY" in text and "stu=2" in text and "usr=3" in text and "tea=3" in text


def test_smoke_plan_shares_gpu0(a2d_dirs, capsys, tmp_path):
    from ask2distill import config
    from ask2distill.tau2 import serve
    cfg = config.load(None, str(REPO_ROOT / "configs" / "smoke" / "tau2_smoke.yaml"))
    specs = serve.plan_servers(cfg, "synrl")
    serve.start(specs, ["0"], tmp_path / "e.txt", dry_run=True)
    text = capsys.readouterr().out
    assert text.count("CUDA_VISIBLE_DEVICES=0 vllm serve") == 3 and text.count("--enforce-eager") == 3


def test_endpoints_roundtrip(tmp_path):
    from ask2distill.tau2.envside import endpoints as E
    eps = [E.Endpoint("stu", "http://10.0.0.1:20030/v1", "Qwen3-4B"),
           E.Endpoint("stu", "http://10.0.0.2:20031/v1", "Qwen3-4B"),
           E.Endpoint("usr", "http://10.0.0.1:20032/v1", "Qwen3-4B-Instruct-2507")]
    (tmp_path / "e.txt").write_text("# comment\n" + E.format_lines(eps))
    back = E.read(tmp_path / "e.txt")
    assert back == eps and E.urls(back, "stu") == [eps[0].url, eps[1].url]
    assert E.served_name(back, "usr") == "Qwen3-4B-Instruct-2507"
    cfgj = E.model_config(E.by_kind(back, "stu"), "/x.json")
    assert cfgj == {"Qwen3-4B": [{"ip_addr": "10.0.0.1", "port": "20030"},
                                 {"ip_addr": "10.0.0.2", "port": "20031"}],
                    "vllm_model_config_path": "/x.json"}
    with pytest.raises(ValueError, match="unknown kind"):
        E.parse("judge http://a:1/v1 X\n")
