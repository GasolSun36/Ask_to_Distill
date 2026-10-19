"""A new pair built from configs/pairs/TEMPLATE_custom.yaml dry-runs every stage (CPU).

The template is filled in the way the README's "Using your own student-teacher
pair" section describes -- two small local Qwen3 models that share a tokenizer,
δ_ref copied from a calibration, the chosen synthesizer step -- once as a math
pair (the template's bases) and once as a code pair (the template's CODE notes
applied). Every stage then resolves its configuration and prints its command
under ``--dry-run``; a renamed or missing key fails here.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ask2distill import cli
from ask2distill import config as C

TEMPLATE = C.PAIRS_DIR / "TEMPLATE_custom.yaml"
STUDENT, TEACHER = "Qwen3-1.7B", "Qwen3-4B-Instruct-2507"


def _fill(domain: str) -> dict:
    """The template as a user fills it in (README, step 1-4)."""
    doc = yaml.safe_load(TEMPLATE.read_text())
    name = f"{domain.capitalize()}_{STUDENT}_{TEACHER}"
    doc["pair"] = name
    doc["models"].update(student="${paths.models}/" + STUDENT, teacher="${paths.models}/" + TEACHER,
                         qwen3_nothink=True)
    doc["models"]["hf"]["student"] = {"repo": f"Qwen/{STUDENT}", "revision": "a" * 40}
    doc["models"]["hf"]["teacher"] = {"repo": f"Qwen/{TEACHER}", "revision": "b" * 40}
    doc["reward"]["delta_ref"] = 1.2345
    doc["select"]["chosen_step"] = 40
    doc["bank"]["row_id_prefix"] = f"{domain}_custom"
    doc["opd"]["chat_template_args"] = "+data.apply_chat_template_kwargs.enable_thinking=False"
    if domain == "code":                      # the template's CODE notes
        doc["_base_"] = ["configs/base/opd.yaml", "configs/base/code.yaml", "configs/base/eval_code.yaml"]
        doc["reward"]["gen_bs"] = 4
        doc["synrl"]["save_freq"] = 5
        doc["select"]["sweep"] = {"steps": list(range(5, 61, 5))}
        doc["filter"].update(slices=[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
                             agg_order=[0, 1, 2, 3])
        del doc["filter"]["aggregate"]
        doc["bank"]["pool"] = "${filter.pool}"
        doc["opd"]["arms"]["vanilla"]["train"] = "${paths.data}/real/OpenCodeReasoning-10k.parquet"
        doc["eval"] = {"prompt_style": "Qwen3NoThink"}
    return doc


@pytest.fixture(params=["math", "code"])
def custom(request, tmp_path, monkeypatch):
    for var, d in (("A2D_OUT", "out"), ("A2D_CKPT", "ckpt")):
        monkeypatch.setenv(var, str(tmp_path / d))
    monkeypatch.delenv("A2D_CONFIG", raising=False)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(_fill(request.param), sort_keys=False))
    return request.param, path


def test_template_is_not_a_pair_and_needs_filling():
    assert "TEMPLATE_custom" not in C.list_pairs()
    doc = yaml.safe_load(TEMPLATE.read_text())
    assert doc["reward"]["delta_ref"] is None                 # set after `a2d calibrate`
    text = TEMPLATE.read_text()
    assert text.count("FILL") >= 8 and "CODE:" in text


def test_filled_template_validates(custom):
    domain, path = custom
    cfg = C.load(None, path)
    assert cfg.domain == domain
    assert C.validate(cfg) == []
    assert cfg.reward.diversity.mode == "bleu" and cfg.calib.n_questions == 200
    assert cfg.calib.max_new == cfg.opd.max_response_length == cfg.reward.max_new
    assert Path(str(cfg.synrl.topics)).name == f"{domain}.yaml"


def test_unfilled_delta_ref_blocks_synthesizer_rl(custom, capsys):
    _, path = custom
    cfg = C.load(None, path, ["reward.delta_ref=null"])
    assert any("delta_ref is not set" in w for w in C.validate(cfg))
    assert cli.main(["calibrate", "--config", str(path), "reward.delta_ref=null", "--dry-run"]) == 0
    from ask2distill.synrl.reward_config import RewardConfigError
    with pytest.raises(RewardConfigError, match="a2d calibrate"):
        cli.main(["synrl", "--config", str(path), "reward.delta_ref=null", "--dry-run"])


STAGES = [["calibrate"], ["prompts", "build"], ["synrl"], ["select", "sweep"], ["select", "pick"],
          ["harvest"], ["harvest", "--shard", "0"], ["filter"], ["filter", "--slice", "0"], ["aggregate"],
          ["bank"], ["opd", "--arm", "ours"], ["opd", "--arm", "vanilla"],
          ["merge", "--arm", "ours", "--step", "5"], ["eval", "{domain}", "--arm", "ours", "--step", "5"]]


@pytest.mark.parametrize("stage", STAGES, ids=lambda s: " ".join(s).replace("{domain}", "<domain>"))
def test_every_stage_dry_runs(custom, stage, capsys):
    domain, path = custom
    argv = [w.replace("{domain}", domain) for w in stage]
    assert cli.main([*argv, "--config", str(path), "--dry-run"]) == 0
    assert "# ---- command ----" in capsys.readouterr().out
