"""Safe parsing, composition precedence, immutability and resource admission."""
import json

import pytest
from pydantic import ValidationError

from lob.config import ResearchConfig, config_diff, config_hash, load_config


def test_defaults_are_deeply_immutable_and_hash_roundtrips():
    cfg = load_config(environ={})
    with pytest.raises(ValidationError):
        cfg.execution.quantity = 1
    assert isinstance(cfg.evaluation.seeds, tuple)
    assert config_hash(cfg) == config_hash(ResearchConfig.model_validate(json.loads(cfg.model_dump_json())))
    assert cfg.runner_params(101)["strict_model"] is True


@pytest.mark.parametrize("overlay", [
    {"unknown": 1}, {"market": {"bogus": 2}}, {"execution": {"quantity": -1}},
    {"market": {"offset_p": 2}}, {"market": {"latency_base": -1}},
    {"execution": {"decision_dt": 11}}, {"fees": {"taker_bps": float("nan")}},
    {"risk": {"max_order_qty": True}}, {"evaluation": {"seeds": [1, 1]}},
    {"evaluation": {"agents": ["ppo", "twap"]}}, {"evaluation": {"agents": ["twap"], "reference": "ac"}},
    {"resources": {"max_episodes": 1}}, {"execution": {"decision_dt": 0.00001}},
    {"market": {"lot_size": 3}}, {"market": {"market_rate": "6"}},
])
def test_invalid_research_config_rejected(overlay):
    with pytest.raises(ValidationError):
        ResearchConfig.model_validate(overlay)


def test_yaml_composition_cli_environment_precedence(tmp_path):
    (tmp_path / "base.yaml").write_text("execution:\n  quantity: 100\n  horizon: 5.0\n", encoding="utf-8")
    path = tmp_path / "research.yaml"
    path.write_text("extends: base.yaml\nexecution:\n  quantity: 200\n", encoding="utf-8")
    cfg = load_config(path, ("execution.quantity=400",), {"CLEO__EXECUTION__QUANTITY": "300"})
    assert cfg.execution.quantity == 400 and cfg.execution.horizon == 5
    assert load_config(path, environ={}).execution.quantity == 200


@pytest.mark.parametrize("name,content", [
    ("a.json", '{"name":"a","name":"b"}'),
    ("a.yaml", "name: a\nname: b\n"),
    ("a.yaml", "market: &market {}\nrisk: *market\n"),
    ("a.yaml", "!!python/object/apply:os.system [echo unsafe]"),
    ("a.yaml", "extends: a.yaml"),
    ("a.yaml", "- not\n- a mapping"),
])
def test_duplicate_unsafe_recursive_and_wrong_roots_fail(tmp_path, name, content):
    import yaml
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    with pytest.raises((ValueError, yaml.YAMLError)):
        load_config(path, environ={})


def test_unknown_overrides_cannot_escape_schema():
    with pytest.raises(ValueError):
        load_config(overrides=("execution.quanitty=1",), environ={})
    with pytest.raises(ValueError):
        load_config(environ={"CLEO__EXECUTION__QUANTITY": "1.5"})


def test_config_diff_and_schema_describe_research_fields():
    left = ResearchConfig()
    right = load_config(overrides=("execution.quantity=400",), environ={})
    assert config_diff(left.model_dump(), right.model_dump()) == [
        {"field": "execution.quantity", "before": 2000, "after": 400}]
    schema = ResearchConfig.model_json_schema()
    assert schema["additionalProperties"] is False
    assert all("description" in field for field in schema["properties"].values())
