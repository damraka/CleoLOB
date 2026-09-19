"""Leakage boundaries, complete stress families and frozen historical fits."""
import json

import numpy as np
import pandas as pd
import pytest

from lob.config import ResearchConfig, load_config
from lob.robustness import chronological_split, walk_forward_splits, run_stress_suite, run_chronological_calibration
from lob.robustness import verify_study


def frame(n=1000):
    return pd.DataFrame({"timestamp_us": np.arange(n, dtype=np.int64) * 1_000_000,
                         "log_return": np.zeros(n), "value": np.arange(n)})


def test_purge_removes_overlapping_outcomes_and_elapsed_time_embargo():
    parts = chronological_split(frame(), embargo_seconds=10, outcome_horizon_seconds=5)
    assert parts["train"].timestamp_us.max() + 15_000_000 < parts["validation"].timestamp_us.min()
    assert parts["validation"].timestamp_us.max() + 15_000_000 < parts["test"].timestamp_us.min()
    assert parts["test"].timestamp_us.max() == 994_000_000
    assert all(np.isnan(part.log_return.iloc[0]) for part in parts.values())
    assert not set(parts["train"].index) & set(parts["test"].index)


@pytest.mark.parametrize("change", ["duplicate", "reverse", "float", "negative"])
def test_bad_time_axes_never_sorted_or_coerced(change):
    data = frame()
    if change == "duplicate":
        data.loc[2, "timestamp_us"] = data.loc[1, "timestamp_us"]
    elif change == "reverse":
        data = data.iloc[::-1]
    elif change == "float":
        data["timestamp_us"] = data.timestamp_us.astype(float)
    else:
        data.loc[0, "timestamp_us"] = -1
    with pytest.raises(ValueError):
        chronological_split(data)


@pytest.mark.parametrize("kwargs", [dict(train_fraction=.9, validation_fraction=.2),
    dict(embargo_seconds=-1), dict(outcome_horizon_seconds=float("nan")),
    dict(embargo_seconds=10000), dict(train_fraction=True)])
def test_invalid_split_settings_reject(kwargs):
    with pytest.raises(ValueError):
        chronological_split(frame(), **kwargs)


def test_expanding_folds_have_disjoint_test_blocks_and_no_future_training():
    folds = list(walk_forward_splits(frame(), initial_train_rows=500, test_rows=100,
                                    embargo_seconds=10, outcome_horizon_seconds=3))
    seen = set()
    for fold in folds:
        train, test = fold["train"], fold["test"]
        assert train.timestamp_us.max() + 13_000_000 < test.timestamp_us.min()
        assert not seen & set(test.index)
        seen |= set(test.index)
        assert np.isnan(test.log_return.iloc[0])
    assert len(folds) == 5 and len(folds[-1]["train"]) > len(folds[0]["train"])
    with pytest.raises(ValueError):
        list(walk_forward_splits(frame(), initial_train_rows=10, test_rows=2, max_folds=5))


def small_config():
    return ResearchConfig.model_validate({"market": {"limit_rate": 0.0, "market_rate": 0.0,
        "cancel_rate": 0.0, "resilience": 0.0, "latency_base": 0.001, "latency_jitter": 0.0},
        "execution": {"quantity": 100, "horizon": 1.0, "decision_dt": .2},
        "evaluation": {"agents": ["twap", "ac"], "reference": "twap", "seeds": [1, 2], "bootstrap_samples": 100}})


def test_entire_stress_family_is_corrected_together_and_sealed(tmp_path):
    run = run_stress_suite(small_config(), tmp_path, profiles={"base": {}, "delay": {"market": {"latency_base": .1}}})
    result = json.loads((run / "result.json").read_text())
    assert result["status"] == "COMPLETE" and result["episodes_recorded"] == 8
    assert result["family_inference"] == "AVAILABLE"
    assert len(result["paired"]) == 2
    assert all(row["family_size"] == 2 and row["correction"] == "holm" for row in result["paired"])
    assert (run / "manifest.json").is_file()
    assert json.loads((run / "plan.json").read_text())["episodes"] == 8
    assert verify_study(run)["valid"]
    child_manifest = next((run / "experiments").glob("*/manifest.json"))
    child_manifest.write_text("{}")
    assert not verify_study(run)["valid"]


def test_invalid_scenario_retained_and_withholds_other_scenario_inference(tmp_path):
    run = run_stress_suite(small_config(), tmp_path, profiles={"base": {}, "empty": {
        "market": {"target_level_vol": 1}, "risk": {"kill_switch": True}}})
    result = json.loads((run / "result.json").read_text())
    assert result["status"] == "COMPLETE"  # complete test, unsuccessful economic outcomes
    assert result["outcomes"]["INVALID"] == 4
    assert result["family_inference"] == "WITHHELD" and result["paired"] == []
    assert len(pd.read_csv(run / "episodes.csv")) == 8


def test_design_resource_limit_and_favorable_seed_selection_rejected(tmp_path):
    with pytest.raises(ValueError, match="max_total"):
        run_stress_suite(small_config(), tmp_path, max_total_episodes=1)
    with pytest.raises(ValueError, match="overlays"):
        run_stress_suite(small_config(), tmp_path, profiles={"biased": {"evaluation": {"seeds": [1]}}})
    assert list(tmp_path.iterdir()) == []


def test_child_provenance_failure_cannot_be_reenabled_by_valid_episode_rows(tmp_path, monkeypatch):
    from lob import robustness
    original = robustness.read_experiment
    def invalidate(path):
        stored = original(path)
        stored["result"]["status"] = "INVALID"
        return stored
    monkeypatch.setattr(robustness, "read_experiment", invalidate)
    run = run_stress_suite(small_config(), tmp_path, profiles={"base": {}})
    result = json.loads((run / "result.json").read_text())
    assert result["status"] == "INVALID" and result["family_inference"] == "WITHHELD"
    assert not result["paired"] and not result["provenance_valid"]


def test_changed_child_model_invalidates_whole_family(tmp_path, monkeypatch):
    from lob import robustness
    original = robustness.read_experiment
    def replace_model(path):
        stored = original(path)
        stored["metadata"]["model"] = {"path": "different-model", "sha256": "wrong"}
        return stored
    monkeypatch.setattr(robustness, "read_experiment", replace_model)
    run = run_stress_suite(small_config(), tmp_path, profiles={"base": {}})
    result = json.loads((run / "result.json").read_text())
    assert result["status"] == "INVALID" and result["family_inference"] == "WITHHELD"


def test_settlement_config_roundtrips_and_budgets():
    cfg = load_config(overrides=("execution.settlement_timeout=2.0", "execution.settlement_poll_dt=0.02"), environ={})
    p = cfg.runner_params(101)
    assert p["settlement_timeout"] == 2.0 and p["settlement_poll_dt"] == .02
    with pytest.raises(ValueError):
        load_config(overrides=("execution.settlement_poll_dt=0.0000001",), environ={})


def write_l2(path, start):
    lines = ["exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount"]
    for i in range(1001):
        time = start + i * 1_000_000
        for side, price in (("bid", 99), ("ask", 101)):
            lines.append(f"venue,PAIR,{time},{time},{'true' if i == 0 else 'false'},{side},{price},{100 + i % 40}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_real_pipeline_fits_once_and_scores_disjoint_intervals(tmp_path):
    train, test = tmp_path / "day1.csv", tmp_path / "day2.csv"
    write_l2(train, 1_000_000)
    write_l2(test, 86_400_000_000)
    run = run_chronological_calibration(train, None, test, tmp_path / "studies")
    result = json.loads((run / "result.json").read_text())
    assert result["status"] == "COMPLETE"
    assert result["rows"]["internal_test"] > 0 and result["rows"]["test"] == 1001
    assert len(result["walk_forward"]) == 4
    assert result["frozen_model_sha256"] and (run / "model.json").is_file()
    assert verify_study(run)["valid"]
    for fold in result["walk_forward"]:
        assert fold["train_last_timestamp_us"] + 61_000_000 < fold["test_first_timestamp_us"]


def test_repeated_dataset_not_accepted_as_holdout(tmp_path):
    train = tmp_path / "day.csv"
    write_l2(train, 1_000_000)
    with pytest.raises(ValueError, match="distinct"):
        run_chronological_calibration(train, None, train, tmp_path / "out")
