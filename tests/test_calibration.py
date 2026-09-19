"""Causal sampling, frozen train-only calibration and honest distribution drift."""
import csv
import gzip
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from lob.calibration import (
    FEATURES, evaluate_calibration, extract_features, fit_calibration,
    generate_features, load_calibration, save_calibration,
)
from lob.replay.l2 import L2_COLUMNS


def source(tmp_path, rows, name="observations.csv.gz"):
    path = tmp_path / name
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(L2_COLUMNS)
        for timestamp, snapshot, side, price, amount in rows:
            writer.writerow(["test", "NATIVE-CONTRACT", timestamp, timestamp,
                             str(snapshot).lower(), side, price, amount])
    return path


def frame(n=200, start=0):
    rng = np.random.default_rng(37)
    returns = rng.normal(0, .0001, n)
    values = pd.DataFrame({
        "timestamp_us": np.arange(start, start + n, dtype=np.int64) * 1_000_000,
        "valid": True, "mid_price": 100 * np.exp(np.cumsum(returns)),
        "spread_bps": rng.uniform(1, 3, n), "bid_depth5": rng.uniform(100, 200, n),
        "ask_depth5": rng.uniform(100, 200, n), "log_return": returns,
    })
    values["imbalance5"] = (values.bid_depth5 - values.ask_depth5) / (
        values.bid_depth5 + values.ask_depth5)
    values.loc[0, "log_return"] = np.nan
    values.attrs["calibration_metadata"] = {
        "exchange": "test", "symbol": "NATIVE-CONTRACT", "sample_interval_us": 1_000_000,
        "source_sha256": "f" * 64, "source_complete": True,
    }
    return values


def test_sampling_is_causal_on_capture_grid_and_preserves_native_depth(tmp_path):
    path = source(tmp_path, [
        (200_000, True, "bid", "99", "12.25"),
        (200_000, True, "ask", "101", "4.5"),
        (1_200_000, False, "ask", "101", "8.75"),
        (2_200_000, False, "bid", "99", "25"),
    ])
    samples, meta = extract_features(path)
    assert samples.timestamp_us.tolist() == [1_000_000, 2_000_000]
    assert samples.source_timestamp_us.tolist() == [200_000, 1_200_000]
    assert samples.ask_depth5.tolist() == [4.5, 8.75]
    assert samples.bid_depth5.tolist() == [12.25, 12.25]
    assert samples.log_return.isna().tolist() == [True, False]
    assert samples.log_return.iloc[1] == 0
    assert meta["source_complete"]
    assert len(meta["source_sha256"]) == 64
    assert "no contract/base/share conversion" in meta["units"]["bid_depth5"]


def test_grid_equal_group_uses_completed_atomic_group_and_eof_boundary(tmp_path):
    path = source(tmp_path, [
        (0, True, "bid", "99", "12"), (0, True, "ask", "101", "4"),
        (1_000_000, False, "bid", "99", "7"),
        (1_000_000, False, "ask", "101", "15"),
    ])
    values, _ = extract_features(path)
    assert values.timestamp_us.tolist() == [0, 1_000_000]
    assert values.bid_depth5.tolist() == [12, 7]
    assert values.ask_depth5.tolist() == [4, 15]


def test_stale_samples_stay_invalid_and_returns_do_not_bridge(tmp_path):
    path = source(tmp_path, [
        (0, True, "bid", "99", "12"), (0, True, "ask", "101", "4"),
        (4_000_000, False, "ask", "101", "8"),
        (5_000_000, False, "ask", "101", "10"),
    ])
    values, meta = extract_features(path, max_staleness_seconds=1)
    assert values.valid.tolist() == [True, True, False, False, True, True]
    assert values.log_return.isna().tolist() == [True, False, True, True, True, False]
    assert values.loc[2, "invalid_reason"] == "stale"
    assert np.isnan(values.loc[2, "mid_price"])
    assert meta["invalid_reasons"] == {"stale": 2}


def test_crossed_state_is_not_silently_repaired_or_fit(tmp_path):
    path = source(tmp_path, [
        (0, True, "bid", "99", "12"), (0, True, "ask", "101", "4"),
        (1_000_000, False, "bid", "102", "3"),
        (2_000_000, False, "bid", "102", "0"),
    ])
    values, _ = extract_features(path)
    assert values.valid.tolist() == [True, False, True]
    assert values.loc[1, "invalid_reason"] == "crossed_or_locked"
    assert values.log_return.isna().all()


def test_sample_cap_raises_without_partial_results(tmp_path):
    path = source(tmp_path, [
        (0, True, "bid", "99", "12"), (0, True, "ask", "101", "4"),
        (4_000_000, False, "ask", "101", "8"),
    ])
    with pytest.raises(ValueError, match="max_samples"):
        extract_features(path, max_samples=2)


@pytest.mark.parametrize("kwargs", [
    {"sample_interval_seconds": 0}, {"sample_interval_seconds": True},
    {"sample_interval_seconds": .0000001}, {"sample_interval_seconds": float("inf")},
    {"max_staleness_seconds": -1}, {"max_samples": True}, {"replay_options": {"depth": 10}},
])
def test_extraction_rejects_invalid_bounds(tmp_path, kwargs):
    with pytest.raises(ValueError):
        extract_features(tmp_path / "not-read", **kwargs)


def test_fit_is_frozen_and_bound_to_source_and_exact_training_frame():
    train = frame()
    model = fit_calibration(train, max_pool_rows=64)
    assert model.train_start_us == 0
    assert model.train_end_us == 199_000_000
    assert model.train_rows == 200
    assert model.train_joint_rows == 199
    assert model.train_source_sha256 == ("f" * 64,)
    assert len(model.pool) == 64
    with pytest.raises(FrozenInstanceError):
        model.train_end_us = 0
    with pytest.raises(TypeError):
        model.pool[0][0] = 0
    changed = train.copy()
    changed.loc[4, "spread_bps"] *= 2
    assert fit_calibration(changed).train_frame_sha256 != model.train_frame_sha256
    assert fit_calibration(train, max_pool_rows=64).model_sha256 == model.model_sha256


def test_identical_later_distribution_passes_and_evaluation_does_not_refit():
    model = fit_calibration(frame())
    before = model.model_sha256
    score = evaluate_calibration(model, frame(start=1000))
    assert score["status"] == "PASS"
    assert score["fit_on_holdout"] is False
    assert score["holdout_kind"] == "observed_feature_table"
    assert score["holdout_source_sha256"] == ["f" * 64]
    assert score["holdout_frame_sha256"] != model.train_frame_sha256
    assert score["metrics"]["spread_bps"]["normalized_quantile_distance"] == 0
    assert .94 <= score["metrics"]["spread_bps"]["holdout_coverage_of_train_95pct_range"] <= .96
    assert model.model_sha256 == before


def test_shifted_regime_fails_without_changing_parameters():
    model = fit_calibration(frame())
    before = model.to_dict()
    holdout = frame(start=1000)
    holdout["spread_bps"] *= 20
    score = evaluate_calibration(model, holdout)
    assert score["status"] == "FAIL"
    assert score["metrics"]["spread_bps"]["holdout_coverage_of_train_95pct_range"] == 0
    assert model.to_dict() == before


def test_constant_marginal_shift_is_detected_and_unchanged_constants_pass():
    train, holdout = frame(), frame(start=1000)
    train["spread_bps"] = holdout["spread_bps"] = 2.0
    model = fit_calibration(train)
    assert evaluate_calibration(model, holdout)["status"] == "PASS"
    holdout["spread_bps"] = 2.1
    assert evaluate_calibration(model, holdout)["metrics"]["spread_bps"]["status"] == "FAIL"


def test_overlap_rejected_not_mislabelled_out_of_sample():
    model = fit_calibration(frame())
    with pytest.raises(ValueError, match="strictly later"):
        evaluate_calibration(model, frame(start=199))


def test_sampling_interval_and_instrument_cannot_silently_change():
    model = fit_calibration(frame())
    holdout = frame(start=1000)
    meta = dict(holdout.attrs["calibration_metadata"], sample_interval_us=500_000)
    with pytest.raises(ValueError, match="returns must not bridge|intervals differ"):
        evaluate_calibration(model, holdout, meta)
    meta = dict(holdout.attrs["calibration_metadata"], symbol="OTHER")
    with pytest.raises(ValueError, match="exchange/symbol"):
        evaluate_calibration(model, holdout, meta)


@pytest.mark.parametrize("mutation,match", [
    (lambda values: values.assign(timestamp_us=0), "strictly increase"),
    (lambda values: values.assign(timestamp_us=values.timestamp_us.astype(float)), "exact integer"),
    (lambda values: values.assign(valid=1), "booleans"),
    (lambda values: values.assign(spread_bps=np.inf), "finite"),
    (lambda values: values.assign(bid_depth5=0), "physical domain"),
    (lambda values: values.assign(imbalance5=2), "physical domain"),
    (lambda values: values.assign(log_return=0), "first feature row"),
    (lambda values: values.assign(valid=False), "invalid samples"),
])
def test_unusable_training_frames_rejected(mutation, match):
    with pytest.raises(ValueError, match=match):
        fit_calibration(mutation(frame()))


def test_return_cannot_bridge_gaps_or_invalid_rows():
    values = frame()
    values = values.drop(index=3)
    with pytest.raises(ValueError, match="returns must not bridge"):
        fit_calibration(values)
    values = frame()
    values.loc[3, "valid"] = False
    values.loc[3, ["mid_price", *FEATURES]] = np.nan
    with pytest.raises(ValueError, match="returns must not bridge"):
        fit_calibration(values)
    values.loc[4, "log_return"] = np.nan
    assert fit_calibration(values).train_joint_rows == 197


def test_incomplete_source_and_insufficient_sample_rejected():
    with pytest.raises(ValueError, match="incomplete source"):
        fit_calibration(frame(), {"source_complete": False})
    with pytest.raises(ValueError, match="at least 32"):
        fit_calibration(frame(n=32))
    with pytest.raises(ValueError, match="requires at least 32"):
        evaluate_calibration(fit_calibration(frame()), frame(n=32, start=1000))


def test_generated_features_are_deterministic_joint_native_observations():
    model = fit_calibration(frame())
    left = generate_features(model, 200, seed=23)
    right = generate_features(model, 200, seed=23)
    pd.testing.assert_frame_equal(left, right)
    assert not left.equals(generate_features(model, 200, seed=24))
    assert left.timestamp_us.iloc[0] == model.train_end_us + model.sample_interval_us
    assert left.synthetic.all()
    assert left.log_return.isna().sum() == 1
    assert left.bid_depth5.between(100, 200).all()
    np.testing.assert_allclose(np.diff(np.log(left.mid_price)), left.log_return.iloc[1:])
    np.testing.assert_allclose(left.imbalance5,
                               (left.bid_depth5-left.ask_depth5)/(left.bid_depth5+left.ask_depth5))
    score = evaluate_calibration(model, left)
    assert score["fit_on_holdout"] is False
    assert score["holdout_kind"] == "synthetic_model_sample"


def test_model_roundtrip_and_write_once(tmp_path):
    model = fit_calibration(frame())
    path = save_calibration(model, tmp_path / "model.json")
    loaded = load_calibration(path)
    assert loaded == model
    pd.testing.assert_frame_equal(generate_features(loaded, 64), generate_features(model, 64))
    with pytest.raises(FileExistsError):
        save_calibration(model, path)


def test_corrupted_artifact_detected(tmp_path):
    path = save_calibration(fit_calibration(frame()), tmp_path / "model.json")
    data = json.loads(path.read_text())
    data["model"]["initial_mid_price"] *= 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_calibration(path)


def test_malformed_and_nonfinite_model_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"model": {}, "model": {}, "model_sha256": "bad"}')
    with pytest.raises(ValueError, match="duplicate calibration"):
        load_calibration(path)
    model = fit_calibration(frame())
    with pytest.raises(ValueError, match="finite"):
        replace(model, means=(np.nan, *model.means[1:]))
    with pytest.raises(ValueError, match="physical domain"):
        replace(model, pool=tuple((0, *row[1:]) for row in model.pool))
    with pytest.raises(ValueError, match="immutable tuples"):
        replace(model, means=list(model.means))
    with pytest.raises(ValueError, match="immutable tuples"):
        replace(model, pool=tuple(list(row) for row in model.pool))
    with pytest.raises(ValueError, match="limitations"):
        replace(model, limitations=())
