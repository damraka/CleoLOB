"""Observable calibration must drive the real engine and keep holdout data frozen."""
import csv
import gzip
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from lob.engine import SimConfig
from lob.zi_calibration import (
    DEFAULT_GATE, _normalize, _training_scales, candidate_configs, compare_summaries,
    evaluate_model, extract_top5, fit_model, load_model, refine_model, simulate_features, summarize,
)


def snapshots(path, *, start=0, count=20, symbol="TEST", amount=10.0, times=None):
    columns = ["exchange", "symbol", "timestamp", "local_timestamp"]
    columns += [f"{side}s[{level}].{field}" for level in range(5)
                for side in ("ask", "bid") for field in ("price", "amount")]
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for i, timestamp in enumerate(times if times is not None else range(start, start + count)):
            row = ["test", symbol, timestamp * 1_000_000, timestamp * 1_000_000]
            for level in range(5):
                row.extend([101 + level, amount + i, 99 - level, amount + 2 * i])
            writer.writerow(row)
    return path


def test_snapshot_sampling_grid_group_boundary_and_eof(tmp_path, monkeypatch):
    path = snapshots(tmp_path / "snapshots.csv.gz", times=[0, 1, 1, 2, 3])
    original = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **kw: original(*a, **{**kw, "chunksize": 2}))
    frame, metadata = extract_top5(path)
    assert frame.timestamp_us.tolist() == [0, 1_000_000, 2_000_000, 3_000_000]
    # Timestamp=1 crosses parser chunks; the completed group is row index2.
    assert frame.bid_depth_native.tolist() == [50, 70, 80, 90]
    assert frame.ask_depth_native.tolist() == [50, 60, 65, 70]
    assert metadata["rows"] == 5 and metadata["source_complete"]
    assert frame.return_bps.iloc[1] == 0


def test_snapshot_ofi_uses_best_quote_changes_and_impact_uses_whole_curve(tmp_path):
    path = snapshots(tmp_path / "snapshots.csv.gz", count=3)
    frame, _ = extract_top5(path)
    assert frame.snapshot_ofi.isna().tolist() == [True, False, False]
    # Same prices: bid quantity +2 minus ask quantity +1, normalized by mean top volume.
    assert frame.snapshot_ofi.iloc[1] == pytest.approx(1 / 21.5)
    assert frame["impact_0.1_bps"].iloc[0] == pytest.approx(100)
    assert frame["impact_1.0_bps"].iloc[0] == pytest.approx(300)
    assert (frame["impact_1.0_bps"] >= frame["impact_0.5_bps"]).all()


def test_stale_samples_are_invalid_and_returns_do_not_bridge(tmp_path):
    path = snapshots(tmp_path / "snapshots.csv.gz", times=[0, 5, 6])
    frame, _ = extract_top5(path, max_staleness_seconds=1)
    assert frame.valid.tolist() == [True, True, False, False, False, True, True]
    assert np.isnan(frame.return_bps.iloc[5])
    assert frame.return_bps.iloc[6] == 0


def test_snapshot_reader_rejects_out_of_order_and_enforces_resource_bound(tmp_path):
    path = snapshots(tmp_path / "bad.csv.gz", times=[0, 3, 2])
    with pytest.raises(ValueError, match="nondecreasing"):
        extract_top5(path)
    path = snapshots(tmp_path / "large.csv.gz", count=4)
    with pytest.raises(ValueError, match="resource limit"):
        extract_top5(path, max_rows=3)


def test_depth_mapping_uses_only_training_scale(tmp_path):
    train, meta = extract_top5(snapshots(tmp_path / "train.csv.gz", count=5, amount=10))
    later, _ = extract_top5(snapshots(tmp_path / "later.csv.gz", start=100, count=5, amount=100))
    scales = _training_scales([train], [meta])
    transformed = _normalize(later, scales)
    assert transformed.bid_depth.median() > 7
    assert scales["test:TEST"]["native_depth_per_1000_lots"] == pytest.approx(65)
    assert transformed.bid_depth.median() != pytest.approx(1)


def test_unseen_instrument_cannot_borrow_test_quantity_scale(tmp_path):
    train, meta = extract_top5(snapshots(tmp_path / "train.csv.gz", count=3))
    later, _ = extract_top5(snapshots(tmp_path / "later.csv.gz", count=3, symbol="UNSEEN"))
    with pytest.raises(ValueError, match="training-only quantity scale"):
        _normalize(later, _training_scales([train], [meta]))


def test_simulator_features_are_deterministic_and_change_with_engine_parameters():
    config = SimConfig(record_events=False)
    first = simulate_features(config, seeds=[123], seconds=10, warmup_seconds=1)
    again = simulate_features(config, seeds=[123], seconds=10, warmup_seconds=1)
    changed = simulate_features(replace(config, target_level_vol=1500), seeds=[123],
                                seconds=10, warmup_seconds=1)
    pd.testing.assert_frame_equal(first, again)
    assert not first.bid_depth.equals(changed.bid_depth)
    assert first.return_bps.iloc[0] != first.return_bps.iloc[0]
    assert first.bid_depth.max() < changed.bid_depth.max()


def test_seed_streams_never_create_returns_across_independent_paths():
    frame = simulate_features(SimConfig(), seeds=[11, 12], seconds=3, warmup_seconds=0)
    assert np.isnan(frame.return_bps.iloc[0])
    assert np.isnan(frame.return_bps.iloc[3])


def test_calibration_gate_checks_every_family_and_target_sample_count():
    summary = summarize(simulate_features(SimConfig(), seeds=[7], seconds=5, warmup_seconds=0))
    short = compare_summaries(summary, summary)
    assert short["status"] == "FAIL" and not short["sample_count_passed"]
    relaxed_size = {**DEFAULT_GATE, "minimum_target_samples": 2}
    assert compare_summaries(summary, summary, relaxed_size)["status"] == "PASS"
    changed = json.loads(json.dumps(summary))
    changed["features"]["spread_bps"]["quantiles"] = [100, 100, 100]
    result = compare_summaries(summary, changed, relaxed_size)
    assert result["status"] == "FAIL" and not result["families"]["spread"]["passed"]
    assert result["families"]["depth"]["passed"]


def test_frozen_search_design_is_reproducible_and_bounded():
    scales = {"x": {"tick_bps": .2}}
    first, second = candidate_configs(scales, 5), candidate_configs(scales, 5)
    assert first == second
    assert len({config.limit_rate for config in first}) == 5
    assert all(60 <= config.target_level_vol <= 250 for config in first)
    assert first[0].initial_mid_ticks == 50000


def test_fit_serializes_actual_simconfig_rejects_train_reuse_and_keeps_model_frozen(tmp_path):
    train = snapshots(tmp_path / "train.csv.gz", count=10)
    later = snapshots(tmp_path / "later.csv.gz", start=100, count=10)
    out = tmp_path / "fit"
    model = fit_model([train], candidate_count=1, simulation_seconds=4, warmup_seconds=0, out=out)
    assert load_model(out / "model.json") == model
    SimConfig(**model["simulator_config"])
    before = json.dumps(model, sort_keys=True)
    result = evaluate_model(model, [later], seeds=[42000], simulation_seconds=4)
    assert result["model_unchanged"]
    assert json.dumps(model, sort_keys=True) == before
    with pytest.raises(ValueError, match="reuses training"):
        evaluate_model(model, [train])
    with pytest.raises(ValueError, match="independent"):
        evaluate_model(model, [later], seeds=[41001])
    with pytest.raises(ValueError, match="new or empty"):
        fit_model([train], out=out)
    changed = json.loads((out / "model.json").read_text())
    changed["simulator_config"]["market_rate"] += 1
    (out / "model.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_model(out / "model.json")


def test_evaluation_rejects_earlier_dates_even_with_different_file_hash(tmp_path):
    train = snapshots(tmp_path / "train.csv.gz", start=100, count=5)
    earlier = snapshots(tmp_path / "earlier.csv.gz", start=0, count=5, amount=13)
    model = fit_model([train], candidate_count=1, simulation_seconds=3, warmup_seconds=0)
    with pytest.raises(ValueError, match="strictly earlier training"):
        evaluate_model(model, [earlier], simulation_seconds=3)


def test_amended_search_uses_fresh_confirmation_seeds_and_preserves_gate(tmp_path):
    path = snapshots(tmp_path / "training.csv.gz", count=5)
    model = fit_model([path], candidate_count=1, simulation_seconds=3, warmup_seconds=0)
    amendment = {"candidate_count": 25, "finalists": 2, "screening_seeds": [11],
                 "confirmation_seeds": [12], "screening_seconds_per_seed": 3,
                 "confirmation_seconds_per_seed": 4, "warmup_seconds": 0,
                 "search_seed": 9, "candidate_bounds": {}}
    original = json.dumps(model, sort_keys=True)
    out = tmp_path / "amended"
    result = refine_model(model, amendment, out=out)
    assert result == load_model(out / "model.json")
    assert json.dumps(model, sort_keys=True) == original
    assert result["protocol"]["gate"] == DEFAULT_GATE
    assert result["protocol"]["simulation_seeds"] == [11, 12]
    assert result["simulated_summary"]["samples"] == 4
    records = [json.loads(line) for line in (out / "candidate_results.jsonl").read_text().splitlines()]
    assert len(records) == 27
    assert [record["phase"] for record in records].count("confirmation") == 2
    assert result["selected_candidate"] in json.loads((out / "finalists.json").read_text())["candidate_ids"]
    with pytest.raises(ValueError, match="new or empty"):
        refine_model(model, amendment, out=out)
    with pytest.raises(ValueError, match="independent"):
        refine_model(model, {**amendment, "confirmation_seeds": [11]}, out=tmp_path / "overlap")
    snapshots(path, count=6)
    with pytest.raises(ValueError, match="source changed"):
        refine_model(model, amendment, out=tmp_path / "changed")
