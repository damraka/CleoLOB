import pytest

from lob import scaling


def test_scaling_covers_real_paths_and_counts_work():
    config = {"depths": [2], "event_counts": [3], "snapshot_every": [1, 3],
              "calibration_rows": [128], "experiment_sizes": [1], "repeats": 1, "warmup": 1}
    result = scaling.scaling_suite(config)
    rows = result["measurements"]
    assert not result["production_latency"]
    assert {r["family"] for r in rows} == {
        "engine_mutation_and_snapshot", "mbo_mutation_and_snapshot", "l2_csv_replay",
        "mbo_jsonl_replay", "calibration_fit_and_diagnostic", "json_serialization_roundtrip",
        "heuristic_policy_evaluation", "experiment_orchestration"}
    assert len(rows) == 10
    for row in rows:
        assert len(row["batch_nanoseconds"]) == 1
        assert row["units_per_batch"] > 0
        assert row["python_peak_bytes_separate_pass"] > 0
        assert row["process_rss"] is None
    l2 = next(r for r in rows if r["family"] == "l2_csv_replay")
    assert l2["units_per_batch"] == 7


def test_batch_memory_does_not_contaminate_timing(monkeypatch):
    ticks = iter([0, 1000, 2000, 5000])
    monkeypatch.setattr(scaling.time, "perf_counter_ns", lambda: next(ticks))
    calls = []
    result = scaling._measure_batch(lambda: calls.append(1), units=2, repeats=2, warmup=1)
    assert len(calls) == 4  # warmup, two measured passes, one memory pass
    assert result["batch_nanoseconds"] == [1000, 3000]
    assert result["mean_nanoseconds_per_unit"] == 1000
    assert result["batch_units_per_second"] == pytest.approx([2_000_000, 2_000_000 / 3])


@pytest.mark.parametrize("key,value", [("event_counts", [False]), ("depths", [100, 10]),
    ("experiment_sizes", [10000]), ("repeats", 0), ("warmup", True), ("snapshot_every", [1, 1])])
def test_scaling_workload_limits(key, value):
    config = {**scaling.DEFAULT_CONFIG, key: value}
    with pytest.raises(ValueError):
        scaling.validate_config(config)
