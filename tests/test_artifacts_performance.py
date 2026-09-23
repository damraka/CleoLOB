import json

import pytest

from lob import performance
from lob.artifacts import portable_provenance, seal_artifacts, verify_artifacts
from lob.performance import benchmark_suite, latency_summary


def test_compact_artifact_verification_detects_tamper_extra_missing_and_paths(tmp_path):
    (tmp_path / "result.json").write_text('{"status":"FAIL"}')
    seal_artifacts(tmp_path)
    assert verify_artifacts(tmp_path)["valid"]
    (tmp_path / "extra").write_text("unexpected")
    assert not verify_artifacts(tmp_path)["valid"]
    (tmp_path / "extra").unlink()
    (tmp_path / "result.json").write_text('{"status":"PASS"}')
    assert not verify_artifacts(tmp_path)["valid"]
    (tmp_path / "result.json").unlink()
    assert not verify_artifacts(tmp_path)["valid"]
    manifest = json.loads((tmp_path / "checksums.json").read_text())
    manifest["files"] = {"../outside.json": "0" * 64}
    (tmp_path / "checksums.json").write_text(json.dumps(manifest))
    assert any("unsafe" in x for x in verify_artifacts(tmp_path)["issues"])


def test_compact_provenance_does_not_record_python_executable_or_source_absolute_paths():
    metadata = portable_provenance()
    assert "executable" not in metadata["runtime"]
    assert "lob/engine.py" in metadata["source_files"]
    assert all(not p.startswith("/") and ":" not in p for p in metadata["source_files"])


def test_latency_summary_units_and_workload_reconciliation():
    summary = latency_summary([1000, 2000, 3000])
    assert summary["p50_ns"] == 2000
    assert summary["operations_per_second"] == 500000
    result = benchmark_suite(operations=3, repeats=1, warmup=1, sizes=("small",))
    assert not result["production_latency"]
    for key in ("book_modify", "snapshot_top5", "l2_update", "mbo_update"):
        row = result["sizes"]["small"][key]
        assert row["samples"] == 3
        assert len(row["timed_call_operations_per_second"]) == 1
        assert len(row["batch_operations_per_second"]) == 1
        assert row["median_timed_call_operations_per_second"] > 0
        assert row["median_batch_operations_per_second"] > 0
    assert result["sizes"]["small"]["l2_replay"]["rows_per_batch"] == 23
    assert result["simulation_episode"]["samples"] == 3


def test_measure_separates_call_latency_from_wall_clock_throughput(monkeypatch):
    # Two individually measured calls take 10 ns each, but the separate
    # uninstrumented batch takes 60 ns wall-clock. The two rates therefore
    # must remain distinct.
    ticks = iter([0, 10, 20, 30, 100, 160])
    monkeypatch.setattr(
        performance.time,
        "perf_counter_ns",
        lambda: next(ticks),
    )

    def factory():
        calls = 0

        def operation():
            nonlocal calls
            calls += 1

        def check():
            assert calls == 2

        return operation, check

    result = performance._measure(
        factory,
        operations=2,
        repeats=1,
        warmup=0,
    )

    assert result["samples"] == 2
    assert result["p50_ns"] == 10
    assert result["operations_per_second"] == 100_000_000
    assert result["latency_derived_operations_per_second"] == 100_000_000

    assert result["timed_call_operations_per_second"] == [100_000_000]
    assert result["median_timed_call_operations_per_second"] == 100_000_000

    expected_wall_clock = 2 * 1e9 / 60
    assert result["batch_operations_per_second"] == pytest.approx(
        [expected_wall_clock]
    )
    assert result["median_batch_operations_per_second"] == pytest.approx(
        expected_wall_clock
    )


@pytest.mark.parametrize("kwargs", [{"operations": True}, {"repeats": 0}, {"warmup": -1},
                                    {"sizes": ()}, {"sizes": ("large",)}])
def test_benchmark_limits_reject_invalid_work(kwargs):
    with pytest.raises(ValueError):
        benchmark_suite(**kwargs)
