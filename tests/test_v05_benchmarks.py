"""M11: the v0.5 benchmark suite runs every workload and reports honest units."""
from __future__ import annotations

from lob import benchmarks_v05 as bench


def test_suite_measures_every_workload(monkeypatch):
    monkeypatch.setattr(bench, "SIZES", {"small": 1})
    rows = bench.suite(scale=1.0, repeats=1)
    names = {r["workload"] for r in rows}
    assert names == {"mbo_parsing_normalization", "lifecycle_replay", "mbo_to_l2_aggregation", "fill_bound_updates",
                     "historical_execution_l2", "impact_resilience", "calibration_v2_summary", "extended_simulation",
                     "policy_evaluation_episodes", "evidence_serialization"}
    for row in rows:
        assert row["throughput_per_second"] > 0 and row["peak_traced_python_bytes"] > 0
        assert row["process_rss"] == "NOT_AVAILABLE"
    per_call = [r for r in rows if "per_call" in r]
    assert all(r["per_call"]["p50_us"] <= r["per_call"]["p99_us"] for r in per_call) and per_call
