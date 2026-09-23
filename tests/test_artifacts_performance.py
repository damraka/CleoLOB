import json

import pytest

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
        assert result["sizes"]["small"][key]["samples"] == 3
    assert result["sizes"]["small"]["l2_replay"]["rows_per_batch"] == 23
    assert result["simulation_episode"]["samples"] == 3


@pytest.mark.parametrize("kwargs", [{"operations": True}, {"repeats": 0}, {"warmup": -1},
                                    {"sizes": ()}, {"sizes": ("large",)}])
def test_benchmark_limits_reject_invalid_work(kwargs):
    with pytest.raises(ValueError):
        benchmark_suite(**kwargs)
