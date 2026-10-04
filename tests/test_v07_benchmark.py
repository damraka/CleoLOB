"""v0.7 frozen benchmark tasks, the public replication subset and the evidence bundle."""
from __future__ import annotations

import json
from pathlib import Path

from lob.v07.benchmark import tasks
from lob.v07.reports import export

ROOT = Path(__file__).resolve().parents[1]


def test_public_subset_is_deterministic_and_matches_expectation() -> None:
    first, second = tasks.run(tasks.PUBLIC), tasks.run(tasks.PUBLIC)
    assert tasks.public_digest(first) == tasks.public_digest(second)
    assert first["queue"]["bound_order_holds"] == 500 and first["execution"]["conserved"]
    assert first["reconstruction"]["tapes_equal"] and first["identifiability"]["effective_rank"] == 2
    expected = ROOT / "examples/studies/v07/public-subset-expected.json"
    if expected.is_file():
        assert json.loads(expected.read_text(encoding="utf-8"))["digest"] == tasks.public_digest(first)


def test_task_definitions_are_frozen() -> None:
    defs = tasks.definitions()
    stored = json.loads((ROOT / "configs/v07/benchmark-tasks.json").read_text(encoding="utf-8"))
    assert stored["tasks"].keys() == defs["tasks"].keys() and stored["public_subset"] == defs["public_subset"]
    assert tasks.task_performance()["simulated_seconds_per_wall_second"] > 0


def test_bundle_export_and_verification(tmp_path) -> None:
    extra = tmp_path / "note.md"
    extra.write_text("synthetic bundle content\n", encoding="utf-8")
    manifest = export.export(tmp_path / "bundle", root=ROOT, generated={"note.md": extra})
    assert "TIER_4" in manifest["tiers"] and "not independent" in manifest["meaning"]
    assert export.verify(tmp_path / "bundle")["valid"]
    (tmp_path / "bundle" / "note.md").write_text("tampered C:/Users/someone", encoding="utf-8")
    report = export.verify(tmp_path / "bundle")
    assert not report["valid"] and len(report["issues"]) >= 2
