"""v0.7 power tooling."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lob.v07.protocol import power

ROOT = Path(__file__).resolve().parents[1]


def test_mde_and_required_n_are_consistent() -> None:
    assert power.mde(1.0, alpha=0.05) == pytest.approx(1.96 + 0.8416, abs=1e-3)
    n = power.required_n(6.0, 1.0, alpha=0.05)
    assert power.mde(6.0 / n ** 0.5, alpha=0.05) <= 1.0 + 1e-9
    assert power.mde(1.0, alpha=0.05 / 28) > power.mde(1.0, alpha=0.05)
    assert power.pooled_world_se(6, 0, 1, 36) == pytest.approx(1.0)


def test_power_design_covers_confirmatory_families() -> None:
    families = json.loads((ROOT / "configs/v07/statistical-families.json").read_text(encoding="utf-8"))["families"]
    doc = power.design(families)
    confirmatory = {k for k, v in families.items() if v["kind"] == "confirmatory"}
    assert confirmatory <= set(doc["families"])
    assert doc["families"]["F8_robust_pairs"]["mde_bps"] < 2.0
