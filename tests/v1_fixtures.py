"""Shared v1 test helpers: isolated workspaces (never the repository ledger) and a minimal valid protocol."""
from __future__ import annotations

from pathlib import Path

import pytest

from lob.v1.protocol import document as D


def minimal_protocol(**overrides) -> dict:
    doc = {
        "schema": D.SCHEMA, "protocol_id": "test-protocol", "research_question": "test",
        "statuses": list(D.STATUSES), "evidence_levels": dict(D.EVIDENCE_LEVELS),
        "roles": ["consumed", "development", "explore", "calibrate", "confirm", "replicate", "vault"],
        "namespaces": ["dev", "explore", "calibrate", "confirm", "replicate", "vault"],
        "pools": {"explore": {"role": "explore"}, "confirm": {"role": "confirm", "streams": ["s"]},
                  "replicate": {"role": "replicate", "streams": ["s"]}, "vault": {"role": "vault", "streams": ["s"]}},
        "promotion_rules": {"min_power": 0.8, "planning_effect": "pilot - 1 cluster SE", "min_pilot_clusters": 5,
                            "compute_ceiling_hours": 10},
        "statistics": {"independent_unit": "calendar date", "primary_test": "cluster_t", "min_clusters": 7,
                       "alpha": 0.05, "strictest_family_alpha": 0.0125, "correction": "holm"},
        "realism_gate": {}, "resources": {}, "amendment_policy": "numbered, hashed, posthoc flagged",
        "referenced_configs": [], "fidelities": {},
    }
    doc.update(overrides)
    return doc


@pytest.fixture
def ws(tmp_path) -> D.Workspace:
    root = tmp_path / "root"
    root.mkdir()
    return D.Workspace(Path(root), allow_dirty=True)
