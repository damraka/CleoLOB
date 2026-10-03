"""v0.7 hierarchical variance decomposition: synthetic recovery of between/within components."""
from __future__ import annotations

import numpy as np

from lob.v07.transfer.matrix import variance_decomposition


def test_variance_decomposition_recovers_planted_structure() -> None:
    rng = np.random.default_rng(0)
    vectors, groups = {}, {}
    for inst, shift in (("eth", 0.0), ("btc", 5.0)):
        for m in range(6):
            name = f"{inst}-{m}"
            # coefficient 0: instrument-level (local); coefficient 1: global (stable across everything)
            vectors[name] = np.array([shift + rng.normal(0, 0.1), rng.normal(1.0, 0.1)])
            groups[name] = inst
    out = variance_decomposition(vectors, groups)
    assert out["coefficients"][0]["between_share"] > 0.95
    assert out["coefficients"][1]["between_share"] < 0.5
    assert out["groups"] == ["btc", "eth"]
