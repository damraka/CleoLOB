"""v0.7 world manifests and scenario provenance."""
from __future__ import annotations

import pytest

from lob.sim_v2 import SimulatorSpec
from lob.v07.protocol.taxonomy import TaxonomyError
from lob.v07.uncertainty import worlds as w


def test_deterministic_ids_and_composition() -> None:
    a = w.WorldManifest("G0_post", "draw-03", "POSTERIOR_SAMPLE", market_seed=7)
    b = w.WorldManifest("G0_post", "draw-03", "POSTERIOR_SAMPLE", market_seed=7, note="different note")
    c = w.WorldManifest("G0_post", "draw-03", "POSTERIOR_SAMPLE", market_seed=8)
    assert a.id == b.id != c.id and len(a.id) == 16
    comp = w.composition(w.plausible_set([a, c, w.WorldManifest("G3", "fit", "CALIBRATED")]))
    assert comp["worlds"] == 3 and comp["by_scenario"] == {"POSTERIOR_SAMPLE": 2, "CALIBRATED": 1}
    assert w.spec_hash(SimulatorSpec({"a": 1}, {})) == w.spec_hash(SimulatorSpec({"a": 1}, {}))


def test_schema_enforcement() -> None:
    with pytest.raises(TaxonomyError):
        w.WorldManifest("G0", "x", "REALISTIC")
    with pytest.raises(TaxonomyError):
        w.WorldManifest("G0", "x", "SYNTHETIC_STRESS")
    with pytest.raises(TaxonomyError):
        w.WorldManifest("G0", "x", "CALIBRATED", queue_model="exact_fifo")
    stress = w.WorldManifest("G0", "x", "SYNTHETIC_STRESS", structural_assumption="liquidity_drought")
    with pytest.raises(TaxonomyError):
        w.plausible_set([stress])
    a = w.WorldManifest("G0", "x", "CALIBRATED")
    with pytest.raises(TaxonomyError):
        w.plausible_set([a, a])
