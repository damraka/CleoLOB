"""v0.7 impact subsystem: zero-size and zero-impact limits, monotone impact, recovery, TCA identity, zoo fits."""
from __future__ import annotations

import numpy as np
import pytest

from lob.sim_v2 import SimulatorSpec
from lob.v07.impact import metaorder as mo

CONFIG = {"tick_size": 100 / 2640, "initial_mid_ticks": 2640, "target_level_vol": 26, "limit_rate": 2.9,
          "market_rate": 0.14, "cancel_rate": 0.03, "offset_p": 0.05, "limit_qty_mean": 35, "market_qty_mean": 40,
          "resilience": 0.2, "max_events": 3_000_000}
WORLD = SimulatorSpec(CONFIG, {})


def test_zero_size_order_has_zero_cost_and_leaves_the_market_unchanged() -> None:
    order = mo.MetaOrder(quantity=0, horizon_s=30.0, recovery_s=10.0, warmup_s=10.0)
    out = mo.run_metaorder(WORLD, order, 3)
    assert out["filled"] == 0 and out["shortfall_bps"] == 0.0 and out["completion"] == 1.0
    sim = WORLD.build(3)
    sim.step(10.0)
    mids = [sim.book.mid()]
    while sim.t < 10.0 + 40.0 - 1e-9:
        sim.step(mo.SAMPLE_S)
        mids.append(sim.book.mid())
    assert out["path"]["mid"][: len(mids)] == mids[: len(out["path"]["mid"])]   # zero-impact limit
    t = mo.tca(out, quantity=0)
    assert t["total_bps"] == 0.0


def test_impact_is_monotone_in_size_on_average() -> None:
    small = [mo.run_metaorder(WORLD, mo.MetaOrder(quantity=10, horizon_s=30.0, recovery_s=10.0, warmup_s=10.0), s)
             for s in range(12)]
    large = [mo.run_metaorder(WORLD, mo.MetaOrder(quantity=400, horizon_s=30.0, recovery_s=10.0, warmup_s=10.0), s)
             for s in range(12)]
    assert np.mean([r["peak_move_bps"] for r in large]) > np.mean([r["peak_move_bps"] for r in small])
    assert np.mean([r["shortfall_bps"] for r in large]) > np.mean([r["shortfall_bps"] for r in small])


def test_recovery_after_the_parent_ends() -> None:
    runs = [mo.run_metaorder(WORLD, mo.MetaOrder(quantity=400, horizon_s=20.0, recovery_s=60.0, warmup_s=10.0), s)
            for s in range(8)]
    fractions = [r["recovery_fraction"] for r in runs if r["recovery_fraction"] is not None]
    assert fractions and np.mean([r["residual_impact_bps"] for r in runs]) <= np.mean([r["peak_move_bps"] for r in runs])


@pytest.mark.parametrize("quantity", [14, 200])
def test_tca_components_sum_to_shortfall(quantity) -> None:
    out = mo.run_metaorder(WORLD, mo.MetaOrder(quantity=quantity, horizon_s=30.0, recovery_s=5.0, warmup_s=10.0), 5)
    t = mo.tca(out, quantity=quantity)
    assert t["identity_error"] < 1e-6
    assert sum(t["components_bps"].values()) == pytest.approx(t["total_bps"])


def test_urgency_front_loads_the_schedule() -> None:
    flat, urgent = mo.MetaOrder(quantity=100).schedule(), mo.MetaOrder(quantity=100, urgency=3.0).schedule()
    assert flat[0] == urgent[0] == 0 and flat[-1] == urgent[-1] == 100 and urgent[5] > flat[5]


def test_zoo_fits_known_laws() -> None:
    q = np.linspace(1, 100, 40)
    lin = mo.fit_linear(q, 0.3 * q)
    assert lin["a"] == pytest.approx(0.3)
    sigma, volume = np.full(40, 2.0), np.full(40, 400.0)
    sq = mo.fit_sqrt(q, 0.7 * sigma * np.sqrt(q / volume), sigma, volume)
    assert sq["Y"] == pytest.approx(0.7)
    assert mo.r2(q, q) == 1.0
    out = mo.run_metaorder(WORLD, mo.MetaOrder(quantity=200, horizon_s=20.0, recovery_s=5.0, warmup_s=10.0), 1)
    prop = mo.fit_propagator([out])
    assert prop["model"] == "propagator" and prop["beta"] in (0.2, 0.5, 1.0, 2.0)
