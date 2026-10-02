"""v0.7 generators: event extraction, count models, event-driven engine, regime switching (synthetic data only)."""
from __future__ import annotations

import numpy as np
import pytest

from lob.engine import Side
from lob.v06.tape import tape_from_simulator
from lob.v07.data.tape import build_tape
from lob.v07.generators import counts as cm
from lob.v07.generators.engine import BundleSimulator, EmpiricalTables, GeneratorSpec, bundle_config
from lob.v07.generators.events import MARKS, extract
from lob.v07.generators.regime import RegimeSpec, exit_rates
from tests.v07_fixtures import write_tardis

BASE = {"tick_size": 100 / 2640, "initial_mid_ticks": 2640, "target_level_vol": 20, "max_events": 2_000_000}


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    files = write_tardis(tmp_path_factory.mktemp("gen"), seconds=900.0)
    tape, _ = build_tape(files["l2"], files["trades"], tick=0.05)
    return extract(tape.rescaled(10.0))


def test_extraction_counts_and_samples(bundles) -> None:
    assert bundles.counts.shape[1] == len(MARKS) and bundles.valid.mean() > 0.9
    totals = bundles.counts[bundles.valid].sum(0)
    assert totals[2] > 0 and totals[3] > 0 and totals[0] + totals[1] > 0
    for mark in MARKS:
        assert len(bundles.sizes[mark]) == totals[MARKS.index(mark)]
    assert {k.split("@")[0] for k in bundles.offsets} == set(MARKS[2:])


def test_hawkes_fit_reproduces_mean_rates(bundles) -> None:
    model = cm.HawkesCounts.fit(bundles.counts, bundles.state, bundles.valid)
    assert all(info["converged"] for info in model.fit_info.values())
    rates = model.rates(model.initial(), bundles.state[bundles.valid].mean(0))
    assert np.all(np.isfinite(rates)) and np.all(rates >= 0)


def _tables(b) -> EmpiricalTables:
    return EmpiricalTables(b.sizes, b.offsets)


@pytest.mark.parametrize("family", ["G1", "G3", "G4"])
def test_generator_tapes_are_valid_and_deterministic(bundles, family) -> None:
    if family == "G1":
        model = cm.HawkesCounts.fit(bundles.counts, bundles.state, bundles.valid)
    elif family == "G3":
        model = cm.ConditionalCounts.fit(bundles.counts, bundles.state, bundles.valid)
    else:
        pytest.importorskip("torch")
        model = cm.GRUCounts.fit(bundles.counts, bundles.state, bundles.valid, seed=1, sequences=300, length=20,
                                 epochs=1, hidden=8, threads=1)
    spec = GeneratorSpec(family, BASE, model, _tables(bundles))
    a = tape_from_simulator(spec, 3, seconds=120.0, warmup=10.0)
    b = tape_from_simulator(spec, 3, seconds=120.0, warmup=10.0)
    assert a.valid.mean() > 0.5 and len(a.trade_t) > 0  # mechanics and determinism, not realism
    np.testing.assert_array_equal(a.bq, b.bq)
    assert not np.array_equal(a.bq, tape_from_simulator(spec, 4, seconds=120.0, warmup=10.0).bq)


def test_engine_refuses_zi_rates_and_supports_strategy_orders(bundles) -> None:
    model = cm.ConditionalCounts.fit(bundles.counts, bundles.state, bundles.valid)
    with pytest.raises(ValueError):
        BundleSimulator(bundle_config({**BASE, "seed": 1}, target_level_vol=20, max_events=10**6) .__class__(
            **{**bundle_config({**BASE, "seed": 1}, target_level_vol=20, max_events=10**6).__dict__,
               "limit_rate": 1.0}), model, _tables(bundles))
    sim = GeneratorSpec("G3", BASE, model, _tables(bundles)).build(5)
    sim.step(5.0)
    oid = sim.submit(Side.BUY, 5, "agent")
    sim.step(5.0)
    assert sim.orders[oid].is_terminal or sim.orders[oid].status.name in {"RESTING", "PARTIALLY_FILLED"}
    assert sim.diagnostics["bins"] >= 99
    sim.book.assert_invariants()


@pytest.mark.parametrize("position", ["uniform", "front", "back"])
def test_cancel_positions_keep_book_valid(bundles, position) -> None:
    model = cm.ConditionalCounts.fit(bundles.counts, bundles.state, bundles.valid)
    sim = GeneratorSpec("G3", BASE, model, _tables(bundles), cancel_position=position).build(2)
    sim.step(30.0)
    sim.book.assert_invariants()


def test_regime_switching_spec(monkeypatch) -> None:
    rates = exit_rates(["low", "low", "high", "high", "low"], 300.0)
    assert rates["high"] == pytest.approx(1.5 / 600) and rates["low"] == pytest.approx(1.5 / 900)
    base = {**BASE, "limit_rate": 2.0, "market_rate": 0.2, "cancel_rate": 0.05, "resilience": 0.01,
            "offset_p": 0.2, "limit_qty_mean": 5, "market_qty_mean": 5}
    spec = RegimeSpec("G2", {"low": base, "high": {**base, "limit_rate": 6.0, "market_rate": 1.0}},
                      {"low": {"hawkes_alpha": 0.1, "hawkes_decay": 1.0}, "high": {"hawkes_alpha": 0.2,
                                                                                    "hawkes_decay": 1.0}},
                      {"low": 0.05, "high": 0.05})
    sim = spec.build(9)
    seen = {sim.regime(t) for t in np.linspace(0, 400, 200)}
    assert seen == {"low", "high"}
    sim.step(60.0)
    assert sim.cfg.limit_rate in {2.0, 6.0}
    tape = tape_from_simulator(spec, 9, seconds=60.0, warmup=5.0)
    assert tape.valid.mean() > 0.5
