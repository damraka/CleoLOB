"""M3: calibration-v2 simulator family, shared measurement operator, and study guards."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lob.calibration_v2 import (BASE_BOUNDS, base_config, block_bootstrap_loss_difference, block_samples,
                                empirical_size_tables, fit_family, gate_status, perturb_base, random_base,
                                simulated_summary, tape_from_simulator)
from lob.engine import ExchangeSimulator, SimConfig
from lob.observables_v2 import (FAMILIES, GATE, SAMPLE_DT, Tape, family_errors, group_prints, loss, measure,
                                summarize)
from lob.sim_v2 import ExtendedSimulator, FlowExtensions, SimulatorSpec


def test_extension_free_simulator_is_bit_identical_to_v04_engine():
    cfg = SimConfig(seed=11, record_events=False)
    a, b = ExchangeSimulator(cfg), ExtendedSimulator(cfg, FlowExtensions())
    for _ in range(40):
        a.step(0.1)
        b.step(0.1)
    assert [(t.time, t.price, t.qty) for t in a.book.trades] == [(t.time, t.price, t.qty) for t in b.book.trades]
    assert a.book.depth(10) == b.book.depth(10)


@pytest.mark.parametrize("kwargs", [
    {"inside_spread_prob": 1.2}, {"cancel_depth_exponent": 5}, {"imbalance_beta": -1.5},
    {"regime_multiplier": 0.5}, {"hawkes_alpha": 2.0, "hawkes_decay": 1.0},
    {"limit_size_quantiles": (3.0, 1.0)}, {"market_size_quantiles": (1.0,)}, {"inside_spread_prob": math.nan}])
def test_extension_parameters_are_bounded(kwargs):
    with pytest.raises(ValueError):
        FlowExtensions(**kwargs)


def test_signed_extensions_are_allowed():
    assert FlowExtensions(cancel_depth_exponent=-1.0, imbalance_beta=-0.5).active == [
        "depth_conditioned_cancellation", "imbalance_conditioned_market_flow"]


def _run(ext: FlowExtensions, seconds=60, seed=3, **cfg):
    sim = ExtendedSimulator(SimConfig(seed=seed, record_events=False, check_invariants=True, **cfg), ext)
    for _ in range(int(seconds / 0.5)):
        sim.step(0.5)
    return sim


def test_each_extension_runs_with_book_invariants():
    for ext in (FlowExtensions(limit_size_quantiles=(1, 4, 30), market_size_quantiles=(1, 2, 50)),
                FlowExtensions(inside_spread_prob=0.8), FlowExtensions(cancel_depth_exponent=1.5),
                FlowExtensions(imbalance_beta=0.8),
                FlowExtensions(regime_multiplier=4, regime_switch_rate=0.1, regime_high_share=0.3),
                FlowExtensions(hawkes_alpha=0.8, hawkes_decay=2.0)):
        sim = _run(ext, seconds=20)
        assert sim.event_count > 0 and ext.active


def test_empirical_sizes_follow_the_table():
    sim = _run(FlowExtensions(market_size_quantiles=(7.0, 7.0)), seconds=60, market_rate=5)
    takers = {t.taker_order_id: t for t in sim.book.trades if t.maker_owner == "ZI"}
    totals = {}
    for trade in sim.book.trades:
        totals[trade.taker_order_id] = totals.get(trade.taker_order_id, 0) + trade.qty
    assert takers and max(totals.values()) <= 7


def test_regime_path_is_seeded_and_changes_activity():
    ext = FlowExtensions(regime_multiplier=5, regime_switch_rate=0.05, regime_high_share=0.5, extension_seed=1)
    a = ExtendedSimulator(SimConfig(seed=4, record_events=False), ext)
    b = ExtendedSimulator(SimConfig(seed=4, record_events=False), ext)
    assert a._regime_switches == b._regime_switches
    states = {a.regime_high(t) for t in np.linspace(0, 5000, 400)}
    assert states == {True, False}


def test_hawkes_increases_trade_clustering():
    base = SimConfig(seed=9, record_events=False, market_rate=0.3)
    plain = simulated_summary(SimulatorSpec({**vars(base)}, {}), [1, 2], 300)
    excited = simulated_summary(SimulatorSpec({**vars(base)}, {"hawkes_alpha": 1.6, "hawkes_decay": 2.0}), [1, 2], 300)
    assert excited["trade_dispersion_index"] > plain["trade_dispersion_index"]


def test_group_prints_aggregates_one_aggressor():
    t, p, q, s = group_prints(np.array([1.0, 1.0, 2.0]), np.array([10.0, 11.0, 10.0]), np.array([1.0, 3.0, 2.0]),
                              np.array([1, 1, -1]), np.array([5, 5, 6]))
    assert list(q) == [4.0, 2.0] and p[0] == pytest.approx(10.75) and list(s) == [1, -1]


def _synthetic_tape(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    mid = 100 + np.cumsum(rng.choice([-0.05, 0, 0.05], size=n, p=[.05, .9, .05]))
    bp = mid[:, None] - 0.025 - 0.05 * np.arange(5)
    ap = mid[:, None] + 0.025 + 0.05 * np.arange(5)
    bq = rng.integers(1, 50, size=(n, 5)).astype(float)
    aq = rng.integers(1, 50, size=(n, 5)).astype(float)
    trade_t = np.sort(rng.uniform(0, n * SAMPLE_DT, 60))
    return Tape(np.arange(n) * SAMPLE_DT, bp, bq, ap, aq, trade_t, np.full(60, 100.0), rng.uniform(1, 30, 60),
                rng.choice([-1.0, 1.0], 60))


def test_measurement_and_errors_are_complete_and_zero_on_identity():
    summary = summarize(measure(_synthetic_tape()))
    errors = family_errors(summary, summary)
    assert set(errors) == set(FAMILIES)
    assert all(e["error"] == 0 or e["error"] is None for e in errors.values())
    assert loss(errors) == 0


def test_missing_simulated_quantities_are_penalised_not_ignored():
    target = summarize(measure(_synthetic_tape()))
    empty = _synthetic_tape()
    empty = Tape(empty.t, empty.bp, empty.bq, empty.ap, empty.aq, np.array([]), np.array([]), np.array([]), np.array([]))
    errors = family_errors(target, summarize(measure(empty)))
    assert errors["sizes"]["error"] == 10.0 and errors["sizes"]["passed"] is False


def test_invalid_samples_are_excluded():
    tape = _synthetic_tape()
    tape.ap[:100, 0] = tape.bp[:100, 0] - 1  # crossed
    samples = measure(tape)
    assert len(samples["spread_bps"]) == len(np.arange(0, len(tape.t), 10)) - 10


def test_gate_status_uses_frozen_threshold():
    target = summarize(measure(_synthetic_tape()))
    errors = family_errors(target, target)
    assert gate_status(errors, 0.99, 5000)["status"] == "ESTABLISHED"
    assert gate_status(errors, 0.90, 5000)["status"] == "FAILED"
    errors["spread"] = {"error": GATE + 0.01, "passed": False, "components": []}
    assert gate_status(errors, 0.99, 5000)["failed_families"] == ["spread"]


def test_search_is_bounded_and_deterministic():
    rng = np.random.default_rng(1)
    base = base_config(2.5)
    for _ in range(20):
        config = perturb_base(rng, random_base(rng, base))
        for name, (low, high) in BASE_BOUNDS.items():
            assert low <= config[name] <= high
    target = summarize(measure(tape_from_simulator(SimulatorSpec(base, {}), 1, seconds=60)))
    kwargs = dict(base=base, size_tables={}, baseline_best=None, seeds=[5], seconds=30, candidates=4, search_seed=2)
    a = fit_family("v04_zi_baseline", target, **kwargs)
    b = fit_family("v04_zi_baseline", target, **kwargs)
    assert a["best"]["config"] == b["best"]["config"] and a["candidates"] == 4


def test_size_tables_come_from_training_samples_only():
    samples = measure(_synthetic_tape())
    tables = empirical_size_tables(samples)
    assert len(tables["market_size_quantiles"]) == 21
    assert all(b >= a for a, b in zip(tables["market_size_quantiles"], tables["market_size_quantiles"][1:]))


def test_block_bootstrap_is_paired_and_seeded():
    tape = _synthetic_tape(n=12000)
    blocks = block_samples(tape, block_seconds=200)
    summary = summarize(measure(tape))
    other = dict(summary, spread_bps={"quantiles": [x * 2 for x in summary["spread_bps"]["quantiles"]], "n": 1})
    a = block_bootstrap_loss_difference(blocks, summary, other, samples=50, alpha=0.05, seed=3)
    b = block_bootstrap_loss_difference(blocks, summary, other, samples=50, alpha=0.05, seed=3)
    assert a == b and a["ci_high"] < 0 and a["blocks"] == 6
