"""v0.6 realism measurement, sketches, distances, families and equivalence."""
from __future__ import annotations

import math

import numpy as np
import pytest

from lob.v06 import observables as ob
from lob.v06 import realism as rl
from lob.v06.tape import SAMPLE_DT, BookTape


def _tape(n: int = 6000, *, seed: int = 0, trade_rate: float = 0.2, levels: int = 10, tick: float = 0.05,
          walk: bool = True) -> BookTape:
    """A synthetic but internally consistent book: random-walk mid, noisy depths, random trades."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * SAMPLE_DT
    steps = rng.choice([-1, 0, 0, 0, 0, 0, 0, 0, 0, 1], size=n) if walk else np.zeros(n, int)
    best_bid = 4000 + np.cumsum(steps)
    wide = rng.random(n) < 0.05
    best_ask = best_bid + 1 + wide
    bp = (best_bid[:, None] - np.arange(levels)[None, :]) * tick
    ap = (best_ask[:, None] + np.arange(levels)[None, :]) * tick
    bq = np.round(rng.gamma(2.0, 20.0, size=(n, levels)), 0) + 1
    aq = np.round(rng.gamma(2.0, 20.0, size=(n, levels)), 0) + 1
    k = rng.poisson(trade_rate * n * SAMPLE_DT)
    trade_t = np.sort(rng.uniform(0, n * SAMPLE_DT, k))
    trade_s = rng.choice([-1.0, 1.0], size=k)
    index = np.minimum((trade_t / SAMPLE_DT).astype(int), n - 1)
    trade_p = np.where(trade_s > 0, ap[index, 0], bp[index, 0])
    trade_q = rng.lognormal(1.0, 1.0, size=k)
    return BookTape(t, bp, bq, ap, aq, trade_t, trade_p, trade_q, trade_s, tick)


def _sketches(tapes, design):
    return [ob.sketch(ob.measure(b), design) for tape in tapes for b in tape.blocks(600)]


@pytest.fixture(scope="module")
def setup():
    dev = [_tape(18000, seed=s) for s in (1, 2)]
    raws = [ob.measure(b) for tape in dev for b in tape.blocks(600)]
    design = ob.build_design(raws, tick_bps=0.05 / 200 * 1e4)
    return dev, raws, design


def test_tape_validation() -> None:
    tape = _tape(100)
    with pytest.raises(ValueError, match="regular"):
        BookTape(tape.t * 2, tape.bp, tape.bq, tape.ap, tape.aq, tape.trade_t, tape.trade_p, tape.trade_q,
                 tape.trade_s, tape.tick)
    with pytest.raises(ValueError, match="shape"):
        BookTape(tape.t, tape.bp[:-1], tape.bq, tape.ap, tape.aq, tape.trade_t, tape.trade_p, tape.trade_q,
                 tape.trade_s, tape.tick)
    with pytest.raises(ValueError, match="tick"):
        BookTape(tape.t, tape.bp, tape.bq, tape.ap, tape.aq, tape.trade_t, tape.trade_p, tape.trade_q,
                 tape.trade_s, 0.0)
    with pytest.raises(ValueError, match="positive"):
        tape.rescaled(0)
    assert len(_tape(6000).blocks(600)) == 1 and len(_tape(5999).blocks(600)) == 0


def test_measure_known_book_changes() -> None:
    n, levels, tick = 40, 10, 0.5
    t = np.arange(n) * SAMPLE_DT
    bp = np.tile((100 - np.arange(levels)) * tick, (n, 1))
    ap = np.tile((101 + np.arange(levels)) * tick, (n, 1))
    bq = np.full((n, levels), 10.0)
    aq = np.full((n, levels), 10.0)
    bq[10:, 2] = 14.0          # +4 addition at bid level 3 between samples 9 and 10
    aq[20:, 0] = 7.0           # -3 at best ask between 19 and 20: explained by a 3-lot buy print
    bq[30:, 1] = 4.0           # -6 at bid level 2 between 29 and 30: no print, a cancellation
    trade_t = np.asarray([1.95])
    tape = BookTape(t, bp, bq, ap, aq, trade_t, np.asarray([101 * tick]), np.asarray([3.0]), np.asarray([1.0]), tick)
    raw = ob.measure(tape)
    assert raw["add_size"].tolist() == [4.0]
    assert raw["cancel_size"].tolist() == [6.0]
    assert np.allclose(raw["spread_bps"], (0.5 / 50.25) * 1e4)
    assert np.all(raw["level_gap_ticks"] == 1.0)
    assert raw["trade_size"].tolist() == [3.0]
    markov = raw["markov"].reshape(5, 5)
    assert markov.sum() == n - 2
    assert markov[:, ob.STATES.index("trade")].sum() == 1


def test_measure_invalid_samples_are_excluded() -> None:
    tape = _tape(6000, seed=3)
    tape.bp[100:200] = np.nan
    tape.bq[100:200] = 0
    raw = ob.measure(tape)
    assert len(raw["spread_bps"]) == 600 - 10
    assert np.isfinite(raw["spread_bps"]).all()


def test_runs_require_completed_bounded_runs() -> None:
    flag = np.asarray([1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1], bool)
    valid = np.ones_like(flag)
    assert ob._runs(flag, valid).tolist() == [2.0, 3.0]
    valid[6] = False
    assert ob._runs(flag, valid).tolist() == [2.0]


def test_reference_mid_ranks_with_ties() -> None:
    reference = ob.Reference.fit(np.asarray([0.0, 0.0, 0.0, 1.0, 2.0]))
    assert np.allclose(reference.pit(np.asarray([0.0, 1.0, 2.0])), [0.3, 0.7, 0.9])
    assert reference.pit(np.asarray([-5.0]))[0] == 0.0 and reference.pit(np.asarray([9.0]))[0] == 1.0
    roundtrip = ob.Reference.from_dict(reference.to_dict())
    assert np.allclose(roundtrip.pit(np.asarray([0.5])), reference.pit(np.asarray([0.5])))


def test_sketches_are_additive(setup) -> None:
    dev, raws, design = setup
    pooled = ob.pool([ob.sketch(r, design) for r in raws[:3]])
    merged: dict = {}
    for key in raws[0]:
        values = [r[key] for r in raws[:3]]
        merged[key] = (tuple(np.concatenate([v[i] for v in values]) for i in range(2))
                       if isinstance(values[0], tuple) else
                       np.sum(values, axis=0) if key in {"markov", "nonzero_r1", "recovered", "depletion", "replenished"}
                       else np.concatenate(values))
    direct = ob.sketch(merged, design)
    for key in pooled:
        assert np.allclose(pooled[key], direct[key]), key
    stacked = ob.stack([ob.sketch(r, design) for r in raws[:3]])
    again = ob.weighted(stacked, np.asarray([1.0, 1.0, 1.0]))
    assert all(np.allclose(again[k], pooled[k]) for k in pooled)


def test_design_is_development_only_and_complete(setup) -> None:
    _, raws, design = setup
    assert set(design["dist"]) == set(ob.DIST_SOURCES)
    assert set(design["references"]) == set(ob.PIT_SOURCES)
    assert len(design["quintiles"]) == 4
    with pytest.raises(ValueError, match="too small"):
        ob.build_design(raws[:0] + [{**raws[0], "trade_size": np.asarray([1.0])}], tick_bps=2.5)


def test_registry_documents_every_component() -> None:
    rows = ob.registry()
    assert {r["family"] for r in rows} == set(ob.FAMILIES)
    for row in rows:
        for key in ("definition", "units", "capability", "sampling", "invalid_when", "min_n"):
            assert row[key], (row["name"], key)
    assert len({r["name"] for r in rows}) == len(rows)


def test_distribution_metrics_known_values() -> None:
    spec = {"transform": "linear", "lo": -10.0, "hi": 10.0, "bins": ob.BINS, "scale": 2.0}
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 20000)
    a, b, c = ob._hist(x, spec), ob._hist(x, spec), ob._hist(x + 1.0, spec)
    same = rl.distribution_metrics(a, b, spec)
    assert same["w1"] == 0 and same["ks"] == 0 and same["energy"] == 0 and same["jsd"] == 0
    shifted = rl.distribution_metrics(a, c, spec)
    assert shifted["w1"] == pytest.approx(1.0, abs=0.08)
    assert shifted["w1_normalized"] == pytest.approx(0.5, abs=0.04)
    assert 0 < shifted["jsd"] <= 1 and 0 < shifted["ks"] <= 1
    assert shifted["quantile_errors_normalized"]["q50"] == pytest.approx(0.5, abs=0.05)
    # 1-D energy distance of two normals shifted by 1: E|X-Y|*2 - E|X-X'| - E|Y-Y'|
    exact = 2 * (2 * 1.0 * (0.3989422804 * 2 ** 0.5 * math.exp(-1 / 4) + 0.5 * (1 - 2 * 0.2397500611))
                 ) - 2 * (2 / math.sqrt(math.pi))
    assert shifted["energy"] == pytest.approx(exact, rel=0.1)


def test_overflow_mass_counts_in_w1() -> None:
    spec = {"transform": "linear", "lo": 0.0, "hi": 1.0, "bins": ob.BINS, "scale": 1.0}
    a = ob._hist(np.full(100, 0.5), spec)
    b = ob._hist(np.full(100, 5.5), spec)
    assert rl.distribution_metrics(a, b, spec)["w1"] == pytest.approx(5.0, abs=0.01)


def test_identical_and_shifted_simulations(setup) -> None:
    dev, _, design = setup
    h = ob.pool(_sketches(dev, design))
    same = rl.compare(h, h, design)
    assert same["objective"] == pytest.approx(0.0, abs=1e-12)
    other = ob.pool(_sketches([_tape(18000, seed=9)], design))
    different = ob.pool(_sketches([_tape(18000, seed=9, trade_rate=2.0, walk=False)], design))
    near, far = rl.compare(h, other, design), rl.compare(h, different, design)
    assert far["objective"] > near["objective"]
    assert far["families"]["event_activity"]["error"] > near["families"]["event_activity"]["error"]
    assert far["families"]["returns"]["error"] > near["families"]["returns"]["error"]


def test_too_few_samples_are_not_evaluable(setup) -> None:
    dev, _, design = setup
    h = ob.pool(_sketches(dev, design))
    tiny = ob.pool(_sketches([_tape(6000, seed=4, trade_rate=0.0)], design))
    result = rl.compare(h, tiny, design)
    assert result["components"]["trade_size"]["error"] is None
    assert result["components"]["hill_trade_size"]["error"] is None
    assert result["families"]["sizes"]["evaluable"] < result["families"]["sizes"]["components"]


def test_objective_uses_scales_and_skips_missing_families() -> None:
    families = {f: {"error": 1.0} for f in ob.FAMILIES}
    families["tail"]["error"] = None
    scales = {f: 2.0 for f in ob.FAMILIES}
    assert rl.objective(families, scales) == pytest.approx(0.5)
    assert rl.objective(families) == pytest.approx(1.0)
    assert rl.objective({f: {"error": None} for f in ob.FAMILIES}) is None
    scales["sizes"] = 0.0   # floored at 0.05
    assert rl.objective(families, scales) > 0.5


@pytest.mark.parametrize("upper,lower,margin,status", [
    (0.4, 0.1, 0.5, "EQUIVALENT_WITHIN_MARGIN"), (0.9, 0.6, 0.5, "FAILED_MARGIN"), (0.6, 0.4, 0.5, "NOT_ESTABLISHED"),
    (None, 0.1, 0.5, "NOT_EVALUABLE"), (0.4, 0.1, None, "NOT_EVALUABLE"), (math.nan, 0.1, 0.5, "NOT_EVALUABLE")])
def test_equivalence_status(upper, lower, margin, status) -> None:
    assert rl.equivalence_status(upper, lower, margin) == status
