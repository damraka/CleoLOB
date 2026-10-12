"""v0.8 market core: aggregate-queue book semantics, simulation determinism, generator interface."""
from __future__ import annotations

import json

import numpy as np
import pytest

from lob.v08.generators import (REGISTRY, HawkesGenerator, MarketGenerator, MixtureGenerator, PoissonGenerator,
                                QueueReactiveGenerator, RegimeSwitchingGenerator, StateDependentGenerator, register)
from lob.v08.generators.ecology import EcologyGenerator
from lob.v08.generators.endogenous import EndogenousGenerator
from lob.v08.market.book import BUY, SELL, AggBook, BookError
from lob.v08.market.frame import SUMMARY_NAMES, EventFrame, summaries
from lob.v08.market.sim import BookModel, MarketSim, simulate


def make_book(depth=5, levels=10, **kw) -> AggBook:
    return AggBook(refill=lambda level, rng: depth, rng=np.random.default_rng(0), levels=levels, mid_ticks=1000, **kw)


# ----------------------------------------------------------------------------- book


def test_fifo_allocation_background_ahead_then_strategy_then_behind() -> None:
    book = make_book(depth=5)
    oid = book.submit_limit(BUY, 3, 0.0)             # joins behind 5 background lots at the bid touch
    order = book.own[oid]
    assert order.ahead == 5 and book.bq[0] == 8
    book.add(BUY, 0, 4)                              # arrives behind the strategy order
    assert order.ahead == 5 and book.bq[0] == 12
    filled, _, _ = book.market(SELL, 6, 1.0)         # 5 background, then 1 of ours
    assert filled == 6 and order.remaining == 2 and order.ahead == 0
    book.market(SELL, 2, 2.0)
    assert order.status == "filled" and order.remaining == 0 and book.bq[0] == 4
    book.check()


@pytest.mark.parametrize("position,expected_ahead", [("front", 2.0), ("back", 5.0)])
def test_background_cancel_position_moves_queue_ahead(position, expected_ahead) -> None:
    book = make_book(depth=5, cancel_position=position)
    oid = book.submit_limit(BUY, 1, 0.0)
    book.add(BUY, 0, 5)                              # 5 ahead of us, 5 behind
    assert book.cancel(BUY, 0, 3) == 3
    assert book.own[oid].ahead == pytest.approx(expected_ahead)
    book.check()


def test_uniform_cancel_reduces_ahead_between_extremes() -> None:
    book = make_book(depth=10, cancel_position="uniform")
    oid = book.submit_limit(BUY, 1, 0.0)
    book.add(BUY, 0, 10)
    book.cancel(BUY, 0, 4)
    assert 6.0 <= book.own[oid].ahead <= 10.0


def test_cancel_never_removes_strategy_volume() -> None:
    book = make_book(depth=2)
    oid = book.submit_limit(BUY, 3, 0.0)
    assert book.cancel(BUY, 0, 100) == 2             # only background
    assert book.own[oid].remaining == 3 and book.bq[0] == 3
    book.check()


def test_depleted_touch_shifts_out_and_refills() -> None:
    book = make_book(depth=5)
    ask = book.ask_px
    filled, notional, fills = book.market(BUY, 7, 0.0)
    assert filled == 7 and fills == [(ask, 5), (ask + 1, 2)] and book.ask_px == ask + 1
    assert book.aq[0] == 3 and book.aq[-1] == 5 and len(book.aq) == book.levels
    book.check()


def test_addition_inside_spread_creates_new_touch() -> None:
    book = make_book(depth=5)
    book.market(BUY, 5, 0.0)                         # spread becomes 2
    assert book.spread() == 2
    bid = book.bid_px
    book.add(BUY, -1, 3)
    assert book.bid_px == bid + 1 and book.bq[0] == 3 and book.spread() == 1
    book.check()


def test_walk_cost_does_not_mutate_and_hidden_fraction_replenishes() -> None:
    book = make_book(depth=5)
    before = list(book.aq)
    notional, fillable = book.walk_cost(BUY, 12)
    assert fillable == 12 and book.aq == before
    hidden = make_book(depth=5, hidden_fraction=1.0)
    ask = hidden.ask_px
    _, _, fills = hidden.market(BUY, 9, 0.0)
    assert fills[:2] == [(ask, 5), (ask, 4)]         # the consumed touch replenished once at the same price


def test_self_trade_is_refused() -> None:
    book = make_book(depth=5)
    book.submit_limit(SELL, 2, 0.0)
    with pytest.raises(BookError):
        book.market(BUY, 10, 1.0, strategy=True)


def test_random_operation_sequence_preserves_invariants() -> None:
    rng = np.random.default_rng(5)
    book = AggBook(refill=lambda level, r: int(r.integers(1, 8)), rng=rng, levels=20, mid_ticks=500)
    for i in range(3000):
        u = rng.random()
        side = BUY if rng.random() < 0.5 else SELL
        if u < 0.3:
            book.add(side, int(rng.integers(-2, 6)), int(rng.integers(1, 6)))
        elif u < 0.6:
            book.cancel(side, int(rng.integers(0, 6)), int(rng.integers(1, 6)))
        elif u < 0.85:
            book.market(side, int(rng.integers(1, 8)), float(i), strategy=False)
        elif u < 0.95 and not any(o.side == -side for o in book.resting()):
            book.submit_limit(side, int(rng.integers(1, 4)), float(i), improve=int(rng.integers(0, 2)))
        else:
            for o in book.resting()[:1]:
                book.cancel_own(o.oid)
        book.check()


# ----------------------------------------------------------------------------- simulation


def test_simulation_is_seed_deterministic_and_seed_sensitive() -> None:
    g = HawkesGenerator()
    a, b, c = g.simulate(400, seed=11), g.simulate(400, seed=11), g.simulate(400, seed=12)
    for name in ("counts", "bid_px", "depth", "volume", "mid"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
    assert not np.array_equal(a.counts, c.counts)


def test_frame_round_trip_and_summaries() -> None:
    f = PoissonGenerator().simulate(600, seed=1)
    g = EventFrame.from_arrays(f.to_arrays(), f.meta)
    assert np.array_equal(f.counts, g.counts) and np.array_equal(f.depth, g.depth)
    s = summaries(f)
    assert s.shape == (len(SUMMARY_NAMES),) and np.all(np.isfinite(s))
    assert abs(s[0] - f.counts[:, :2].sum(1).mean()) < 1e-12
    t, _, q, _ = f.trades()
    assert q.sum() == pytest.approx(f.volume.sum())


def test_frame_slice_keeps_trade_clock() -> None:
    f = PoissonGenerator().simulate(600, seed=2)
    part = f.slice(100, 300)
    t, _, q, _ = part.trades()
    assert part.t0 == pytest.approx(f.t0 + 10.0)
    assert q.sum() == pytest.approx(part.volume.sum())


ALL = (PoissonGenerator, HawkesGenerator, QueueReactiveGenerator, RegimeSwitchingGenerator, StateDependentGenerator,
       MixtureGenerator, EcologyGenerator, EndogenousGenerator)


@pytest.mark.parametrize("cls", ALL, ids=lambda c: c.name)
def test_generator_interface_and_serialization_round_trip(cls) -> None:
    g = cls()
    d = json.loads(json.dumps(g.to_dict()))
    h = MarketGenerator.from_dict(d)
    assert h.name == g.name and set(h.capabilities) == set(g.capabilities)
    fa, fb = g.simulate(200, seed=4, warmup_bins=20), h.simulate(200, seed=4, warmup_bins=20)
    assert np.array_equal(fa.counts, fb.counts) and np.array_equal(fa.mid, fb.mid)
    ll = g.loglik(fa)
    assert np.all(np.isfinite(ll[fa.valid])) and g.score(fa)["bins"] == 200
    for key, (lo, hi) in g.parameter_space().items():
        assert lo < hi
        assert key in g.get_flat()


def test_serialized_version_mismatch_is_refused() -> None:
    d = PoissonGenerator().to_dict()
    d["version"] = "999"
    with pytest.raises(ValueError):
        MarketGenerator.from_dict(d)


def test_registry_refuses_duplicates_and_unknown_capabilities() -> None:
    assert {"poisson", "hawkes", "queue_reactive", "regime_switching", "state_dependent", "mixture"} <= set(REGISTRY)

    class Dup(PoissonGenerator):
        name = "poisson"

    with pytest.raises(ValueError):
        register(Dup)

    class Bad(PoissonGenerator):
        name = "bad_caps_test"
        capabilities = frozenset({"telepathy"})

    with pytest.raises(ValueError):
        register(Bad)


# ----------------------------------------------------------------------------- synthetic recovery (fits)


def test_poisson_fit_recovers_rates() -> None:
    truth = PoissonGenerator({"market": 0.8, "add": 3.0, "cancel": 2.6, "market_buy_share": 0.6,
                              "add_bid_share": 0.5, "cancel_bid_share": 0.5})
    fit = PoissonGenerator().fit(truth.simulate(4000, seed=3))
    assert fit.params["market"] == pytest.approx(0.8, rel=0.1)
    assert fit.params["add"] == pytest.approx(3.0, rel=0.05)
    assert fit.params["market_buy_share"] == pytest.approx(0.6, abs=0.04)


def test_hawkes_fit_detects_self_excitation() -> None:
    truth = HawkesGenerator()
    frame = truth.simulate(5000, seed=8)
    fit = HawkesGenerator().fit(frame)
    poisson = PoissonGenerator().fit(frame)
    assert fit.params["branching"] > 0.2
    assert np.nansum(fit.loglik(frame)) > np.nansum(poisson.loglik(frame))


def test_regime_switching_fit_separates_regimes() -> None:
    truth = RegimeSwitchingGenerator()
    frame = truth.simulate(6000, seed=2)
    fit = RegimeSwitchingGenerator().fit(frame)
    totals = np.asarray(fit.params["rates"]).sum(1)
    assert totals[1] / totals[0] > 2.0
    assert np.nansum(fit.loglik(frame)) > np.nansum(PoissonGenerator().fit(frame).loglik(frame))


def test_queue_reactive_and_state_dependent_fits_are_finite() -> None:
    frame = QueueReactiveGenerator().simulate(3000, seed=5)
    qr = QueueReactiveGenerator().fit(frame)
    assert qr.params["gamma"][4] > 0          # cancellations grow with queue size in the truth
    sd = StateDependentGenerator().fit(frame)
    assert np.all(np.isfinite(sd.loglik(frame)[frame.valid]))


def test_mixture_predictive_is_between_components() -> None:
    frame = HawkesGenerator().simulate(2000, seed=6)
    a, b = PoissonGenerator().fit(frame), HawkesGenerator().fit(frame)
    mix = MixtureGenerator.of([a, b], [0.5, 0.5])
    la, lb, lm = (np.nansum(g.loglik(frame)) for g in (a, b, mix))
    assert min(la, lb) - 1.0 <= lm <= max(la, lb) + 1.0
    assert lm >= max(la, lb) - np.log(2) - 1e-6        # Bayesian mixture loses at most log(2) to the best component


def test_ecology_scaling_and_endogenous_response() -> None:
    eco = EcologyGenerator()
    stressed = eco.scaled("liquidity_taker", 3.0)
    x = np.asarray([1.0, 0.0, 3.0, 10.0, 10.0, 0.0])
    s0 = eco.expected(eco.initial_state(), x)
    s1 = stressed.expected(stressed.initial_state(), x)
    assert s1[0] > s0[0] and np.allclose(s1[2:], s0[2:])
    endo = EndogenousGenerator.wrap(PoissonGenerator())
    state = endo.initial_state(simulate=True)
    base = endo.expected(state, x)
    state = endo.update(state, np.zeros(6, int), x, own=np.asarray([20.0, 0.0]))
    after = endo.expected(state, x)
    assert after[3] < base[3] and after[5] > base[5]   # strategy buying depletes ask additions, raises ask cancels


def test_book_model_fit_from_frame_samples() -> None:
    model = BookModel(add_level_p=0.3, market_size_mean=3.0, limit_size_mean=4.0)
    frame = simulate(PoissonGenerator(book=model), model, 3000, seed=1)
    fitted = BookModel.fit(frame)
    assert fitted.add_level_p == pytest.approx(0.3, abs=0.08)
    assert fitted.market_size_mean == pytest.approx(3.0, rel=0.15)


def test_market_sim_strategy_messages_are_time_ordered() -> None:
    sim = MarketSim(PoissonGenerator(), BookModel(), seed=1)
    order = []
    sim.run_bin([(0.05, 1, lambda: order.append("late")), (0.01, 0, lambda: order.append("early"))])
    assert order == ["early", "late"]


def test_regression_uniform_cancels_keep_integer_queue_positions() -> None:
    """Regression: continuous cancel positions made ``ahead`` fractional; FIFO then left an epsilon-sized
    strategy remainder resting at an emptied touch ("strategy volume at an empty touch")."""
    rng = np.random.default_rng(7)
    for trial in range(200):
        book = AggBook(refill=lambda level, r: 5, rng=np.random.default_rng(trial), levels=10, mid_ticks=1000)
        oid = book.submit_limit(BUY, int(rng.integers(1, 4)), 0.0)
        book.add(BUY, 0, int(rng.integers(1, 10)))
        for _ in range(int(rng.integers(1, 4))):
            book.cancel(BUY, 0, int(rng.integers(1, 4)))
        assert float(book.own[oid].ahead).is_integer()
        book.market(SELL, book.bq[0], 1.0)            # consume exactly the visible touch
        book.check()
        assert book.own[oid].status == "filled" and book.own[oid].remaining == 0


def test_regression_strategy_order_never_leaves_the_window() -> None:
    """Regression: a long rally of touch improvements dropped the level holding a strategy bid."""
    book = make_book(depth=5, levels=10)
    oid = book.submit_limit(BUY, 2, 0.0)
    for _ in range(30):                                # widen the spread, then improve the bid touch
        book.market(BUY, book.aq[0], 0.0)
        book.add(BUY, -1, 1)
    book.check()
    order = book.own[oid]
    assert order.status == "resting" and book.index(BUY, order.price) is not None
    assert len(book.bq) > book.levels
    book.cancel_own(oid)
    for _ in range(40):
        book.market(SELL, book.bq[0], 0.0)
    book.check()
    assert len(book.bq) == book.levels               # the grown window shrinks back


def test_fixed_share_survives_a_zero_weight_dominant_component() -> None:
    # regression: with switch 0 a component weight can reach exactly 0; when that component later has the
    # highest log-likelihood every weighted term underflowed and math.log(0) raised (ood-study-v08 INVALID)
    from lob.v08.generators.families import fixed_share
    comp = np.array([[0.0, -1000.0], [-1000.0, 0.0], [-2.0, -1.0]])
    out, path = fixed_share(comp, np.array([0.5, 0.5]), 0.0)
    assert np.all(np.isfinite(out))
    assert out[1] == pytest.approx(-1000.0)
    assert np.allclose(path.sum(1), 1.0)
    # the ordinary (non-underflow) path is unchanged
    ok = np.array([[-1.0, -2.0], [-0.5, -0.7]])
    w = np.array([0.3, 0.7])
    expected0 = np.log(0.3 * np.exp(-1.0) + 0.7 * np.exp(-2.0))
    assert fixed_share(ok, w, 0.01)[0][0] == pytest.approx(expected0, abs=1e-12)
