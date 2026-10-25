"""Audit Issue A (M0, M70): the v0.9 book-sample estimator cannot observe spread-improving additions; v1 can."""
import numpy as np
import pytest

from lob.v08.generators import PoissonGenerator
from lob.v08.market.sim import BookModel, simulate
from lob.v09.generators import classical as C9
from lob.v1.generators.classical import book as B

L = 10


def _frames_with_improvements(n_cycles: int = 60):
    """Frames where the bid touch depletes (spread 1 -> 2) and an order then joins inside the spread."""
    bid, ask, depth, vol = [], [], [], []
    for _ in range(n_cycles):
        # 1: tight book
        bid.append(1000), ask.append(1001), depth.append(np.full((2, L), 5.0)), vol.append([0.0, 0.0])
        # 2: the bid touch is consumed by a sell -> bid 999, spread 2
        d = np.full((2, L), 5.0)
        bid.append(999), ask.append(1001), depth.append(d), vol.append([0.0, 5.0])
        # 3: a new bid joins inside the spread at 1000 (size 3) -> spread 1 again
        d = np.full((2, L), 5.0)
        d[0, 0] = 3.0
        bid.append(1000), ask.append(1001), depth.append(d), vol.append([0.0, 0.0])
    n = len(bid)
    return (np.asarray(bid), np.asarray(ask), np.asarray(depth), np.asarray(vol), np.ones(n, bool))


def _cache(bid, ask, depth, vol, valid, pad: int = 3400):
    n = len(bid)
    reps = -(-pad // n)
    return {"bid_px": np.tile(bid, reps)[:pad], "ask_px": np.tile(ask, reps)[:pad],
            "depth": np.tile(depth, (reps, 1, 1))[:pad], "volume": np.tile(vol, (reps, 1))[:pad],
            "valid": np.tile(valid, reps)[:pad]}


def test_v09_estimator_never_sees_improvements():
    """Regression record of the frozen v0.9 defect: improvements exist, the v0.9 samples contain none."""
    cache = _cache(*_frames_with_improvements())
    s = C9.book_samples(cache, windows=(30,))
    assert len(s["add_level"]) > 0
    assert (s["add_level"] < 0).sum() == 0
    frame_samples = {k: s[k] for k in ("add_level", "add_size", "cancel_level", "cancel_frac")}

    class F:                    # BookModel.fit only reads samples, valid and depth
        samples = frame_samples
        valid = np.zeros(1, bool)
        depth = np.zeros((1, 2, L))
    assert BookModel.fit(F()).improve_prob == 0.0


def test_v1_estimator_recovers_inside_spread_additions():
    ev = B.infer_book_events(*_frames_with_improvements())
    inside = ev["add_level"] < 0
    assert inside.sum() == 60                      # one improvement per cycle
    assert np.all(ev["add_level"][inside] == -1) and np.all(ev["add_spread"][inside] == 2)
    assert np.all(ev["add_size"][inside] == 3.0)
    tab = B.improvement_table(ev)
    assert tab.identified[2] and tab.probs[2] > 0.5    # every addition at spread 2 improved the touch
    assert not tab.identified[3] and tab.prob(1) == 0.0


def test_cancellation_explained_by_prints_is_not_a_cancel():
    bid, ask = np.array([1000, 1000]), np.array([1001, 1001])
    d0 = np.full((2, L), 5.0)
    d1 = d0.copy()
    d1[0, 0] = 2.0
    ev = B.infer_book_events(bid, ask, np.stack([d0, d1]), np.array([[0.0, 0.0], [0.0, 3.0]]), np.ones(2, bool))
    assert len(ev["cancel_level"]) == 0
    ev = B.infer_book_events(bid, ask, np.stack([d0, d1]), np.array([[0.0, 0.0], [0.0, 1.0]]), np.ones(2, bool))
    assert ev["cancel_level"].tolist() == [0.0] and ev["cancel_frac"][0] == pytest.approx(2.0 / 5.0)


def test_invalid_and_crossed_frames_are_skipped():
    bid, ask = np.array([1000, 1001, 1000]), np.array([1001, 1001, 1001])
    d = np.full((3, 2, L), 5.0)
    ev = B.infer_book_events(bid, ask, d, np.zeros((3, 2)), np.ones(3, bool))
    assert ev["pairs"] == 2 and len(ev["add_level"]) == 0   # the locked frame contributes nothing


def _sim_pair(seed=3):
    base = BookModel(refill_mean=6.0, limit_size_mean=3.0, market_size_mean=2.0, mid_ticks=4000)
    params = {**PoissonGenerator.default_params(), "market": 1.2, "add": 4.0, "cancel": 3.0}
    v08 = simulate(PoissonGenerator(params, base), base, 3000, seed=seed, warmup_bins=300)
    v1_model = B.BookModelV1(**base.to_dict(), improve_by_spread=((2, 0.6), (3, 0.7), (4, 0.8)),
                             improve_depth_p=0.7)
    v1 = B.simulate_v1(PoissonGenerator(params, v1_model), v1_model, 3000, seed=seed, warmup_bins=300)
    return v08, v1


def test_improvement_closes_simulated_spreads():
    v08, v1 = _sim_pair()
    s08, s1 = B.spread_summary(v08), B.spread_summary(v1)
    assert s1["mean"] < s08["mean"]
    assert s1["p_ge3"] < s08["p_ge3"]


def test_round_trip_identifies_positive_improvement_probability():
    _, v1 = _sim_pair(seed=11)
    fitted = B.fit_from_frame(v1)
    tab = fitted.calibration["improvement"]
    k2, n2 = tab["counts"]["2"]
    assert n2 >= 20 and k2 > 0
    assert fitted.improve(2) > 0.0 and fitted.improve(1) == 0.0
    # 100 ms aggregation merges events, so the recovered rate is biased; it must not be the v0.9 zero
    assert 0.05 < fitted.improve(2) < 0.95
