"""v0.9 bounded replay engine: positive controls on synthetic day caches with known truth (M26-M31)."""
from __future__ import annotations

import numpy as np
import pytest

from lob.v09.data import daycache as dc
from lob.v09.execution import policies as P
from lob.v09.replay import engine as E

L = 12_000      # bins (window 0 starts at 600 s = bin 6000)


def synthetic_cache(*, touch_size=100.0, deep_size=50.0, prints=(), bid=1000, ask=1001) -> dict:
    n = L
    depth = np.zeros((n, 2, 10), np.float32)
    depth[:, 0, :] = touch_size
    depth[:, 1, :] = touch_size
    secs = dc.deep_seconds()
    deep_px = np.zeros((len(secs), 2, dc.DEEP_LEVELS), np.int64)
    deep_q = np.zeros((len(secs), 2, dc.DEEP_LEVELS), np.float32)
    for k in range(dc.DEEP_LEVELS):
        deep_px[:, 0, k], deep_px[:, 1, k] = bid - k, ask + k
        deep_q[:, :, k] = touch_size if k < 10 else deep_size
    t = np.asarray([p[0] for p in prints], float)
    return {"tick": 0.1, "lot": 0.001, "day": "2000-01-01", "valid": np.ones(n, bool),
            "bid_px": np.full(n, bid, np.int64), "ask_px": np.full(n, ask, np.int64), "depth": depth,
            "mid": np.full(n + 1, (bid + ask) / 2.0), "state": np.zeros((n, 6), np.float32),
            "volume": np.zeros((n, 2), np.float32), "counts": np.zeros((n, 6), np.int32),
            "deep_s": secs, "deep_px": deep_px, "deep_q": deep_q, "deep_ok": np.ones(len(secs), bool),
            "trade_t": t, "trade_px": np.asarray([p[1] for p in prints], np.int64),
            "trade_q": np.asarray([p[2] for p in prints], float), "trade_s": np.asarray([p[3] for p in prints], np.int8)}


def _run(cache, policy, qty, assumption=E.Assumption(), side=1):
    book = E.DayBook(cache)
    res = E.Replay(book, policy, window=0, qty=qty, side=side, assumption=assumption).run()
    return res, E.metrics(res, book)


def test_sweep_walks_the_ladder_exactly() -> None:
    res, m = _run(synthetic_cache(), P.make("sweep"), 250.0)
    prices = sorted(f[1] for f in res.fills)
    assert prices == [1001, 1002, 1003] and [f[2] for f in sorted(res.fills, key=lambda f: f[1])] == [100, 100, 50]
    vwap = (1001 * 100 + 1002 * 100 + 1003 * 50) / 250
    assert m["is_gross_bps"] == pytest.approx((vwap - 1000.5) / 1000.5 * 1e4)
    assert m["is_net_bps"] == pytest.approx(m["is_gross_bps"] + 5.5) and m["maker_fraction"] == 0.0


def test_beyond_the_recorded_book_is_penalized_and_counted() -> None:
    cache = synthetic_cache(touch_size=1.0, deep_size=1.0)
    res, m = _run(cache, P.make("sweep"), 500.0)
    assert res.beyond_book == pytest.approx(300.0) and m["penalty_bps"] == pytest.approx(25.0 * 300 / 500)


def test_passive_order_fills_from_prints_through_and_at_its_price() -> None:
    t0 = 600.0
    through = synthetic_cache(prints=[(t0 + 7.0, 999, 1.0, -1)])               # a sell print below our bid
    res, m = _run(through, P.make("passive_naive"), 40.0, E.Assumption("conservative"))
    assert m["maker_fraction"] == 1.0 and res.fills[0][1] == 1000
    at = synthetic_cache(prints=[(t0 + 7.0, 1000, 130.0, -1)])                 # 100 ahead, then 30 for us
    res, m = _run(at, P.make("passive_naive"), 40.0, E.Assumption("neutral"))
    assert sum(f[2] for f in res.fills if f[3] == "maker") == pytest.approx(30.0)
    assert sum(f[2] for f in res.fills if f[3] == "taker") == pytest.approx(10.0)     # hard completion


def test_conservative_queue_never_fills_more_than_optimistic() -> None:
    rng = np.random.default_rng(3)
    prints = [(600 + float(x), 1000, float(q), -1) for x, q in zip(np.sort(rng.uniform(0, 300, 40)),
                                                                  rng.uniform(1, 40, 40))]
    cache = synthetic_cache(prints=prints)
    cache["depth"][6100:9000, 0, 0] = np.linspace(100, 20, 2900)               # visible decreases (cancels)
    got = {}
    for rule in E.QUEUE_RULES:
        res, m = _run(cache, P.make("passive_naive"), 200.0, E.Assumption(rule, hidden_fraction=0.5))
        got[rule] = m["maker_fraction"]
    assert got["conservative"] <= got["neutral"] + 1e-12 <= got["optimistic"] + 2e-12


def test_persistent_impact_costs_at_least_no_impact() -> None:
    cache = synthetic_cache()
    _, none = _run(cache, P.make("twap"), 3000.0, E.Assumption(impact="none"))
    _, pers = _run(cache, P.make("twap"), 3000.0, E.Assumption(impact="persistent"))
    _, trans = _run(cache, P.make("twap"), 3000.0, E.Assumption(impact="transient"))
    assert none["is_gross_bps"] <= trans["is_gross_bps"] + 1e-12 <= pers["is_gross_bps"] + 2e-12
    assert pers["is_gross_bps"] > none["is_gross_bps"]


def test_replay_is_deterministic_and_bounds_order_endpoints() -> None:
    cache = synthetic_cache(prints=[(605.0, 1000, 50.0, -1), (650.0, 999, 5.0, -1)])
    book = E.DayBook(cache)
    rows = [E.episode(book, P.make("passive_deadline"), window=0, notional_usd=10.0, assumption=a)
            for a in E.grid(latencies=(0.0, 0.25), hidden=(0.0,), impacts=("none",))]
    again = [E.episode(book, P.make("passive_deadline"), window=0, notional_usd=10.0, assumption=a)
             for a in E.grid(latencies=(0.0, 0.25), hidden=(0.0,), impacts=("none",))]
    assert rows == again
    b = E.bounds(rows)
    assert b["is_net_bps"][0] <= b["is_net_bps"][1]


def test_sell_side_mirrors_buy_side() -> None:
    cache = synthetic_cache()
    _, buy = _run(cache, P.make("sweep"), 150.0, side=1)
    _, sell = _run(cache, P.make("sweep"), 150.0, side=-1)
    assert buy["is_gross_bps"] == pytest.approx(sell["is_gross_bps"])


def test_invalid_window_is_not_evaluable() -> None:
    cache = synthetic_cache()
    cache["valid"][6500] = False
    out = E.episode(E.DayBook(cache), P.make("twap"), window=0, notional_usd=10.0, assumption=E.Assumption())
    assert out["status"] == "NOT_EVALUABLE"
