"""v0.7 queue-uncertainty models: bound ordering, known cases, seeded draws and learned positions."""
from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from lob.v07.data.schema import Record
from lob.v07.queue import learned
from lob.v07.queue.models import MODELS, LevelEvent, all_models, fill, fill_distribution, level_events, uniform_cdf
from lob.v07.data.tape import build_tape
from tests.v07_fixtures import write_tardis

E = LevelEvent


def test_known_case() -> None:
    # Join behind 100; others join behind (150); 40 cancelled (110); 70 printed at the price; 30 through.
    events = [E(1, "size", 150), E(1.5, "size", 110), E(2, "trade_at", 70), E(3, "trade_through", 30)]
    out = all_models(events, qty=20, level_at_join=100)
    assert out["conservative"]["first_fill_t"] == 3 and out["conservative"]["filled"] == 20
    assert out["optimistic"]["first_fill_t"] == 2 and out["optimistic"]["filled"] == 20
    assert out["fifo_upper"]["first_fill_t"] == 2       # cancels ahead: 60 ahead, the 70 print fills 10
    assert out["fifo_lower"]["first_fill_t"] == 3       # cancels behind: 100 ahead, print leaves 30 ahead
    assert out["probabilistic"]["first_fill_t"] == 3    # pro-rata: 40 * 100/150 ahead -> 3.3 left ahead
    assert out["probabilistic"]["remaining_ahead"] == 0


def test_no_prints_no_fill() -> None:
    assert all(v["filled"] == 0 for v in all_models([E(1, "size", 10)], qty=5, level_at_join=50).values())
    with pytest.raises(ValueError):
        fill("exact", [], qty=1, level_at_join=1)


def _random_events(rng) -> tuple[list, float]:
    level, events, t = float(rng.integers(10, 200)), [], 0.0
    join = level
    for _ in range(int(rng.integers(1, 40))):
        t += 1
        r = rng.random()
        if r < 0.5:
            level = max(0.0, level + float(rng.integers(-40, 30)))
            events.append(E(t, "size", level))
        elif r < 0.9:
            q = float(rng.integers(1, 40))
            events.append(E(t, "trade_at", q))
            level = max(0.0, level - q)
        else:
            events.append(E(t, "trade_through", float(rng.integers(1, 40))))
            level = 0.0
    return events, join


@pytest.mark.parametrize("seed", range(200))
def test_bound_ordering_property(seed) -> None:
    rng = np.random.default_rng(seed)
    events, join = _random_events(rng)
    qty = float(rng.integers(1, 60))
    cdf = learned.EmpiricalCDF(rng.beta(rng.uniform(0.3, 3), rng.uniform(0.3, 3), 200))
    out = all_models(events, qty=qty, level_at_join=join, cdf=cdf)
    filled = [out[m]["filled"] for m in MODELS]
    assert all(a <= b + 1e-9 for a, b in zip(filled, filled[1:])), dict(zip(MODELS, filled))
    dist = fill_distribution(events, qty=qty, level_at_join=join, draws=20, seed=seed, cdf=cdf)
    assert out["fifo_lower"]["filled"] - 1e-9 <= dist["quantiles"]["p5"] <= dist["quantiles"]["p95"] \
        <= out["fifo_upper"]["filled"] + 1e-9


def test_fill_distribution_is_seed_deterministic() -> None:
    events = [E(1, "size", 50), E(2, "trade_at", 30), E(3, "size", 10), E(4, "trade_at", 15)]
    a = fill_distribution(events, qty=10, level_at_join=80, draws=50, seed=3)
    assert a == fill_distribution(events, qty=10, level_at_join=80, draws=50, seed=3)


def test_level_events_from_tape(tmp_path) -> None:
    files = write_tardis(tmp_path, seconds=120)
    tape, _ = build_tape(files["l2"], files["trades"], tick=0.05)
    price = float(tape.bp[50, 0])
    join, events = level_events(tape, side=1, price=price, start=float(tape.t[50]), stop=float(tape.t[50]) + 30)
    assert join == float(tape.bq[50, 0]) and events
    assert {e.kind for e in events} <= {"size", "trade_at", "trade_through"}
    out = all_models(events, qty=10, level_at_join=join)
    assert out["conservative"]["filled"] <= out["optimistic"]["filled"]


def _rec(kind, oid, price="100", qty="1", side="bid", t=0):
    return Record(kind, t, t, side, Decimal(price) if price else None, Decimal(qty) if qty else None, order_id=oid)


def test_cancellation_positions_under_tracked_priority() -> None:
    records = [_rec("order_add", "a"), _rec("order_add", "b"), _rec("order_add", "c"), _rec("order_add", "d"),
               _rec("order_cancel", "c", qty=None),      # 2 ahead of 4 -> 0.5
               _rec("order_cancel", "a", qty=None),      # front -> 0.0
               _rec("order_modify", "b", qty="5"),        # increase re-queues b behind d
               _rec("order_cancel", "d", qty=None),      # d now ahead of b: 0 ahead -> 0.0
               _rec("order_cancel", "zzz", qty=None)]
    out = learned.cancellation_positions(records)
    assert out["positions"].tolist() == [0.5, 0.0, 0.0]
    assert out["skipped"]["unknown_order"] == 1
    cdf = learned.EmpiricalCDF(np.asarray([0.05, 0.15, 0.95]))
    assert cdf(0) == 0 and cdf(1) == 1 and cdf(0.5) == pytest.approx(2 / 3)
    assert uniform_cdf(0.3) == 0.3
    s = learned.summary(np.random.default_rng(0).uniform(size=500))
    assert s["ks_from_uniform"] < 0.1 and s["ci95"][0] < s["mean"] < s["ci95"][1]
