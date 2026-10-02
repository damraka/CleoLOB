"""v0.7 property-based, fuzz and metamorphic tests (workstreams 74-76).

Property tests (Hypothesis): order and quantity conservation in the matching engine, book ordering,
nonnegative sizes, lifecycle validity, queue-model monotonicity, record serialization round trips and
ledger hash-chain integrity under arbitrary appends.

Fuzzing: random event sequences with duplicate ids, invalid and repeated cancels, timestamp reversal,
empty book sides, negative sizes, crossing orders, partial-fill and replace races. Every rejection must
be an explicit error or a ``False`` return, never a crash or silent corruption (invariants checked).

Metamorphic relations (assumptions stated per test):
* price-scale invariance — dimensionless (bps, ratio) metrics are unchanged when every price is scaled;
* deterministic reseeding — the same seed reproduces a simulation bit for bit;
* equivalent event transformations — splitting one displayed-size update into two consecutive updates
  with the same final size leaves the 100 ms tape unchanged;
* aggregation consistency — pooling per-block sketches equals sketching the concatenated blocks' sums.
"""
from __future__ import annotations

from decimal import Decimal

from hypothesis import HealthCheck, given, settings, strategies as st
import numpy as np
import pytest

from lob.engine import ExchangeSimulator, Order, OrderBook, OrderType, Side, SimConfig, TimeInForce
from lob.v07.data.schema import Record
from lob.v07.data.validation import from_json, to_json
from lob.v07.queue.models import MODELS, LevelEvent, all_models

FAST = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])

orders = st.lists(st.tuples(st.sampled_from([Side.BUY, Side.SELL]), st.integers(1, 30), st.integers(95, 105),
                            st.booleans(), st.sampled_from(list(TimeInForce)[:3])), min_size=1, max_size=60)


@FAST
@given(orders)
def test_matching_conserves_quantity_and_keeps_book_ordered(sequence) -> None:
    book = OrderBook(check_invariants=True)
    submitted = filled = 0
    for i, (side, qty, price, is_market, tif) in enumerate(sequence, 1):
        order = Order(i, side, qty, OrderType.MARKET if is_market else OrderType.LIMIT, f"o{i % 3}", 0.0,
                      price=None if is_market else price, time_in_force=TimeInForce.GTC if is_market else tif)
        trades = book.process(order, float(i))
        submitted += qty
        filled += sum(t.qty for t in trades)
        assert all(t.qty > 0 for t in trades)
        bb, ba = book.best_bid(), book.best_ask()
        assert bb is None or ba is None or bb < ba                       # never locked or crossed at rest
        assert all(v > 0 for v in list(book.bid_vol.values()) + list(book.ask_vol.values()))
    resting = sum(book.bid_vol.values()) + sum(book.ask_vol.values())
    cancelled = sum(o.remaining for o in book.order_history.values() if o.is_terminal)
    assert resting + 2 * filled + cancelled == submitted                 # every unit is resting, traded (both sides) or gone
    assert book.bid_prices == sorted(book.bid_prices) and book.ask_prices == sorted(book.ask_prices)


@FAST
@given(st.lists(st.tuples(st.sampled_from(["size", "trade_at", "trade_through"]), st.integers(0, 60)),
                max_size=30), st.integers(1, 50), st.integers(0, 200))
def test_queue_models_are_ordered_and_bounded(events, qty, join) -> None:
    seq = [LevelEvent(float(i), k, float(v)) for i, (k, v) in enumerate(events)]
    out = all_models(seq, qty=qty, level_at_join=join)
    filled = [out[m]["filled"] for m in MODELS]
    assert all(0 <= f <= qty + 1e-9 for f in filled)
    assert all(a <= b + 1e-9 for a, b in zip(filled, filled[1:]))


decimals = st.decimals(min_value=Decimal("0.01"), max_value=Decimal("100000"), places=2)


@FAST
@given(st.sampled_from(["book_delta", "book_snapshot"]), st.sampled_from(["bid", "ask"]), decimals,
       st.decimals(min_value=Decimal("0"), max_value=Decimal("1000000"), places=4), st.integers(0, 2**52))
def test_record_serialization_round_trip(kind, side, price, amount, t) -> None:
    record = Record(kind, t, t + 1, side, price, amount).validate("aggregate_l2")
    assert from_json(to_json(record)) == record


@settings(max_examples=25, deadline=None)
@given(st.lists(st.text(alphabet="abcdefgh", min_size=1, max_size=8), min_size=1, max_size=6))
def test_ledger_chain_holds_under_arbitrary_reasons(tmp_path_factory, reasons) -> None:
    from tests.test_v07_protocol import COPIED, ROOT
    import shutil
    from lob.v07.protocol import core as pr
    root = tmp_path_factory.mktemp("ledger")
    for name in COPIED + tuple(f"configs/v07/{p.name}" for p in (ROOT / "configs/v07").glob("*.json")):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, root / name)
    pr.initialize_ledger(root)
    for i, reason in enumerate(reasons):
        pr.append_event(root / pr.LEDGER_PATH, "attempt", reason=reason, design=f"d{i}", root=root,
                        protocol=pr.load_protocol(root / pr.PROTOCOL_PATH),
                        payload={"outcome": "COMPLETED", "directory": f"x/{i}", "note": reason})
    assert pr.verify_protocol_files(root)["valid"]


# ----------------------------------------------------------------------------- fuzzing (75)

actions = st.lists(st.tuples(st.sampled_from(["limit", "market", "cancel", "cancel_again", "modify", "replace",
                                              "negative", "duplicate", "rewind", "empty_side"]),
                             st.integers(1, 40), st.integers(90, 110)), max_size=80)


@FAST
@given(actions)
def test_fuzzed_event_sequences_never_corrupt_the_book(sequence) -> None:
    book = OrderBook(check_invariants=True)
    now, next_id, ids = 0.0, 1, []
    for action, qty, price in sequence:
        now += 0.1
        try:
            if action in {"limit", "market", "empty_side"}:
                side = Side.SELL if action == "empty_side" else (Side.BUY if price % 2 else Side.SELL)
                order = Order(next_id, side, qty, OrderType.MARKET if action == "market" else OrderType.LIMIT,
                              "F", now, price=None if action == "market" else price)
                book.process(order, now)
                ids.append(next_id)
                next_id += 1
            elif action in {"cancel", "cancel_again"} and ids:
                target = ids[qty % len(ids)]
                first = book.cancel(target, now)
                second = book.cancel(target, now)                           # repeated cancel must be a no-op
                assert not (first and second)
            elif action == "modify" and ids:
                book.modify(ids[qty % len(ids)], qty, None, now)
            elif action == "replace" and ids:
                target = ids[qty % len(ids)]
                if book.cancel(target, now):
                    book.process(Order(next_id, Side.BUY, qty, OrderType.LIMIT, "F", now, price=price), now)
                    ids.append(next_id)
                    next_id += 1
            elif action == "negative":
                with pytest.raises(ValueError):
                    Order(next_id, Side.BUY, -qty, OrderType.LIMIT, "F", now, price=price).validate()
            elif action == "duplicate" and ids:
                with pytest.raises(ValueError):
                    book.process(Order(ids[0], Side.BUY, qty, OrderType.LIMIT, "F", now, price=price), now)
            elif action == "rewind":
                with pytest.raises(ValueError):
                    book.cancel(ids[0] if ids else 1, now - 10.0)
        except ValueError as exc:                                          # explicit, typed rejection only
            assert str(exc)
        book.assert_invariants()


@FAST
@given(st.lists(st.tuples(st.sampled_from(["bid", "ask"]), st.integers(1, 30), st.integers(0, 50),
                          st.integers(-3, 3)), max_size=60))
def test_fuzzed_l2_records_are_flagged_not_crashing(rows) -> None:
    from lob.v07.data.schema import venue_spec
    from lob.v07.data.validation import validate_stream
    t, records = 1_604_188_800_000_000, []
    for side, price, amount, jump in rows:
        t += jump * 1000
        records.append(Record("book_delta", t, t, side, Decimal(price), Decimal(amount)))
    report = validate_stream(records, venue_spec("deribit", "ETH-PERPETUAL"))
    assert report["status"] in {"PASS", "WARN", "FAIL"} and report["records"] == len(records)


# ----------------------------------------------------------------------------- metamorphic (76)


def test_price_scale_invariance_of_dimensionless_metrics() -> None:
    """Assumption: bps and ratio metrics depend on prices only through ratios."""
    from lob.v06.tape import BookTape
    from lob.v07.generators.events import book_state
    rng = np.random.default_rng(0)
    bp = np.cumsum(rng.normal(0, 1, (50, 1)), axis=0) + 1000
    tape = BookTape(np.arange(50) * 0.1, np.hstack([bp, bp - 1]), np.ones((50, 2)) * 3, np.hstack([bp + 1, bp + 2]),
                    np.ones((50, 2)) * 4, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), 1.0)
    spread_bps = (tape.ap[:, 0] - tape.bp[:, 0]) / tape.mid * 1e4
    scaled = (tape.ap[:, 0] * 7 - tape.bp[:, 0] * 7) / (tape.mid * 7) * 1e4
    np.testing.assert_allclose(spread_bps, scaled)
    assert book_state(100.0, 101.0, np.ones(5), np.ones(5) * 2, 1.0)[:2] == book_state(
        700.0, 707.0, np.ones(5), np.ones(5) * 2, 7.0)[:2]


def test_deterministic_reseeding() -> None:
    """Assumption: every stochastic stream derives from the configured seed."""
    def run(seed):
        sim = ExchangeSimulator(SimConfig(seed=seed))
        sim.step(30.0)
        return [(t.time, t.price, t.qty) for t in sim.book.trades]
    assert run(5) == run(5) != run(6)


def test_equivalent_event_transformation_preserves_the_tape(tmp_path) -> None:
    """Assumption: the 100 ms grid samples the last state of each interval, so an intermediate size that is
    immediately overwritten within the same local-timestamp group does not change the tape."""
    import gzip
    from lob.v07.data.tape import build_tape
    header = "exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount"
    base = [header, "deribit,ETH-PERPETUAL,1000,1000,true,bid,99.95,10", "deribit,ETH-PERPETUAL,1000,1000,true,ask,100.05,10"]
    rows = [f"deribit,ETH-PERPETUAL,{t},{t},false,bid,99.95,{10 + i % 5}" for i, t in enumerate(range(2000, 4_000_000, 50_000))]
    split = []
    for row in rows:
        parts = row.split(",")
        split.append(",".join(parts[:7] + ["999"]))
        split.append(row)
    trades = tmp_path / "t.csv.gz"
    with gzip.open(trades, "wt") as h:
        h.write("exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n")
    tapes = []
    for name, body in (("a", base + rows), ("b", base + split)):
        path = tmp_path / f"{name}.csv.gz"
        with gzip.open(path, "wt") as h:
            h.write("\n".join(body) + "\n")
        tapes.append(build_tape(path, trades, tick=0.05)[0])
    np.testing.assert_array_equal(tapes[0].bq, tapes[1].bq)


def test_aggregation_consistency_of_sketches() -> None:
    """Assumption: sketches are additive sufficient statistics on frozen bins."""
    from lob.v06.observables import pool
    rng = np.random.default_rng(3)
    blocks = [{"a": rng.integers(0, 5, 10).astype(float), "b": rng.normal(size=4)} for _ in range(5)]
    pooled = pool(blocks)
    np.testing.assert_allclose(pooled["a"], sum(b["a"] for b in blocks))
    np.testing.assert_allclose(pool([pool(blocks[:2]), pool(blocks[2:])])["b"], pooled["b"])
