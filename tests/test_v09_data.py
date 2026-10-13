"""v0.9 data layer: reconstruction differential tests, quality counters, normalization, adapters, acquisition."""
from __future__ import annotations

import csv
import gzip
import io
import itertools
import json
from pathlib import Path
import zipfile

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest

from lob.v09 import protocol as pr, quality
from lob.v09.data import binance, bybit, daycache, normalize, reconstruct as rc, sources, store
from tests.v09_fixtures import first, v09_root, v09_template  # noqa: F401

DEV_L2 = Path("data/v09/raw/bybit-linear/BTCUSDT/2023-07-15/l2.zip")


def _msg(kind, ts, u, bids, asks):
    return {"topic": "orderbook.500.X", "type": kind, "ts": ts,
            "data": {"s": "X", "b": [[f"{p / 10:.1f}", f"{q / 1000:.3f}"] for p, q in bids],
                     "a": [[f"{p / 10:.1f}", f"{q / 1000:.3f}"] for p, q in asks], "u": u, "seq": u}}


@st.composite
def streams(draw):
    """A random but valid snapshot+delta stream around price 1000 ticks (bids below, asks above)."""
    n = draw(st.integers(5, 60))
    msgs = [_msg("snapshot", 1000, 1, [(1000 - k, 5 + k) for k in range(1, 8)], [(1000 + k, 5 + k) for k in range(8)])]
    for i in range(n):
        bids = draw(st.lists(st.tuples(st.integers(985, 999), st.integers(0, 40)), max_size=6))
        asks = draw(st.lists(st.tuples(st.integers(1000, 1015), st.integers(0, 40)), max_size=6))
        msgs.append(_msg("delta", 1000 + 100 * (i + 1), 2 + i, bids, asks))
    return msgs


def _trace(book, m):
    return (book.best(1), book.best(-1), tuple(book.visible(1)), tuple(book.visible(-1)))


@settings(max_examples=60, deadline=None)
@given(streams())
def test_dense_book_matches_the_reference_book(msgs) -> None:
    _, ca, ta = rc.replay(msgs, tick=0.1, lot=0.001, book=rc.ReferenceBook(), every=_trace)
    _, cb, tb = rc.replay(msgs, tick=0.1, lot=0.001, book=rc.DenseBook(width=4096), every=_trace)
    assert ta == tb
    assert ca.crossed == cb.crossed and ca.empty_side == cb.empty_side and ca.sequence_gaps == cb.sequence_gaps


@pytest.mark.skipif(not DEV_L2.is_file(), reason="development-day raw archive is local research data")
def test_dense_book_matches_reference_on_real_development_messages() -> None:
    msgs = list(itertools.islice(bybit.book_messages(DEV_L2), 3000))
    _, _, ta = rc.replay(msgs, tick=0.1, lot=0.001, book=rc.ReferenceBook(), every=_trace)
    _, _, tb = rc.replay(msgs, tick=0.1, lot=0.001, every=_trace)
    assert ta == tb


def test_sequence_duplicates_out_of_order_crossed_and_gaps_are_counted() -> None:
    msgs = [_msg("snapshot", 0, 10, [(999, 5)], [(1001, 5)]), _msg("delta", 100, 11, [], []),
            _msg("delta", 200, 11, [], []), _msg("delta", 300, 14, [], []), _msg("delta", 400, 13, [], []),
            _msg("delta", 9_000, 15, [(1002, 3)], []), _msg("delta", 8_000, 16, [(1002, 0)], [])]
    _, c, _ = rc.replay(msgs, tick=0.1, lot=0.001)
    assert (c.duplicates, c.sequence_gaps, c.out_of_order, c.crossed, c.ts_backwards) == (1, 1, 1, 1, 1)
    assert c.max_gap_ms == 8_600 and c.gap_starts_ms == [400]


def test_off_grid_and_negative_sizes_are_counted_not_hidden() -> None:
    m = _msg("snapshot", 0, 1, [(999, 5)], [(1001, 5)])
    m["data"]["b"].append(["99.95", "1.000"])
    m["data"]["a"].append(["100.30", "-1.000"])
    book, c, _ = rc.replay([m], tick=0.1, lot=0.001)
    assert c.off_tick == 1 and c.negative_size == 1
    assert book.best(-1) == 1001


def test_dense_book_recenters_and_counts_truncation() -> None:
    msgs = [_msg("snapshot", 0, 1, [(1000, 5)], [(1001, 5)]), _msg("delta", 100, 2, [(1000, 0), (3000, 5)],
                                                                      [(1001, 0), (3001, 5)])]
    book, c, _ = rc.replay(msgs, tick=0.1, lot=0.001, book=rc.DenseBook(width=1024))
    assert c.recenters >= 1 and book.best(1) == 3000 and book.best(-1) == 3001


def test_grid_inference_and_registered_normalization_rule() -> None:
    msgs = [_msg("snapshot", 0, 1, [(30000 + 10 * k, 1500) for k in range(5)], [(30100, 2000)])]
    obs = normalize.infer_grid(msgs)
    assert obs["tick"] == 1.0 and obs["lot"] == 0.1        # prices 3000.0, 3001.0 ...; sizes 1.5, 2.0
    g = normalize.grid(0.1, 0.001, obs)
    assert g["tick"] == 1.0 and g["lot"] == 0.001 and g["anomalies"]
    assert normalize.grid(1.0, 0.1, obs)["anomalies"] == []


def test_quality_decision_thresholds() -> None:
    base = {"reconstruction": rc.Counters(messages=10_000).to_dict(), "trades": 5_000, "trades_inconsistent_20bps": 0,
            "trade_side_unknown": 0, "valid_bin_fraction": 0.999}
    assert quality.decide(base)["status"] == "AVAILABLE"
    bad = json.loads(json.dumps(base))
    bad["reconstruction"]["sequence_gaps"] = 50
    out = quality.decide(bad)
    assert out["status"] == "QUALITY_REJECTED" and out["failed"] == ["sequence"]
    assert quality.decide({**base, "valid_bin_fraction": 0.5})["failed"] == ["valid_bins"]


def _zip(path: Path, member: str, text: str) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(member, text)
    return path


def test_binance_adapters_with_and_without_headers(tmp_path) -> None:
    rows = "1,100.0,2.0,100.1,3.0,1689379200100,1689379200101\n2,100.0,1.0,100.1,3.0,1689379200200,1689379200201\n"
    a = binance.read_book_ticker(_zip(tmp_path / "a.zip", "a.csv", rows))
    b = binance.read_book_ticker(_zip(tmp_path / "b.zip", "b.csv", ",".join(binance.BOOK_COLUMNS) + "\n" + rows))
    assert all(np.array_equal(a[k], b[k]) for k in a)
    t = binance.read_trades(_zip(tmp_path / "t.zip", "t.csv", "id,price,qty,quote_qty,time,is_buyer_maker\n"
                                 "5,100.1,0.5,50.05,1689379200150,false\n6,100.0,0.2,20,1689379200250,true\n"))
    assert t["side"].tolist() == [1, -1] and t["t"][0] == pytest.approx(1689379200.15)


def _bybit_files(tmp_path: Path, day: str) -> tuple[Path, Path]:
    t0 = int(daycache.day_start(day) * 1000)
    msgs = [_msg("snapshot", t0 + 50, 1, [(1000 - k, 5000) for k in range(12)], [(1001 + k, 5000) for k in range(12)])]
    for i in range(1, 40):
        msgs.append(_msg("delta", t0 + 50 + 100 * i, 1 + i, [(1000, 5000 - 100 * i)], [(1001, 5000 + 50 * i)]))
    l2 = _zip(tmp_path / "l2.zip", "x.data", "".join(json.dumps(m) + "\n" for m in msgs))
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["timestamp", "symbol", "side", "size", "price", "tickDirection", "trdMatchID", "grossValue",
                "homeNotional", "foreignNotional"])
    for i in range(1, 30):
        w.writerow([f"{t0 / 1000 + 0.06 + 0.1 * i:.4f}", "X", "Sell", "0.100", "100.0", "ZeroMinusTick", f"id{i}", 0, 0, 0])
    trades = tmp_path / "trades.csv.gz"
    with gzip.open(trades, "wt", newline="") as f:
        f.write(buf.getvalue())
    return l2, trades


def test_day_cache_infers_market_and_cancel_events_and_is_deterministic(tmp_path) -> None:
    l2, trades = _bybit_files(tmp_path, "2023-07-15")
    cache = daycache.build_bybit(l2, trades, day="2023-07-15", tick=0.1, lot=0.001)
    c = cache["counts"]
    assert c[:, 1].sum() == 29                         # sell prints
    assert c[:, 4].sum() == 10                         # 29 decreases explained by prints; 10 unexplained
    assert c[:, 3].sum() >= 30                         # ask increases are adds
    assert cache["valid"][1:40].all() and not cache["valid"][100:].any()   # stale after 5 s without messages
    assert cache["quality"]["reconstruction"]["sequence_gaps"] == 0
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    assert daycache.save(a, cache) == daycache.save(b, cache)
    loaded = daycache.load(a)
    assert np.array_equal(loaded["depth"], cache["depth"]) and loaded["quality"] == json.loads(
        json.dumps(cache["quality"]))


def test_acquisition_failure_is_ledgered_and_backup_slots_resolve(v09_root) -> None:
    protocol = pr.load_protocol(v09_root / pr.PROTOCOL_PATH)
    stream = "bybit-linear:BTCUSDT"
    _, state = pr.state_of(v09_root)
    before = sources.resolve(protocol, state, "explore", stream, 8)

    def failing(url, target):
        raise sources.DataNotAvailable("HTTP 404 (test)")

    victim = before["datasets"][0]
    with pytest.raises(sources.DataNotAvailable):
        sources.acquire(victim, use="explore", design="pilot", purpose="test", root=v09_root, downloader=failing)
    _, state = pr.state_of(v09_root)
    after = sources.resolve(protocol, state, "explore", stream, 8)
    assert victim not in after["datasets"] and after["failed"] == [victim] and after["status"] == "AVAILABLE"
    assert after["datasets"][-1] not in before["datasets"]          # the first registered backup took the slot
    with pytest.raises(sources.DataNotAvailable, match="ledgered"):
        sources.acquire(victim, use="explore", design="pilot", purpose="test", root=v09_root)


def test_checksum_mismatch_is_not_available(v09_root, tmp_path) -> None:
    protocol = pr.load_protocol(v09_root / pr.PROTOCOL_PATH)
    ds = first(protocol, pool="explore", venue="binance-um")

    def fake(url, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return "0" * 64, 1

    with pytest.raises(sources.DataNotAvailable, match="checksum"):
        sources.acquire(ds, use="explore", design="pilot", purpose="t", root=v09_root, downloader=fake,
                        checksum=lambda url: "1" * 64)
    report = pr.verify_protocol_files(v09_root)
    assert report["valid"] and report["attempts"][-1]["outcome"] == "NOT_AVAILABLE"


def test_store_refuses_a_cache_that_differs_from_the_ledger(v09_root, tmp_path) -> None:
    protocol = pr.load_protocol(v09_root / pr.PROTOCOL_PATH)
    ds = first(protocol, pool="explore", venue="bybit-linear")
    sources.mark(ds, stage="parsed", use="explore", design="pilot", root=v09_root, reason="t", cache_sha256="e" * 64)
    path = daycache.cache_path(v09_root, ds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not the ledgered cache")
    with pytest.raises(ValueError, match="differ"):
        store.prepare(ds, use="explore", design="pilot", purpose="t", root=v09_root)
