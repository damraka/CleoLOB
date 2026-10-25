"""v1 data acquisition: connectors, licences, store, streaming ingestion, failover and firewall (M10-M19)."""
import gzip
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from lob.v1.data import acquire as A, firewall as F, ingest as I, schema as SC, sources as SRC, store as S
from lob.v1.data.connectors import base as CB, binance as BN, bitstamp as BS, bybit as BY
from lob.v1.protocol import document as D
from tests.v1_fixtures import minimal_protocol, ws  # noqa: F401


# ------------------------------------------------------------------------------------------------ fixtures
def bybit_zip(path, messages):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("day.data", "\n".join(json.dumps(m) for m in messages) + "\n")
    return path


def bybit_messages():
    return [
        {"topic": "orderbook.200.BTCUSDT", "type": "snapshot", "ts": 1000, "data": {
            "s": "BTCUSDT", "b": [["100.0", "1.000"], ["99.9", "2.000"]], "a": [["100.1", "1.500"]], "u": 1}},
        {"topic": "orderbook.200.BTCUSDT", "type": "delta", "ts": 1100, "data": {
            "s": "BTCUSDT", "b": [["100.0", "0"]], "a": [["100.2", "0.250"]], "u": 2}},
        {"topic": "orderbook.200.BTCUSDT", "type": "delta", "ts": 1200, "data": {
            "s": "BTCUSDT", "b": [["100.05", "0.5"]], "a": [], "u": 3}},
    ]


def bybit_trades(path, rows):
    with gzip.open(path, "wt", newline="") as f:
        f.write("timestamp,symbol,side,size,price,tickDirection,trdMatchID,grossValue,homeNotional,foreignNotional\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    return path


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def fake_opener(objects):
    import urllib.error

    def open_(url):
        if url not in objects:
            raise urllib.error.HTTPError(url, 404, "nf", {}, None)
        return FakeResponse(objects[url])
    return open_


# ------------------------------------------------------------------------------------------------ M10 / M16
def test_connector_registry_and_capabilities():
    assert {"bybit", "binance", "bitstamp"} <= set(CB.known())
    cap = CB.get("bybit").capability("linear", "BTCUSDT", "2025-09-10")
    assert cap.book == "l2" and cap.depth_levels == 200 and not cap.permits("queue_position")
    assert cap.permits("queue_clearance") and not cap.permits("l3")
    assert CB.get("bybit").capability("linear", "BTCUSDT", "2025-08-20").depth_levels == 500
    l3 = CB.get("bitstamp").capability("exchange", "btcusd", "2026-10-26")
    assert l3.permits("l3") and l3.permits("queue_position") and l3.permits("explicit_cancels")
    with pytest.raises(ValueError):
        SC.Capability(book="l2", depth_levels=10, order_ids=False, trades=True, trade_side=True, trade_ids=True,
                      exchange_ts_resolution_ns=1, receive_ts=False, sequence_numbers=True, snapshots=True,
                      queue_observability="exact_order_id", cancel_visibility="none")
    with pytest.raises(KeyError):
        CB.get("nonexistent")


def test_licence_catalogue_is_complete():
    cat = SRC.catalogue()
    assert len(cat) == len(SRC.STREAMS)
    for row in cat:
        assert row["licence"]["terms_url"].startswith("https://") and row["licence"]["checked"]
        assert row["licence"]["redistribution"] is False
    depth = next(r for r in cat if r["stream"] == "binance-um:BTCUSDT")
    assert depth["capability"]["book"] == "depth_bands" and depth["capability"]["queue_observability"] == "none"


def test_connector_bybit_urls_follow_depth_eras():
    c = CB.get("bybit")
    assert c.sources("linear", "BTCUSDT", "2025-08-20")[0].url.endswith("2025-08-20_BTCUSDT_ob500.data.zip")
    assert c.sources("linear", "BTCUSDT", "2025-08-21")[0].url.endswith("2025-08-21_BTCUSDT_ob200.data.zip")
    assert c.sources("spot", "ETHUSDT", "2026-01-02")[1].url == "https://public.bybit.com/spot/ETHUSDT/ETHUSDT_2026-01-02.csv.gz"


# ------------------------------------------------------------------------------------------------ M11 / M13
def test_bybit_book_events_are_canonical_and_exact(tmp_path):
    p = bybit_zip(tmp_path / "ob.data.zip", bybit_messages())
    grid = BY.Grid(0.1, 0.001)
    ev = SC.concat(BY.book_events(BY.book_messages(p), grid))
    v = SC.validate_events(ev)
    assert v["by_kind"] == {"SNAPSHOT_BEGIN": 1, "SNAPSHOT_END": 1, "LEVEL_SET": 6}
    first = ev[ev["kind"] == SC.EventKind.LEVEL_SET][0]
    assert first["price"] == 1000 and first["size"] == 1000.0 and first["ts_exchange"] == 1_000_000_000
    # 100.05 is off the 0.1 grid: counted, never silent
    assert grid.off_tick == 1


def test_bybit_vectorized_parser_equals_scalar_grid(tmp_path):
    msgs = bybit_messages()
    rng = np.random.default_rng(0)
    for u in range(4, 200):
        msgs.append({"type": "delta", "ts": 1200 + u, "data": {"u": u, "b": [[f"{99.0 + rng.integers(0, 10) / 10:.1f}",
                    f"{rng.integers(0, 5) / 1000:.3f}"]], "a": [[f"{100.1 + rng.integers(0, 10) / 10:.1f}", "0.001"]]}})
    p = bybit_zip(tmp_path / "ob.data.zip", msgs)
    ev = SC.concat(BY.book_events(BY.book_messages(p), BY.Grid(0.1, 0.001), chunk=64))
    g = BY.Grid(0.1, 0.001)
    lv = ev[ev["kind"] == SC.EventKind.LEVEL_SET]
    scalar = [(g.price(px), g.size(sz)) for m in msgs for k in ("b", "a") for px, sz in m["data"].get(k, ())]
    assert [(int(a), float(b)) for a, b in zip(lv["price"], lv["size"])] == scalar


def test_bybit_spot_trades_and_exact_timestamps(tmp_path):
    p = bybit_trades(tmp_path / "t.csv.gz", [["1755820800.1395", "BTCUSDT", "Buy", "0.002", "100.1", "PlusTick", "a"],
                                              ["1755820800.2", "BTCUSDT", "Sell", "0.001", "100.0", "MinusTick", "b"]])
    tr = BY.trade_events(p, "linear", BY.Grid(0.1, 0.001))
    assert tr["ts_exchange"].tolist() == [1755820800139500000, 1755820800200000000]
    assert tr["side"].tolist() == [1, -1] and tr["size"].tolist() == [2.0, 1.0]
    assert BY.seconds_to_ns("12.5") == 12_500_000_000
    with pytest.raises(BY.ParseError):
        BY.seconds_to_ns("1.0000000001")
    sp = tmp_path / "s.csv.gz"
    with gzip.open(sp, "wt", newline="") as f:
        f.write("id,timestamp,price,volume,side,rpi\n1,1755820800139,100.10,0.5,buy,0\n")
    st = BY.trade_events(sp, "spot", BY.Grid(0.01, 0.000001))
    assert st["ts_exchange"][0] == 1755820800139_000_000 and st["price"][0] == 10010


# ------------------------------------------------------------------------------------------------ M12
def test_binance_trades_and_depth_bands(tmp_path):
    tp = tmp_path / "t.zip"
    with zipfile.ZipFile(tp, "w") as z:
        z.writestr("t.csv", "id,price,qty,quote_qty,time,is_buyer_maker\n1,100.1,0.002,0.2,1755820800871,true\n"
                            "2,100.2,0.001,0.1,1755820800900,false\n")
    inst = BN.CONNECTOR.instrument("um", "BTCUSDT", "2025-08-22")
    tr, meta = BN.trade_events(tp, "um", inst)
    assert tr["side"].tolist() == [-1, 1] and tr["ts_exchange"][0] == 1755820800871_000_000
    assert meta["time_unit"] == "ms" and meta["id_gaps"] == 0
    dp = tmp_path / "d.zip"
    with zipfile.ZipFile(dp, "w") as z:
        z.writestr("d.csv", "timestamp,percentage,depth,notional\n2025-08-22 00:00:07,-1.00,10.5,1.0e6\n"
                            "2025-08-22 00:00:07,1.00,9.5,9.0e5\n")
    ev, dm = BN.depth_band_events(dp)
    assert ev["kind"].tolist() == [SC.EventKind.DEPTH_BAND] * 2 and ev["price"].tolist() == [-100, 100]
    assert ev["ts_exchange"][0] == 1755820807 * 10**9 and dm["bands_bp"] == [-100, 100]


# ------------------------------------------------------------------------------------------------ M14
def test_bitstamp_capture_to_mbo_events(tmp_path):
    seg = tmp_path / "capture.jsonl.gz"
    census = {"timestamp": "1", "microtimestamp": "1000000", "bids": [["80574.23", "0.5", "11"]],
              "asks": [["80574.24", "0.25", "12"]]}
    recs = [
        {"kind": "capture_start", "recv_ns": 1},
        {"kind": "rest_snapshot", "recv_ns": 2, "body": json.dumps(census)},
        {"kind": "ws", "recv_ns": 3, "raw": json.dumps({"event": "order_created", "channel": "live_orders_btcusd",
            "data": {"id": 13, "order_type": 1, "microtimestamp": "1000100", "amount_str": "0.1",
                     "price_str": "80574.30"}})},
        {"kind": "ws", "recv_ns": 4, "raw": json.dumps({"event": "trade", "channel": "live_trades_btcusd", "data": {
            "id": 77, "type": 0, "microtimestamp": "1000200", "amount_str": "0.05", "price_str": "80574.24",
            "buy_order_id": 99, "sell_order_id": 12}})},
        {"kind": "ws", "recv_ns": 5, "raw": json.dumps({"event": "order_changed", "channel": "live_orders_btcusd",
            "data": {"id": 12, "order_type": 1, "microtimestamp": "1000200", "amount_str": "0.2",
                     "price_str": "80574.24"}})},
        {"kind": "ws", "recv_ns": 6, "raw": json.dumps({"event": "order_deleted", "channel": "live_orders_btcusd",
            "data": {"id": 13, "order_type": 1, "microtimestamp": "1000300", "amount_str": "0.1",
                     "price_str": "80574.30"}})},
        {"kind": "capture_end", "recv_ns": 7, "status": "GAP", "reason": "x"},
    ]
    with gzip.open(seg, "wt") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    inst = BS.CONNECTOR.instrument("exchange", "btcusd", "x")
    ev = SC.concat(BS.CONNECTOR.events({"a": seg}, inst))
    kinds = [SC.EventKind(k).name for k in ev["kind"]]
    assert kinds == ["SNAPSHOT_BEGIN", "ORDER_ADD", "ORDER_ADD", "SNAPSHOT_END", "ORDER_ADD", "TRADE",
                     "ORDER_MODIFY", "ORDER_DELETE", "GAP"]
    trade = ev[ev["kind"] == SC.EventKind.TRADE][0]
    assert trade["side"] == 1 and trade["order_id"] == 12 and trade["aux"] == 77 and trade["size"] == 5_000_000
    assert ev[1]["price"] == 8057423 and ev[1]["order_id"] == 11 and ev[4]["ts_receive"] == 3


# ------------------------------------------------------------------------------------------------ M15 / M17
def test_stream_ingest_roundtrip(tmp_path):
    p = bybit_zip(tmp_path / "ob.data.zip", bybit_messages())
    t = bybit_trades(tmp_path / "t.csv.gz", [["1.5", "BTCUSDT", "Buy", "0.001", "100.1", "x", "a"]])
    inst = BY.CONNECTOR.instrument("linear", "BTCUSDT", "2025-09-10")
    meta = I.ingest(BY.CONNECTOR, {"l2": p, "trades": t}, inst, tmp_path / "ev",
                    capability=BY.CONNECTOR.capability("linear", "BTCUSDT", "2025-09-10").to_dict(), sources={})
    assert meta["book"]["events"] == 8 and meta["trades"]["events"] == 1
    back = I.read_events(tmp_path / "ev" / "book.parquet")
    direct = SC.concat(BY.book_events(BY.book_messages(p), BY.Grid(0.1, 0.001)))
    assert np.array_equal(back, direct)
    assert np.array_equal(SC.concat(I.iter_events(tmp_path / "ev" / "book.parquet")), direct)


def test_store_content_addressing_checksums_and_changes(tmp_path):
    raw = S.RawStore(tmp_path / "raw")
    body = b"hello archive" * 1000
    objs = {"https://x/a.zip": body}
    got = raw.fetch("https://x/a.zip", opener=fake_opener(objs))
    assert got.sha256 == hashlib.sha256(body).hexdigest() and got.path.read_bytes() == body
    assert raw.fetch("https://x/a.zip", opener=fake_opener({})).path == got.path        # served from the store
    with pytest.raises(S.NotAvailable):
        raw.fetch("https://x/missing.zip", opener=fake_opener(objs))
    with pytest.raises(S.ChecksumMismatch):
        raw.fetch("https://x/b.zip", opener=fake_opener({"https://x/b.zip": b"x"}), checksum="0" * 64)
    assert not list((tmp_path / "raw" / "partial").iterdir())                          # no half files left
    with pytest.raises(ValueError):
        raw.fetch("https://x/c.zip", opener=fake_opener({"https://x/c.zip": b"y" * 100}), max_bytes=10, retries=1)
    assert raw.verify()["valid"]
    got.path.write_bytes(b"corrupt")
    assert not raw.verify()["valid"]


# ------------------------------------------------------------------------------------------------ M18
def test_acquire_failover_uses_registered_backups_in_order(ws):
    D.freeze(ws, minimal_protocol(), reason="freeze")
    raw = S.RawStore(ws.root / "raw")
    conn = CB.get("bybit")
    present = {}
    for day in ("2025-09-03", "2025-09-05"):
        for s in conn.sources("linear", "BTCUSDT", day):
            present[s.url] = b"data-" + day.encode() + s.name.encode()
    specs = []
    for slot, (day, kind) in enumerate([("2025-09-02", "primary"), ("2025-09-03", "primary"),
                                        ("2025-09-04", "backup"), ("2025-09-05", "backup")]):
        ds = A.dataset_id("bybit", "linear", "BTCUSDT", day)
        ws.append("declare_fresh", dataset=ds, reason="slot", payload={"role": "explore", "pool": "p"})
        specs.append(A.DatasetSpec(ds, "bybit", "linear", "BTCUSDT", day, "explore", "p", slot, kind))
    fetch = lambda url, **kw: raw.fetch(url, opener=fake_opener(present), retries=1, **kw)  # noqa: E731
    got = A.fill_slots(specs, 2, raw, ws=ws, fetch=fetch)
    assert [(a.spec.date, a.status) for a in got] == [("2025-09-02", "NOT_AVAILABLE"), ("2025-09-03", "AVAILABLE"),
                                                      ("2025-09-04", "NOT_AVAILABLE"), ("2025-09-05", "AVAILABLE")]
    st = ws.state()
    assert set(st.acquired) == {specs[1].id, specs[3].id} and len(st.attempts) == 2
    assert not st.accesses                                                              # acquisition is not access


# ------------------------------------------------------------------------------------------------ M19
def test_firewall_rules():
    assert F.date_fresh("2025-08-31") == (False, "period-consumed (on or before 2025-08-31)")
    assert not F.date_fresh("2025-09-20")[0] and not F.date_fresh("2026-08-01")[0]
    assert F.date_fresh("2025-09-02")[0]
    ok, why = F.eligible("bybit-linear-btcusdt-2025-09-02", "2025-09-02", "vault", consumed=set(),
                         v1_dates={"2025-09-02": "confirm"})
    assert not ok and "already assigned" in why
    assert F.eligible("x", "2025-01-01", "explore", consumed=set(), v1_dates={})[0]
    assert not F.eligible("old", "2026-01-01", "confirm", consumed={"old"}, v1_dates={})[0]


def test_firewall_inventory_and_ledger_import(ws):
    from pathlib import Path
    inv = F.inventory(Path("."))
    assert inv["dataset_count"] >= 160 and "2025-08-19" in inv["dates_with_reads"]
    assert inv["datasets"]["bybit-linear-solusdt-2025-08-19"]["version"] == "v0.9"
    n = F.import_into_ledger(ws, inv)
    assert n == inv["dataset_count"] + 1 + len(F.EXCLUDED_DATES)
    assert F.import_into_ledger(ws, inv) == 0                                            # idempotent
    with pytest.raises(Exception):
        ws.append("declare_fresh", dataset="bybit-linear-solusdt-2025-08-19", reason="x",
                  payload={"role": "confirm"})
