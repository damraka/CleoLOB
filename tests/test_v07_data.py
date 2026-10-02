"""v0.7 data layer: canonical schema, adapters, quality, capability checks, tapes and staged ledgered access."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import shutil

import numpy as np
import pytest

from lob.v06.tape import tape_from_tardis
from lob.v07.adapters.base import conformance
from lob.v07.adapters.tardis import TardisAdapter, tick_consistency
from lob.v07.data import access, capability
from lob.v07.data.quality import QualityThresholds, assess
from lob.v07.data.schema import Record, SchemaError, VenueSpec, digest, venue_spec
from lob.v07.data.tape import build_tape
from lob.v07.protocol import core as pr
from tests.test_v07_protocol import COPIED, ROOT
from tests.v07_fixtures import write_tardis

ETH_NOV, DEV = "deribit-eth-perp-2020-11-01", "deribit-eth-perp-2020-04-01"


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> dict[str, Path]:
    return write_tardis(tmp_path_factory.mktemp("tardis"))


def test_record_schema_refuses_fabricated_identity_and_bad_values() -> None:
    Record("book_delta", 1, 2, "bid", Decimal("1"), Decimal("0")).validate("aggregate_l2")
    with pytest.raises(SchemaError):
        Record("book_delta", 1, 2, "bid", Decimal("1"), Decimal("1"), order_id="x").validate("aggregate_l2")
    with pytest.raises(SchemaError):
        Record("order_add", 1, 2, "bid", Decimal("1"), Decimal("1"), order_id="x").validate("aggregate_l2")
    with pytest.raises(SchemaError):
        Record("order_add", 1, 2, "bid", Decimal("1"), Decimal("1")).validate("order_level_mbo")
    with pytest.raises(SchemaError):
        Record("trade", 1, 2, "buy", Decimal("1"), Decimal("0")).validate("aggregate_l2")
    with pytest.raises(SchemaError):
        Record("book_delta", 1, 2, "buy", Decimal("1"), Decimal("1")).validate("aggregate_l2")
    with pytest.raises(SchemaError):
        VenueSpec("x", "y", Decimal("0"), "u", "spot", "X", "aggregate_l2")
    with pytest.raises(SchemaError):
        venue_spec("binance", "BTCUSDT")


def test_venue_contracts_are_honest() -> None:
    for venue, symbol in (("deribit", "ETH-PERPETUAL"), ("deribit", "BTC-PERPETUAL"), ("bitmex", "XBTUSD")):
        contract = venue_spec(venue, symbol).contract
        assert contract.supports("trade_prints") and contract.supports("aggregate_depth")
        for missing in ("order_identity", "exact_fifo_position", "quantity_ahead", "hidden_order_quantity"):
            assert not contract.supports(missing)
    bitstamp = venue_spec("bitstamp", "btcusd").contract
    assert bitstamp.supports("order_identity") and not bitstamp.supports("exact_fifo_position")


def test_tardis_adapter_conformance_and_determinism(files) -> None:
    adapter = TardisAdapter("deribit", "ETH-PERPETUAL", l2=files["l2"], trades=files["trades"])
    report = conformance(adapter)
    assert report["valid"], report["issues"]
    assert set(report["kinds"]) == {"book_snapshot", "book_delta", "trade"}
    assert digest(adapter.records()) == digest(adapter.records())
    assert tick_consistency(adapter)["off_grid"] == 0


def test_tardis_adapter_refuses_wrong_instrument(files) -> None:
    adapter = TardisAdapter("deribit", "BTC-PERPETUAL", l2=files["l2"], trades=None)
    assert not conformance(adapter)["valid"]


def test_bitmex_files_use_the_same_reader(tmp_path) -> None:
    bitmex = write_tardis(tmp_path, venue="bitmex", symbol="XBTUSD", date="2020-11-01", tick=0.5)
    assert conformance(TardisAdapter("bitmex", "XBTUSD", l2=bitmex["l2"], trades=bitmex["trades"]))["valid"]


def test_v07_tape_equals_frozen_v06_tape(files) -> None:
    old = tape_from_tardis(files["l2"], files["trades"], tick=0.05)
    new, quality = build_tape(files["l2"], files["trades"], tick=0.05)
    for name in ("t", "bp", "bq", "ap", "aq", "trade_t", "trade_p", "trade_q", "trade_s"):
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name), err_msg=name)
    assert quality["status"] == "FAIL" and any("duration" in f for f in quality["fails"])


def test_quality_flags_crossed_books(tmp_path) -> None:
    crossed = write_tardis(tmp_path, crossed_every=3)
    tape, quality = build_tape(crossed["l2"], crossed["trades"], tick=0.05)
    assert quality["crossed_sample_fraction"] > 0
    relaxed = QualityThresholds(min_duration_s=60.0, max_crossed_fraction=1.0, min_valid_fraction=0.0)
    report = assess(tape, {"complete": True, "rows": 1}, {"rows": 1, "unknown_side": 0, "duplicate_ids": 0},
                    {"samples": len(tape.t), "stale_samples": 0, "crossed_samples": 0}, relaxed)
    assert report["status"] in {"PASS", "WARN"}


def test_capability_matrix_marks_unavailable_analyses() -> None:
    protocol = pr.load_protocol(ROOT / pr.PROTOCOL_PATH)
    assert capability.check(protocol, ETH_NOV, "realism")["status"] == "AVAILABLE"
    blocked = capability.check(protocol, "bitmex-xbtusd-2020-11-01", "queue_position")
    assert blocked["status"] == "NOT_AVAILABLE" and "exact_fifo_position" in blocked["missing"]
    assert capability.check(protocol, "bitstamp-btcusd-mbo-dev", "order_survival")["status"] == "AVAILABLE"
    assert capability.check(protocol, "bitstamp-btcusd-mbo-dev", "queue_position")["status"] == "NOT_AVAILABLE"
    with pytest.raises(ValueError):
        capability.check(protocol, ETH_NOV, "unregistered")


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    for name in COPIED + tuple(f"configs/v07/{p.name}" for p in (ROOT / "configs/v07").glob("*.json")):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, tmp_path / name)
    pr.initialize_ledger(tmp_path)
    return tmp_path


def _fake_downloader(source: dict[str, Path]):
    calls = []

    def download(venue, symbol, date, data_type, directory, **kwargs):
        calls.append((venue, symbol, date, data_type, kwargs))
        directory.mkdir(parents=True, exist_ok=True)
        kind = "l2" if data_type == "incremental_book_L2" else "trades"
        target = directory / access.file_name(venue, data_type, date, symbol)
        shutil.copyfile(source[kind], target)
        return target
    return download, calls


def _seal(root: Path, name: str, reads: list[str]) -> None:
    pr.append_event(root / pr.LEDGER_PATH, "seal_design", design=name, reason="t", root=root,
                    protocol=pr.load_protocol(root / pr.PROTOCOL_PATH),
                    payload={"design_sha256": "a" * 64, "reads": reads, "posthoc": False})


def test_holdout_access_is_ledgered_in_stages_before_download(root, files) -> None:
    download, calls = _fake_downloader(files)
    with pytest.raises(pr.ProtocolError):
        access.acquire(ETH_NOV, use="evaluate", design="unsealed", purpose="t", root=root, downloader=download)
    assert calls == []
    _seal(root, "holdout", [ETH_NOV])
    tape, meta = access.load_tape(ETH_NOV, use="evaluate", design="holdout", purpose="t", root=root,
                                  downloader=download)
    assert len(calls) == 2 and calls[0][4]["max_bytes"] == 2 * 1024**3
    access.mark_evaluated(ETH_NOV, use="evaluate", design="holdout", run="results/v07/x", root=root)
    access.mark_inspected(ETH_NOV, run="results/v07/x", root=root)
    report = pr.verify_protocol_files(root)
    assert report["valid"] and report["holdouts"][ETH_NOV]["stages"] == list(pr.STAGES)
    assert report["holdouts"][ETH_NOV]["outcome_inspected_index"] is not None
    assert meta["quality"]["status"] == "FAIL"  # 10-minute synthetic file: below the registered duration


def test_failed_holdout_download_is_not_available_without_substitution(root) -> None:
    calls = []

    def failing(*args, **kwargs):
        calls.append(args)
        raise OSError("HTTP 404")
    _seal(root, "holdout", [ETH_NOV])
    with pytest.raises(access.DataNotAvailable):
        access.acquire(ETH_NOV, use="evaluate", design="holdout", purpose="t", root=root, downloader=failing)
    assert len(calls) == 2 and {c[2] for c in calls} == {"2020-11-01"}
    report = pr.verify_protocol_files(root)
    assert report["attempts"][-1]["outcome"] == "NOT_AVAILABLE"
    assert report["holdouts"][ETH_NOV]["freshness"] == "consumed"


def test_wrong_use_is_refused_before_any_request(root, files) -> None:
    download, calls = _fake_downloader(files)
    with pytest.raises(pr.ProtocolError):
        access.acquire(DEV, use="evaluate", design="d", purpose="t", root=root, downloader=download)
    with pytest.raises(pr.ProtocolError):
        access.acquire(ETH_NOV, use="download", design="d", purpose="t", root=root, downloader=download)
    assert calls == []


BITSTAMP = ROOT / "data/v05/bitstamp/dev-900s/capture.jsonl.gz"


@pytest.mark.skipif(not BITSTAMP.is_file(), reason="consumed v0.5 Bitstamp capture is local-only (never committed)")
def test_bitstamp_adapter_conformance_on_local_capture() -> None:
    from lob.v07.adapters.bitstamp import BitstampAdapter
    report = conformance(BitstampAdapter(BITSTAMP), limit=10**7)
    assert report["valid"], report["issues"]
    assert "order_add" in report["kinds"] and "book_delta" not in report["kinds"]
