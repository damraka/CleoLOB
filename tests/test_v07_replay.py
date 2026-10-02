"""v0.7 replay v3: reference-snapshot agreement, counterfactual separation, corrupt-interval policy."""
from __future__ import annotations

import gzip

import pytest

from lob.replay.l2 import L2Replay
from lob.v07.queue.models import LevelEvent
from lob.v07.replay.reference import CounterfactualChild, exclude_corrupt, snapshot_agreement
from tests.v07_fixtures import write_tardis


def _write_reference(l2, path, corrupt_every=None):
    header = ["exchange", "symbol", "timestamp", "local_timestamp"]
    for i in range(5):
        header += [f"asks[{i}].price", f"asks[{i}].amount", f"bids[{i}].price", f"bids[{i}].amount"]
    lines = [",".join(header)]
    for n, state in enumerate(L2Replay(l2, depth=5)):
        row = ["deribit", "ETH-PERPETUAL", str(state.timestamp_us + 1), str(state.local_timestamp_us + 1)]
        for i in range(5):
            a = state.asks[i] if i < len(state.asks) else ("", "")
            b = state.bids[i] if i < len(state.bids) else ("", "")
            amount = str(a[1]) if not (corrupt_every and n % corrupt_every == 0 and a[1] != "") else str(a[1] + 1)
            row += [str(a[0]), amount, str(b[0]), str(b[1])]
        lines.append(",".join(row))
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(lines) + "\n")


def test_reference_agreement(tmp_path) -> None:
    files = write_tardis(tmp_path, seconds=120.0)
    good, bad = tmp_path / "good.csv.gz", tmp_path / "bad.csv.gz"
    _write_reference(files["l2"], good)
    _write_reference(files["l2"], bad, corrupt_every=2)
    ok = snapshot_agreement(files["l2"], good, every=1)
    assert ok["compared"] > 100 and ok["exact_agreement"] == pytest.approx(1.0)
    worse = snapshot_agreement(files["l2"], bad, every=1)
    assert worse["exact_agreement"] < 0.7 and worse["price_agreement"] == pytest.approx(1.0)


def test_counterfactual_child_never_mutates_observed_state() -> None:
    child = CounterfactualChild(side=1, price=100.0, qty=5, model="fifo_lower", level_at_join=10)
    for e in (LevelEvent(1, "size", 8), LevelEvent(2, "trade_at", 12)):
        child.observe(e)
    out = child.outcome()
    assert out["hypothetical"] and out["observed_state_modified"] is False and out["filled"] == 4  # 8 ahead after the cancel cap; 12 printed -> 4
    with pytest.raises(ValueError):
        CounterfactualChild(side=1, price=1, qty=1, model="exact_fifo")


def test_corrupt_interval_policy() -> None:
    out = exclude_corrupt([(0, 10), (10, 20), (20, 30)], [(12, 15)])
    assert out["excluded"] == [(10, 20)] and len(out["kept"]) == 2
