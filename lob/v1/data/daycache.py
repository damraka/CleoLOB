"""v1 day caches: acquisition -> canonical events -> reconstructed frames -> quality decision -> columnar cache.

``prepare`` is the only path from a registered dataset to model-ready arrays. It

1. records a ledgered ``access`` (namespace, design, use) *before* parsing - an access that the ledger refuses
   (wrong namespace for the dataset's role, unsealed design, closed namespace) stops here and nothing is read;
2. ingests the raw sources streaming into canonical events (``ingest``);
3. reconstructs 100 ms frames with gap invalidation (``reconstruction.l2.frames``) plus deep snapshots inside the
   registered windows;
4. computes the registered quality checks (``QUALITY`` thresholds, the v0.9 thresholds carried unchanged plus the
   v1 gap rule) and returns ``AVAILABLE`` or ``QUALITY_REJECTED`` with every counter;
5. writes the columnar cache (``data.columnar``); ``load`` returns a memory-mapped dict with the v0.9 cache keys, so
   unchanged v0.9 model code (point processes, survival, fills) can be applied to v1 data for exact replication.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import columnar, ingest
from .connectors import base as CB
from .reconstruction import l2

CACHE_DIR = "data/v1/cache"
EVENTS_DIR = "data/v1/events"
WINDOW_STARTS_S = tuple(h * 3600 + 600 for h in range(24))
WARMUP_S, HORIZON_S, SETTLE_S = 30, 300, 30
QUALITY = {
    "max_sequence_problem_rate": 0.001,
    "max_crossed_rate": 0.001,
    "min_valid_bin_fraction": 0.98,
    "max_long_gap_minutes": 10.0,
    "max_inconsistent_trade_rate": 0.005,
    "max_unknown_side_rate": 0.001,
    "max_off_grid_rate": 0.001,
    "min_trades": 1000,
    "max_unreliable_message_rate": 0.001,
}


def deep_seconds() -> np.ndarray:
    secs = set()
    for s in WINDOW_STARTS_S:
        secs.update(range(s - WARMUP_S, s + HORIZON_S + SETTLE_S + 1))
    return np.asarray(sorted(secs), np.int64)


def features(bid_px, ask_px, depth, mid) -> np.ndarray:
    from ...v09.data.daycache import features as f09
    return f09(bid_px, ask_px, depth, mid)


def quality(frames: dict, book_meta: dict) -> dict:
    c = frames["counters"]
    msgs = max(c["messages"], 1)
    t = frames["trade_t"]
    bins = len(frames["valid"])
    tb = np.floor(t * 10).astype(np.int64)
    ok = (tb >= 0) & (tb < bins)
    b = tb[ok]
    px = frames["trade_px"][ok]
    valid, bid, ask = frames["valid"], frames["bid_px"], frames["ask_px"]
    lo_b, hi_b = np.clip(b - 1, 0, bins - 1), np.clip(b + 1, 0, bins - 1)
    v = valid[b] & valid[lo_b] & valid[hi_b]
    lo = np.minimum.reduce([bid[lo_b], bid[b], bid[hi_b]]).astype(float)
    hi = np.maximum.reduce([ask[lo_b], ask[b], ask[hi_b]]).astype(float)
    far = v & ((px < lo - 0.002 * hi) | (px > hi + 0.002 * hi))
    n_tr = int(len(t))
    grid = book_meta.get("grid", {})
    levels = max(book_meta.get("events", 1), 1)
    rates = {
        "sequence_problem_rate": (c["sequence_gaps"] + c["duplicates"] + c["out_of_order"]) / msgs,
        "crossed_rate": c["crossed"] / msgs,
        "valid_bin_fraction": float(valid.mean()),
        "long_gap_minutes": c["long_gap_ms_total"] / 60_000,
        "inconsistent_trade_rate": float(far.sum()) / max(n_tr, 1),
        "unknown_side_rate": float((frames["trade_s"] == 0).mean()) if n_tr else 0.0,
        "off_grid_rate": (grid.get("off_tick", 0) + grid.get("off_lot", 0)) / levels,
        "trades": n_tr,
        "unreliable_message_rate": c["unreliable_messages"] / msgs,
    }
    q = QUALITY
    checks = {
        "sequence": rates["sequence_problem_rate"] <= q["max_sequence_problem_rate"],
        "crossed": rates["crossed_rate"] <= q["max_crossed_rate"],
        "valid_bins": rates["valid_bin_fraction"] >= q["min_valid_bin_fraction"],
        "long_gaps": rates["long_gap_minutes"] <= q["max_long_gap_minutes"],
        "trade_consistency": rates["inconsistent_trade_rate"] <= q["max_inconsistent_trade_rate"],
        "trade_side": rates["unknown_side_rate"] <= q["max_unknown_side_rate"],
        "grid": rates["off_grid_rate"] <= q["max_off_grid_rate"],
        "min_trades": n_tr >= q["min_trades"],
        "reliability": rates["unreliable_message_rate"] <= q["max_unreliable_message_rate"],
    }
    spread = (ask - bid)[valid]
    return {"status": "AVAILABLE" if all(checks.values()) else "QUALITY_REJECTED", "checks": checks,
            "rates": rates, "thresholds": dict(q),
            "profile": {"spread_ticks_p50": float(np.median(spread)) if len(spread) else None,
                        "spread_ticks_mean": float(spread.mean()) if len(spread) else None,
                        "trades_per_hour": n_tr / 24.0}}


def cache_path(root: Path, dataset_id: str) -> Path:
    return Path(root) / CACHE_DIR / f"{dataset_id}.arrow"


def build(dataset_id: str, connector: str, segment: str, symbol: str, day: str, files: dict[str, Path], *,
          root: Path, raw_sha256: dict) -> dict:
    """Pure function of the raw files: events, frames, quality, cache (no ledger interaction)."""
    conn = CB.get(connector)
    inst = conn.instrument(segment, symbol, day)
    cap = conn.capability(segment, symbol, day)
    ev_dir = Path(root) / EVENTS_DIR / dataset_id
    from dataclasses import replace

    from ...v09.data import normalize
    from .connectors import bybit as BY
    if connector != "bybit":
        raise ValueError(f"day caches are defined for L2 sources; {connector} is {cap.book}")
    # registered grid normalization (v0.9 amendment 2, carried unchanged): coarser tick, finer lot
    norm = normalize.grid(inst.tick, inst.lot, normalize.infer_grid(BY.book_messages(files["l2"])))
    inst = replace(inst, tick=norm["tick"], lot=norm["lot"])
    grid = BY.Grid(inst.tick, inst.lot)
    if connector == "bybit":
        book_meta = ingest.write_events(BY.book_events(BY.book_messages(files["l2"]), grid), ev_dir / "book.parquet")
        trades = BY.trade_events(files["trades"], segment, BY.Grid(inst.tick, inst.lot))
        ingest.write_events([trades], ev_dir / "trades.parquet")
    book_meta["grid"] = grid.counters()
    fr = l2.frames(ingest.iter_events(ev_dir / "book.parquet"), trades, day=day, deep_seconds=deep_seconds())
    q = quality(fr, book_meta)
    fr["state"] = features(fr["bid_px"], fr["ask_px"], fr["depth"], fr["mid"])
    meta = {"dataset": dataset_id, "day": day, "tick": inst.tick, "lot": inst.lot, "stream": inst.stream,
            "grid_normalization": norm,
            "capability": cap.to_dict(), "raw_sha256": raw_sha256, "quality": q, "book_events": book_meta}
    out = cache_path(root, dataset_id)
    columnar.write_frames(fr, out, meta=meta)
    (ev_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")
    return {"dataset": dataset_id, "status": q["status"], "quality": q, "cache": out.as_posix()}


def prepare(ws, spec, raw, *, namespace: str, design: str | None, use: str, root: Path) -> dict:
    """Ledgered access + build (or reuse an existing cache) for one acquired dataset ``spec``."""
    from ..protocol import document as D
    from .connectors import base
    state = ws.state()
    if spec.id not in state.acquired:
        raise ValueError(f"{spec.id} was not acquired")
    D.record_access(ws, spec.id, namespace=namespace, stage="parsed", use=use, design=design,
                    reason=f"{use}: day cache of {spec.id}")
    conn = base.get(spec.connector)
    files = {}
    for s in conn.sources(spec.segment, spec.symbol, spec.date):
        got = raw.lookup(s.url)
        if got is not None:
            files[s.name] = got.path
    path = cache_path(root, spec.id)
    if path.exists():
        f = columnar.open_frames(path)
        return {"dataset": spec.id, "status": f.meta["meta"]["quality"]["status"], "cache": path.as_posix(),
                "reused": True}
    return build(spec.id, spec.connector, spec.segment, spec.symbol, spec.date, files, root=root,
                 raw_sha256=dict(state.acquired[spec.id]))


def load(path: Path) -> dict:
    """Memory-mapped cache dict with the v0.9 keys (``counts``, ``state``, ..., ``tick``, ``lot``, ``day``)."""
    f = columnar.open_frames(Path(path))
    out = {k: f[k] for k in f.keys()}
    m = f.meta["meta"]
    out.update({"tick": m["tick"], "lot": m["lot"], "day": m["day"], "quality": m["quality"],
                "dataset": m["dataset"], "stream": m["stream"]})
    return out


__all__ = ["QUALITY", "prepare", "build", "load", "quality", "cache_path", "deep_seconds", "WINDOW_STARTS_S"]
