"""Day caches: reconstructed, quality-checked, compressed day representations (M17-M21, M26 foundations).

One cache per dataset (``data/v09/cache/<dataset id>.npz``, gitignored) holds:

* **frames** - 100 ms bins over the UTC day in the v0.8 ``EventFrame`` layout (six inferred event counts, six state
  features, touch prices in ticks, ten contiguous visible levels per side in lots, traded volume per aggressor side,
  mid at every bin boundary and a validity mask), so v0.8 summaries and bounded replay apply unchanged;
* **deep** - top-200 non-contiguous levels per side every second inside the registered execution windows
  (warmup start to horizon end + 30 s), for walking large orders;
* **trades** - every print (seconds since the day start, price ticks, size lots, aggressor sign);
* **quality** - reconstruction counters, trade/book consistency checks and the day profile (M19).

Event inference (registered): within a bin, a visible-level size increase is an *add*; a decrease is first explained
by prints at that price on the passive side in the same or previous bin (*market*), and the unexplained remainder is
a *cancel*. Market counts are prints per aggressor side. This is an inference from aggregate data, labelled as such.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from . import bybit, reconstruct as rc

CACHE_DIR = "data/v09/cache"
SCHEMA = "cleolob-v09-daycache-1"
BIN_MS = 100
BINS = 864_000
DEEP_LEVELS = 200
WINDOW_STARTS_S = tuple(h * 3600 + 600 for h in range(24))   # HH:10:00 UTC
WARMUP_S, HORIZON_S, SETTLE_S = 30, 300, 30
MARKS = ("market_buy", "market_sell", "add_bid", "add_ask", "cancel_bid", "cancel_ask")


def day_start(day: str) -> float:
    d = date.fromisoformat(day)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp()


def deep_seconds() -> np.ndarray:
    secs = []
    for s in WINDOW_STARTS_S:
        secs.extend(range(s - WARMUP_S, s + HORIZON_S + SETTLE_S + 1))
    return np.asarray(sorted(set(secs)), np.int64)


def cache_path(root: Path, dataset_id: str) -> Path:
    return Path(root) / CACHE_DIR / f"{dataset_id}.npz"


def build_bybit(l2: Path, trades: Path, *, day: str, tick: float, lot: float, segment: str = "linear") -> dict:
    """Reconstruct one Bybit day into cache arrays (pure function of the raw files and the registered grid)."""
    from . import normalize
    norm = normalize.grid(tick, lot, normalize.infer_grid(bybit.book_messages(l2)))
    tick, lot = norm["tick"], norm["lot"]
    t0_ms = int(day_start(day) * 1000)
    counters = rc.Counters()
    grid = rc._Grid(tick, lot, counters)
    book = rc.DenseBook(counters=counters)
    state: dict = {}
    raw = bybit.read_trades(trades, segment)
    tr_t = raw["t"] - t0_ms / 1000.0
    order = np.argsort(tr_t, kind="stable")
    trades_out = {"t": tr_t[order], "px": np.rint(raw["price"][order] / tick).astype(np.int64),
                  "q": raw["size"][order] / lot, "s": raw["side"][order].astype(np.int8)}
    unsorted_trades = int(np.count_nonzero(np.diff(tr_t) < 0))
    tb = np.floor(trades_out["t"] * 1000 / BIN_MS).astype(np.int64)
    # passive-side traded volume per (bin, price): buy prints consume asks, sell prints consume bids
    traded: dict[tuple[int, int, int], float] = {}
    for b, p, q, s in zip(tb.tolist(), trades_out["px"].tolist(), trades_out["q"].tolist(), trades_out["s"].tolist()):
        key = (b, -1 if s > 0 else 1, p)
        traded[key] = traded.get(key, 0.0) + q

    counts = np.zeros((BINS, 6), np.int32)
    bid_px = np.zeros(BINS, np.int64)
    ask_px = np.zeros(BINS, np.int64)
    depth = np.zeros((BINS, 2, rc.VISIBLE), np.float32)
    valid = np.zeros(BINS, bool)
    secs = deep_seconds()
    deep_px = np.zeros((len(secs), 2, DEEP_LEVELS), np.int64)
    deep_q = np.zeros((len(secs), 2, DEEP_LEVELS), np.float32)
    deep_ok = np.zeros(len(secs), bool)
    deep_bin_arr = (secs * 1000 // BIN_MS).astype(np.int64)
    captured = np.zeros(BINS, bool)
    cur = -1
    decreases: dict[tuple[int, int], float] = {}

    def capture(lo: int, hi: int) -> None:
        """State at the start of bins lo..hi (no message falls between them): captured at lo, forward-filled later."""
        if lo > hi or lo >= BINS:
            return
        hi = min(hi, BINS - 1)
        captured[lo] = True
        bb, ba = book.best(1), book.best(-1)
        if bb is None or ba is None or bb >= ba:
            return
        bid_px[lo], ask_px[lo] = bb, ba
        depth[lo, 0] = book.visible(1)
        depth[lo, 1] = book.visible(-1)
        valid[lo] = True
        a, z = np.searchsorted(deep_bin_arr, lo, "left"), np.searchsorted(deep_bin_arr, hi, "right")
        if z > a:
            levels = [book.levels(side, DEEP_LEVELS) for side in (1, -1)]
            for i in range(a, z):
                for k, (p, q) in enumerate(levels):
                    deep_px[i, k, :len(p)] = p
                    deep_q[i, k, :len(q)] = q
                deep_ok[i] = True

    def close(b: int) -> None:
        """Explain each decrease by passive-side prints of the previous then the same bin; prints are consumed once."""
        for (side, price), dq in decreases.items():
            left = dq
            for key in ((b - 1, side, price), (b, side, price)):
                avail = traded.get(key, 0.0)
                use = min(avail, left)
                if use > 0:
                    traded[key] = avail - use
                    left -= use
            if left > 1e-9:
                counts[b, 4 if side > 0 else 5] += 1
        decreases.clear()

    for m in bybit.book_messages(l2):
        b = (int(m["ts"]) - t0_ms) // BIN_MS
        if b >= BINS:
            break
        if b > cur:
            if cur >= 0:
                close(cur)
            if b >= 0:
                capture(max(cur + 1, 0), b)
            cur = b
        if b < 0:
            rc.apply_message(book, m, grid, state)
            continue
        bb, ba = book.best(1), book.best(-1)
        data = m["data"]
        if m.get("type") != "snapshot" and bb is not None and ba is not None:
            for side, key, best in ((1, "b", bb), (-1, "a", ba)):
                arr = book.bid if side > 0 else book.ask
                for px, sz in data.get(key, ()):
                    p = int(round(float(px) / tick))
                    if (best - p if side > 0 else p - best) >= rc.VISIBLE or (best - p if side > 0 else p - best) < -rc.VISIBLE:
                        continue
                    i = p - book.base
                    old = float(arr[i]) if 0 <= i < book.width else 0.0
                    new = float(sz) / lot
                    if new > old + 1e-9:
                        counts[b, 2 if side > 0 else 3] += 1
                    elif new < old - 1e-9:
                        decreases[(side, p)] = decreases.get((side, p), 0.0) + (old - new)
        rc.apply_message(book, m, grid, state)
    if cur >= 0:
        close(cur)
        capture(cur + 1, BINS - 1)
    # forward-fill uncaptured bins from the last captured bin; bins more than 5 s after it are stale (invalid)
    src = np.where(captured, np.arange(BINS), -1)
    np.maximum.accumulate(src, out=src)
    has = src >= 0
    j = np.where(has, src, 0)
    stale = (np.arange(BINS) - j) * BIN_MS > 5_000
    bid_px, ask_px, depth = bid_px[j], ask_px[j], depth[j]
    valid = has & valid[j] & ~stale
    # market counts and traded volume per bin from prints
    inside = (tb >= 0) & (tb < BINS)
    np.add.at(counts[:, 0], tb[inside & (trades_out["s"] > 0)], 1)
    np.add.at(counts[:, 1], tb[inside & (trades_out["s"] < 0)], 1)
    volume = np.zeros((BINS, 2), np.float32)
    np.add.at(volume[:, 0], tb[inside & (trades_out["s"] > 0)], trades_out["q"][inside & (trades_out["s"] > 0)])
    np.add.at(volume[:, 1], tb[inside & (trades_out["s"] < 0)], trades_out["q"][inside & (trades_out["s"] < 0)])
    mid_ticks = np.where(valid, (bid_px + ask_px) / 2.0, np.nan)
    mid = _ffill(np.r_[mid_ticks, mid_ticks[-1]])
    state_arr = features(bid_px, ask_px, depth, mid)
    quality = _quality(counters, trades_out, bid_px, ask_px, valid, unsorted_trades)
    quality["grid"] = norm
    return {"schema": SCHEMA, "day": day, "tick": tick, "lot": lot, "counts": counts, "state": state_arr,
            "bid_px": bid_px, "ask_px": ask_px, "depth": depth, "volume": volume, "mid": mid, "valid": valid,
            "deep_s": secs, "deep_px": deep_px, "deep_q": deep_q, "deep_ok": deep_ok,
            "trade_t": trades_out["t"], "trade_px": trades_out["px"], "trade_q": trades_out["q"],
            "trade_s": trades_out["s"], "quality": quality}


def build_binance(l1: Path, trades: Path, *, day: str, tick: float, lot: float) -> dict:
    """L1-only day cache from Binance bookTicker + trades (visible levels beyond the touch stay 0 = unobserved)."""
    from . import binance
    t0_ms = int(day_start(day) * 1000)
    bt = binance.read_book_ticker(l1)
    raw = binance.read_trades(trades)
    ts_backwards = int(np.count_nonzero(np.diff(bt["t"]) < 0))
    u_backwards = int(np.count_nonzero(np.diff(bt["u"]) < 0))
    order = np.lexsort((bt["u"], bt["t"]))
    t = bt["t"][order] - t0_ms
    bid = np.rint(bt["bid"][order] / tick).astype(np.int64)
    ask = np.rint(bt["ask"][order] / tick).astype(np.int64)
    bq, aq = bt["bid_q"][order] / lot, bt["ask_q"][order] / lot
    off_tick = int(np.count_nonzero(np.abs(bt["bid"] / tick - np.rint(bt["bid"] / tick)) > 1e-6))
    edges = np.arange(BINS, dtype=np.int64) * BIN_MS
    idx = np.searchsorted(t, edges, side="left") - 1
    has = idx >= 0
    j = np.clip(idx, 0, len(t) - 1)
    stale = np.where(has, edges - t[j], 10**9)
    bid_px = np.where(has, bid[j], 0)
    ask_px = np.where(has, ask[j], 0)
    valid = has & (bid_px < ask_px) & (stale <= 5_000)
    depth = np.zeros((BINS, 2, rc.VISIBLE), np.float32)
    depth[:, 0, 0] = np.where(valid, bq[j], 0.0)
    depth[:, 1, 0] = np.where(valid, aq[j], 0.0)
    tr_t = raw["t"] - t0_ms / 1000.0
    o = np.argsort(tr_t, kind="stable")
    trades_out = {"t": tr_t[o], "px": np.rint(raw["price"][o] / tick).astype(np.int64), "q": raw["size"][o] / lot,
                  "s": raw["side"][o]}
    counts = np.zeros((BINS, 6), np.int32)
    tb = np.floor(trades_out["t"] * 1000 / BIN_MS).astype(np.int64)
    inside = (tb >= 0) & (tb < BINS)
    np.add.at(counts[:, 0], tb[inside & (trades_out["s"] > 0)], 1)
    np.add.at(counts[:, 1], tb[inside & (trades_out["s"] < 0)], 1)
    volume = np.zeros((BINS, 2), np.float32)
    np.add.at(volume[:, 0], tb[inside & (trades_out["s"] > 0)], trades_out["q"][inside & (trades_out["s"] > 0)])
    np.add.at(volume[:, 1], tb[inside & (trades_out["s"] < 0)], trades_out["q"][inside & (trades_out["s"] < 0)])
    # L1 add/cancel inference from consecutive updates at an unchanged touch price
    ub = np.floor(t / BIN_MS).astype(np.int64)
    traded: dict[tuple[int, int, int], float] = {}
    for b_, p_, q_, s_ in zip(tb[inside].tolist(), trades_out["px"][inside].tolist(), trades_out["q"][inside].tolist(),
                              trades_out["s"][inside].tolist()):
        key = (b_, -1 if s_ > 0 else 1, p_)          # buy prints consume asks (side -1), sell prints bids (side 1)
        traded[key] = traded.get(key, 0.0) + q_
    for side, px, q, col_add, col_cancel in ((1, bid, bq, 2, 4), (-1, ask, aq, 3, 5)):
        same = np.r_[False, px[1:] == px[:-1]]
        dq = np.r_[0.0, np.diff(q)]
        ok = same & (ub >= 0) & (ub < BINS)
        np.add.at(counts[:, col_add], ub[ok & (dq > 1e-9)], 1)
        unexplained = []
        for i in np.flatnonzero(ok & (dq < -1e-9)).tolist():
            left = -float(dq[i])
            for key in ((int(ub[i]) - 1, side, int(px[i])), (int(ub[i]), side, int(px[i]))):
                avail = traded.get(key, 0.0)
                use = min(avail, left)
                if use > 0:
                    traded[key] = avail - use
                    left -= use
            if left > 1e-9:
                unexplained.append(i)
        np.add.at(counts[:, col_cancel], ub[unexplained], 1)
    mid_ticks = np.where(valid, (bid_px + ask_px) / 2.0, np.nan)
    mid = _ffill(np.r_[mid_ticks, mid_ticks[-1]])
    counters = rc.Counters(messages=len(t), deltas=len(t), ts_backwards=ts_backwards, out_of_order=u_backwards,
                           crossed=int(np.count_nonzero(bid >= ask)), off_tick=off_tick,
                           max_gap_ms=int(np.diff(t).max()) if len(t) > 1 else 0,
                           first_ts=int(t[0] + t0_ms) if len(t) else None, last_ts=int(t[-1] + t0_ms) if len(t) else None)
    secs = deep_seconds()
    quality = _quality(counters, trades_out, bid_px, ask_px, valid, int(np.count_nonzero(np.diff(tr_t) < 0)))
    quality["level_capability"] = "L1 only (deeper visible levels unobserved, stored as 0)"
    return {"schema": SCHEMA, "day": day, "tick": tick, "lot": lot, "counts": counts,
            "state": features(bid_px, ask_px, depth, mid), "bid_px": bid_px, "ask_px": ask_px, "depth": depth,
            "volume": volume, "mid": mid, "valid": valid, "deep_s": secs,
            "deep_px": np.zeros((len(secs), 2, DEEP_LEVELS), np.int64),
            "deep_q": np.zeros((len(secs), 2, DEEP_LEVELS), np.float32), "deep_ok": np.zeros(len(secs), bool),
            "trade_t": trades_out["t"], "trade_px": trades_out["px"], "trade_q": trades_out["q"],
            "trade_s": trades_out["s"], "quality": quality, "levels": 1}


def features(bid_px, ask_px, depth, mid) -> np.ndarray:
    """Vectorized v0.8 FEATURES: spread ticks (capped 10), imbalance5, log L1 depth, bid q0, ask q0, |ret10|."""
    b5, a5 = depth[:, 0, :5].sum(1).astype(float), depth[:, 1, :5].sum(1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        imb = np.where(b5 + a5 > 0, (b5 - a5) / (b5 + a5), 0.0)
        l1 = np.log(np.maximum(depth[:, 0, 0].astype(float) + depth[:, 1, 0], 1e-6))
    ret10 = np.zeros(len(bid_px))
    ret10[10:] = np.abs(mid[10:len(bid_px)] - mid[:len(bid_px) - 10])
    spread = np.minimum((ask_px - bid_px).astype(float), 10.0)
    return np.column_stack([spread, imb, l1, depth[:, 0, 0], depth[:, 1, 0], ret10]).astype(np.float32)


def _ffill(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, float).copy()
    ok = np.isfinite(out)
    if not ok.any():
        return np.zeros_like(out)
    idx = np.where(ok, np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = out[idx]
    first = int(np.argmax(ok))
    out[:first] = out[first]
    return out


def _quality(c: rc.Counters, trades: dict, bid_px, ask_px, valid, unsorted_trades: int) -> dict:
    tb = np.floor(trades["t"] * 1000 / BIN_MS).astype(np.int64)
    ok = (tb >= 0) & (tb < BINS)
    b = tb[ok]
    px = trades["px"][ok]
    lo_b, hi_b = np.clip(b - 1, 0, BINS - 1), np.clip(b + 1, 0, BINS - 1)
    v = valid[b] & valid[lo_b] & valid[hi_b]
    lo = np.minimum.reduce([bid_px[lo_b], bid_px[b], bid_px[hi_b]]).astype(float)
    hi = np.maximum.reduce([ask_px[lo_b], ask_px[b], ask_px[hi_b]]).astype(float)
    # gross inconsistency: a print more than 20 bps outside the touch range of the surrounding +-1 bins (sweeps and
    # 100 ms book quantization legitimately put prints several ticks beyond the touch of their own bin)
    band = 0.002 * np.maximum(hi, 1.0)
    far = v & ((px < lo - band) | (px > hi + band))
    beyond = v & ((px < bid_px[b]) | (px > ask_px[b]))
    spread = (ask_px - bid_px)[valid]
    return {"reconstruction": c.to_dict(), "trades": int(len(trades["t"])), "trades_outside_day": int((~ok).sum()),
            "trades_unsorted": unsorted_trades, "trades_inconsistent_20bps": int(far.sum()),
            "trades_beyond_own_bin_touch": int(beyond.sum()),
            "trade_side_unknown": int((trades["s"] == 0).sum()),
            "valid_bin_fraction": float(valid.mean()),
            "profile": {"spread_ticks_p50": float(np.median(spread)) if len(spread) else None,
                        "spread_ticks_p99": float(np.percentile(spread, 99)) if len(spread) else None,
                        "trades_per_hour": float(len(trades["t"]) / 24.0),
                        "volume_lots": float(trades["q"].sum()),
                        "buy_fraction": float((trades["s"] > 0).mean()) if len(trades["s"]) else None}}


def save(path: Path, cache: dict) -> str:
    """Byte-deterministic compressed archive (fixed member timestamps and order), so rebuilds reproduce digests."""
    import zipfile
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in cache.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in cache.items() if not isinstance(v, np.ndarray)}
    arrays["__meta__"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
    tmp = path.with_name(path.stem + ".tmp.npz")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with archive.open(info, "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, np.ascontiguousarray(arrays[name]), allow_pickle=False)
    tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def load(path: Path, *, mmap: bool = False) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "__meta__"}
        out.update(json.loads(bytes(z["__meta__"]).decode()))
    return out


def frame(cache: dict, start_bin: int = 0, stop_bin: int | None = None):
    """A v0.8 EventFrame view of a cache slice (trade samples relative to the slice start)."""
    from ...v08.market.frame import EventFrame
    stop_bin = BINS if stop_bin is None else stop_bin
    sl = slice(start_bin, stop_bin)
    t_lo, t_hi = start_bin * BIN_MS / 1000.0, stop_bin * BIN_MS / 1000.0
    k = (cache["trade_t"] >= t_lo) & (cache["trade_t"] < t_hi)
    samples = {"trade_t": cache["trade_t"][k] - t_lo, "trade_px": cache["trade_px"][k].astype(float),
               "trade_q": cache["trade_q"][k], "trade_s": cache["trade_s"][k].astype(float),
               "market_size": cache["trade_q"][k]}
    return EventFrame(cache["counts"][sl].astype(np.int64), cache["state"][sl].astype(float), cache["bid_px"][sl],
                      cache["ask_px"][sl], cache["depth"][sl].astype(float), cache["volume"][sl].astype(float),
                      cache["mid"][start_bin:stop_bin + 1], cache["valid"][sl], samples=samples,
                      meta={"source": "historical", "tick": cache["tick"], "t0": 0.0, "day": cache["day"],
                            "start_bin": start_bin})
