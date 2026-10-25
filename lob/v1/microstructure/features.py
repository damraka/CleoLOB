"""Canonical microstructure feature store (M50), order-flow imbalance (M51) and liquidity metrics (M52).

``FeatureStore`` computes named features from a v1 day cache (or any frame-like dict with the cache keys) at a
chosen sampling interval and persists them as one Arrow IPC file per (dataset, interval) with the feature
definitions and the cache fingerprint in the metadata, so every study reads the *same* definition of, say,
"5-level order-flow imbalance at 1 s". Registered features (all on the 100 ms frames, aggregated to the interval):

* ``ofi_l{k}`` - multi-level order-flow imbalance (Cont, Kukanov & Stoikov 2014 generalized to k levels): sum
  over levels of signed depth changes, bid increases and ask decreases count as buy pressure, with level resets
  when the touch moves (the standard event-based definition at the frame resolution);
* ``imbalance_l{k}`` - depth imbalance over k levels; ``microprice_offset`` - microprice minus mid (ticks);
* ``spread`` (ticks), ``log_touch_depth``, ``signed_volume``, ``trade_count``, ``return_bps``, ``rv_60s``;
* liquidity (M52): ``depth_resiliency`` (touch depth relative to its 60 s median), ``refill_rate`` (inferred
  additions at the touch per second), ``cancel_rate``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

BIN_MS = 100


def _agg(x: np.ndarray, k: int, how: str = "sum") -> np.ndarray:
    n = len(x) // k
    y = x[: n * k].reshape(n, k, *x.shape[1:])
    return y.sum(1) if how == "sum" else y.mean(1) if how == "mean" else y[:, -1]


def ofi(bid_px, ask_px, depth, levels: int = 1) -> np.ndarray:
    """Per-bin multi-level OFI (lots): e^b - e^a summed over the first ``levels`` levels."""
    bid_px, ask_px = np.asarray(bid_px), np.asarray(ask_px)
    d = np.asarray(depth, float)
    n = len(bid_px)
    out = np.zeros(n)
    for k in range(levels):
        qb, qa = d[:, 0, k], d[:, 1, k]
        pb, pa = bid_px - k, ask_px + k
        eb = np.zeros(n)
        ea = np.zeros(n)
        up, same, dn = pb[1:] > pb[:-1], pb[1:] == pb[:-1], pb[1:] < pb[:-1]
        eb[1:] = np.where(up, qb[1:], 0) + np.where(same, qb[1:] - qb[:-1], 0) - np.where(dn, qb[:-1], 0)
        dn_a, same_a, up_a = pa[1:] < pa[:-1], pa[1:] == pa[:-1], pa[1:] > pa[:-1]
        ea[1:] = np.where(dn_a, qa[1:], 0) + np.where(same_a, qa[1:] - qa[:-1], 0) - np.where(up_a, qa[:-1], 0)
        out += eb - ea
    return out


def imbalance(depth, levels: int) -> np.ndarray:
    d = np.asarray(depth, float)
    b, a = d[:, 0, :levels].sum(1), d[:, 1, :levels].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(a + b > 0, (b - a) / (a + b), 0.0)


def compute(cache: dict, *, interval_ms: int = 1000) -> dict[str, np.ndarray]:
    k = max(1, interval_ms // BIN_MS)
    bid, ask, depth = np.asarray(cache["bid_px"]), np.asarray(cache["ask_px"]), np.asarray(cache["depth"], float)
    valid = np.asarray(cache["valid"], bool)
    mid = np.asarray(cache["mid"], float)[: len(bid)]
    vol = np.asarray(cache["volume"], float)
    counts = np.asarray(cache["counts"], float)
    f: dict[str, np.ndarray] = {}
    for lv in (1, 5, 10):
        f[f"ofi_l{lv}"] = _agg(np.where(valid, ofi(bid, ask, depth, lv), 0.0), k)
        f[f"imbalance_l{lv}"] = _agg(imbalance(depth, lv), k, "last")
    qb, qa = depth[:, 0, 0], depth[:, 1, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        micro = np.where(qb + qa > 0, (bid * qa + ask * qb) / (qa + qb), (bid + ask) / 2) - (bid + ask) / 2
    f["microprice_offset"] = _agg(micro, k, "last")
    f["spread"] = _agg((ask - bid).astype(float), k, "last")
    touch = qb + qa
    f["log_touch_depth"] = _agg(np.log(np.maximum(touch, 1.0)), k, "last")
    f["signed_volume"] = _agg(vol[:, 0] - vol[:, 1], k)
    f["trade_count"] = _agg(counts[:, 0] + counts[:, 1], k)
    m = _agg(mid, k, "last")
    r = np.r_[0.0, np.diff(m) / np.maximum(m[:-1], 1e-9) * 1e4]
    f["return_bps"] = r
    w = max(1, 60_000 // interval_ms)
    f["rv_60s"] = np.sqrt(np.convolve(r ** 2, np.ones(w), "full")[: len(r)])
    med = np.array([np.median(touch[max(0, i - 600):i + 1]) for i in range(0, len(touch), k)])[: len(f["spread"])]
    f["depth_resiliency"] = _agg(touch, k, "last") / np.maximum(med, 1.0)
    f["refill_rate"] = _agg(counts[:, 2] + counts[:, 3], k) / (interval_ms / 1000)
    f["cancel_rate"] = _agg(counts[:, 4] + counts[:, 5], k) / (interval_ms / 1000)
    f["valid"] = _agg(valid.astype(float), k, "mean") >= 0.999
    n = min(len(v) for v in f.values())
    return {name: v[:n] for name, v in f.items()}


def cache_fingerprint(cache: dict) -> str:
    h = hashlib.sha256()
    for key in ("bid_px", "ask_px", "valid"):
        a = np.ascontiguousarray(cache[key][::997])
        h.update(a.tobytes())
    h.update(str(cache.get("dataset", "")).encode())
    return h.hexdigest()[:16]


class FeatureStore:
    def __init__(self, base: Path) -> None:
        self.base = Path(base)

    def path(self, dataset: str, interval_ms: int, fp: str) -> Path:
        """Content-addressed by the cache fingerprint: a changed cache gets a new file, nothing is overwritten
        (memory-mapped files cannot be replaced while mapped on Windows)."""
        return self.base / f"{dataset}@{interval_ms}ms-{fp}.arrow"

    def get(self, cache: dict, *, interval_ms: int = 1000) -> dict[str, np.ndarray]:
        from ..data import columnar
        ds = cache.get("dataset", "unknown")
        fp = cache_fingerprint(cache)
        p = self.path(ds, interval_ms, fp)
        if p.exists():
            f = columnar.open_frames(p)
            return {k: f[k] for k in f.keys()}
        feats = compute(cache, interval_ms=interval_ms)
        columnar.write_frames(feats, p, meta={"fingerprint": fp, "interval_ms": interval_ms, "dataset": ds,
                                              "definitions": json.dumps(sorted(feats))})
        return feats


__all__ = ["FeatureStore", "compute", "ofi", "imbalance", "cache_fingerprint"]
