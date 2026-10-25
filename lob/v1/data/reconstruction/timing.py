"""Exchange-time versus receive-time alignment (M24).

Sources with local receive clocks (prospective captures) allow measuring the observation delay
``d = ts_receive - ts_exchange`` per event. Because local and venue clocks are not synchronized, ``d`` contains an
unknown constant offset; the registered estimator is the minimum-delay offset (the fastest messages are assumed to
have ~zero network delay - a lower-envelope assumption, never a latency measurement of the venue). Reported:

* raw delay quantiles, the offset estimate and offset-corrected delay quantiles;
* ``negative_after_offset`` (impossible delays after correction, i.e. clock drift during the capture);
* exchange-time inversions in receive order (messages received out of exchange order), per event kind;
* drift: the slope of the per-minute minimum delay against time (us per hour).

Archives without receive time return ``status = NOT_AVAILABLE``: their latency is never imputed.
"""
from __future__ import annotations

import numpy as np

from ..schema import NO_TS, EventKind


def alignment(ev: np.ndarray) -> dict:
    has = ev["ts_receive"] != NO_TS
    ev = ev[has & (ev["kind"] != EventKind.GAP)]
    if len(ev) < 10:
        return {"status": "NOT_AVAILABLE", "reason": "source has no receive timestamps"}
    d = (ev["ts_receive"] - ev["ts_exchange"]).astype(np.float64)
    offset = float(np.min(d))
    corrected = d - offset
    minute = (ev["ts_receive"] - ev["ts_receive"][0]) // 60_000_000_000
    mins = np.array([d[minute == m].min() for m in np.unique(minute)])
    drift = float(np.polyfit(np.arange(len(mins)), mins, 1)[0] * 60 / 1e3) if len(mins) >= 3 else None
    inv = {}
    for k in np.unique(ev["kind"]):
        t = ev["ts_exchange"][ev["kind"] == k]
        inv[EventKind(int(k)).name] = int((np.diff(t) < 0).sum())
    q = [0.0, 0.01, 0.5, 0.99, 1.0]
    return {"status": "OK", "events": int(len(ev)),
            "raw_delay_ms": dict(zip(map(str, q), (np.quantile(d, q) / 1e6).round(3).tolist())),
            "offset_ms": offset / 1e6, "assumption": "minimum observed delay = zero network delay",
            "corrected_delay_ms": dict(zip(map(str, q), (np.quantile(corrected, q) / 1e6).round(3).tolist())),
            "drift_us_per_hour": drift, "exchange_time_inversions": inv,
            "inversions_total": int((np.diff(ev["ts_exchange"]) < 0).sum())}


def aligned_receive(ev: np.ndarray, offset_ns: float) -> np.ndarray:
    """Receive times mapped onto the exchange clock with the estimated offset (NO_TS stays NO_TS)."""
    out = ev["ts_receive"].astype(np.int64).copy()
    ok = out != NO_TS
    out[ok] = out[ok] - int(round(offset_ns))
    return out


__all__ = ["alignment", "aligned_receive"]
