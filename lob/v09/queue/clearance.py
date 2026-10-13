"""Observable queue-clearance events from public L2 + prints (M22, M23, M66).

At a probe time t, for the touch price p of one side with visible queue q_t, the *back-of-queue clearance time* is

    T = min( first passive-side print strictly through p after t,
             first time the cumulative passive-side prints at p since t exceed q_t )  - t

A hypothetical order that joined the back of the queue at t is certainly filled by T whatever happens to the
orders ahead of it (cancellations ahead only make it earlier), so T is an exact upper bound of its fill time and an
observable, assumption-free survival target (BOUNDED: it is not the fill time of any actual order). Probes are taken
every second inside the registered windows; T is right-censored at the registered 30 s horizon (or the end of the
day). ``queue_confidence`` reports how well the 100 ms aggregate book pins the queue down (M22): the fraction of probes
whose level changed by both adds and decreases inside the bin (ambiguous ordering) and the visible-size uncertainty.

Covariates (observed at t only): log queue at p, log opposite touch size, five-level imbalance (signed so positive
favours a fill), spread in ticks, passive-side print volume over the previous 5 s relative to q_t, mid volatility
over the previous 60 s, hour of day (sin, cos) and side.
"""
from __future__ import annotations

import math

import numpy as np

from ..data.daycache import BIN_MS, HORIZON_S, WINDOW_STARTS_S

HORIZON = 30.0
FEATURES = ("log_queue", "log_opposite", "imbalance_fav", "spread", "flow_ratio", "vol60", "hour_sin", "hour_cos",
            "side")


def probe_times(step_s: float = 1.0) -> np.ndarray:
    return np.concatenate([np.arange(s, s + HORIZON_S, step_s) for s in WINDOW_STARTS_S]).astype(float)


def events(cache: dict, *, step_s: float = 1.0, horizon: float = HORIZON) -> dict:
    """(features, durations, observed flags, side, probe time) for both sides at every probe."""
    t_tr, px, q, s = cache["trade_t"], cache["trade_px"], cache["trade_q"], cache["trade_s"]
    order = np.argsort(t_tr, kind="stable")
    t_tr, px, q, s = t_tr[order], px[order], q[order].astype(float), s[order]
    probes = probe_times(step_s)
    rows, dur, obs, side_out, times = [], [], [], [], []
    mid = cache["mid"]
    for t in probes.tolist():
        b = int(t * 1000 // BIN_MS)
        if b >= len(cache["valid"]) or not cache["valid"][b]:
            continue
        lo = int(np.searchsorted(t_tr, t, "right"))
        hi = int(np.searchsorted(t_tr, t + horizon, "right"))
        prev = int(np.searchsorted(t_tr, t - 5.0, "right"))
        m60 = mid[max(0, b - 600):b + 1:10]
        vol = float(np.std(np.diff(m60))) if len(m60) > 3 else 0.0
        hour = t / 3600.0
        for side in (1, -1):                         # +1 = resting bid (filled by sell prints), -1 = resting ask
            k = 0 if side > 0 else 1
            p = int(cache["bid_px"][b] if side > 0 else cache["ask_px"][b])
            queue = float(cache["depth"][b, k, 0])
            opp = float(cache["depth"][b, 1 - k, 0])
            if queue <= 0:
                continue
            passive_sign = -1 if side > 0 else 1
            seg_s, seg_p, seg_q, seg_t = s[lo:hi], px[lo:hi], q[lo:hi], t_tr[lo:hi]
            hit = seg_s == passive_sign
            through = hit & ((seg_p < p) if side > 0 else (seg_p > p))
            at = hit & (seg_p == p)
            t_through = float(seg_t[through][0]) if through.any() else math.inf
            cum = np.cumsum(np.where(at, seg_q, 0.0))
            cleared = np.flatnonzero(cum > queue + 1e-9)
            t_at = float(seg_t[cleared[0]]) if len(cleared) else math.inf
            t_event = min(t_through, t_at)
            observed = math.isfinite(t_event)
            d = (t_event - t) if observed else horizon
            flow = float(q[prev:lo][s[prev:lo] == passive_sign].sum()) / max(queue, 1e-9)
            imb = float(cache["state"][b, 1])
            rows.append([math.log1p(queue), math.log1p(opp), -side * imb, float(cache["ask_px"][b] - cache["bid_px"][b]),
                         math.log1p(flow), vol, math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
                         float(side)])
            dur.append(max(d, 1e-4))
            obs.append(observed)
            side_out.append(side)
            times.append(t)
    return {"X": np.asarray(rows, float).reshape(-1, len(FEATURES)), "T": np.asarray(dur, float),
            "E": np.asarray(obs, bool), "side": np.asarray(side_out, np.int8), "t": np.asarray(times, float)}


def queue_confidence(cache: dict) -> dict:
    """M22: how ambiguous the aggregate book is about touch queues (descriptive, per day)."""
    c = cache["counts"][cache["valid"]]
    both = ((c[:, 2] > 0) & ((c[:, 4] > 0) | (c[:, 0] > 0) | (c[:, 1] > 0))).mean() if len(c) else None
    return {"bins_with_add_and_decrease_fraction": float(both) if both is not None else None,
            "note": "adds and decreases inside one 100 ms bin have unknown order; queue position is never observed",
            "order_ids": False}
