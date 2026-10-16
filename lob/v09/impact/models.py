"""Observational impact models (M71-M73) on aggressive-flow bursts (ASSUMPTION_DEPENDENT, never causal by default).

Bursts (registered): per UTC second, signed aggressive volume v = buy prints - sell prints (lots). A burst is a second
whose |v| is in the top 1 % of that day's nonzero seconds. Its responses are the mid moves from the start of the
second to +1 s, +10 s and +60 s, signed by the burst direction, in bps of the start mid:

* temporary = response at 1 s, transient = response at 10 s minus the 60 s level, persistent = response at 60 s
  (a descriptive decomposition, M72).

Models of the 10 s response:

* ``SqrtImpact`` - Y sigma sqrt(|v| / V) with sigma the day's 10 s mid volatility (bps) and V the mean absolute
  signed volume per second; one parameter Y by least squares (the registered baseline);
* ``StateImpact`` - gradient-boosted regression trees (squared loss, depth 2, registered size) on log|v|/V, log
  opposite touch depth, spread, signed imbalance, recent volatility, recent signed flow and the square-root term.

``kernel`` estimates a transient propagator G(l) (mid return at lag l per unit signed flow) by least squares over
lags 0..30 s (M71); ``cross_impact`` regresses one instrument's mid moves on another's bursts (M74, descriptive).
"""
from __future__ import annotations

import math

import numpy as np

from ..queue.survival import _Tree

HORIZONS = (1, 10, 60)
QUANTILE = 0.99


def seconds(cache: dict) -> dict:
    """Per-second signed aggressive volume, mid (start of second, ticks), and state at the second start."""
    n_s = len(cache["mid"]) // 10
    t = cache["trade_t"]
    sec = np.floor(t).astype(np.int64)
    ok = (sec >= 0) & (sec < n_s)
    v = np.zeros(n_s)
    np.add.at(v, sec[ok], (cache["trade_q"][ok] * cache["trade_s"][ok]).astype(float))
    mid = cache["mid"][::10][:n_s + 1].astype(float)
    b = np.arange(n_s) * 10
    return {"v": v, "mid": mid, "valid": cache["valid"][b], "state": cache["state"][b].astype(float),
            "depth": cache["depth"][b, :, 0].astype(float), "tick": float(cache["tick"])}


def bursts(cache: dict) -> dict:
    s = seconds(cache)
    v, mid = s["v"], s["mid"]
    n = len(v)
    nz = np.abs(v[v != 0])
    if len(nz) < 100:
        return {"X": np.zeros((0, 7)), "y": {h: np.zeros(0) for h in HORIZONS}, "sec": np.zeros(0, int)}
    thr = float(np.quantile(nz, QUANTILE))
    ret10 = np.diff(mid[::10]) / mid[:-10:10] * 1e4 if len(mid) > 20 else np.zeros(1)
    sigma = float(np.std(ret10)) or 1e-9
    V = float(np.mean(np.abs(v))) or 1e-9
    idx = np.flatnonzero((np.abs(v) >= thr) & s["valid"])
    idx = idx[(idx + max(HORIZONS) < n) & (idx >= 60)]
    sign = np.sign(v[idx])
    y = {h: sign * (mid[idx + h] - mid[idx]) / mid[idx] * 1e4 for h in HORIZONS}
    opp = np.where(sign > 0, s["depth"][idx, 1], s["depth"][idx, 0])
    recent_flow = np.asarray([v[i - 30:i].sum() for i in idx]) * sign / V
    recent_vol = np.asarray([np.std(np.diff(mid[i - 60:i + 1])) / mid[i] * 1e4 for i in idx])
    X = np.column_stack([np.log(np.abs(v[idx]) / V), np.log1p(opp), s["state"][idx, 0],
                         sign * s["state"][idx, 1], recent_vol, recent_flow,
                         sigma * np.sqrt(np.abs(v[idx]) / V)])
    return {"X": X, "y": y, "sec": idx, "sigma": sigma, "V": V, "threshold": thr}


class SqrtImpact:
    name = "sqrt"

    def fit(self, X, y):
        z = X[:, 6]
        self.Y = float((z * y).sum() / max((z * z).sum(), 1e-12))
        return self

    def predict(self, X):
        return self.Y * X[:, 6]


class StateImpact:
    name = "state_dependent"

    def __init__(self, trees: int = 150, lr: float = 0.05, n_bins: int = 32) -> None:
        self.trees, self.lr, self.n_bins = trees, lr, n_bins

    def _bin(self, X):
        return np.stack([np.clip(np.searchsorted(self.cuts[j], X[:, j], side="right"), 0, self.n_bins - 1)
                         for j in range(X.shape[1])], axis=1).astype(np.int64)

    def fit(self, X, y):
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.cuts = [np.unique(np.quantile(X[:, j], qs)) for j in range(X.shape[1])]
        B = self._bin(X)
        self.base = float(np.mean(y))
        f = np.full(len(y), self.base)
        self.ensemble = []
        h = np.ones(len(y))
        for _ in range(self.trees):
            tree = _Tree().fit(B, f - y, h, self.n_bins, lam=1.0, min_leaf=20.0)
            f += self.lr * tree.predict(B)
            self.ensemble.append(tree)
        return self

    def predict(self, X):
        B = self._bin(X)
        return self.base + self.lr * sum(t.predict(B) for t in self.ensemble)


def decomposition(b: dict) -> dict:
    y1, y10, y60 = (np.asarray(b["y"][h]) for h in HORIZONS)
    if not len(y1):
        return {}
    return {"temporary_1s_bps": float(y1.mean()), "transient_10s_minus_60s_bps": float((y10 - y60).mean()),
            "persistent_60s_bps": float(y60.mean()), "bursts": int(len(y1)),
            "label": "observational, ASSUMPTION_DEPENDENT (information and flow are confounded)"}


def kernel(cache: dict, lags: int = 30, ridge: float = 1e-3) -> dict:
    """Transient propagator: r_t = sum_l G(l) v_{t-l} / V + e (per-second, bps), ridge least squares."""
    s = seconds(cache)
    v, mid = s["v"], s["mid"]
    r = np.diff(mid) / mid[:-1] * 1e4
    n = len(r)
    V = float(np.mean(np.abs(v))) or 1e-9
    X = np.column_stack([np.r_[np.zeros(lag), v[: n - lag]] / V for lag in range(lags + 1)])
    ok = s["valid"][:n]
    A, y = X[ok], r[ok]
    G = np.linalg.solve(A.T @ A + ridge * len(y) * np.eye(lags + 1), A.T @ y)
    cum = np.cumsum(G)
    return {"G_bps_per_unit_flow": G.tolist(), "cumulative": cum.tolist(),
            "half_life_s": float(np.argmax(cum >= 0.5 * cum[-1])) if cum[-1] > 0 else None}


def cross_impact(target: dict, source: dict, horizon: int = 10) -> dict:
    """Signed response of ``target`` mid to ``source`` bursts in the same seconds (M74, descriptive)."""
    b = bursts(source)
    if not len(b["sec"]):
        return {}
    ts = seconds(target)
    mid = ts["mid"]
    idx = b["sec"][b["sec"] + horizon < len(mid)]
    sign = np.sign(seconds(source)["v"][idx])
    resp = sign * (mid[idx + horizon] - mid[idx]) / mid[idx] * 1e4
    return {"mean_bps": float(resp.mean()), "se_bps": float(resp.std(ddof=1) / math.sqrt(len(resp))),
            "bursts": int(len(resp)), "label": "descriptive cross impact; common information is not separated"}
