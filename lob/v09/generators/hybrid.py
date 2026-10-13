"""Hybrid historical twin (F3, M49): historical backbone + strategy intervention + local stochastic response.

Response realism (H1, H12). For every aggressive-flow burst of a day (top 1 % |signed volume| seconds, as in the
impact study) the observed response has six targets: the signed mid move (bps) at +1 s, +5 s, +30 s and the hit-side
touch-depth recovery ratio (depth at +h / depth before) at the same horizons.

* ``HybridResponse`` predicts the six targets from the pre-burst state (relative size, opposite depth, spread,
  imbalance, recent volatility and flow) with the registered boosted regression, fitted on training-day bursts;
* a generative model predicts them from its simulated probe-response table (``classical.probe_responses``) by
  log-linear interpolation in relative size (|v| / median touch depth of the training days).

``response_error`` is the mean over bursts and targets of the squared error standardized by the training-day
standard deviation of each target.

Hybrid execution twin (``HybridReplay``): the bounded replay engine with (a) a self-impact price shift - every own
market child adds I(q, state) [p + (1 - p) exp(-(t - t_c) / tau)] ticks to all later prices, with I from the fitted
10 s response model and the persistent fraction p from the training decomposition, tau = 10 s - and (b) stochastic
resilience: consumed liquidity refills with an exponential half-life drawn per episode from LogNormal(log 10 s, 0.5)
(common random numbers across policies). Results are HYBRID (E3), never historical fills.
"""
from __future__ import annotations

import math

import numpy as np

from ..impact import models as IM
from ..replay import engine as E

TARGETS = ("mid_1", "mid_5", "mid_30", "rec_1", "rec_5", "rec_30")
HORIZONS = (1, 5, 30)
TAU_S = 10.0


def burst_targets(cache: dict) -> dict:
    """Burst features (impact-model layout) and the six observed response targets."""
    b = IM.bursts(cache)
    if not len(b["sec"]):
        return {"X": np.zeros((0, 7)), "Y": np.zeros((0, 6)), "rel": np.zeros(0)}
    s = IM.seconds(cache)
    sec, mid, depth = b["sec"], s["mid"], s["depth"]
    sign = np.sign(s["v"][sec])
    ys = []
    for h in HORIZONS:
        ok = sec + h < len(mid)
        ys.append(np.where(ok, sign * (mid[np.minimum(sec + h, len(mid) - 1)] - mid[sec]) / mid[sec] * 1e4, np.nan))
    hit = np.where(sign > 0, 1, 0)
    pre = depth[sec, hit]
    for h in HORIZONS:
        j = np.minimum(sec + h, len(depth) - 1)
        ys.append(depth[j, hit] / np.maximum(pre, 1.0))
    Y = np.column_stack(ys)
    touch = float(np.median(depth[depth > 0])) if (depth > 0).any() else 1.0
    rel = np.abs(s["v"][sec]) / max(touch, 1.0)
    ok = np.isfinite(Y).all(1)
    return {"X": b["X"][ok], "Y": Y[ok], "rel": rel[ok], "touch": touch}


class HybridResponse:
    name = "hybrid_response"

    def fit(self, X, Y):
        self.models = [IM.StateImpact(trees=100).fit(X, Y[:, k]) for k in range(Y.shape[1])]
        return self

    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.models])


def table_prediction(table: dict, rel: np.ndarray) -> np.ndarray:
    """Generative prediction of the six targets by log-linear interpolation of a probe-response table."""
    sizes = np.asarray(sorted(float(k) for k in table))
    out = np.zeros((len(rel), 6))
    logs = np.log(sizes)
    x = np.log(np.clip(rel, sizes[0], sizes[-1]))
    for k, (kind, h) in enumerate([("mid_bps", h) for h in HORIZONS] + [("depth_recovery", h) for h in HORIZONS]):
        vals = np.asarray([table[str(s) if str(s) in table else f"{s:g}"][kind][str(h)] or 0.0 for s in sizes])
        out[:, k] = np.interp(x, logs, vals)
    return out


def response_error(pred: np.ndarray, Y: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Per-burst mean standardized squared error over the six targets."""
    return (((pred - Y) / np.maximum(sd, 1e-9)) ** 2).mean(1)


# ----------------------------------------------------------------------------- hybrid execution twin


class ShiftedBook(E.DayBook):
    """A DayBook whose prices are displaced by the strategy's cumulative self-impact (ticks, at time t)."""

    def __init__(self, cache: dict, shift) -> None:
        super().__init__(cache)
        self.shift = shift

    def touch(self, t):
        b, a, ok = super().touch(t)
        s = int(round(self.shift(t)))
        return b + s, a + s, ok

    def mid(self, t):
        return super().mid(t) + self.shift(t)

    def ladder(self, t, side_taken):
        p, q = super().ladder(t, side_taken)
        return p + int(round(self.shift(t))), q


class HybridReplay(E.Replay):
    def __init__(self, cache: dict, policy, *, impact_model, persistent: float, seed: int, **kw) -> None:
        self.children: list[tuple[float, float]] = []      # (time, impact ticks signed)
        self.impact_model, self.persistent = impact_model, float(np.clip(persistent, 0.0, 1.0))
        rng = np.random.default_rng(np.random.SeedSequence([9, 49, int(seed)]))
        self.half_life = float(np.exp(rng.normal(math.log(10.0), 0.5)))
        book = ShiftedBook(cache, self._shift)
        super().__init__(book, policy, **kw)

    def _shift(self, t: float) -> float:
        s = 0.0
        for tc, imp in self.children:
            if t >= tc:
                s += imp * (self.persistent + (1 - self.persistent) * math.exp(-(t - tc) / TAU_S))
        return s

    def _remaining_consumed(self, price: int, t: float) -> float:
        return sum(q * 0.5 ** ((t - at) / self.half_life) for at, q in self.consumed.get(price, ()))

    def _advance_passive(self, t_from: float, t_to: float) -> None:
        """Historical prints are compared with order prices in un-shifted (historical) coordinates."""
        s = int(round(self._shift(t_from)))
        for o in self.orders:
            o.price -= s
        try:
            super()._advance_passive(t_from, t_to)
        finally:
            for o in self.orders:
                o.price += s

    def _features(self, qty: float, t: float) -> np.ndarray:
        """The registered impact-model features of an own child order (same layout as ``impact.models.bursts``)."""
        c = self.book.c
        sec = int(t)
        lo = max(0, sec - 600)
        q, s, tt = c["trade_q"], c["trade_s"], c["trade_t"]
        a, z = np.searchsorted(tt, [lo, sec])
        per_sec = np.bincount(np.floor(tt[a:z]).astype(int) - lo, weights=(q[a:z] * s[a:z]).astype(float),
                              minlength=sec - lo + 1)
        V = max(float(np.mean(np.abs(per_sec))), 1e-9)
        f = np.searchsorted(tt, sec - 30)
        flow = float((q[f:z] * s[f:z]).sum()) * self.side / V
        b = self.book.bin_of(t)
        mid = c["mid"][max(0, b - 600):b + 1:10].astype(float)
        vol = float(np.std(np.diff(mid)) / max(mid[-1], 1e-9) * 1e4) if len(mid) > 3 else 0.0
        m10 = c["mid"][::100].astype(float)
        sigma = float(np.std(np.diff(m10) / np.maximum(m10[:-1], 1e-9) * 1e4)) if len(m10) > 3 else 0.0
        opp = float(c["depth"][b, 1 if self.side > 0 else 0, 0])
        return np.asarray([[math.log(max(qty, 1e-9) / V), math.log1p(opp), float(c["state"][b, 0]),
                            self.side * float(c["state"][b, 1]), vol, flow, sigma * math.sqrt(qty / V)]])

    def _market(self, qty: float, t: float) -> None:
        before = len(self.res.fills)
        super()._market(qty, t)
        if qty <= 0 or len(self.res.fills) == before or self.impact_model is None:
            return
        bps = float(self.impact_model.predict(self._features(qty, t))[0])
        ticks = max(bps, 0.0) * self.res.arrival_mid / 1e4
        self.children.append((t, self.side * ticks))
