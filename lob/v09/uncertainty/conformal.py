"""Sequential conformal prediction under drift (§G of the brief; H17).

Target (registered): realized volatility of 100 ms mid changes over the next 60 s (bps of mid, floored at 0.01 bps so
quiet periods are kept, not dropped), sampled every 10 s inside the registered windows. Base predictor: least squares on log realized volatility over the previous 10 s, 60 s
and 300 s (HAR-type), fitted on training days. Conformity score: |log y - log y_hat|. Methods for a nominal 90 %
interval exp(log y_hat -/+ q):

* ``split`` - q = the (1 - alpha)(1 + 1/n) quantile of training-day scores (exchangeable reference);
* ``weighted`` - time-decayed weights (half-life 1 h) over the scores seen so far on the evaluation day plus the
  training scores (non-exchangeable weighted conformal);
* ``rolling`` - quantile of the last 500 scores of the evaluation day (training scores until 50 exist);
* ``regime`` - split quantile conditional on the training-day volatility tercile of the current state;
* ``aci`` - adaptive conformal inference (Gibbs & Candes 2021): alpha_t+1 = alpha_t + gamma (alpha - err_t),
  gamma = 0.005, quantile of all scores seen so far.

Metrics: marginal coverage, mean width (log scale), conditional coverage by volatility tercile, coverage after the
largest registered volatility shock of the day (post-drift) and recovery time (steps until the rolling 50-step coverage
returns within 0.05 of nominal).
"""
from __future__ import annotations

import numpy as np

from ..data.daycache import BIN_MS, HORIZON_S, WARMUP_S, WINDOW_STARTS_S

ALPHA = 0.10
FLOOR = 0.01          # bps: realized volatility below the floor (no mid change) is floored, never dropped
GAMMA = 0.005
METHODS = ("split", "weighted", "rolling", "regime", "aci")


def samples(cache: dict) -> dict:
    """(features [n, 3], target [n], time [n]) every 10 s inside the registered windows (with 300 s look-back)."""
    mid = cache["mid"].astype(float)
    step = np.abs(np.diff(mid)) / np.maximum(mid[:-1], 1e-9) * 1e4
    feats, target, times = [], [], []
    for s in WINDOW_STARTS_S:
        for t in range(int(s - WARMUP_S), int(s + HORIZON_S) - 60, 10):
            b = int(t * 1000 // BIN_MS)
            if b < 3000 or not cache["valid"][b - 3000:b + 600].all():
                continue
            past = [max(float(np.sqrt(np.mean(step[b - k:b] ** 2))), FLOOR) for k in (100, 600, 3000)]
            fut = max(float(np.sqrt(np.mean(step[b:b + 600] ** 2))), FLOOR)
            feats.append(np.log(past))
            target.append(np.log(fut))
            times.append(t)
    return {"X": np.asarray(feats).reshape(-1, 3), "y": np.asarray(target), "t": np.asarray(times, float)}


def fit_base(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ coef


def _q(scores: np.ndarray, level: float, weights=None) -> float:
    if not len(scores):
        return np.inf
    level = min(max(level, 0.0), 1.0)
    if weights is None:
        k = int(np.ceil(level * (len(scores) + 1))) - 1
        return float(np.sort(scores)[min(max(k, 0), len(scores) - 1)]) if k < len(scores) else np.inf
    order = np.argsort(scores)
    cw = np.cumsum(weights[order]) / weights.sum()
    j = int(np.searchsorted(cw, level))
    return float(scores[order][min(j, len(scores) - 1)])


def run_day(coef, train_scores, train_vol, test: dict) -> dict:
    X, y, t = test["X"], test["y"], test["t"]
    yhat = predict(coef, X)
    score = np.abs(y - yhat)
    terciles = np.quantile(train_vol, [1 / 3, 2 / 3])
    tr_reg = np.digitize(train_vol, terciles)
    reg = np.digitize(X[:, 1], terciles)
    out = {}
    for method in METHODS:
        cover, width = np.zeros(len(y), bool), np.zeros(len(y))
        a_t = ALPHA
        for i in range(len(y)):
            seen = score[:i]
            if method == "split":
                q = _q(train_scores, 1 - ALPHA)
            elif method == "regime":
                q = _q(train_scores[tr_reg == reg[i]], 1 - ALPHA)
            elif method == "rolling":
                q = _q(seen[-500:], 1 - ALPHA) if i >= 50 else _q(train_scores, 1 - ALPHA)
            elif method == "weighted":
                pool = np.r_[train_scores, seen]
                w = np.r_[np.full(len(train_scores), 0.5 ** ((t[i] - t[0] + 3600.0) / 3600.0)),
                          0.5 ** ((t[i] - t[:i]) / 3600.0)]
                q = _q(pool, 1 - ALPHA, w)
            else:
                pool = np.r_[train_scores, seen]
                q = _q(pool, 1 - a_t)
            cover[i] = score[i] <= q
            width[i] = 2 * q if np.isfinite(q) else np.nan
            if method == "aci":
                a_t = a_t + GAMMA * (ALPHA - (0.0 if cover[i] else 1.0))
        shock = int(np.argmax(np.abs(np.diff(y)))) + 1 if len(y) > 2 else 0
        post = cover[shock:shock + 60]
        roll = np.convolve(cover.astype(float), np.ones(50) / 50, mode="valid") if len(y) >= 50 else np.zeros(0)
        after = np.flatnonzero(np.abs(roll[shock:] - (1 - ALPHA)) <= 0.05) if len(roll) > shock else []
        out[method] = {"coverage": float(cover.mean()), "width": float(np.nanmean(width)),
                       "coverage_by_tercile": {int(k): float(cover[reg == k].mean()) for k in range(3)
                                               if (reg == k).any()},
                       "post_shock_coverage": float(post.mean()) if len(post) else None,
                       "recovery_steps": int(after[0]) if len(after) else None, "n": int(len(y)),
                       "odd": float(cover[(t // 3600).astype(int) % 2 == 1].mean()) if len(y) else None,
                       "even": float(cover[(t // 3600).astype(int) % 2 == 0].mean()) if len(y) else None}
    return out
