"""Conformal uncertainty for execution endpoints (M20).

* ``split`` — split conformal interval ``y_hat +/- q`` with ``q`` the
  ``ceil((1 - alpha)(n + 1)) / n`` quantile of calibration absolute residuals;
* ``rolling`` — the same with only the most recent ``window`` residuals (for
  drifting sequences; validity then rests on local exchangeability);
* ``mondrian`` — per-group (e.g. regime) quantiles; groups with fewer than
  ``min_group`` calibration points fall back to the pooled quantile (recorded);
* ``classification_sets`` — conformal prediction sets for a binary endpoint
  (completion) from predicted probabilities.

``coverage`` records empirical coverage with a Wilson interval. Every result
carries ``validity``: distribution-free coverage holds only under
exchangeability of calibration and test points; a residual-trend diagnostic is
reported and the result is marked ASSUMPTION_DEPENDENT when the trend is
material.
"""
from __future__ import annotations

import math

import numpy as np

from .statistics import spearman, wilson


def quantile(residuals: np.ndarray, alpha: float) -> float:
    r = np.sort(np.abs(np.asarray(residuals, float)))
    n = len(r)
    if n == 0:
        return math.inf
    k = int(math.ceil((1 - alpha) * (n + 1)))
    return float(r[k - 1]) if k <= n else math.inf


def split(cal_true, cal_pred, test_pred, alpha: float) -> dict:
    q = quantile(np.asarray(cal_true) - np.asarray(cal_pred), alpha)
    tp = np.asarray(test_pred, float)
    return {"lower": (tp - q).tolist(), "upper": (tp + q).tolist(), "q": q, "n_cal": len(cal_true)}


def rolling(true, pred, alpha: float, window: int) -> dict:
    """Sequential: each point's interval uses only the previous ``window`` residuals."""
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    lower, upper, covered = [], [], []
    for i in range(len(true)):
        past = (true - pred)[max(0, i - window):i]
        q = quantile(past, alpha) if len(past) >= max(5, int(1 / alpha)) else math.inf
        lower.append(pred[i] - q)
        upper.append(pred[i] + q)
        covered.append(bool(pred[i] - q <= true[i] <= pred[i] + q) if math.isfinite(q) else None)
    return {"lower": lower, "upper": upper, "covered": covered, "window": window}


def mondrian(cal_true, cal_pred, cal_groups, test_pred, test_groups, alpha: float, min_group: int = 20) -> dict:
    res = np.asarray(cal_true, float) - np.asarray(cal_pred, float)
    groups = np.asarray(cal_groups)
    pooled = quantile(res, alpha)
    qs, fallback = {}, []
    for g in sorted(set(np.asarray(test_groups).tolist()) | set(groups.tolist())):
        mask = groups == g
        if mask.sum() >= min_group:
            qs[g] = quantile(res[mask], alpha)
        else:
            qs[g] = pooled
            fallback.append(g)
    tp = np.asarray(test_pred, float)
    q = np.asarray([qs[g] for g in np.asarray(test_groups).tolist()])
    return {"lower": (tp - q).tolist(), "upper": (tp + q).tolist(), "group_q": {str(k): v for k, v in qs.items()},
            "fallback_groups": [str(g) for g in fallback]}


def classification_sets(cal_labels, cal_prob, test_prob, alpha: float) -> dict:
    """Binary conformal sets with nonconformity 1 - p(true label)."""
    cal_labels, cal_prob = np.asarray(cal_labels, int), np.asarray(cal_prob, float)
    score = 1 - np.where(cal_labels == 1, cal_prob, 1 - cal_prob)
    q = quantile(score, alpha)
    sets = []
    for p in np.asarray(test_prob, float):
        s = [label for label, prob in ((0, 1 - p), (1, p)) if 1 - prob <= q]
        sets.append(s)
    return {"sets": sets, "q": q}


def coverage(true, lower, upper, *, nominal: float, order=None) -> dict:
    true, lower, upper = (np.asarray(x, float) for x in (true, lower, upper))
    ok = np.isfinite(lower) & np.isfinite(upper)
    hit = (true >= lower) & (true <= upper)
    k, n = int(hit[ok].sum()), int(ok.sum())
    lo, hi = wilson(k, n)
    out = {"coverage": k / n if n else None, "ci": [lo, hi], "n": n, "nominal": nominal,
           "mean_width": float(np.mean((upper - lower)[ok])) if n else None, "infinite_intervals": int((~ok).sum())}
    if order is not None and n > 10:
        trend = spearman(np.asarray(order, float)[ok], np.abs(true - (lower + upper) / 2)[ok])
        out["residual_trend_spearman"] = trend
        out["validity"] = ("ASSUMPTION_DEPENDENT: residuals trend with time (|rho| > 0.2); exchangeability doubtful"
                           if abs(trend) > 0.2 else "exchangeability not contradicted by the residual-trend check")
    else:
        out["validity"] = "valid under exchangeability of calibration and test points (assumed)"
    return out
