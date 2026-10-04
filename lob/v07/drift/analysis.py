"""Nonstationarity, change points and calibration half-life (workstreams 22, 23).

* ``cusum`` — two-sided CUSUM on standardized block statistics with a registered threshold; change
  points are reported with their block index (synthetic known change points are recovered in tests).
* ``drift_chronology`` — per dataset, the real-vs-real distance of its history to the development day
  (observable drift), support drift and model objectives (performance decay).
* ``half_life`` — calibration-quality decay. For one model/metric, the relative degradation
  d(t) = (e(t) - e(0)) / e(0) is fitted by d(t) = d_inf (1 - 2^(-t / tau)) over a tau grid; tau is
  the half-life (months) of the approach to the long-run degradation. A flat sequence returns
  ``stable`` (no decay detected); nonmonotone noise returns ``INCONCLUSIVE``. Block-bootstrap
  intervals come from resampled per-period errors. Exploratory: not a registered hypothesis.
"""
from __future__ import annotations

import numpy as np

TAU_GRID = np.round(np.exp(np.linspace(np.log(0.25), np.log(48.0), 120)), 6)


def _baseline(x: np.ndarray) -> tuple[float, float]:
    """Robust location and scale (median, 1.4826 x MAD) of a baseline window."""
    median = float(np.median(x))
    return median, max(1.4826 * float(np.median(np.abs(x - median))), 1e-9)


def cusum(values: np.ndarray, *, threshold: float = 8.0, drift: float = 0.5, burn: int = 20) -> dict:
    """Two-sided CUSUM with a robust baseline re-estimated on ``burn`` blocks after each detection."""
    x = np.asarray(values, float)
    mean, sd = _baseline(x[:burn])
    hi = lo = 0.0
    points, i = [], 0
    while i < len(x):
        v = (x[i] - mean) / sd
        hi, lo = max(0.0, hi + v - drift), min(0.0, lo + v + drift)
        if hi > threshold or lo < -threshold:
            points.append(i)
            hi = lo = 0.0
            if i + burn >= len(x):
                break
            mean, sd = _baseline(x[i:i + burn])
            i += burn
            continue
        i += 1
    return {"change_points": points, "threshold": threshold, "drift": drift, "burn_in_blocks": burn,
            "baseline": "median and 1.4826 x MAD of the burn-in window"}


def change_points(values: np.ndarray, *, penalty_factor: float = 3.0, min_size: int = 5) -> dict:
    """Binary segmentation for mean shifts (primary retrospective diagnostic).

    Noise variance from first differences (1.4826 x MAD / sqrt 2, robust to the shifts themselves); a split is
    accepted when it reduces the sum of squares by more than ``penalty_factor * sigma^2 * log(n)``.
    """
    x = np.asarray(values, float)
    n = len(x)
    sigma = max(1.4826 * float(np.median(np.abs(np.diff(x) - np.median(np.diff(x))))) / np.sqrt(2), 1e-9)
    penalty = penalty_factor * sigma ** 2 * np.log(max(n, 2))
    found = []

    def split(lo: int, hi: int) -> None:
        segment = x[lo:hi]
        if len(segment) < 2 * min_size:
            return
        total = float(np.sum((segment - segment.mean()) ** 2))
        best = None
        for k in range(min_size, len(segment) - min_size + 1):
            left, right = segment[:k], segment[k:]
            sse = float(np.sum((left - left.mean()) ** 2) + np.sum((right - right.mean()) ** 2))
            if best is None or sse < best[0]:
                best = (sse, k)
        if best is not None and total - best[0] > penalty:
            found.append(lo + best[1])
            split(lo, lo + best[1])
            split(lo + best[1], hi)
    split(0, n)
    return {"change_points": sorted(found), "sigma": sigma, "penalty": penalty, "min_size": min_size,
            "method": "binary segmentation on the mean with a BIC-scale penalty"}


def fit_decay(t: np.ndarray, d: np.ndarray) -> dict:
    t, d = np.asarray(t, float), np.asarray(d, float)
    best = None
    for tau in TAU_GRID:
        basis = 1 - 2.0 ** (-t / tau)
        if not np.any(basis):
            continue
        d_inf = float(np.sum(basis * d) / np.sum(basis * basis))
        sse = float(np.sum((d - d_inf * basis) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(tau), d_inf)
    sse, tau, d_inf = best
    return {"tau": tau, "d_inf": d_inf, "sse": sse}


def half_life(months: list[float], errors: list[float], *, per_period_draws: list[np.ndarray] | None = None,
              samples: int = 1000, seed: int = 0, stable_tolerance: float = 0.02) -> dict:
    t = np.asarray(months, float)
    e = np.asarray(errors, float)
    if len(t) < 3 or not np.isfinite(e).all() or e[0] <= 0:
        return {"status": "NOT_AVAILABLE", "reason": "fewer than three dated periods or invalid errors"}
    d = (e - e[0]) / e[0]
    if np.max(np.abs(d[1:])) < stable_tolerance:
        return {"status": "stable", "relative_degradation": d.tolist(), "half_life_months": None}
    fit = fit_decay(t, d)
    out = {"relative_degradation": d.tolist(), "half_life_months": fit["tau"], "long_run_degradation": fit["d_inf"],
           "fit_sse": fit["sse"]}
    if fit["d_inf"] <= 0:
        out["status"] = "INCONCLUSIVE"
        out["reason"] = "errors do not increase after calibration"
        return out
    if per_period_draws is not None:
        rng = np.random.default_rng(seed)
        taus = []
        for _ in range(samples):
            draw = np.asarray([p[rng.integers(len(p))] for p in per_period_draws], float)
            if draw[0] <= 0:
                continue
            f = fit_decay(t, (draw - draw[0]) / draw[0])
            if f["d_inf"] > 0:
                taus.append(f["tau"])
        if taus:
            out["half_life_ci"] = [float(np.quantile(taus, 0.025)), float(np.quantile(taus, 0.975))]
            out["bootstrap_positive_fraction"] = len(taus) / samples
    out["status"] = "EXPLORATORY"
    return out
