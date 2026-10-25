"""Liquidity events (M53), multiscale dependence and long memory (M57), intraday seasonality (M58) and the
microstructure fingerprint and similarity engine (M59).

* ``liquidity_events``: liquidity droughts (touch depth below 10% of its rolling 10 min median for >= 2 s),
  liquidity shocks (a single-second touch depth drop > 80%) and cancel storms (cancel rate above the day's 99.5th
  percentile for >= 2 s); each event is returned with start, duration and magnitude.
* ``hurst_dfa``: detrended fluctuation analysis exponent of a series (0.5 = no memory; > 0.5 persistent), with the
  fitted scales; ``acf_decay``: ACF at log-spaced lags and the fitted power-law decay exponent.
* ``seasonality``: per-hour-of-day mean and SD of a feature across days, and ``deseasonalize`` (z-score by hour),
  the calendar control used when comparing days.
* ``fingerprint``: a fixed-length vector of standardized summary statistics of a day (spread, depth, flow, volatility,
  memory, event rates); ``similarity`` returns cosine similarities and nearest days, used to match test days to
  training regimes and to describe how far a holdout is from the calibration data.
"""
from __future__ import annotations

import numpy as np


def _runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(int))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(a), int(b - a)) for a, b in zip(starts, ends) if b - a >= min_len]


def liquidity_events(features: dict[str, np.ndarray], *, interval_s: float = 1.0) -> dict:
    touch = np.exp(features["log_touch_depth"])
    n = len(touch)
    w = int(600 / interval_s)
    med = np.array([np.median(touch[max(0, i - w):i + 1]) for i in range(n)])
    drought = _runs(touch < 0.1 * med, int(2 / interval_s))
    drop = np.r_[0.0, 1.0 - touch[1:] / np.maximum(touch[:-1], 1e-9)]
    shocks = [(int(i), 1) for i in np.flatnonzero(drop > 0.8)]
    cr = features["cancel_rate"]
    storms = _runs(cr > np.percentile(cr, 99.5), int(2 / interval_s)) if len(cr) else []
    return {"droughts": [{"start_s": a * interval_s, "duration_s": d * interval_s,
                          "min_ratio": float((touch[a:a + d] / np.maximum(med[a:a + d], 1e-9)).min())}
                         for a, d in drought],
            "shocks": [{"start_s": a * interval_s, "drop": float(drop[a])} for a, _ in shocks],
            "cancel_storms": [{"start_s": a * interval_s, "duration_s": d * interval_s,
                               "peak_rate": float(cr[a:a + d].max())} for a, d in storms],
            "rates_per_hour": {"droughts": len(drought) * 3600 / (n * interval_s),
                               "shocks": len(shocks) * 3600 / (n * interval_s),
                               "cancel_storms": len(storms) * 3600 / (n * interval_s)}}


def hurst_dfa(x: np.ndarray, scales=None) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    y = np.cumsum(x - x.mean())
    scales = scales or [int(s) for s in np.unique(np.logspace(1, np.log10(max(len(x) // 8, 20)), 12).astype(int))]
    F = []
    for s in scales:
        n = len(y) // s
        if n < 4:
            continue
        seg = y[: n * s].reshape(n, s)
        t = np.arange(s)
        A = np.column_stack([np.ones(s), t])
        coef, *_ = np.linalg.lstsq(A, seg.T, rcond=None)
        res = seg.T - A @ coef
        F.append((s, float(np.sqrt((res ** 2).mean()))))
    if len(F) < 3:
        return {"hurst": None, "scales": []}
    s, f = np.log([a for a, _ in F]), np.log([b for _, b in F])
    h = float(np.polyfit(s, f, 1)[0])
    return {"hurst": h, "scales": [a for a, _ in F], "fluctuation": [b for _, b in F]}


def acf_decay(x: np.ndarray, lags=(1, 2, 4, 8, 16, 32, 64, 128)) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)] - np.nanmean(x)
    v = (x * x).mean()
    acf = {int(k): float((x[:-k] * x[k:]).mean() / v) if len(x) > k + 2 and v > 0 else np.nan for k in lags}
    pos = [(k, a) for k, a in acf.items() if np.isfinite(a) and a > 0]
    exp = float(-np.polyfit(np.log([k for k, _ in pos]), np.log([a for _, a in pos]), 1)[0]) if len(pos) >= 3 \
        else None
    return {"acf": acf, "power_law_decay": exp}


def seasonality(days: list[dict[str, np.ndarray]], feature: str, *, interval_s: float = 1.0) -> dict:
    per_hour = int(3600 / interval_s)
    rows = []
    for f in days:
        x = np.asarray(f[feature], float)
        n = len(x) // per_hour
        rows.append([np.nanmean(x[h * per_hour:(h + 1) * per_hour]) for h in range(n)])
    H = min(len(r) for r in rows)
    M = np.asarray([r[:H] for r in rows])
    return {"feature": feature, "hour_mean": M.mean(0).tolist(), "hour_sd": M.std(0, ddof=1).tolist()
            if len(M) > 1 else [0.0] * H, "days": len(days)}


def deseasonalize(x: np.ndarray, profile: dict, *, interval_s: float = 1.0) -> np.ndarray:
    per_hour = int(3600 / interval_s)
    mu = np.repeat(profile["hour_mean"], per_hour)[: len(x)]
    sd = np.repeat(np.maximum(profile["hour_sd"], 1e-9), per_hour)[: len(x)]
    return (np.asarray(x, float)[: len(mu)] - mu) / sd


FINGERPRINT = ("spread_mean", "spread_p_gt1", "log_depth_median", "trade_rate", "abs_ofi_mean", "rv_mean",
               "imbalance_sd", "hurst_returns", "acf1_trades", "drought_rate", "storm_rate")


def fingerprint(features: dict[str, np.ndarray]) -> np.ndarray:
    v = np.asarray(features["valid"], bool)
    ev = liquidity_events(features)
    tr = features["trade_count"][v]
    vals = {
        "spread_mean": features["spread"][v].mean(), "spread_p_gt1": (features["spread"][v] > 1).mean(),
        "log_depth_median": np.median(features["log_touch_depth"][v]), "trade_rate": tr.mean(),
        "abs_ofi_mean": np.abs(features["ofi_l1"][v]).mean(), "rv_mean": np.nanmean(features["rv_60s"][v]),
        "imbalance_sd": features["imbalance_l5"][v].std(),
        "hurst_returns": hurst_dfa(features["return_bps"][v])["hurst"] or 0.5,
        "acf1_trades": acf_decay(tr, lags=(1,))["acf"][1], "drought_rate": ev["rates_per_hour"]["droughts"],
        "storm_rate": ev["rates_per_hour"]["cancel_storms"]}
    return np.asarray([float(vals[k]) for k in FINGERPRINT])


def similarity(reference: dict[str, np.ndarray], query: np.ndarray, *, k: int = 3) -> dict:
    """Cosine similarity of standardized fingerprints; ``reference`` maps day ids to fingerprints."""
    ids = list(reference)
    R = np.vstack([reference[i] for i in ids])
    mu, sd = R.mean(0), R.std(0) + 1e-9
    Rz, qz = (R - mu) / sd, (np.asarray(query) - mu) / sd
    cos = Rz @ qz / (np.linalg.norm(Rz, axis=1) * np.linalg.norm(qz) + 1e-12)
    order = np.argsort(-cos)
    dist = float(np.linalg.norm(qz) / np.sqrt(len(qz)))
    return {"nearest": [(ids[i], float(cos[i])) for i in order[:k]], "standardized_distance": dist}


__all__ = ["liquidity_events", "hurst_dfa", "acf_decay", "seasonality", "deseasonalize", "fingerprint", "similarity",
           "FINGERPRINT"]
