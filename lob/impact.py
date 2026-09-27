"""Empirical conditional price impact around aggressive trade events (M4).

For each aggressive event on a ``lob.observables_v2.Tape`` the response is
measured from the last sample *at or before* the event to later samples, so no
post-event information enters the conditioning variables. Responses are
associations with signed flow; they mix impact, information and concurrent
flow, and are not causal effects. Almgren-Chriss quantities are empirical
proxies, not structurally identified parameters.
"""
from __future__ import annotations

import math

import numpy as np

from .observables_v2 import SAMPLE_DT, Tape

HORIZONS_S = (0.1, 1.0, 10.0, 60.0)
BUCKET_DIMENSIONS = ("size", "participation", "spread", "volatility", "depth", "imbalance_alignment", "activity",
                     "direction")


def event_table(tape: Tape, horizons: tuple[float, ...] = HORIZONS_S) -> dict[str, np.ndarray]:
    """One row per aggressive event with pre-event state and signed responses (NaN if unavailable)."""
    valid = tape.valid
    mid = tape.mid
    n = len(tape.t)
    k = np.searchsorted(tape.t, tape.trade_t, side="right") - 1
    keep = (k >= 0) & (k < n)
    k = np.where(keep, k, 0)
    sign = tape.trade_s
    spread = (tape.ap[:, 0] - tape.bp[:, 0]) / mid * 1e4
    d5b, d5a = tape.bq.sum(axis=1), tape.aq.sum(axis=1)
    imbalance = (d5b - d5a) / np.where(d5b + d5a > 0, d5b + d5a, np.nan)
    opposite_l1 = np.where(sign > 0, tape.aq[k, 0], tape.bq[k, 0])
    opposite_5 = np.where(sign > 0, d5a[k], d5b[k])
    one = int(round(1 / SAMPLE_DT))
    # trailing 60 s realized volatility and activity at the pre-event sample
    returns_1s = np.full(n, np.nan)
    returns_1s[one:] = np.log(mid[one:] / mid[:-one]) * 1e4
    returns_1s[~valid] = np.nan
    returns_1s[one:][~valid[:-one]] = np.nan
    sq = np.nan_to_num(returns_1s ** 2)
    csum = np.r_[0.0, np.cumsum(sq[::one])]
    csum_count = np.r_[0.0, np.cumsum(np.isfinite(returns_1s[::one]))]
    second = k // one
    lo = np.maximum(0, second - 60)
    rv = np.sqrt((csum[second] - csum[lo]) / np.maximum(1, csum_count[second] - csum_count[lo]))
    trade_second = np.floor(tape.trade_t - tape.t[0]).astype(int) if len(tape.trade_t) else np.array([], int)
    counts = np.bincount(np.clip(trade_second, 0, max(1, n // one)), minlength=n // one + 2).astype(float)
    ccount = np.r_[0.0, np.cumsum(counts)]
    activity = ccount[np.clip(second, 0, len(ccount) - 1)] - ccount[np.clip(second - 60, 0, len(ccount) - 1)]
    table = {"time": tape.trade_t, "sign": sign, "size": tape.trade_q, "valid_pre": keep & valid[k],
             "spread_pre": spread[k], "depth5_pre": (d5b + d5a)[k], "opposite_l1_pre": opposite_l1,
             "participation": tape.trade_q / np.where(opposite_5 > 0, opposite_5, np.nan),
             "imbalance_pre": imbalance[k], "imbalance_alignment": sign * imbalance[k], "rv60_pre": rv,
             "activity60_pre": activity}
    for h in horizons:
        j = k + int(round(h / SAMPLE_DT))
        ok = keep & (j < n)
        j = np.where(ok, j, 0)
        ok &= valid[k] & valid[j]
        table[f"mid_bps_{h:g}s"] = np.where(ok, sign * np.log(mid[j] / mid[k]) * 1e4, np.nan)
        table[f"spread_change_bps_{h:g}s"] = np.where(ok, spread[j] - spread[k], np.nan)
        table[f"imbalance_change_{h:g}s"] = np.where(ok, sign * (imbalance[j] - imbalance[k]), np.nan)
        opposite_after = np.where(sign > 0, d5a[j], d5b[j])
        table[f"opposite_depth_ratio_{h:g}s"] = np.where(ok & (opposite_5 > 0), opposite_after / opposite_5, np.nan)
    # absolute mid movement in the 10 s after versus before (volatility response)
    ten = 10 * one
    after = np.where(keep & (k + ten < n), np.abs(np.log(mid[np.minimum(k + ten, n - 1)] / mid[k])) * 1e4, np.nan)
    before = np.where(keep & (k - ten >= 0), np.abs(np.log(mid[k] / mid[np.maximum(k - ten, 0)])) * 1e4, np.nan)
    table["abs_move_after_10s_bps"] = after
    table["abs_move_before_10s_bps"] = before
    return table


def thresholds(development: dict[str, np.ndarray]) -> dict:
    """Bucket edges frozen from the development period only."""
    ok = development["valid_pre"]
    def q(name, probs):
        values = development[name][ok]
        values = values[np.isfinite(values)]
        return [float(v) for v in np.quantile(values, probs)]
    return {"size": q("size", [1 / 3, 2 / 3]), "participation": q("participation", [1 / 3, 2 / 3]),
            "spread": q("spread_pre", [0.5]), "volatility": q("rv60_pre", [0.5]), "depth": q("depth5_pre", [0.5]),
            "imbalance_alignment": [0.0], "activity": q("activity60_pre", [0.5]), "direction": [0.0]}


_SOURCE = {"size": "size", "participation": "participation", "spread": "spread_pre", "volatility": "rv60_pre",
           "depth": "depth5_pre", "imbalance_alignment": "imbalance_alignment", "activity": "activity60_pre",
           "direction": "sign"}


def buckets(table: dict[str, np.ndarray], edges: dict, dimension: str) -> np.ndarray:
    values = table[_SOURCE[dimension]]
    labels = np.digitize(values, edges[dimension], right=False).astype(float)
    labels[~np.isfinite(values)] = np.nan
    return labels


def ac_proxies(table: dict[str, np.ndarray], one_second_returns: np.ndarray) -> dict:
    """Empirical Almgren-Chriss-style proxies; not a structural identification."""
    ok = table["valid_pre"]
    def slope(x_name, y_name):
        x, y = table[x_name][ok], table[y_name][ok]
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.sum() < 30 or np.var(x[keep]) == 0:
            return None
        return float(np.polyfit(x[keep], y[keep], 1)[0])
    returns = one_second_returns[np.isfinite(one_second_returns)]
    return {"sigma_bps_per_sqrt_s": float(returns.std()) if len(returns) else None,
            "temporary_impact_proxy_bps_per_participation": slope("participation", "mid_bps_0.1s"),
            "persistent_impact_proxy_bps_per_lot": slope("size", "mid_bps_60s"),
            "liquidity_scale_lots": float(np.nanmedian(table["depth5_pre"][ok])) if ok.any() else None,
            "caveat": "OLS slopes of signed responses; confounded by concurrent flow and information; "
                      "no unique structural impact model is identified."}


def conditional_means(table: dict[str, np.ndarray], edges: dict, dimension: str, response: str) -> dict:
    labels = buckets(table, edges, dimension)
    ok = table["valid_pre"] & np.isfinite(table[response]) & np.isfinite(labels)
    result = {}
    for label in sorted(set(labels[ok].astype(int))):
        values = table[response][ok & (labels == label)]
        result[str(label)] = {"n": int(len(values)), "mean": float(values.mean()),
                              "sd": float(values.std(ddof=1)) if len(values) > 1 else None}
    return result


def safe_log_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a == 0 or b == 0 or (a > 0) != (b > 0):
        return None
    return float(math.log(abs(a) / abs(b)))
