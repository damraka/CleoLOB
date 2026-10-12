"""Posterior predictive checks 2.0 (M16).

For each registered horizon (1, 10, 50, 300, 600 bins = 100 ms, 1 s, 5 s, 30 s,
60 s) the observed frame and every replicated frame are reduced to the same
statistics; each statistic gets a two-sided posterior predictive p-value
``2 min(P(T_rep >= T_obs), P(T_rep <= T_obs))``. A check *fails* when that
p-value is below the registered 0.01. Horizons longer than a fifth of the frame
are skipped (reported as NOT_EVALUABLE), so short data never pretends to
support long-horizon checks.

Statistics: marginal means/variances and 99th percentiles of aggregated
market/add/cancel counts, kurtosis (tails), autocorrelation, market-cancel
cross-correlation, dispersion (event clustering), trade clustering,
volatility clustering (|return| autocorrelation), variance ratio
(multi-timescale), depth-change spread, imbalance autocorrelation, plus
horizon-free spread transitions, spread dwell times and touch queue depletion
times.
"""
from __future__ import annotations

import numpy as np

HORIZONS = {"100ms": 1, "1s": 10, "5s": 50, "30s": 300, "60s": 600}
FAIL_P = 0.01


def _agg(x: np.ndarray, h: int) -> np.ndarray:
    n = (len(x) // h) * h
    return x[:n].reshape(-1, h, *x.shape[1:]).sum(1) if n else x[:0]


def _acf1(x: np.ndarray) -> float:
    if len(x) < 4 or np.std(x) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _kurt(x: np.ndarray) -> float:
    s = np.std(x)
    return float(np.mean(((x - x.mean()) / s) ** 4)) if s > 0 else 0.0


def horizon_stats(frame, h: int) -> dict:
    c = frame.counts.astype(float)
    groups = {"market": c[:, 0] + c[:, 1], "add": c[:, 2] + c[:, 3], "cancel": c[:, 4] + c[:, 5]}
    out = {}
    for name, series in groups.items():
        a = _agg(series, h)
        out[f"{name}_mean"] = float(a.mean())
        out[f"{name}_var"] = float(a.var())
        out[f"{name}_p99"] = float(np.quantile(a, 0.99)) if len(a) else 0.0
        out[f"{name}_dispersion"] = float(a.var() / a.mean()) if a.mean() > 0 else 1.0
    m = _agg(groups["market"], h)
    out["market_kurtosis"] = _kurt(m)
    out["market_acf1"] = _acf1(m)
    cancel = _agg(groups["cancel"], h)
    out["market_cancel_xcorr"] = float(np.corrcoef(m, cancel)[0, 1]) if len(m) > 3 and m.std() > 0 and \
        cancel.std() > 0 else 0.0
    trade = (m > 0).astype(float)
    out["trade_clustering"] = float(np.mean(trade[1:][trade[:-1] > 0])) if trade[:-1].sum() > 0 else 0.0
    mid = frame.mid.astype(float)
    r = mid[h::h] - mid[:-h:h] if len(mid) > 2 * h else np.zeros(2)
    out["return_sd"] = float(np.std(r))
    out["vol_clustering"] = _acf1(np.abs(r))
    r1 = np.diff(mid)
    out["variance_ratio"] = float(np.var(r) / (h * np.var(r1))) if np.var(r1) > 0 else 1.0
    depth = np.log(np.maximum(frame.state[:, 3] + frame.state[:, 4], 1e-6))
    dd = depth[h::h] - depth[:-h:h] if len(depth) > 2 * h else np.zeros(2)
    out["depth_change_sd"] = float(np.std(dd))
    imb = frame.state[:, 1]
    out["imbalance_acf_h"] = float(np.corrcoef(imb[:-h], imb[h:])[0, 1]) if len(imb) > h + 3 and imb.std() > 0 \
        else 0.0
    return out


def static_stats(frame) -> dict:
    spread = np.clip(np.round(frame.state[:, 0]), 1, 3).astype(int)
    out = {}
    for a in (1, 2, 3):
        mask = spread[:-1] == a
        for b in (1, 2, 3):
            out[f"spread_{a}_to_{b}"] = float(np.mean(spread[1:][mask] == b)) if mask.any() else 0.0
    runs, current = [], 1
    for i in range(1, len(spread)):
        if spread[i] == 1 and spread[i - 1] == 1:
            current += 1
        elif spread[i - 1] == 1:
            runs.append(current)
            current = 1
    out["tight_dwell_mean_bins"] = float(np.mean(runs)) if runs else 0.0
    moves = np.nonzero((np.diff(frame.bid_px) != 0) | (np.diff(frame.ask_px) != 0))[0]
    out["queue_depletion_mean_bins"] = float(np.mean(np.diff(moves))) if len(moves) > 2 else float(len(frame))
    return out


def check(observed, replicates: list) -> dict:
    n = len(observed)
    table = {}
    for label, h in HORIZONS.items():
        if h * 5 > n:
            table[label] = {"status": "NOT_EVALUABLE", "reason": f"frame of {n} bins is shorter than 5 x {h} bins"}
            continue
        obs = horizon_stats(observed, h)
        reps = [horizon_stats(r, h) for r in replicates]
        table[label] = _pvalues(obs, reps)
    obs = static_stats(observed)
    table["static"] = _pvalues(obs, [static_stats(r) for r in replicates])
    failures = {k: v.get("failures", 0) for k, v in table.items() if isinstance(v, dict)}
    checks = {k: v.get("checks", 0) for k, v in table.items() if isinstance(v, dict)}
    return {"horizons": table, "failures": failures, "checks": checks,
            "failure_fraction": sum(failures.values()) / max(sum(checks.values()), 1), "fail_p": FAIL_P,
            "replicates": len(replicates)}


def _pvalues(obs: dict, reps: list[dict]) -> dict:
    stats, failures = {}, 0
    for key, value in obs.items():
        rv = np.asarray([r[key] for r in reps], float)
        upper, lower = float(np.mean(rv >= value)), float(np.mean(rv <= value))
        p = min(1.0, 2 * min(upper, lower))
        failed = p < FAIL_P
        failures += int(failed)
        stats[key] = {"observed": value, "rep_mean": float(rv.mean()), "rep_sd": float(rv.std()), "p": p,
                      "failed": failed}
    return {"stats": stats, "failures": failures, "checks": len(stats)}
