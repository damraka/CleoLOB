"""Market-data quality decisions, dataset profiles and distribution drift (M13-M16, M19).

``decide`` turns a day cache's quality counters into AVAILABLE or QUALITY_REJECTED using the thresholds registered
by protocol amendment 1 (before any exploratory access). Nothing is repaired silently: rejected days keep their
counters, and the slot is replaced by the next registered backup (``data.sources.resolve``).

``profile`` summarizes a day (spread, depth, trade rate and size, volatility, imbalance, event rates) and
``drift`` compares profiles against a reference set with standardized differences and two-sample distances, so
exploratory, confirmatory and vault days can be described without inspecting any outcome.
"""
from __future__ import annotations

import math

import numpy as np

THRESHOLDS = {
    "max_sequence_problem_rate": 0.001,     # (gaps + duplicates + out-of-order) / messages
    "max_crossed_rate": 0.001,              # crossed books after a message / messages
    "min_valid_bin_fraction": 0.98,
    "max_long_gap_minutes": 10.0,           # total duration of book gaps longer than 5 s
    "max_inconsistent_trade_rate": 0.005,   # prints > 20 bps outside the +-1-bin touch range / prints
    "max_unknown_side_rate": 0.001,
    "max_off_grid_rate": 0.001,             # off-tick prices / messages
    "min_trades": 1_000,
}


def decide(quality: dict, thresholds: dict | None = None) -> dict:
    th = {**THRESHOLDS, **(thresholds or {})}
    r = quality["reconstruction"]
    messages = max(int(r["messages"]), 1)
    trades = max(int(quality["trades"]), 1)
    if "long_gap_ms_total" not in r:
        raise ValueError("quality counters predate the gap-duration fix (incident 1); rebuild the cache")
    long_gap_min = float(r["long_gap_ms_total"]) / 60_000.0
    checks = {
        "sequence": (r["sequence_gaps"] + r["duplicates"] + r["out_of_order"]) / messages
        <= th["max_sequence_problem_rate"],
        "crossed": r["crossed"] / messages <= th["max_crossed_rate"],
        "valid_bins": quality["valid_bin_fraction"] >= th["min_valid_bin_fraction"],
        "gaps": long_gap_min <= th["max_long_gap_minutes"],
        "trade_consistency": quality.get("trades_inconsistent_20bps", 0) / trades <= th["max_inconsistent_trade_rate"],
        "trade_side": quality["trade_side_unknown"] / trades <= th["max_unknown_side_rate"],
        "grid": r["off_tick"] / messages <= th["max_off_grid_rate"],
        "volume": quality["trades"] >= th["min_trades"],
    }
    return {"status": "AVAILABLE" if all(checks.values()) else "QUALITY_REJECTED", "checks": checks,
            "failed": sorted(k for k, ok in checks.items() if not ok), "thresholds": th}


def profile(cache: dict) -> dict:
    v = cache["valid"].astype(bool)
    spread = (cache["ask_px"] - cache["bid_px"])[v].astype(float)
    l1 = (cache["depth"][v, 0, 0] + cache["depth"][v, 1, 0]).astype(float)
    mid = cache["mid"].astype(float)
    tick_bps = 1e4 / float(np.median(mid[mid > 0])) if np.any(mid > 0) else float("nan")
    r1 = np.diff(mid[::10]) * tick_bps
    r60 = np.diff(mid[::600]) * tick_bps
    q = cache["trade_q"].astype(float)
    counts = cache["counts"][v].astype(float)
    imb = cache["state"][v, 1].astype(float)
    return {"spread_ticks_mean": float(spread.mean()) if len(spread) else None,
            "spread_ticks_p95": float(np.percentile(spread, 95)) if len(spread) else None,
            "l1_depth_lots_median": float(np.median(l1)) if len(l1) else None,
            "trades": int(len(q)), "trade_size_median": float(np.median(q)) if len(q) else None,
            "trade_size_p99": float(np.percentile(q, 99)) if len(q) else None,
            "vol_1s_bps": float(np.std(r1)) if len(r1) else None, "vol_60s_bps": float(np.std(r60)) if len(r60) else None,
            "imbalance_sd": float(imb.std()) if len(imb) else None,
            "event_rates": dict(zip(("market_buy", "market_sell", "add_bid", "add_ask", "cancel_bid", "cancel_ask"),
                                    (float(x) for x in counts.mean(0)))) if len(counts) else {},
            "tick_bps": tick_bps, "mid_start": float(mid[0]) * float(cache["tick"]),
            "mid_end": float(mid[-1]) * float(cache["tick"])}


_DRIFT_KEYS = ("spread_ticks_mean", "l1_depth_lots_median", "trades", "trade_size_median", "vol_1s_bps",
               "vol_60s_bps", "imbalance_sd")


def drift(profiles: dict[str, dict], reference: list[str]) -> dict:
    """Standardized log-ratio drift of each profile metric versus the reference days (descriptive)."""
    out = {}
    for key in _DRIFT_KEYS:
        ref = np.asarray([math.log(max(profiles[d][key], 1e-12)) for d in reference
                          if profiles[d].get(key) is not None], float)
        if len(ref) < 2:
            continue
        mu, sd = float(ref.mean()), float(ref.std(ddof=1)) or 1e-9
        out[key] = {d: (math.log(max(p[key], 1e-12)) - mu) / sd for d, p in profiles.items()
                    if p.get(key) is not None}
    flagged = sorted({d for row in out.values() for d, z in row.items() if abs(z) > 3.0})
    return {"metrics": out, "flagged_days": flagged, "rule": "|z| > 3 on the log scale versus the reference days",
            "meaning": "descriptive drift; never used to choose or drop confirmatory days"}
