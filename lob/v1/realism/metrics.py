"""Realism metrics: marginal, conditional, temporal, path and tail (M132).

Every metric is computed the same way on real day windows and on generated frames (v0.8 ``EventFrame`` layout with
trade samples), so a generator is compared with history on identical definitions. Each metric is tagged

* ``marginal`` - an unconditional rate or distribution moment (what distribution-only realism scores look at);
* ``conditional`` - a response of one quantity to the state of another (imbalance -> next order-flow side, spread
  state -> cancellation intensity, burst -> spread and depth response, volatility clustering);
* ``temporal`` - a dependence across time (autocorrelations, multiscale dispersion of counts);
* ``tail`` - extremes (Hill tail index, wide-spread and depth-collapse frequencies);

and grouped by capability (``CAPABILITIES``), which is what the gate decides on.
"""
from __future__ import annotations

import math

import numpy as np

from ...v09.realism.benchmark import _acf, _bursts, _clearance

BIN_S = 0.1
CAPABILITIES = {
    "spread_dynamics": ("p_spread_gt1", "mean_spread", "acf1_spread_change"),
    "depth_dynamics": ("median_log_touch", "imbalance_sd", "acf10_log_touch"),
    "event_timing": ("rate_market", "rate_add", "rate_cancel", "disp_market_1s", "disp_market_10s",
                     "disp_market_60s", "acf1_market"),
    "queue_survival": ("clear_1s", "clear_5s", "clear_30s"),
    "fill_prediction": ("clear_5s",),
    "impact": ("burst_mid_10s_bps",),
    "tail_behavior": ("hill_abs_ret10", "p_spread_ge3", "p_depth_collapse"),
    "conditional_response": ("imbalance_flow_response", "cancel_ratio_wide_spread", "vol_clustering_acf10",
                             "burst_depth_recovery_5s"),
}
KIND = {
    "p_spread_gt1": "marginal", "mean_spread": "marginal", "acf1_spread_change": "temporal",
    "median_log_touch": "marginal", "imbalance_sd": "marginal", "acf10_log_touch": "temporal",
    "rate_market": "marginal", "rate_add": "marginal", "rate_cancel": "marginal", "disp_market_1s": "temporal",
    "disp_market_10s": "temporal", "disp_market_60s": "temporal", "acf1_market": "temporal",
    "clear_1s": "conditional", "clear_5s": "conditional", "clear_30s": "conditional",
    "burst_mid_10s_bps": "conditional", "hill_abs_ret10": "tail", "p_spread_ge3": "tail",
    "p_depth_collapse": "tail", "imbalance_flow_response": "conditional", "cancel_ratio_wide_spread": "conditional",
    "vol_clustering_acf10": "temporal", "burst_depth_recovery_5s": "conditional",
}
METRICS = tuple(KIND)


def _dispersion(counts: np.ndarray, scale_bins: int) -> float:
    n = len(counts) // scale_bins
    if n < 5:
        return math.nan
    agg = counts[: n * scale_bins].reshape(n, scale_bins).sum(1)
    m = agg.mean()
    return float(agg.var() / m) if m > 0 else math.nan


def frame_metrics(frames: list) -> dict[str, float]:
    """Pooled metrics over a list of frames (windows); returns {metric: value} (NaN when not estimable)."""
    v = np.concatenate([f.valid for f in frames]).astype(bool)
    spread = np.concatenate([(f.ask_px - f.bid_px).astype(float) for f in frames])
    touch = np.concatenate([(f.depth[:, 0, 0] + f.depth[:, 1, 0]).astype(float) for f in frames])
    imb = np.concatenate([f.state[:, 1] for f in frames])
    c = np.concatenate([f.counts for f in frames]).astype(float)
    market = c[:, 0] + c[:, 1]
    out: dict[str, float] = {}
    sv, tv = spread[v], touch[v]
    out["p_spread_gt1"] = float((sv > 1).mean())
    out["mean_spread"] = float(sv.mean())
    out["acf1_spread_change"] = _acf((np.diff(spread) != 0).astype(float), 1)
    lt = np.log(np.maximum(touch, 1.0))
    out["median_log_touch"] = float(np.median(lt[v]))
    out["imbalance_sd"] = float(imb[v].std())
    out["acf10_log_touch"] = _acf(lt, 10)
    out["rate_market"] = float(market[v].mean())
    out["rate_add"] = float((c[v, 2] + c[v, 3]).mean())
    out["rate_cancel"] = float((c[v, 4] + c[v, 5]).mean())
    per = [np.asarray(f.counts[:, 0] + f.counts[:, 1], float) for f in frames]
    for name, k in (("disp_market_1s", 10), ("disp_market_10s", 100), ("disp_market_60s", 600)):
        vals = [x for x in (_dispersion(p, k) for p in per) if np.isfinite(x)]
        out[name] = float(np.mean(vals)) if vals else math.nan
    out["acf1_market"] = _acf(market, 1)
    clear = np.mean([_clearance(f) for f in frames], axis=0)
    out["clear_1s"], out["clear_5s"], out["clear_30s"] = (float(x) for x in clear)
    burst = np.mean([_bursts(f) for f in frames], axis=0)
    out["burst_mid_10s_bps"] = float(burst[0])
    out["burst_depth_recovery_5s"] = float(burst[1])
    mids = [f.mid.astype(float) for f in frames]
    r10 = np.concatenate([np.diff(m[::100]) / np.maximum(m[:-100:100], 1e-9) * 1e4 for m in mids if len(m) > 200]) \
        if any(len(m) > 200 for m in mids) else np.zeros(0)
    a = np.sort(np.abs(r10[r10 != 0]))[::-1]
    k = max(5, int(0.05 * len(a)))
    out["hill_abs_ret10"] = float(1.0 / np.mean(np.log(a[:k] / a[k]))) if len(a) > k + 1 and a[k] > 0 else math.nan
    out["p_spread_ge3"] = float((sv >= 3).mean())
    med = float(np.median(tv)) if len(tv) else 1.0
    out["p_depth_collapse"] = float((tv < 0.1 * med).mean())
    # conditional responses
    nxt_buy = np.concatenate([np.r_[f.counts[1:, 0] - f.counts[1:, 1], 0] for f in frames]).astype(float)
    hi, lo = v & (imb > 0.5), v & (imb < -0.5)
    out["imbalance_flow_response"] = float(nxt_buy[hi].mean() - nxt_buy[lo].mean()) if hi.any() and lo.any() \
        else math.nan
    canc = c[:, 4] + c[:, 5]
    wide, tight = v & (spread > 1), v & (spread == 1)
    out["cancel_ratio_wide_spread"] = float(canc[wide].mean() / max(canc[tight].mean(), 1e-9)) if wide.sum() > 20 \
        and tight.any() else math.nan
    r1 = np.concatenate([np.abs(np.diff(m[::10])) for m in mids if len(m) > 20])
    out["vol_clustering_acf10"] = _acf(r1, 10)
    return out


def window_bootstrap_se(frames: list, *, draws: int = 30, seed: int = 0) -> dict[str, float]:
    """Bootstrap SE of every metric over the frame list (windows resampled)."""
    rng = np.random.default_rng(seed)
    vals = {m: [] for m in METRICS}
    for _ in range(draws):
        pick = [frames[i] for i in rng.integers(0, len(frames), len(frames))]
        mm = frame_metrics(pick)
        for m in METRICS:
            vals[m].append(mm[m])
    return {m: float(np.nanstd(v)) if np.isfinite(v).any() else math.nan for m, v in vals.items()}


__all__ = ["CAPABILITIES", "KIND", "METRICS", "frame_metrics", "window_bootstrap_se"]
