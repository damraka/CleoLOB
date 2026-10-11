"""Decision-focused calibration (M21): does better calibration improve execution decisions?

For every (truth world T, candidate calibration C) pair, where T is a known
data-generating world and C a model calibrated to data from T:

* ``calibration_loss`` — standardized summary distance between C's simulated
  data and T's observed data (generic) and over execution-sensitive summaries
  (spread, depth, market rate, volatility, depletion) only;
* ``predictive_fit`` — mean one-step log predictive density of T's data under C;
* ``cost_error`` — mean absolute error of C's predicted policy costs versus T's;
* ``completion_error`` — mean absolute error of predicted completion rates;
* ``kendall_tau`` / ``pairwise_accuracy`` — ranking agreement;
* ``top_retention`` — C's best policy is T's best;
* ``regret`` — T's cost of C's chosen policy minus T's best cost.

``associations`` reports Spearman correlations (with a bootstrap interval over
pairs) between each calibration metric and regret, so the question "does
better calibration help decisions" is answered from data, with the direction
and its uncertainty.
"""
from __future__ import annotations

import numpy as np

from .market.frame import SUMMARY_NAMES
from .statistics import kendall_tau, spearman

EXECUTION_SUMMARIES = ("rate_market", "spread_mean", "p_spread1", "log_depth_mean", "vol_1s", "vol_10s",
                       "mean_market_size", "depletion_rate")


def summary_distance(a: np.ndarray, b: np.ndarray, scale: np.ndarray, keys: tuple[str, ...] | None = None) -> float:
    idx = [SUMMARY_NAMES.index(k) for k in (keys or SUMMARY_NAMES)]
    a, b, s = (np.asarray(x, float)[idx] for x in (a, b, scale))
    return float(np.sqrt((((a - b) / s) ** 2).mean()))


def pair_metrics(true_costs: dict[str, float], model_costs: dict[str, float], *, true_completion: dict | None = None,
                 model_completion: dict | None = None) -> dict:
    names = sorted(true_costs)
    t = np.asarray([true_costs[n] for n in names])
    m = np.asarray([model_costs[n] for n in names])
    agree = total = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if t[i] == t[j]:
                continue
            total += 1
            agree += int(np.sign(t[i] - t[j]) == np.sign(m[i] - m[j]))
    chosen = names[int(np.argmin(m))]
    out = {"cost_error": float(np.mean(np.abs(t - m))), "kendall_tau": kendall_tau(m, t),
           "pairwise_accuracy": agree / total if total else None, "top_retention": float(chosen == names[int(np.argmin(t))]),
           "regret": float(true_costs[chosen] - t.min()), "chosen": chosen}
    if true_completion and model_completion:
        out["completion_error"] = float(np.mean([abs(true_completion[n] - model_completion[n]) for n in names]))
    return out


def associations(rows: list[dict], *, metrics: tuple[str, ...], target: str = "regret", n_boot: int = 2000,
                 seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    y = np.asarray([r[target] for r in rows], float)
    for metric in metrics:
        x = np.asarray([r.get(metric, np.nan) for r in rows], float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5:
            out[metric] = {"status": "NOT_EVALUABLE", "n": int(ok.sum())}
            continue
        xs, ys = x[ok], y[ok]
        rho = spearman(xs, ys)
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(xs), len(xs))
            boots.append(spearman(xs[i], ys[i]))
        out[metric] = {"spearman": rho, "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
                       "n": int(ok.sum())}
    return out
