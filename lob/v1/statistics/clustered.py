"""Cluster-level inference for dependent research units (M130, M7).

v0.9 resampled instrument-days as if independent although several instruments shared each calendar date (audit
Issue B). v1 makes the independent unit explicit: every value carries a cluster key (normally the calendar date, or
the venue-date), and every interval, test and power calculation is computed at the cluster level.

Provided procedures (all deterministic given ``seed``):

* ``cluster_t`` - size-weighted mean with the CR1 cluster-robust standard error and a Student t reference on G - 1
  degrees of freedom (the conservative small-G choice);
* ``multiway_t`` - Cameron-Gelbach-Miller two-way cluster-robust variance (e.g. date x instrument);
* ``sign_flip`` - exact (G <= 16) or Monte Carlo randomization test that flips the sign of whole cluster sums of
  (value - threshold); its minimum attainable p-value is 2^-G, reported as ``p_floor`` so a design can be rejected
  as infeasible before any confirmatory data are read;
* ``cluster_bootstrap`` - percentile interval resampling whole clusters;
* ``design_effect`` - one-way ANOVA intra-cluster correlation and the implied effective sample size;
* ``feasible_clusters`` / ``cluster_power`` - pre-confirmatory feasibility from a pilot.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import itertools
import math
from typing import Hashable, Mapping, Sequence

import numpy as np

from .distributions import nct_power, t_ppf, t_sf


@dataclass(frozen=True)
class ClusterTest:
    method: str
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_one_sided: float
    threshold: float
    units: int
    clusters: int
    df: float

    def to_dict(self) -> dict:
        return asdict(self)


def _arrays(values: Sequence[float], clusters: Sequence[Hashable]) -> tuple[np.ndarray, list]:
    x = np.asarray(values, float)
    keys = list(clusters)
    if x.ndim != 1 or len(keys) != len(x):
        raise ValueError("values and clusters must be 1-d and of equal length")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite")
    if len(x) == 0:
        raise ValueError("no values")
    return x, keys


def groups(values: Sequence[float], clusters: Sequence[Hashable]) -> dict[Hashable, np.ndarray]:
    x, keys = _arrays(values, clusters)
    out: dict[Hashable, list[float]] = defaultdict(list)
    for v, k in zip(x, keys):
        out[k].append(float(v))
    return {k: np.asarray(v) for k, v in out.items()}


def _cr_meat(resid: np.ndarray, keys: list) -> float:
    sums: dict[Hashable, float] = defaultdict(float)
    for r, k in zip(resid, keys):
        sums[k] += float(r)
    return float(sum(s * s for s in sums.values()))


def cluster_t(values: Sequence[float], clusters: Sequence[Hashable], *, threshold: float = 0.0,
              alpha: float = 0.05) -> ClusterTest:
    """Mean with CR1 cluster-robust SE; one-sided p of H0: mean <= threshold; two-sided (1 - alpha) interval."""
    x, keys = _arrays(values, clusters)
    n, g = len(x), len(set(keys))
    est = float(x.mean())
    if g < 2:
        return ClusterTest("cluster_t", est, math.inf, -math.inf, math.inf, 1.0, threshold, n, g, 0.0)
    meat = _cr_meat(x - est, keys)
    var = meat / (n * n) * g / (g - 1)
    se = math.sqrt(max(var, 0.0))
    df = g - 1
    if se == 0.0:
        p = 0.0 if est > threshold else 1.0
        return ClusterTest("cluster_t", est, 0.0, est, est, p, threshold, n, g, float(df))
    q = t_ppf(1.0 - alpha / 2.0, df)
    p = t_sf((est - threshold) / se, df)
    return ClusterTest("cluster_t", est, se, est - q * se, est + q * se, float(p), threshold, n, g, float(df))


def multiway_t(values: Sequence[float], first: Sequence[Hashable], second: Sequence[Hashable], *,
               threshold: float = 0.0, alpha: float = 0.05) -> ClusterTest:
    """Two-way cluster-robust mean (CGM 2011): V = V_first + V_second - V_intersection, df = min(G) - 1."""
    x, a = _arrays(values, first)
    _, b = _arrays(values, second)
    n = len(x)
    est = float(x.mean())
    r = x - est
    ga, gb = len(set(a)), len(set(b))
    inter = list(zip(a, b))
    gi = len(set(inter))
    if min(ga, gb) < 2:
        return ClusterTest("multiway_t", est, math.inf, -math.inf, math.inf, 1.0, threshold, n, min(ga, gb), 0.0)

    def v(keys, g):
        return _cr_meat(r, keys) / (n * n) * (g / (g - 1) if g > 1 else 1.0)

    var = v(a, ga) + v(b, gb) - v(inter, gi)
    if var <= 0:                       # the CGM variance can be non-positive; fall back to the larger one-way term
        var = max(v(a, ga), v(b, gb))
    se = math.sqrt(var)
    df = min(ga, gb) - 1
    q = t_ppf(1.0 - alpha / 2.0, df)
    p = t_sf((est - threshold) / se, df) if se > 0 else (0.0 if est > threshold else 1.0)
    return ClusterTest("multiway_t", est, se, est - q * se, est + q * se, float(p), threshold, n, min(ga, gb),
                       float(df))


def sign_flip(values: Sequence[float], clusters: Sequence[Hashable], *, threshold: float = 0.0,
              n_mc: int = 100_000, seed: int = 0) -> dict:
    """Randomization test of H0: cluster-level effects are symmetric about ``threshold`` (one-sided, greater)."""
    gmap = groups(values, clusters)
    sums = np.array([float((v - threshold).sum()) for v in gmap.values()])
    g = len(sums)
    observed = float(sums.sum())
    if g <= 16:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=g)))
        stats = signs @ sums
        p = float(np.mean(stats >= observed - 1e-12))
        exact = True
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((1.0, -1.0), size=(n_mc, g))
        stats = signs @ sums
        p = float((1 + np.sum(stats >= observed - 1e-12)) / (n_mc + 1))
        exact = False
    return {"method": "cluster_sign_flip", "p_one_sided": p, "p_floor": 2.0 ** -g, "clusters": g,
            "exact": exact, "statistic": observed, "threshold": threshold}


def cluster_bootstrap(values: Sequence[float], clusters: Sequence[Hashable], *, alpha: float = 0.05,
                      n_boot: int = 20_000, seed: int = 0) -> dict:
    gmap = groups(values, clusters)
    keys = sorted(gmap, key=str)
    sums = np.array([gmap[k].sum() for k in keys])
    counts = np.array([len(gmap[k]) for k in keys])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    draws = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"method": "cluster_bootstrap_percentile", "estimate": float(sums.sum() / counts.sum()),
            "ci_low": float(lo), "ci_high": float(hi), "clusters": len(keys), "n_boot": n_boot,
            "note": "percentile cluster bootstrap under-covers with few clusters; read with cluster_t and sign_flip"}


def design_effect(values: Sequence[float], clusters: Sequence[Hashable]) -> dict:
    gmap = groups(values, clusters)
    gs = list(gmap.values())
    n, k = sum(len(v) for v in gs), len(gs)
    grand = float(np.concatenate(gs).mean())
    if k < 2 or n <= k:
        return {"units": n, "clusters": k, "icc": None, "design_effect": None, "effective_units": float(k)}
    msb = sum(len(v) * (v.mean() - grand) ** 2 for v in gs) / (k - 1)
    msw = sum(float(((v - v.mean()) ** 2).sum()) for v in gs) / (n - k)
    m0 = (n - sum(len(v) ** 2 for v in gs) / n) / (k - 1)
    denom = msb + (m0 - 1) * msw
    icc = float((msb - msw) / denom) if denom > 0 else 0.0
    deff = 1.0 + (n / k - 1.0) * max(icc, 0.0)
    return {"units": n, "clusters": k, "icc": icc, "design_effect": deff, "effective_units": n / deff}


def min_clusters_for(alpha: float) -> int:
    """Smallest G whose exact sign-flip test can reach one-sided p <= alpha (2^-G <= alpha)."""
    return max(1, math.ceil(math.log2(1.0 / alpha)))


def cluster_power(effect: float, sd_cluster_means: float, clusters: int, *, alpha: float) -> float:
    """Power of the one-sided cluster-mean t test for an effect above its threshold of ``effect``."""
    if clusters < 2 or sd_cluster_means <= 0:
        return 0.0 if clusters < 2 else 1.0
    return float(nct_power(effect / (sd_cluster_means / math.sqrt(clusters)), clusters - 1, alpha))


def feasible_clusters(effect: float, sd_cluster_means: float, *, alpha: float, power: float = 0.8,
                      max_clusters: int = 400) -> int | None:
    """Smallest G with cluster-t power >= ``power`` and a sign-flip floor <= alpha; None if above ``max_clusters``."""
    if effect <= 0:
        return None
    start = max(2, min_clusters_for(alpha))
    for g in range(start, max_clusters + 1):
        if cluster_power(effect, sd_cluster_means, g, alpha=alpha) >= power:
            return g
    return None


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


__all__ = ["ClusterTest", "groups", "cluster_t", "multiway_t", "sign_flip", "cluster_bootstrap", "design_effect",
           "min_clusters_for", "cluster_power", "feasible_clusters", "holm_adjust"]
