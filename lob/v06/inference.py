"""Centralized v0.6 statistical design: resampling, intervals, multiplicity, equivalence and power.

Resampling units are explicit: historical time blocks (600 s, or 5-minute
regime blocks) and simulated seeds are resampled as wholes, so within-block
temporal dependence is preserved; IID resampling is used only for independent
units (market seeds, historical episodes, simulator worlds). Every function
takes an explicit seed; no global random state is used.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..stats import adjust_pvalues


def counts(n: int, rng: np.random.Generator) -> np.ndarray:
    """Resampling weights: how often each of ``n`` units is drawn (with replacement)."""
    if n <= 0:
        raise ValueError("cannot resample zero units")
    return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)


def moving_block_counts(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap weights over an ordered sequence of ``n`` units."""
    if not 1 <= block <= n:
        raise ValueError("block length must lie in [1, n]")
    weights = np.zeros(n)
    drawn = 0
    while drawn < n:
        start = int(rng.integers(0, n - block + 1))
        take = min(block, n - drawn)
        weights[start:start + take] += 1
        drawn += take
    return weights


def stationary_counts(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap weights (geometric block lengths, circular wrap)."""
    if n <= 0 or mean_block < 1:
        raise ValueError("stationary bootstrap needs n > 0 and mean block >= 1")
    p = 1.0 / mean_block
    weights = np.zeros(n)
    position = int(rng.integers(0, n))
    for _ in range(n):
        weights[position] += 1
        position = int(rng.integers(0, n)) if rng.random() < p else (position + 1) % n
    return weights


def interval(draws: Sequence[float], alpha: float, *, sides: str = "two") -> tuple[float, float]:
    """Percentile interval; ``sides='upper'`` gives (-inf, 1 - alpha quantile], 'lower' the mirror."""
    values = np.asarray(draws, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    if sides == "two":
        low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    elif sides == "upper":
        low, high = -math.inf, np.quantile(values, 1 - alpha)
    elif sides == "lower":
        low, high = np.quantile(values, alpha), math.inf
    else:
        raise ValueError("sides must be two, upper or lower")
    return float(low), float(high)


def bootstrap_pvalue(draws: Sequence[float]) -> float:
    """Two-sided percentile-bootstrap p-value for a zero effect."""
    values = np.asarray(draws, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    tail = min(np.mean(values <= 0), np.mean(values >= 0))
    return float(min(1.0, 2 * tail))


def mean_interval(values: Sequence[float], *, alpha: float, samples: int, seed: int) -> dict:
    """IID percentile bootstrap of a mean over independent units."""
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        return {"status": "WITHHELD", "reason": "fewer than 2 units or non-finite values", "n": int(np.size(x))}
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), (samples, len(x)))].mean(1)
    low, high = interval(draws, alpha)
    return {"status": "AVAILABLE", "mean": float(x.mean()), "ci_low": low, "ci_high": high, "n": int(len(x)),
            "alpha": alpha, "samples": samples, "p_value": bootstrap_pvalue(draws)}


def difference_interval(a: Sequence[float], b: Sequence[float], *, alpha: float, samples: int, seed: int,
                        paired: bool) -> dict:
    """Mean(a) - mean(b): paired (same units) or independent percentile bootstrap."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if paired:
        if a.shape != b.shape:
            raise ValueError("paired samples must align")
        return mean_interval(a - b, alpha=alpha, samples=samples, seed=seed)
    if len(a) < 2 or len(b) < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return {"status": "WITHHELD", "reason": "fewer than 2 units or non-finite values"}
    rng = np.random.default_rng(seed)
    draws = (a[rng.integers(0, len(a), (samples, len(a)))].mean(1)
             - b[rng.integers(0, len(b), (samples, len(b)))].mean(1))
    low, high = interval(draws, alpha)
    return {"status": "AVAILABLE", "mean": float(a.mean() - b.mean()), "ci_low": low, "ci_high": high,
            "n_a": int(len(a)), "n_b": int(len(b)), "alpha": alpha, "samples": samples,
            "p_value": bootstrap_pvalue(draws)}


def direction(result: dict) -> int | None:
    """+1 / -1 when the interval excludes zero, 0 when it does not, None when withheld."""
    if result.get("status") != "AVAILABLE":
        return None
    if result["ci_low"] > 0:
        return 1
    if result["ci_high"] < 0:
        return -1
    return 0


def family_alpha(alpha: float, size: int) -> float:
    if size < 1:
        raise ValueError("family size must be positive")
    return alpha / size


def holm(pvalues: Sequence[float]) -> list[float]:
    return adjust_pvalues(np.asarray(pvalues, float), "holm").tolist() if len(pvalues) else []


def bonferroni(pvalues: Sequence[float]) -> list[float]:
    return adjust_pvalues(np.asarray(pvalues, float), "bonferroni").tolist() if len(pvalues) else []


def equivalence_upper(draws: Sequence[float], *, alpha: float) -> tuple[float, float]:
    """One-sided bounds for a nonnegative distance: (alpha lower quantile, 1 - alpha upper quantile)."""
    values = np.asarray(draws, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan, math.nan
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


# ----------------------------------------------------------------------------- rank statistics


def ranks(x: Sequence[float]) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    values, inverse, tie_counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=r)
    return sums[inverse] / tie_counts[inverse]


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return math.nan
    rx, ry = ranks(x[keep]), ranks(y[keep])
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_interval(x: Sequence[float], y: Sequence[float], *, alpha: float, samples: int, seed: int) -> dict:
    """Unit-level bootstrap of Spearman's rho (units are independent, e.g. simulator worlds)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 4:
        return {"status": "WITHHELD", "reason": "fewer than 4 units", "n": int(len(x))}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        pick = rng.integers(0, len(x), len(x))
        draws.append(spearman(x[pick], y[pick]))
    draws = np.asarray(draws)
    low, high = interval(draws, alpha)
    one_sided_low = float(np.nanquantile(draws, alpha))
    p_one_sided = float(np.mean(draws[np.isfinite(draws)] <= 0)) if np.isfinite(draws).any() else math.nan
    return {"status": "AVAILABLE", "rho": spearman(x, y), "ci_low": low, "ci_high": high,
            "one_sided_lower": one_sided_low, "p_one_sided": p_one_sided, "n": int(len(x)), "samples": samples}


def kendall_tau(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Kendall's tau-a between two rankings over their common items."""
    common = [x for x in a if x in b]
    if len(common) < 2:
        return None
    ra, rb = {x: i for i, x in enumerate(a)}, {x: i for i, x in enumerate(b)}
    pairs = [(i, j) for i in range(len(common)) for j in range(i + 1, len(common))]
    score = sum(np.sign(ra[common[i]] - ra[common[j]]) * np.sign(rb[common[i]] - rb[common[j]]) for i, j in pairs)
    return float(score / len(pairs))


# ----------------------------------------------------------------------------- power


def _z(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation, |error| < 1.2e-9)."""
    if not 0 < p < 1:
        raise ValueError("p must lie in (0, 1)")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
    low = 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - low:
        return -_z(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def minimum_detectable_effect(sd: float, n: int, *, alpha: float, power: float = 0.8) -> float:
    """Two-sided normal-approximation MDE of a mean with unit sd ``sd`` over ``n`` independent units."""
    if n < 2 or sd < 0:
        return math.nan
    return float((_z(1 - alpha / 2) + _z(power)) * sd / math.sqrt(n))


def required_units(sd: float, effect: float, *, alpha: float, power: float = 0.8) -> int | None:
    if effect <= 0 or sd <= 0:
        return None
    return int(math.ceil(((_z(1 - alpha / 2) + _z(power)) * sd / effect) ** 2))


def expected_half_width(sd: float, n: int, *, alpha: float) -> float:
    return float(_z(1 - alpha / 2) * sd / math.sqrt(n)) if n >= 2 else math.nan
