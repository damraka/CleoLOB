"""v0.8 statistical decision framework (M33).

Everything here is dependency-free (numpy + the standard library) and seeded.

* paired, hierarchical (world -> seed) and Bayesian bootstraps;
* exact sign test; Wilson intervals;
* superiority, equivalence (TOST via the (1 - 2 alpha) interval) and
  non-inferiority decisions, kept distinct: "not significant" never becomes
  "equivalent";
* Holm, Bonferroni and Benjamini-Hochberg corrections;
* minimum detectable effect, power and required sample sizes (for a target
  effect or a target interval width);
* group-sequential boundaries (O'Brien-Fleming and Pocock spending, seeded
  Monte Carlo calibration);
* ranking uncertainty, regret distributions and rank correlations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Callable, Sequence

import numpy as np

Z = NormalDist()


def zq(p: float) -> float:
    return Z.inv_cdf(p)


# ============================================================================= intervals


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    se: float
    n: int
    method: str

    def to_dict(self) -> dict:
        return {"estimate": self.estimate, "ci_low": self.low, "ci_high": self.high, "se": self.se, "n": self.n,
                "method": self.method}


def _clean(x) -> np.ndarray:
    a = np.asarray(x, float)
    return a[np.isfinite(a)]


def paired_bootstrap(diffs, *, alpha: float = 0.05, n_boot: int = 4000, seed: int = 0,
                     stat: Callable[[np.ndarray], float] = np.mean) -> Interval:
    d = _clean(diffs)
    if len(d) < 2:
        raise ValueError("need at least two paired differences")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = np.apply_along_axis(stat, 1, d[idx]) if stat is not np.mean else d[idx].mean(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return Interval(float(stat(d)), float(lo), float(hi), float(boots.std(ddof=1)), len(d), "paired percentile bootstrap")


def hierarchical_bootstrap(groups: dict, *, alpha: float = 0.05, n_boot: int = 2000, seed: int = 0) -> Interval:
    """Two-stage bootstrap: resample groups (e.g. worlds), then units (seeds) within each; mean of all units."""
    keys = sorted(groups)
    arrays = [_clean(groups[k]) for k in keys]
    arrays = [a for a in arrays if len(a)]
    if len(arrays) < 2:
        raise ValueError("need at least two non-empty groups")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, len(arrays), len(arrays))
        means = [arrays[g][rng.integers(0, len(arrays[g]), len(arrays[g]))].mean() for g in chosen]
        boots[b] = float(np.mean(means))
    point = float(np.mean([a.mean() for a in arrays]))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return Interval(point, float(lo), float(hi), float(boots.std(ddof=1)), sum(len(a) for a in arrays),
                    "hierarchical bootstrap (groups, then units)")


def bayes_prob_below(diffs, threshold: float = 0.0, *, n_draws: int = 4000, seed: int = 0) -> float:
    """Bayesian-bootstrap posterior probability that the mean difference is below ``threshold``."""
    d = _clean(diffs)
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(len(d)), size=n_draws)
    return float(np.mean(w @ d < threshold))


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z = zq(1 - alpha / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def sign_test(diffs) -> dict:
    """Exact two-sided sign test (zeros dropped)."""
    d = _clean(diffs)
    pos, neg = int(np.sum(d > 0)), int(np.sum(d < 0))
    n = pos + neg
    if n == 0:
        return {"positive": 0, "negative": 0, "p_value": 1.0}
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"positive": pos, "negative": neg, "n": n, "p_value": float(min(1.0, 2 * tail))}


# ============================================================================= decisions


def superiority(interval: Interval, *, threshold: float = 0.0, lower_is_better: bool = True) -> str:
    """ESTABLISHED if the whole interval lies beyond ``threshold`` in the good direction; FAILED if beyond in
    the bad direction (interval excludes zero the wrong way); NOT_ESTABLISHED otherwise."""
    if lower_is_better:
        if interval.high < -threshold:
            return "ESTABLISHED"
        if interval.low > 0:
            return "FAILED"
    else:
        if interval.low > threshold:
            return "ESTABLISHED"
        if interval.high < 0:
            return "FAILED"
    return "NOT_ESTABLISHED"


def equivalence(diffs, margin: float, *, alpha: float = 0.05, n_boot: int = 4000, seed: int = 0) -> dict:
    """TOST via the (1 - 2 alpha) bootstrap interval; never inferred from a non-significant test."""
    d = _clean(diffs)
    if len(d) < 2 or margin <= 0:
        return {"status": "NOT_EVALUABLE", "reason": "fewer than two differences or no margin"}
    iv = paired_bootstrap(d, alpha=2 * alpha, n_boot=n_boot, seed=seed)
    if -margin < iv.low and iv.high < margin:
        status = "EQUIVALENT_WITHIN_MARGIN"
    elif iv.low > margin or iv.high < -margin:
        status = "NOT_EQUIVALENT"
    else:
        status = "NOT_ESTABLISHED"
    return {"status": status, "margin": margin, **iv.to_dict(), "interval_level": 1 - 2 * alpha}


def noninferiority(diffs, margin: float, *, alpha: float = 0.05, lower_is_better: bool = True, n_boot: int = 4000,
                   seed: int = 0) -> dict:
    """New minus reference; lower is better: non-inferior if the one-sided upper bound is below ``margin``."""
    d = _clean(diffs)
    if len(d) < 2:
        return {"status": "NOT_EVALUABLE", "reason": "fewer than two differences"}
    iv = paired_bootstrap(d, alpha=2 * alpha, n_boot=n_boot, seed=seed)
    bound = iv.high if lower_is_better else -iv.low
    lower = iv.low if lower_is_better else -iv.high
    if bound < margin:
        status = "NONINFERIOR_WITHIN_MARGIN"
    elif lower > margin:
        status = "INFERIOR_BEYOND_MARGIN"
    else:
        status = "NOT_ESTABLISHED"
    return {"status": status, "margin": margin, "one_sided_bound": bound, **iv.to_dict()}


def practical(estimate: float, threshold: float) -> bool:
    return abs(estimate) >= threshold


# ============================================================================= multiplicity


def holm(pvalues: Sequence[float], alpha: float) -> list[bool]:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    reject = np.zeros(len(p), bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (len(p) - rank):
            reject[i] = True
        else:
            break
    return reject.tolist()


def bonferroni(pvalues: Sequence[float], alpha: float) -> list[bool]:
    return [p <= alpha / len(pvalues) for p in pvalues]


def benjamini_hochberg(pvalues: Sequence[float], q: float) -> list[bool]:
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    passed = [rank for rank, i in enumerate(order, 1) if p[i] <= q * rank / m]
    cutoff = max(passed) if passed else 0
    reject = np.zeros(m, bool)
    reject[order[:cutoff]] = True
    return reject.tolist()


def adjust(pvalues: Sequence[float], method: str, alpha: float) -> list[bool]:
    return {"holm": holm, "bonferroni": bonferroni, "benjamini_hochberg": benjamini_hochberg,
            "none": lambda p, a: [x <= a for x in p]}[method](pvalues, alpha)


def bootstrap_pvalue(diffs, *, n_boot: int = 4000, seed: int = 0) -> float:
    """Two-sided bootstrap p-value for a zero mean (centred resampling)."""
    d = _clean(diffs)
    rng = np.random.default_rng(seed)
    centred = d - d.mean()
    boots = centred[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float(min(1.0, (np.sum(np.abs(boots) >= abs(d.mean())) + 1) / (n_boot + 1)))


# ============================================================================= power and planning


def mde(sd: float, n: int, *, alpha: float = 0.05, power: float = 0.8, sided: int = 2) -> float:
    if n < 2 or sd <= 0:
        return math.inf
    za = zq(1 - alpha / sided)
    return (za + zq(power)) * sd / math.sqrt(n)


def power_of(effect: float, sd: float, n: int, *, alpha: float = 0.05, sided: int = 2) -> float:
    if sd <= 0:
        return 1.0
    za = zq(1 - alpha / sided)
    return float(Z.cdf(abs(effect) * math.sqrt(n) / sd - za))


def required_n(sd: float, effect: float, *, alpha: float = 0.05, power: float = 0.8, sided: int = 2) -> int:
    if effect <= 0:
        raise ValueError("effect must be positive")
    za = zq(1 - alpha / sided)
    return int(math.ceil(((za + zq(power)) * sd / effect) ** 2))


def required_n_for_width(sd: float, width: float, *, alpha: float = 0.05) -> int:
    """Units so that a normal-theory (1 - alpha) interval has total width at most ``width``."""
    if width <= 0:
        raise ValueError("width must be positive")
    return int(math.ceil((2 * zq(1 - alpha / 2) * sd / width) ** 2))


def sequential_boundaries(looks: int, alpha: float = 0.05, *, kind: str = "obrien_fleming", n_paths: int = 200_000,
                          seed: int = 0) -> list[float]:
    """Two-sided z boundaries for ``looks`` equally spaced analyses (Lan-DeMets spending, Monte Carlo)."""
    if kind not in ("obrien_fleming", "pocock"):
        raise ValueError("kind must be obrien_fleming or pocock")
    t = np.arange(1, looks + 1) / looks
    if kind == "obrien_fleming":
        spent = 2 * (1 - np.asarray([Z.cdf(zq(1 - alpha / 2) / math.sqrt(x)) for x in t]))
    else:
        spent = alpha * np.log(1 + (math.e - 1) * t)
    rng = np.random.default_rng(seed)
    increments = rng.standard_normal((n_paths, looks)) * math.sqrt(1.0 / looks)
    paths = np.cumsum(increments, 1) / np.sqrt(t)
    alive = np.ones(n_paths, bool)
    bounds = []
    used = 0.0
    for k in range(looks):
        target = spent[k] - used
        z = np.abs(paths[alive, k])
        needed = int(round(target * n_paths))
        if needed <= 0 or len(z) == 0:
            bound = math.inf
        else:
            bound = float(np.sort(z)[::-1][min(needed, len(z)) - 1])
        crossed = np.zeros(n_paths, bool)
        crossed[alive] = np.abs(paths[alive, k]) >= bound
        used += crossed.sum() / n_paths
        alive &= ~crossed
        bounds.append(bound)
    return bounds


# ============================================================================= rankings and regret


def ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (1 = lowest), ties averaged."""
    v = np.asarray(values, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b) -> float:
    ra, rb = ranks(a), ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def kendall_tau(a, b) -> float:
    """Kendall tau-b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    concordant = discordant = ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif da == db:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_a) * (concordant + discordant + ties_b))
    return float((concordant - discordant) / denominator) if denominator > 0 else 0.0


def rank_uncertainty(costs: np.ndarray, names: Sequence[str], *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap over units (rows) of a units x policies cost matrix: rank distribution and P(best)."""
    c = np.asarray(costs, float)
    rng = np.random.default_rng(seed)
    counts = np.zeros((c.shape[1], c.shape[1]))
    for _ in range(n_boot):
        mean = c[rng.integers(0, len(c), len(c))].mean(0)
        r = ranks(mean).round().astype(int) - 1
        counts[np.arange(c.shape[1]), np.clip(r, 0, c.shape[1] - 1)] += 1
    prob = counts / n_boot
    return {n: {"rank_probabilities": prob[i].tolist(), "p_best": float(prob[i, 0]),
                "expected_rank": float(np.dot(prob[i], np.arange(1, c.shape[1] + 1)))}
            for i, n in enumerate(names)}


def regret(costs: np.ndarray) -> np.ndarray:
    """Per-unit regret of every column relative to the best column on that unit."""
    c = np.asarray(costs, float)
    return c - c.min(1, keepdims=True)


def cvar(values, level: float = 0.9, weights=None) -> float:
    """Mean of the worst (1 - level) tail (higher values are worse); weighted if ``weights`` is given."""
    v = np.asarray(values, float)
    w = np.ones(len(v)) / len(v) if weights is None else np.asarray(weights, float) / np.sum(weights)
    order = np.argsort(v)[::-1]
    v, w = v[order], w[order]
    tail = 1.0 - level
    cum = np.cumsum(w)
    take = np.minimum(w, np.maximum(tail - (cum - w), 0.0))
    return float(np.dot(take, v) / tail) if tail > 0 else float(v[0])


def summarize(values) -> dict:
    v = _clean(values)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "p05": float(np.quantile(v, 0.05)), "median": float(np.median(v)), "p95": float(np.quantile(v, 0.95)),
            "max": float(v.max()), "cvar90": cvar(v, 0.9)}
