"""Reference distribution functions without SciPy (Student t, normal, regularized incomplete beta).

The base installation has no SciPy, yet cluster-level inference with few clusters needs exact t quantiles rather
than normal approximations. ``betainc`` uses the Lentz continued fraction (Numerical Recipes 6.4); quantiles are
found by bracketed bisection, so every value is reproducible to ~1e-12.
"""
from __future__ import annotations

import math

_EPS = 3e-16
_FPMIN = 1e-300


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > _FPMIN else _FPMIN)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _FPMIN else _FPMIN)
        c = 1.0 + aa / c
        c = c if abs(c) > _FPMIN else _FPMIN
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > _FPMIN else _FPMIN)
        c = 1.0 + aa / c
        c = c if abs(c) > _FPMIN else _FPMIN
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(ln)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    if df <= 0:
        raise ValueError("df must be positive")
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    x = df / (df + t * t)
    tail = 0.5 * betainc(0.5 * df, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_sf(t: float, df: float) -> float:
    return t_cdf(-t, df)


def _bisect(f, lo: float, hi: float, target: float) -> float:
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def t_ppf(p: float, df: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    if p == 0.5:
        return 0.0
    hi = 1.0
    while t_cdf(hi, df) < max(p, 1 - p):
        hi *= 2.0
        if hi > 1e12:
            break
    x = _bisect(lambda v: t_cdf(v, df), 0.0, hi, max(p, 1 - p))
    return x if p > 0.5 else -x


def norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    hi = 1.0
    while norm_cdf(hi) < max(p, 1 - p):
        hi *= 2.0
    x = _bisect(norm_cdf, 0.0, hi, max(p, 1 - p))
    return x if p > 0.5 else -x


def nct_power(effect_over_se: float, df: float, alpha: float, *, sided: int = 1) -> float:
    """Power of a one-sample t test at noncentrality ``effect/SE`` (shifted-central-t approximation).

    Exact noncentral-t needs a series; the shifted central t (P(T_df > t_crit - delta)) is accurate to a few
    percent for delta < 6 and is conservative for small df, which is the direction a feasibility gate needs.
    """
    crit = t_ppf(1.0 - alpha / sided, df)
    return t_sf(crit - effect_over_se, df)


__all__ = ["betainc", "t_cdf", "t_sf", "t_ppf", "norm_cdf", "norm_ppf", "nct_power"]
