"""Conjugate Dirichlet-categorical models of the next-bin event class (M58, M59; H14, H15).

State buckets (registered): spread (1 tick vs wider) x five-level imbalance tercile (< -1/3, middle, > 1/3) x the
current bin's event class (7) = 42 buckets. Within a bucket the next-bin class (7 values, as in the sequence models)
is categorical with a Dirichlet prior; the predictive is the posterior mean.

* sequential posterior updating (H14): counts accumulate day by day with a registered power-prior discount 0.9 per
  day; the frozen model keeps the posterior after the first day;
* hierarchical empirical Bayes (H15): the prior mean is the pooled class frequency of the bucket over earlier days
  and the concentration is chosen per bucket on a registered grid by maximizing the Dirichlet-multinomial marginal
  likelihood of the earlier days' bucket counts.
"""
from __future__ import annotations

import math

import numpy as np

N_CLASSES = 7
N_BUCKETS = 42
DISCOUNT = 0.9
KAPPAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def buckets(X: np.ndarray, y_current: np.ndarray) -> np.ndarray:
    """Bucket per bin from raw (unstandardized) inputs: X[:, 6] spread ticks, X[:, 7] imbalance5."""
    wide = (X[:, 6] > 1.5).astype(int)
    imb = np.digitize(X[:, 7], [-1 / 3, 1 / 3])
    return (wide * 3 + imb) * N_CLASSES + y_current


def current_class(X: np.ndarray) -> np.ndarray:
    c = np.expm1(X[:, :6])
    return np.where(c.sum(1) > 0, 1 + np.argmax(c, 1), 0)


def counts(b: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros((N_BUCKETS, N_CLASSES))
    np.add.at(out, (b, y), 1.0)
    return out


def nll(alpha: np.ndarray, b: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = alpha / alpha.sum(1, keepdims=True)
    return -np.log(np.maximum(p[b, y], 1e-12))


def sequential(day_counts: list[np.ndarray], prior: float = 1.0, discount: float = DISCOUNT) -> list[np.ndarray]:
    """Posterior Dirichlet parameters after each day (power-prior discounting of past counts)."""
    acc = np.zeros((N_BUCKETS, N_CLASSES))
    out = []
    for c in day_counts:
        acc = discount * acc + c
        out.append(prior + acc)
    return out


def _dm_loglik(c: np.ndarray, a: np.ndarray) -> float:
    """Dirichlet-multinomial log marginal likelihood of count vector c under parameters a (constants dropped)."""
    n, A = c.sum(), a.sum()
    return float(math.lgamma(A) - math.lgamma(A + n) + sum(math.lgamma(ai + ci) - math.lgamma(ai) for ai, ci in zip(a, c)))


def empirical_bayes(day_counts: list[np.ndarray]) -> np.ndarray:
    """Hierarchical prior: pooled bucket means and a per-bucket concentration chosen by marginal likelihood."""
    pooled = np.sum(day_counts, axis=0) + 0.5
    mean = pooled / pooled.sum(1, keepdims=True)
    prior = np.zeros_like(mean)
    for k in range(N_BUCKETS):
        best = max(KAPPAS, key=lambda kap: sum(_dm_loglik(c[k], kap * mean[k]) for c in day_counts))
        prior[k] = best * mean[k]
    return prior
