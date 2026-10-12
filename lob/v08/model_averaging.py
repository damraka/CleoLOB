"""Model-class uncertainty and dynamic model averaging (M17).

Given fitted generators (one per family) and per-bin one-step predictive
log-likelihoods on evaluation frames:

* ``posterior_model_probabilities`` — BIC approximation to marginal likelihoods
  (equal prior model probabilities);
* ``information_weights`` — Akaike weights;
* ``stacking_weights`` — weights maximizing the held-out mixture log score (EM
  on the simplex), i.e. holdout-predictive weights;
* ``regime_weights`` — stacking within each regime label;
* ``dynamic`` — dynamic model averaging (Raftery-style forgetting): the weight
  path ``w_t ∝ w_{t-1}^alpha p_k(y_t)`` with forgetting ``alpha``, giving a
  sequential mixture predictive;
* ``mixture`` — a ``MixtureGenerator`` for mixture posterior predictive sampling.

``material`` reports model uncertainty as material when no family holds at
least the registered 0.9 weight; the code never forces a single generator when
that is the case.
"""
from __future__ import annotations

import math

import numpy as np

from .generators.families import MixtureGenerator

MATERIAL_MAX_WEIGHT = 0.9


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    e = np.exp(x - x.max())
    return e / e.sum()


def posterior_model_probabilities(loglik: dict[str, float], n_params: dict[str, int], n: int) -> dict[str, float]:
    names = sorted(loglik)
    bic = np.asarray([loglik[k] - 0.5 * n_params[k] * math.log(max(n, 2)) for k in names])
    return dict(zip(names, _softmax(bic).tolist()))


def information_weights(loglik: dict[str, float], n_params: dict[str, int]) -> dict[str, float]:
    names = sorted(loglik)
    return dict(zip(names, _softmax(np.asarray([loglik[k] - n_params[k] for k in names])).tolist()))


def stacking_weights(component_ll: np.ndarray, *, iterations: int = 500) -> np.ndarray:
    """EM for mixture weights given per-bin component log predictive densities (rows: bins)."""
    ll = np.asarray(component_ll, float)
    ll = ll[np.all(np.isfinite(ll), 1)]
    k = ll.shape[1]
    w = np.full(k, 1.0 / k)
    dens = np.exp(ll - ll.max(1, keepdims=True))
    for _ in range(iterations):
        r = dens * w
        r /= r.sum(1, keepdims=True)
        new = r.mean(0)
        if np.max(np.abs(new - w)) < 1e-9:
            w = new
            break
        w = new
    return w


def mixture_score(component_ll: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ll = np.asarray(component_ll, float)
    top = ll.max(1, keepdims=True)
    return (np.log(np.exp(ll - top) @ np.asarray(weights, float)) + top[:, 0])


def dynamic(component_ll: np.ndarray, *, alpha: float = 0.99, prior: np.ndarray | None = None) -> dict:
    """Dynamic model averaging with forgetting ``alpha``; returns predictive scores and the weight path."""
    ll = np.asarray(component_ll, float)
    k = ll.shape[1]
    w = np.full(k, 1.0 / k) if prior is None else np.asarray(prior, float) / np.sum(prior)
    scores = np.full(len(ll), np.nan)
    path = np.zeros((len(ll), k))
    for t in range(len(ll)):
        pred = w ** alpha
        pred /= pred.sum()
        path[t] = pred
        row = ll[t]
        if not np.all(np.isfinite(row)):
            w = pred
            continue
        top = row.max()
        dens = pred * np.exp(row - top)
        scores[t] = math.log(dens.sum()) + top
        w = dens / dens.sum()
    return {"scores": scores, "path": path, "alpha": alpha}


def regime_weights(component_ll: np.ndarray, labels: np.ndarray) -> dict[int, list[float]]:
    out = {}
    for lab in sorted(set(int(x) for x in labels if x >= 0)):
        rows = component_ll[np.asarray(labels) == lab]
        if len(rows) >= 50:
            out[lab] = stacking_weights(rows).tolist()
    return out


def material(weights: dict[str, float] | np.ndarray, threshold: float = MATERIAL_MAX_WEIGHT) -> dict:
    values = np.asarray(list(weights.values()) if isinstance(weights, dict) else weights, float)
    top = float(values.max())
    entropy = float(-(values[values > 0] * np.log(values[values > 0])).sum())
    return {"max_weight": top, "material": bool(top < threshold), "entropy": entropy, "threshold": threshold,
            "selection_allowed": bool(top >= threshold)}


def mixture(generators: list, weights) -> MixtureGenerator:
    return MixtureGenerator.of(generators, list(map(float, weights)))
