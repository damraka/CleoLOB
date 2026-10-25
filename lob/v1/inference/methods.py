"""Simulation-based inference (M92), simulation-based calibration (M93), active simulation by expected information
gain (M95), parameter and model-class identifiability (M96) and posterior predictive checks (M97).

* ``abc_smc``: the v0.8 sequential Monte Carlo ABC (adaptive tolerance, population annealing) on a ``Problem``
  (generator template, box prior, observed summaries), with regression adjustment - the registered SBI method.
* ``sbc``: Talts et al. (2018): draw theta* from the prior, simulate data, run the inference, record the rank of
  theta* among L posterior draws; uniform ranks (chi-square over bins and the KS distance) indicate a calibrated
  posterior. ``conjugate_poisson_posterior`` is an exact reference inference used to validate the SBC machinery.
* ``active_design``: among candidate simulation designs (parameter points), choose the one with the largest expected
  reduction of posterior variance under a local-linear Gaussian approximation (the EIG of a linear-Gaussian model,
  0.5 log det(I + J Sigma J^T R^-1)), from pilot simulations.
* ``fisher``: numerical Fisher information of the summary means (central differences x summary covariance) -> rank,
  condition number and the weakly identified directions; ``model_class_confusion``: simulate from each class, select
  by a scoring rule, report the confusion matrix (a class that is not recovered from its own data is not
  identifiable from that summary set).
* ``ppc``: posterior predictive p-values per summary (share of replicated summaries at least as extreme).
"""
from __future__ import annotations

import math

import numpy as np

from ...v08.inference import abc as ABC

Prior = ABC.Prior
Problem = ABC.Problem


def abc_smc(problem: ABC.Problem, *, particles: int = 64, generations: int = 4, adjust: bool = True) -> dict:
    post = ABC.smc(problem, particles=particles, generations=generations)
    if adjust:
        post = ABC.regression_adjust(post, problem)
    return post


def conjugate_poisson_posterior(counts: np.ndarray, *, a0: float = 2.0, b0: float = 1.0, draws: int = 200,
                                seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.gamma(a0 + counts.sum(), 1.0 / (b0 + len(counts)), size=draws)


def sbc(prior_draw, simulate, infer, *, n: int = 200, draws: int = 99, bins: int = 10, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ranks = []
    for i in range(n):
        theta = prior_draw(rng)
        data = simulate(theta, rng)
        post = np.asarray(infer(data, draws=draws, seed=seed + i + 1))
        ranks.append(int((post < theta).sum()))
    ranks = np.asarray(ranks)
    hist = np.histogram(ranks, bins=bins, range=(0, draws + 1))[0]
    exp = n / bins
    chi2 = float(((hist - exp) ** 2 / exp).sum())
    u = np.sort((ranks + 0.5) / (draws + 1))
    ks = float(np.max(np.abs(u - (np.arange(1, n + 1) - 0.5) / n)))
    # chi-square 95% quantile with bins - 1 df (Wilson-Hilferty approximation, no SciPy)
    k = bins - 1
    crit = k * (1 - 2 / (9 * k) + 1.6448536 * math.sqrt(2 / (9 * k))) ** 3
    return {"ranks": ranks.tolist(), "histogram": hist.tolist(), "chi2": chi2, "chi2_crit_95": crit,
            "ks": ks, "calibrated": bool(chi2 < crit), "n": n, "draws": draws}


def fisher(summary_mean, theta: np.ndarray, *, cov: np.ndarray, rel_step: float = 0.05) -> dict:
    theta = np.asarray(theta, float)
    J = []
    for j in range(len(theta)):
        h = rel_step * max(abs(theta[j]), 1e-3)
        up, dn = theta.copy(), theta.copy()
        up[j] += h
        dn[j] -= h
        J.append((np.asarray(summary_mean(up)) - np.asarray(summary_mean(dn))) / (2 * h))
    J = np.asarray(J).T                                   # (summaries, params)
    inv = np.linalg.pinv(np.asarray(cov, float))
    F = J.T @ inv @ J
    ev, vecs = np.linalg.eigh(F)
    return {"fisher": F.tolist(), "eigenvalues": ev.tolist(), "rank": int((ev > 1e-8 * max(ev.max(), 1e-300)).sum()),
            "condition": float(ev.max() / max(ev.min(), 1e-300)), "weakest_direction": vecs[:, 0].tolist()}


def model_class_confusion(classes: dict, simulate, score, *, reps: int = 10, seed: int = 0) -> dict:
    """``simulate(cls, rng)`` -> data; ``score(cls, data)`` -> fit quality (higher better)."""
    names = list(classes)
    M = np.zeros((len(names), len(names)), int)
    rng = np.random.default_rng(seed)
    for i, c in enumerate(names):
        for _ in range(reps):
            data = simulate(classes[c], rng)
            j = int(np.argmax([score(classes[o], data) for o in names]))
            M[i, j] += 1
    return {"classes": names, "confusion": M.tolist(), "recovery": (np.diag(M) / reps).tolist(),
            "identifiable": bool(np.all(np.diag(M) / reps >= 0.8))}


def active_design(candidates: np.ndarray, jacobian, prior_cov: np.ndarray, noise_cov: np.ndarray) -> dict:
    """EIG of each candidate design under a local-linear Gaussian model; returns the argmax and all gains."""
    S = np.asarray(prior_cov, float)
    Rinv = np.linalg.inv(np.asarray(noise_cov, float))
    gains = []
    for c in candidates:
        J = np.atleast_2d(jacobian(c))
        M = np.eye(J.shape[0]) + J @ S @ J.T @ Rinv
        gains.append(0.5 * float(np.linalg.slogdet(M)[1]))
    return {"gains": gains, "best": int(np.argmax(gains))}


def ppc(observed: np.ndarray, replicated: np.ndarray, names=None) -> dict:
    obs = np.asarray(observed, float)
    rep = np.asarray(replicated, float)
    med = np.median(rep, 0)
    p = ((np.abs(rep - med) >= np.abs(obs - med)).mean(0))
    names = names or [f"s{i}" for i in range(len(obs))]
    return {"p_values": dict(zip(names, p.tolist())), "flagged": [n for n, x in zip(names, p) if x < 0.05],
            "replicates": int(len(rep))}


__all__ = ["Prior", "Problem", "abc_smc", "sbc", "conjugate_poisson_posterior", "fisher", "model_class_confusion",
           "active_design", "ppc"]
