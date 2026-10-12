"""Identifiability v3 (M15).

* ``local`` — finite-difference Jacobian of the summary means with respect to
  each parameter (log scale for positive parameters, common random numbers),
  standardized by the summary noise at the reference point; the Gaussian
  synthetic-likelihood Fisher approximation ``F = J'J / replicates`` gives
  eigenvalues, effective rank, participation ratio, sloppiness (log10 eigenvalue
  span) and condition number. Valid only locally and only under the Gaussian
  summary approximation (stated in the output).
* ``global_indices`` — first-order variance indices of every summary with
  respect to every parameter over the prior (binning estimator).
* ``profile`` — profile-distance proxy from an ABC population (minimum distance
  per parameter bin); a flat profile signals practical non-identifiability.
* ``posterior_diagnostics`` — contraction, correlation and mode counts from a
  posterior population.
* ``STRUCTURAL_FLAGS`` — known structural weak-identification issues per family.

``assess`` combines them; parameters are ``PRACTICALLY_IDENTIFIED`` only when the
posterior contracts (>= 0.5), the profile is not flat and the local Fisher
direction is not sloppy. Otherwise ``WEAKLY_IDENTIFIED`` or ``NOT_IDENTIFIED``.
"""
from __future__ import annotations

import math

import numpy as np

from ..generators.base import MarketGenerator
from .abc import Prior, _seed, simulate_summary

STRUCTURAL_FLAGS = {
    "poisson": [],
    "hawkes": ["baseline rate and branching ratio trade off in the mean event rate (mean = mu / (1 - branching)); "
               "separation relies on clustering summaries"],
    "queue_reactive": ["queue elasticities are identified only through variation of touch queue sizes in the data"],
    "regime_switching": ["regime labels are exchangeable (label switching); transition rates weakly identified "
                         "when regimes are short relative to the window"],
    "state_dependent": ["coefficients of rarely varying features (e.g. spread in one-tick markets) are weakly "
                        "identified"],
    "mixture": ["component weights and components are not jointly identified from a single path"],
    "ecology": ["agent intensities with similar rate signatures are confounded (additive rates)"],
    "endogenous": ["response parameters are not identified from data without strategy flow"],
}


def local(generator: MarketGenerator, prior: Prior, theta0: np.ndarray, *, n_bins: int, replicates: int, seed: int,
          summary_index: list[int], rel_step: float = 0.15, map_fn=map) -> dict:
    names = list(prior.names)
    tmpl = generator.to_dict()
    theta0 = np.asarray(theta0, float)

    def run(theta: np.ndarray, stage: int) -> np.ndarray:
        args = [(tmpl, names, list(map(float, theta)), n_bins, _seed(seed, 500, r)) for r in range(replicates)]
        return np.asarray(list(map_fn(_call, args)), float)[:, summary_index]

    base = run(theta0, 0)
    noise = base.std(0, ddof=1)
    noise = np.where(noise > 1e-9, noise, 1e-9 + np.abs(base.mean(0)) * 1e-3)
    jac = np.zeros((len(summary_index), len(names)))
    for j in range(len(names)):
        up, down = theta0.copy(), theta0.copy()
        if prior.log[j]:
            up[j] *= math.exp(rel_step)
            down[j] *= math.exp(-rel_step)
            step = 2 * rel_step
        else:
            h = rel_step * (prior.high[j] - prior.low[j])
            up[j] = min(up[j] + h / 2, prior.high[j])
            down[j] = max(down[j] - h / 2, prior.low[j])
            step = (up[j] - down[j]) / (prior.high[j] - prior.low[j])
        jac[:, j] = (run(up, 1).mean(0) - run(down, 2).mean(0)) / step
    std_jac = jac / noise[:, None]
    fisher = std_jac.T @ std_jac
    eig = np.sort(np.clip(np.linalg.eigvalsh(fisher), 0.0, None))[::-1]
    top = eig[0] if len(eig) and eig[0] > 0 else 1.0
    rank = int(np.sum(eig > 1e-3 * top))
    pr = float(eig.sum() ** 2 / (eig ** 2).sum()) if (eig ** 2).sum() > 0 else 0.0
    positive = eig[eig > 1e-12]
    return {"names": names, "theta0": theta0.tolist(), "jacobian_standardized": std_jac.tolist(),
            "fisher": fisher.tolist(), "eigenvalues": eig.tolist(), "effective_rank": rank,
            "participation_ratio": pr,
            "sloppiness_log10": float(math.log10(positive[0] / positive[-1])) if len(positive) > 1 else 0.0,
            "condition_number": float(positive[0] / positive[-1]) if len(positive) > 1 else math.inf,
            "per_parameter_information": np.diag(fisher).tolist(),
            "validity": "local, Gaussian synthetic-likelihood approximation at theta0 (log-scale steps for positive "
                        "parameters); not a global identification result"}


def _call(args):
    return simulate_summary(*args)


def global_indices(thetas: np.ndarray, summaries: np.ndarray, names: list[str], bins: int = 8) -> dict:
    """First-order indices Var(E[S | theta_j]) / Var(S) by quantile binning (bias-corrected, floored at 0)."""
    thetas, summaries = np.asarray(thetas, float), np.asarray(summaries, float)
    n = len(thetas)
    out = {}
    for j, name in enumerate(names):
        edges = np.quantile(thetas[:, j], np.linspace(0, 1, bins + 1)[1:-1])
        b = np.searchsorted(edges, thetas[:, j])
        row = []
        for k in range(summaries.shape[1]):
            y = summaries[:, k]
            total = y.var()
            if total <= 0:
                row.append(0.0)
                continue
            means = np.asarray([y[b == g].mean() if np.any(b == g) else y.mean() for g in range(bins)])
            counts = np.asarray([np.sum(b == g) for g in range(bins)])
            between = float(np.sum(counts * (means - y.mean()) ** 2) / n)
            within = float(np.mean([y[b == g].var(ddof=1) / max(np.sum(b == g), 1) for g in range(bins)
                                    if np.sum(b == g) > 1]))
            row.append(max(0.0, (between - within) / total))
        out[name] = row
    return {"first_order": out, "estimator": "binning (quantile bins), bias-corrected"}


def profile(posterior: dict, bins: int = 10) -> dict:
    theta = np.asarray(posterior["particles"], float)
    d = np.asarray(posterior["distances"], float)
    out = {}
    if not np.all(np.isfinite(d)):
        return {n: {"status": "NOT_EVALUABLE", "reason": "population has no distances"} for n in posterior["names"]}
    prior = Prior(**{k: tuple(v) for k, v in posterior["prior"].items()})
    u = prior.to_unit(theta)
    for j, name in enumerate(posterior["names"]):
        b = np.clip((u[:, j] * bins).astype(int), 0, bins - 1)
        curve = [float(d[b == k].min()) if np.any(b == k) else None for k in range(bins)]
        values = [c for c in curve if c is not None]
        span = (max(values) - min(values)) / (np.median(values) or 1.0) if len(values) > 1 else 0.0
        occupied = sum(c is not None for c in curve)
        out[name] = {"profile": curve, "relative_span": float(span), "occupied_bins": occupied,
                     "flat": bool(span < 0.1 and occupied >= bins // 2)}
    return out


def posterior_diagnostics(posterior: dict) -> dict:
    s = posterior["summary"]
    return {"contraction": {n: p["contraction"] for n, p in s["parameters"].items()},
            "modes": {n: p["modes"] for n, p in s["parameters"].items()},
            "correlation": s.get("correlation"), "ess": s["ess"]}


def assess(family: str, local_result: dict | None, posterior: dict | None, profile_result: dict | None) -> dict:
    names = (local_result or posterior or {}).get("names", [])
    verdicts = {}
    for j, name in enumerate(names):
        contraction = posterior["summary"]["parameters"][name]["contraction"] if posterior else None
        flat = profile_result.get(name, {}).get("flat") if profile_result else None
        info = local_result["per_parameter_information"][j] if local_result else None
        if contraction is None:
            verdict = "NOT_EVALUABLE"
        elif contraction >= 0.5 and not flat and (info is None or info >= 1.0):
            verdict = "PRACTICALLY_IDENTIFIED"
        elif contraction >= 0.2:
            verdict = "WEAKLY_IDENTIFIED"
        else:
            verdict = "NOT_IDENTIFIED"
        verdicts[name] = {"verdict": verdict, "contraction": contraction, "profile_flat": flat,
                          "fisher_information": info}
    return {"family": family, "parameters": verdicts, "structural_flags": STRUCTURAL_FLAGS.get(family, []),
            "local": local_result and {k: local_result[k] for k in ("effective_rank", "participation_ratio",
                                                                    "sloppiness_log10", "condition_number",
                                                                    "eigenvalues", "validity")}}
