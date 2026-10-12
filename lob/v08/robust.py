"""Robust selection over registered plausible worlds (M24).

Input: a candidate x world matrix of world-mean costs (lower is better), world
weights (posterior probabilities) and optionally a candidate x world matrix of
completion probabilities. Criteria:

* ``nominal`` — best in the registered nominal world;
* ``posterior_expected`` — minimum weighted mean;
* ``worst_case`` — minimax cost;
* ``minimax_regret`` — minimum of the maximum regret ``C[c, w] - min_c' C[c', w]``;
* ``cvar`` — minimum weighted CVaR of world costs at the registered level;
* ``kl_dro`` — distributionally robust expectation over a KL ball of radius
  ``rho`` around the world weights (dual: ``min_lam lam rho + lam log E_p exp(C / lam)``);
* any criterion ``+ completion``: candidates whose completion probability is
  below the threshold in any world (or in expectation, ``completion_mode``) are
  removed first.

Worlds come only from registered world sets. ``select_all`` reports every
criterion's choice with its value, so disagreement between criteria is visible.
"""
from __future__ import annotations

import math

import numpy as np

from .statistics import cvar

CRITERIA = ("nominal", "posterior_expected", "worst_case", "minimax_regret", "cvar", "kl_dro")


def kl_dro_value(costs: np.ndarray, weights: np.ndarray, rho: float) -> float:
    c = np.asarray(costs, float)
    p = np.asarray(weights, float) / np.sum(weights)
    if rho <= 0:
        return float(np.dot(p, c))
    best = math.inf
    span = max(float(c.max() - c.min()), 1e-6)
    for lam in np.geomspace(span * 1e-3, span * 1e3, 300):
        top = c.max()
        value = lam * rho + lam * math.log(float(np.dot(p, np.exp((c - top) / lam)))) + top
        best = min(best, value)
    return float(min(best, c.max()))


def criterion_values(cost: np.ndarray, weights: np.ndarray, *, nominal: int = 0, level: float = 0.8,
                     rho: float = 0.1) -> dict[str, np.ndarray]:
    c = np.asarray(cost, float)
    w = np.asarray(weights, float) / np.sum(weights)
    regret = c - c.min(0, keepdims=True)
    return {"nominal": c[:, nominal], "posterior_expected": c @ w, "worst_case": c.max(1),
            "minimax_regret": regret.max(1), "cvar": np.asarray([cvar(row, level, w) for row in c]),
            "kl_dro": np.asarray([kl_dro_value(row, w, rho) for row in c])}


def select_all(names: list[str], cost: np.ndarray, weights: np.ndarray, *, completion: np.ndarray | None = None,
               completion_threshold: float | None = None, completion_mode: str = "every_world", nominal: int = 0,
               level: float = 0.8, rho: float = 0.1) -> dict:
    c = np.asarray(cost, float)
    feasible = np.ones(len(names), bool)
    if completion is not None and completion_threshold is not None:
        comp = np.asarray(completion, float)
        w = np.asarray(weights, float) / np.sum(weights)
        feasible = (comp.min(1) >= completion_threshold) if completion_mode == "every_world" else \
            (comp @ w >= completion_threshold)
    values = criterion_values(c, weights, nominal=nominal, level=level, rho=rho)
    out = {"criteria": {}, "feasible": [n for n, f in zip(names, feasible) if f],
           "completion_threshold": completion_threshold, "level": level, "rho": rho}
    for crit, v in values.items():
        if not feasible.any():
            out["criteria"][crit] = {"choice": None, "status": "NO_FEASIBLE_CANDIDATE"}
            continue
        masked = np.where(feasible, v, np.inf)
        i = int(np.argmin(masked))
        out["criteria"][crit] = {"choice": names[i], "value": float(v[i]),
                                 "values": {n: float(x) for n, x in zip(names, v)}}
    choices = {k: v["choice"] for k, v in out["criteria"].items()}
    out["criteria_agree"] = len(set(choices.values())) == 1
    return out


def evaluate_choice(choice_costs: np.ndarray, reference_costs: np.ndarray, *, level: float = 0.8) -> dict:
    """Out-of-sample world-mean costs of a chosen candidate vs a reference candidate (per world, paired)."""
    a, b = np.asarray(choice_costs, float), np.asarray(reference_costs, float)
    return {"mean_diff": float((a - b).mean()), "worst_choice": float(a.max()), "worst_reference": float(b.max()),
            "cvar_choice": cvar(a, level), "cvar_reference": cvar(b, level)}
