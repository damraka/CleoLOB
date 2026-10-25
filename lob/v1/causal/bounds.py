"""Historical counterfactual partial identification (M135) and the causal estimand registry (M136).

Partial identification. For one unit (a day, a window) the effect of treatment vs control is evaluated under every
assumption of a registered grid (queue rule x latency x hidden liquidity x impact x participant reaction). The true
world is one unknown grid point, so the identified set is the interval [min_a d_a, max_a d_a]; a direction is
IDENTIFIED only when the whole interval lies on one side of zero (otherwise INDETERMINATE - reported as an
indeterminate ranking, never broken by a default). ``breakdown`` reports, for an identified direction, how many
grid dimensions must be moved to their least favourable level before the sign is no longer identified (a
robustness value), and which dimension drives the interval width (variance decomposition over grid factors).

``manski_fill`` gives assumption-free bounds for a passive order's fill outcome: with fill probability only known to
lie in [p_lo, p_hi] (L2 queue bounds), the expected cost lies between the costs at the two extremes.

Estimand registry. Every causal question CleoLOB studies is registered with its identification status
(``IDENTIFIED`` / ``ASSUMPTION_DEPENDENT`` / ``NOT_IDENTIFIED``), the assumptions it needs and the diagnostic that
checks them; ``check`` refuses a claim whose estimand is not at least assumption-dependent and whose required
diagnostics are missing.
"""
from __future__ import annotations

import itertools

import numpy as np

STATUSES = ("IDENTIFIED", "ASSUMPTION_DEPENDENT", "NOT_IDENTIFIED")


def identified_set(effects: dict[str, float]) -> dict:
    vals = np.asarray([v for v in effects.values() if v is not None and np.isfinite(v)], float)
    if not len(vals):
        return {"status": "NOT_EVALUABLE", "lower": None, "upper": None}
    lo, hi = float(vals.min()), float(vals.max())
    status = "IDENTIFIED" if lo > 0 or hi < 0 else "INDETERMINATE"
    return {"lower": lo, "upper": hi, "status": status, "direction": 1 if lo > 0 else -1 if hi < 0 else 0,
            "assumptions": int(len(vals))}


def factor_importance(effects: dict[str, float], factors: dict[str, dict[str, str]]) -> dict:
    """Share of the effect's variance across the grid explained by each factor (main-effect decomposition).

    ``factors[assumption_key] = {factor: level}``.
    """
    keys = [k for k in effects if effects[k] is not None and np.isfinite(effects[k])]
    y = np.asarray([effects[k] for k in keys], float)
    total = float(y.var())
    out = {}
    names = sorted({f for k in keys for f in factors[k]})
    for f in names:
        levels = {}
        for k, v in zip(keys, y):
            levels.setdefault(factors[k][f], []).append(v)
        between = sum(len(v) * (np.mean(v) - y.mean()) ** 2 for v in levels.values()) / len(y)
        out[f] = float(between / total) if total > 0 else 0.0
    return {"variance": total, "share": out, "dominant": max(out, key=out.get) if out else None}


def breakdown(effects: dict[str, float], factors: dict[str, dict[str, str]], least_favourable: dict[str, str]) -> dict:
    """Number of factors that must be set to their least favourable level before the sign stops being identified."""
    base = identified_set(effects)
    if base["status"] != "IDENTIFIED":
        return {"robustness": 0, "status": base["status"]}
    sign = base["direction"]
    names = sorted(least_favourable)
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            sub = [v for k, v in effects.items()
                   if v is not None and all(factors[k][f] == least_favourable[f] for f in combo)]
            if sub and (min(sub) * sign <= 0 or max(sub) * sign <= 0):
                return {"robustness": r - 1, "breaking_factors": list(combo), "status": "IDENTIFIED"}
    return {"robustness": len(names), "status": "IDENTIFIED", "breaking_factors": None}


def manski_fill(cost_if_filled: float, cost_if_unfilled: float, p_lo: float, p_hi: float) -> dict:
    a = p_lo * cost_if_filled + (1 - p_lo) * cost_if_unfilled
    b = p_hi * cost_if_filled + (1 - p_hi) * cost_if_unfilled
    return {"lower": float(min(a, b)), "upper": float(max(a, b)),
            "label": "assumption-free bound given only P(fill) in [p_lo, p_hi]"}


ESTIMANDS = {
    "policy_cost_difference_historical": {
        "question": "cost difference of two execution policies had they traded on a recorded day",
        "status": "ASSUMPTION_DEPENDENT", "assumptions": ["queue rule", "latency", "hidden liquidity", "impact rule",
                                                          "participant reaction"],
        "diagnostics": ["identified_set", "factor_importance", "breakdown"]},
    "passive_fill_probability": {
        "question": "probability a hypothetical resting order would have filled", "status": "ASSUMPTION_DEPENDENT",
        "assumptions": ["queue position (bounded on L2, observed on L3)", "hidden liquidity"],
        "diagnostics": ["l2_queue_bounds", "manski_fill"]},
    "queue_position": {"question": "lots ahead of a resting order", "status": "IDENTIFIED",
                       "assumptions": ["L3 order ids", "census priority for census orders"],
                       "diagnostics": ["mbo_reconciliation"], "requires_capability": "exact_order_id"},
    "own_price_impact": {"question": "causal effect of an own order on later prices", "status": "NOT_IDENTIFIED",
                         "assumptions": ["no confounding of flow and information (fails observationally)"],
                         "diagnostics": [], "note": "observational impact estimates are associations"},
    "cross_instrument_impact": {"question": "effect of flow in A on the price of B", "status": "NOT_IDENTIFIED",
                                "assumptions": ["no common information shocks"], "diagnostics": []},
    "synthetic_policy_ranking": {"question": "ranking of policies in a simulator", "status": "ASSUMPTION_DEPENDENT",
                                 "assumptions": ["the simulator's mechanics"], "diagnostics": ["realism_gate"]},
}


def check(estimand: str, *, diagnostics: dict | None = None, capability: dict | None = None) -> dict:
    e = ESTIMANDS[estimand]
    if e["status"] == "NOT_IDENTIFIED":
        return {"allowed": False, "reason": f"{estimand} is NOT_IDENTIFIED", "status": e["status"]}
    need = e.get("requires_capability")
    if need and (capability or {}).get("queue_observability") != need:
        return {"allowed": False, "reason": f"needs capability {need}", "status": e["status"]}
    missing = [d for d in e["diagnostics"] if d not in (diagnostics or {})]
    if missing:
        return {"allowed": False, "reason": f"missing diagnostics {missing}", "status": e["status"]}
    return {"allowed": True, "status": e["status"]}


__all__ = ["identified_set", "factor_importance", "breakdown", "manski_fill", "ESTIMANDS", "check", "STATUSES"]
