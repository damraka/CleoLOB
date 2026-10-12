"""Completion-constrained execution (M22) and multi-objective Pareto analysis (M23).

``actual_completion`` is a primary endpoint. ``completion_summary`` reports, per
policy: completion probability with a Wilson interval, deadline risk (not
complete within the horizon), residual-inventory distribution, and economic
cost conditional on and unconditional of completion. Cost comparisons never
reward an incomplete low-cost execution: ``constrained_choice`` minimizes
expected economic cost **subject to** the registered completion threshold on
the Wilson lower bound; candidates that miss it are infeasible regardless of
cost.

``pareto_front`` returns the non-dominated set over registered objectives
(implementation shortfall, completion probability, CVaR, participation, tail
loss, residual inventory, footprint, latency sensitivity). It does not collapse
objectives into an arbitrary scalar.
"""
from __future__ import annotations

import numpy as np

from ..statistics import cvar, wilson

OBJECTIVES = {   # name -> (row field or derived key, direction: +1 minimize, -1 maximize)
    "economic_cost_bps": +1, "completion_probability": -1, "cvar90_cost_bps": +1, "participation": +1,
    "tail_cost_p95_bps": +1, "residual_inventory": +1, "footprint_children": +1, "latency_sensitivity_bps": +1}


def completion_summary(rows: list[dict], *, threshold: float = 0.95) -> dict:
    valid = [r for r in rows if r.get("status") == "VALID"]
    n = len(valid)
    if n == 0:
        return {"n": 0, "status": "NOT_EVALUABLE", "invalid": len(rows)}
    complete = sum(bool(r["actual_completion"]) for r in valid)
    within = sum(bool(r["within_horizon_completion"]) for r in valid)
    lo, hi = wilson(complete, n)
    costs = np.asarray([r["economic_cost_bps"] for r in valid], float)
    comp = np.asarray([bool(r["actual_completion"]) for r in valid])
    residual = np.asarray([r["terminal_inventory"] for r in valid], float)
    return {
        "n": n, "invalid": len(rows) - n, "completion_probability": complete / n, "completion_ci": [lo, hi],
        "deadline_risk": 1 - within / n, "meets_threshold": bool(lo >= threshold), "threshold": threshold,
        "residual_inventory_mean": float(residual.mean()), "residual_inventory_p95": float(np.quantile(residual, 0.95)),
        "economic_cost_mean_bps": float(costs.mean()),
        "economic_cost_given_complete_bps": float(costs[comp].mean()) if comp.any() else None,
        "economic_cost_given_incomplete_bps": float(costs[~comp].mean()) if (~comp).any() else None,
        "optimization_cost_mean_bps": _mean([r["optimization_cost_bps"] for r in valid]),
        "completion_penalty_mean_bps": float(np.mean([r["completion_penalty_bps"] for r in valid])),
        "hypothetical_residual_mean_bps": _mean([r["hypothetical_residual_cost_bps"] for r in valid]),
        "unpriceable_residuals": int(sum(bool(r["residual_unpriceable"]) for r in valid))}


def _mean(values) -> float | None:
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else None


def constrained_choice(summaries: dict[str, dict], *, threshold: float) -> dict:
    """Minimum expected economic cost subject to the completion lower bound >= ``threshold``."""
    feasible = {k: s for k, s in summaries.items() if s.get("n") and s["completion_ci"][0] >= threshold}
    if not feasible:
        return {"choice": None, "status": "NO_FEASIBLE_POLICY", "threshold": threshold}
    choice = min(feasible, key=lambda k: (feasible[k]["economic_cost_mean_bps"], k))
    return {"choice": choice, "status": "FEASIBLE", "threshold": threshold, "feasible": sorted(feasible),
            "infeasible": sorted(set(summaries) - set(feasible))}


def objectives(rows: list[dict], *, latency_rows: list[dict] | None = None) -> dict:
    valid = [r for r in rows if r.get("status") == "VALID"]
    if not valid:
        return {}
    cost = np.asarray([r["economic_cost_bps"] for r in valid], float)
    out = {"economic_cost_bps": float(cost.mean()),
           "completion_probability": float(np.mean([r["actual_completion"] for r in valid])),
           "cvar90_cost_bps": cvar(cost, 0.9), "participation": float(np.mean([r["participation"] for r in valid])),
           "tail_cost_p95_bps": float(np.quantile(cost, 0.95)),
           "residual_inventory": float(np.mean([r["terminal_inventory"] for r in valid])),
           "footprint_children": float(np.mean([r["children"] for r in valid]))}
    if latency_rows:
        slow = np.asarray([r["economic_cost_bps"] for r in latency_rows if r.get("status") == "VALID"], float)
        out["latency_sensitivity_bps"] = float(slow.mean() - cost.mean()) if len(slow) else None
    return out


def pareto_front(points: dict[str, dict], keys: list[str] | None = None) -> dict:
    keys = keys or [k for k in OBJECTIVES if all(points[p].get(k) is not None for p in points)]
    names = sorted(points)

    def dominates(a: dict, b: dict) -> bool:
        better = False
        for k in keys:
            x, y = OBJECTIVES[k] * a[k], OBJECTIVES[k] * b[k]
            if x > y + 1e-12:
                return False
            if x < y - 1e-12:
                better = True
        return better

    front = [n for n in names if not any(dominates(points[m], points[n]) for m in names if m != n)]
    dominated_by = {n: sorted(m for m in names if m != n and dominates(points[m], points[n])) for n in names}
    return {"objectives": keys, "front": front, "dominated": sorted(set(names) - set(front)),
            "dominated_by": dominated_by, "points": points}
