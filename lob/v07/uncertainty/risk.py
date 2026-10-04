"""Risk-sensitive execution metrics and robust selection rules (workstreams 34, 43, 44).

Costs are completion-adjusted bps (lower is better). ``risk_metrics`` summarizes one
policy's cost distribution; ``robust_selection`` applies the registered selection
criteria across a world x policy cube (world means over identical market seeds):

* expectation-optimal  — minimum pooled mean over worlds;
* worst-case           — minimum over policies of the maximum world mean;
* CVaR over worlds     — minimum mean of the worst ``q`` fraction of world means;
* distributionally robust — minimum worst-case weighted mean over world weights within
  total-variation distance ``radius`` of uniform (closed form: shift ``radius`` mass to
  the worst world);
* domain-randomized    — reported separately (learned policies trained across worlds).

None of these is a statement about real markets.
"""
from __future__ import annotations

import math

import numpy as np


def cvar(costs, alpha: float = 0.95) -> float:
    """Mean of the worst (1 - alpha) fraction of costs (higher cost = worse)."""
    x = np.sort(np.asarray(costs, float))
    k = max(1, int(math.ceil(round((1 - alpha) * len(x), 9))))
    return float(x[-k:].mean())


def downside_semivariance(costs) -> float:
    x = np.asarray(costs, float)
    excess = np.maximum(x - x.mean(), 0.0)
    return float(np.mean(excess ** 2))


def risk_metrics(costs, *, completion=None, time_to_completion=None, residual_inventory=None) -> dict:
    x = np.asarray(costs, float)
    x = x[np.isfinite(x)]
    out = {"n": int(len(x)), "mean_bps": float(x.mean()), "sd_bps": float(x.std(ddof=1)) if len(x) > 1 else None,
           "cvar95_bps": cvar(x, 0.95), "cvar99_bps": cvar(x, 0.99), "tail_p95_bps": float(np.quantile(x, 0.95)),
           "downside_semivariance": downside_semivariance(x)}
    if completion is not None:
        c = np.asarray(completion, float)
        out["completion_risk"] = float(np.mean(c < 1.0))
    if time_to_completion is not None:
        t = np.asarray([v for v in time_to_completion if v is not None], float)
        out["fill_time_p90_s"] = float(np.quantile(t, 0.9)) if len(t) else None
    if residual_inventory is not None:
        r = np.asarray(residual_inventory, float)
        out["inventory_risk_mean_abs"] = float(np.mean(np.abs(r)))
    return out


def worst_plausible(world_means: dict) -> dict:
    name = max(world_means, key=world_means.get)
    return {"world": name, "cost_bps": float(world_means[name]),
            "mean_over_worlds_bps": float(np.mean(list(world_means.values())))}


def robust_selection(world_means: dict[str, dict[str, float]], *, q: float = 0.25, radius: float = 0.2) -> dict:
    """``world_means[world][policy]`` -> selected policy under each registered criterion."""
    policies = sorted(next(iter(world_means.values())))
    worlds = sorted(world_means)
    m = np.asarray([[world_means[w][p] for w in worlds] for p in policies])   # policies x worlds
    k = max(1, int(math.ceil(round(q * len(worlds), 9))))
    criteria = {
        "expectation": m.mean(1),
        "worst_case": m.max(1),
        "cvar_worlds": np.sort(m, axis=1)[:, -k:].mean(1),
        "distributionally_robust": (1 - radius) * m.mean(1) + radius * m.max(1),
    }
    selected = {name: policies[int(np.argmin(v))] for name, v in criteria.items()}
    table = {p: {name: float(v[i]) for name, v in criteria.items()} for i, p in enumerate(policies)}
    return {"selected": selected, "table": table, "q": q, "radius": radius, "worlds": len(worlds),
            "agreement": len(set(selected.values())) == 1}
