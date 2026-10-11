"""Uncertainty decomposition (M18).

Sources: aleatoric (simulation seed), parameter (posterior sample), calibration
window, model family, queue position rule, latency profile, hidden-liquidity
sensitivity and regime. A *design* lists worlds, each with a level for every
source, and several seeds per world (common seeds across worlds).

For a quantity ``y`` (a policy cost, or a paired cost difference between two
policies) per (world, seed):

* aleatoric variance = mean within-world variance over seeds;
* epistemic variance = variance of world means minus aleatoric / seeds (floored at 0);
* each source's first-order share = variance of the world-mean grouped by that
  source's level (bias-corrected for the finite number of worlds per level),
  divided by the total; the remainder is interaction/residual.

Bootstrap intervals resample worlds. The decomposition depends on the
registered design (which levels exist); it is reported with that design.
"""
from __future__ import annotations

import numpy as np

SOURCES = ("parameter", "calibration_window", "model_family", "queue", "latency", "hidden_liquidity", "regime")


def decompose(values: np.ndarray, levels: dict[str, list], *, n_boot: int = 500, seed: int = 0) -> dict:
    """``values``: worlds x seeds; ``levels``: source -> per-world level labels (sources may be absent)."""
    y = np.asarray(values, float)
    out = _decompose(y, levels)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        boots.append(_decompose(y[idx], {k: [v[i] for i in idx] for k, v in levels.items()}))
    for key in ("aleatoric_var", "epistemic_var", "epistemic_share", *[f"share_{s}" for s in levels]):
        vals = np.asarray([b[key] for b in boots], float)
        out[f"{key}_ci"] = [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
    out["n_boot"] = n_boot
    return out


def _decompose(y: np.ndarray, levels: dict[str, list]) -> dict:
    worlds, seeds = y.shape
    world_means = y.mean(1)
    aleatoric = float(np.mean(y.var(1, ddof=1))) if seeds > 1 else 0.0
    epistemic = max(0.0, float(world_means.var(ddof=1)) - aleatoric / seeds) if worlds > 1 else 0.0
    total = aleatoric + epistemic
    out = {"worlds": worlds, "seeds": seeds, "aleatoric_var": aleatoric, "epistemic_var": epistemic,
           "total_var": total, "epistemic_share": epistemic / total if total > 0 else 0.0,
           "epistemic_sd": float(np.sqrt(epistemic))}
    explained = 0.0
    for source, labels in levels.items():
        labels = np.asarray(labels)
        groups = [world_means[labels == g] for g in np.unique(labels)]
        if len(groups) < 2:
            out[f"share_{source}"] = 0.0
            continue
        between = float(np.sum([len(g) * (g.mean() - world_means.mean()) ** 2 for g in groups]) / worlds)
        within = float(np.mean([g.var(ddof=1) / len(g) for g in groups if len(g) > 1])) if any(
            len(g) > 1 for g in groups) else 0.0
        share = max(0.0, between - within) / total if total > 0 else 0.0
        out[f"share_{source}"] = share
        explained += share
    out["share_aleatoric"] = aleatoric / total if total > 0 else 0.0
    out["share_interaction_residual"] = max(0.0, out["epistemic_share"] - explained)
    return out


def summarize_policy_costs(rows: list[dict], *, value: str, policies: list[str], sources: list[str],
                           n_boot: int = 300, seed: int = 0) -> dict:
    """Decompose per-policy values and pairwise differences from flat episode rows."""
    worlds = sorted({r["world_id"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    meta = {r["world_id"]: r for r in rows}
    table = {(r["world_id"], r["seed"], r["policy"]): r[value] for r in rows if r.get(value) is not None}
    levels = {s: [meta[w].get(f"level_{s}") for w in worlds] for s in sources}
    out = {"policies": {}, "pairs": {}, "design": {"worlds": len(worlds), "seeds": len(seeds), "sources": sources}}
    for p in policies:
        y = np.asarray([[table.get((w, s, p), np.nan) for s in seeds] for w in worlds], float)
        if np.isnan(y).any():
            out["policies"][p] = {"status": "INVALID", "reason": "missing episodes"}
            continue
        out["policies"][p] = decompose(y, levels, n_boot=n_boot, seed=seed)
    for i, a in enumerate(policies):
        for b in policies[i + 1:]:
            y = np.asarray([[table.get((w, s, a), np.nan) - table.get((w, s, b), np.nan) for s in seeds]
                            for w in worlds], float)
            if not np.isnan(y).any():
                out["pairs"][f"{a}-{b}"] = decompose(y, levels, n_boot=n_boot, seed=seed)
    return out
