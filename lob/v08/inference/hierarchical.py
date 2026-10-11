"""Hierarchical calibration with partial pooling (M13).

Group-level parameter estimates (posterior mean and variance in the prior's free
space) are pooled with a normal random-effects model, ``theta_g ~ N(mu, tau^2)``,
with ``tau^2`` estimated by DerSimonian-Laird. Levels nest (global -> instrument
-> venue -> regime -> day): ``pool_tree`` pools the leaves of each parent, then
pools the parents, keeping the posterior structure explicit at every level.

Shrunk estimates: ``(tau^2 m_g + v_g mu) / (tau^2 + v_g)`` with variance
``tau^2 v_g / (tau^2 + v_g)``. Nothing here claims identification: every node
carries the contraction of its source posterior, and nodes whose posterior did
not contract (``contraction < 0.5``) are flagged ``NOT_IDENTIFIED``.
"""
from __future__ import annotations

import numpy as np

LEVELS = ("global", "instrument", "venue", "regime", "day")


def dersimonian_laird(means: np.ndarray, variances: np.ndarray) -> dict:
    m, v = np.asarray(means, float), np.maximum(np.asarray(variances, float), 1e-12)
    k = len(m)
    if k < 2:
        return {"mu": float(m[0]) if k else float("nan"), "tau2": 0.0, "q": 0.0, "k": k, "i2": 0.0}
    w = 1 / v
    mu_fixed = float(np.sum(w * m) / w.sum())
    q = float(np.sum(w * (m - mu_fixed) ** 2))
    c = float(w.sum() - (w * w).sum() / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    ws = 1 / (v + tau2)
    mu = float(np.sum(ws * m) / ws.sum())
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return {"mu": mu, "tau2": tau2, "mu_se": float(np.sqrt(1 / ws.sum())), "q": q, "k": k, "i2": i2}


def shrink(means, variances, pooled: dict) -> tuple[np.ndarray, np.ndarray]:
    m, v = np.asarray(means, float), np.maximum(np.asarray(variances, float), 1e-12)
    tau2 = pooled["tau2"]
    if tau2 <= 0:
        return np.full(len(m), pooled["mu"]), np.full(len(m), pooled.get("mu_se", 0.0) ** 2)
    return (tau2 * m + v * pooled["mu"]) / (tau2 + v), tau2 * v / (tau2 + v)


def pool_parameters(groups: dict[str, dict]) -> dict:
    """``groups``: name -> {"mean": {param: m}, "var": {param: v}, "contraction": {param: c}}."""
    names = sorted(groups)
    params = sorted(next(iter(groups.values()))["mean"])
    out = {"groups": {}, "pooled": {}}
    for p in params:
        means = [groups[g]["mean"][p] for g in names]
        varis = [groups[g]["var"][p] for g in names]
        pooled = dersimonian_laird(means, varis)
        sm, sv = shrink(means, varis, pooled)
        out["pooled"][p] = pooled
        for g, a, b in zip(names, sm, sv):
            node = out["groups"].setdefault(g, {"shrunk_mean": {}, "shrunk_var": {}, "identification": {}})
            node["shrunk_mean"][p] = float(a)
            node["shrunk_var"][p] = float(b)
            c = groups[g].get("contraction", {}).get(p)
            node["identification"][p] = "NOT_IDENTIFIED" if c is None or c < 0.5 else "CONTRACTED"
    return out


def pool_tree(leaves: list[dict], levels: tuple[str, ...]) -> dict:
    """Recursive partial pooling. ``leaves``: dicts with keys for every level in ``levels`` plus mean/var.

    Returns the tree: each internal node holds the pooled estimate of its children (treated as groups whose
    mean/var are the children's pooled mu / mu_se^2) and the shrunk estimates of each child.
    """
    if any(level not in LEVELS for level in levels):
        raise ValueError(f"levels must be drawn from {LEVELS}")

    def build(items: list[dict], depth: int) -> dict:
        if depth == len(levels):
            if len(items) != 1:
                raise ValueError("leaf keys must be unique along the hierarchy")
            leaf = items[0]
            return {"mean": leaf["mean"], "var": leaf["var"], "contraction": leaf.get("contraction", {}),
                    "leaf": True}
        key = levels[depth]
        children = {}
        for name in sorted({str(i[key]) for i in items}):
            children[name] = build([i for i in items if str(i[key]) == name], depth + 1)
        groups = {n: {"mean": c["mean"], "var": c["var"], "contraction": c.get("contraction", {})}
                  for n, c in children.items()}
        pooled = pool_parameters(groups)
        mean = {p: v["mu"] for p, v in pooled["pooled"].items()}
        var = {p: v.get("mu_se", 0.0) ** 2 + v["tau2"] for p, v in pooled["pooled"].items()}
        contraction = {p: float(np.mean([c.get("contraction", {}).get(p, 0.0) for c in children.values()]))
                       for p in mean}
        return {"level": key, "children": children, "pooling": pooled, "mean": mean, "var": var,
                "contraction": contraction}

    return build(leaves, 0)
