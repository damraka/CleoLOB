"""Cross-time, cross-instrument and cross-venue transfer (M137).

``transfer`` decides whether conclusions established on a source domain hold on target domains: per target domain
(a pool, an instrument, a venue segment) and per conclusion the registered decision rule is applied to the target's
date-cluster values (``research.decide``), Holm over the conclusions. ``heterogeneity`` measures how much the effect
varies across domains (DerSimonian-Laird tau^2 and I^2 of the per-domain cluster means), so "transfers" is never
claimed from a pooled mean that hides a domain where the effect is absent (audit: C3 on SOLUSDT ~ 0).
"""
from __future__ import annotations

import numpy as np

from ...v08.inference.hierarchical import dersimonian_laird
from ..research import decide as DE


def heterogeneity(domain_values: dict[str, dict[str, float]]) -> dict:
    means, var, n = [], [], {}
    for dom, vals in domain_values.items():
        x = np.asarray(list(vals.values()), float)
        if len(x) < 2:
            continue
        means.append(x.mean())
        var.append(x.var(ddof=1) / len(x))
        n[dom] = {"mean": float(x.mean()), "se": float(np.sqrt(x.var(ddof=1) / len(x))), "clusters": int(len(x))}
    if len(means) < 2:
        return {"domains": n, "tau2": None, "i2": None}
    dl = dersimonian_laird(np.asarray(means), np.asarray(var))
    return {"domains": n, "tau2": dl["tau2"], "i2": dl["i2"], "pooled_mean": dl["mu"]}


def transfer(conclusion_values: dict[str, dict[str, dict[str, float]]], thresholds: dict[str, float], *,
             any_: bool = False, seed: int = 0) -> dict:
    """``conclusion_values[domain][conclusion] = {date: value}``."""
    out = {}
    for dom, per in conclusion_values.items():
        out[dom] = DE.decide_transfer(per, {c: thresholds[c] for c in per}, any_=any_, seed=seed)
    by_conclusion = {}
    for c in thresholds:
        by_conclusion[c] = heterogeneity({d: v[c] for d, v in conclusion_values.items() if c in v})
    return {"domains": out, "heterogeneity": by_conclusion,
            "transfers_everywhere": {c: all(out[d]["conclusions"].get(c, {}).get("status") == "ESTABLISHED"
                                            for d in out) for c in thresholds}}


__all__ = ["transfer", "heterogeneity"]
