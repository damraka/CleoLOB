"""Paired model contrasts on historical blocks for the generic and the execution-sensitive objective.

Resampling follows the v0.6 ``bootstrap_realism`` rule exactly: each draw
resamples historical 600 s blocks once (iid multinomial counts; shared by every
model, so contrasts are paired on history) and each model's simulated units
(seeds, or posterior (draw, seed) units) independently. Point estimates use all
blocks and units once. Every draw computes the full component comparison, from
which both objectives are read:

* generic objective — the sealed v0.6 objective (mean family error / development
  half scale);
* execution-sensitive (ES) objective — the mean over the registered ES components
  (H10) of component error / max(sealed real-vs-real component error, 0.05).
"""
from __future__ import annotations

import math

import numpy as np

from ...v06.inference import bootstrap_pvalue, counts, interval
from ...v06.observables import pool, stack, weighted
from ...v06.realism import compare

ES_COMPONENTS = ("spread", "depth_l1", "depletion_hazard", "replenishment_probability", "trade_size",
                 "acf_signed_volume_10s_l1", "acf_signed_volume_10s_l2", "acf_signed_volume_10s_l5")
SCALE_FLOOR = 0.05


def component_scales(frozen: dict) -> dict:
    """Sealed real-vs-real component errors from the v0.6 design (the ES normalization)."""
    rvr = frozen["real_vs_real"]["components"]
    return {name: max(float(rvr[name]["error"]), SCALE_FLOOR) if rvr.get(name, {}).get("error") is not None
            else 1.0 for name in ES_COMPONENTS}


def es_objective(result: dict, scales: dict) -> float | None:
    values = [result["components"][c]["error"] / scales[c] for c in ES_COMPONENTS
              if result["components"].get(c, {}).get("error") is not None]
    return float(np.mean(values)) if values else None


def _metrics(result: dict, es_scales: dict) -> tuple[float, float]:
    g, e = result["objective"], es_objective(result, es_scales)
    return (math.inf if g is None else g), (math.inf if e is None else e)


def bootstrap(hist_blocks: list[dict], sims: dict[str, list[dict]], design: dict, scales: dict, es_scales: dict, *,
              samples: int, seed: int, alpha: float, contrasts: tuple[tuple[str, str], ...]) -> dict:
    if len(hist_blocks) < 2 or any(len(v) < 2 for v in sims.values()):
        raise ValueError("at least two historical blocks and two simulated units per model are required")
    hist_stack = stack(hist_blocks)
    sim_stacks = {k: stack(v) for k, v in sims.items()}
    h_full = pool(hist_blocks)
    point = {k: _metrics(compare(h_full, pool(v), design, scales), es_scales) for k, v in sims.items()}
    rng = np.random.default_rng(seed)
    draws = {k: np.empty((samples, 2)) for k in sims}
    for b in range(samples):
        h = weighted(hist_stack, counts(len(hist_blocks), rng))
        for name, matrix in sim_stacks.items():
            s = weighted(matrix, counts(len(sims[name]), rng))
            draws[name][b] = _metrics(compare(h, s, design, scales), es_scales)
    models = {}
    for name in sims:
        g_low, g_high = interval(draws[name][:, 0], alpha)
        e_low, e_high = interval(draws[name][:, 1], alpha)
        models[name] = {"objective": point[name][0], "objective_ci": [g_low, g_high],
                        "es_objective": point[name][1], "es_ci": [e_low, e_high], "units": len(sims[name])}
    out = {}
    for left, right in contrasts:
        entry = {}
        for j, metric in enumerate(("objective", "es_objective")):
            diff = draws[left][:, j] - draws[right][:, j]
            low, high = interval(diff, alpha)
            entry[metric] = {"difference": point[left][j] - point[right][j], "ci_low": low, "ci_high": high,
                             "p_value": bootstrap_pvalue(diff), "one_sided_upper": float(np.nanquantile(diff, 1 - alpha)),
                             "one_sided_lower": float(np.nanquantile(diff, alpha))}
        out[f"{left}-{right}"] = entry
    return {"models": models, "contrasts": out, "alpha": alpha, "samples": samples, "blocks": len(hist_blocks)}


def improvement(entry: dict) -> str:
    """Directional rule: ESTABLISHED if the interval lies below zero, FAILED if above, else NOT_ESTABLISHED."""
    if not (math.isfinite(entry["ci_low"]) and math.isfinite(entry["ci_high"])):
        return "INVALID"
    return "ESTABLISHED" if entry["ci_high"] < 0 else "FAILED" if entry["ci_low"] > 0 else "NOT_ESTABLISHED"


def noninferiority(entry: dict, delta: float) -> str:
    """H11 (one-sided at the family alpha): ESTABLISHED if the one-sided upper bound is below +delta,
    FAILED if the one-sided lower bound exceeds it."""
    upper, lower = entry["one_sided_upper"], entry["one_sided_lower"]
    if not (math.isfinite(upper) and math.isfinite(lower)):
        return "INVALID"
    return "ESTABLISHED" if upper < delta else "FAILED" if lower > delta else "NOT_ESTABLISHED"


def conjunction(statuses: list[str]) -> str:
    if any(s == "NOT_AVAILABLE" for s in statuses):
        return "NOT_AVAILABLE" if all(s == "NOT_AVAILABLE" for s in statuses) else "NOT_ESTABLISHED"
    if any(s == "INVALID" for s in statuses):
        return "INVALID"
    if all(s == "ESTABLISHED" for s in statuses):
        return "ESTABLISHED"
    return "FAILED" if any(s == "FAILED" for s in statuses) else "NOT_ESTABLISHED"
