"""Final decision benchmark and cross-cutting analyses (workstreams 19, 94-100).

All functions are pure transformations of sealed results (no new simulation):

* ``realism_to_decision`` — Spearman association, across plausible worlds, between a world's realism error
  and how far its policy-pair differences deviate from the pooled conclusion (associational only);
* ``ablations`` — each sophisticated component against its simpler registered comparison;
* ``complexity_penalty`` — parameters, compute and development/selection/fresh scores per family; a family whose
  added capacity does not improve the fresh score over a simpler family is recorded as not justified;
* ``overfitting_curve`` — development -> selection -> fresh deterioration ordered by capacity;
* ``scaling`` (exploratory) — rank association of realism with parameter count and compute;
* ``compute_value`` — compute against realism gain and changed conclusions;
* ``failure_catalogue`` — registered failure modes linked to the experiments that exhibit them;
* ``decision_benchmark`` — per policy pair: pooled difference, interval, margin, MDE, world sign fractions,
  indeterminate fraction, worst plausible, regime and queue sensitivity, both historical bounds, and the
  certification status (abstention preferred over unsupported certainty).
"""
from __future__ import annotations

import numpy as np

from ..robustness.certification import Evidence, certify


def _rank(x):
    x = np.asarray(x, float)
    order = np.argsort(np.argsort(x))
    return order.astype(float)


def spearman(a, b) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return None
    return float(np.corrcoef(_rank(a[ok]), _rank(b[ok]))[0, 1])


def realism_to_decision(edges: dict, world_realism: dict[str, float]) -> dict:
    """``edges``: H8 edges with per_world means; ``world_realism``: world -> realism objective (lower better)."""
    out = {}
    for pair, edge in edges.items():
        worlds = [w for w in edge["per_world"] if w in world_realism]
        dev = [abs(edge["per_world"][w]["mean"] - edge["pooled_mean_bps"]) for w in worlds]
        out[pair] = {"spearman_error_vs_deviation": spearman([world_realism[w] for w in worlds], dev), "worlds": len(worlds)}
    values = [v["spearman_error_vs_deviation"] for v in out.values() if v["spearman_error_vs_deviation"] is not None]
    verdict = ("UNRESOLVED" if len(values) < 3 else "PREDICTIVE" if np.median(values) > 0.3 else
               "UNINFORMATIVE" if abs(np.median(values)) < 0.1 else "UNRESOLVED")
    return {"pairs": out, "median_spearman": float(np.median(values)) if values else None, "verdict": verdict,
            "interpretation": "associational; few worlds; no causal claim"}


def ablations(table: dict) -> dict:
    """``table``: component -> {'complex': value, 'simple': value, 'lower_is_better': bool, 'status': str}."""
    out = {}
    for name, t in table.items():
        c, s = t.get("complex"), t.get("simple")
        if c is None or s is None:
            out[name] = {**t, "attribution": "NOT_AVAILABLE"}
            continue
        better = c < s if t.get("lower_is_better", True) else c > s
        out[name] = {**t, "difference": c - s, "attribution": "component helps" if better else "component does not help"}
    return out


def complexity_penalty(models: dict[str, dict]) -> dict:
    """``models``: name -> {parameters, compute_s, development, selection, fresh}. Ordered by parameters."""
    ordered = sorted(models, key=lambda m: models[m].get("parameters") or 0)
    out = {}
    best_fresh = None
    for name in ordered:
        m = models[name]
        fresh = m.get("fresh")
        justified = None if fresh is None or best_fresh is None else fresh < best_fresh
        if fresh is not None and (best_fresh is None or fresh < best_fresh):
            best_fresh = fresh
        out[name] = {**m, "capacity_justified_on_fresh_data": justified,
                     "status": "NOT_ESTABLISHED" if justified is False else ("NOT_AVAILABLE" if fresh is None else
                                                                             "ESTABLISHED" if justified else "BASELINE")}
    return {"models": out, "order": ordered,
            "rule": "a family is justified only if its fresh score beats every simpler family's"}


def overfitting_curve(models: dict[str, dict]) -> dict:
    out = {}
    for name, m in sorted(models.items(), key=lambda kv: kv[1].get("parameters") or 0):
        stages = [m.get(k) for k in ("development", "selection", "validation", "fresh")]
        known = [s for s in stages if s is not None]
        out[name] = {"stages": dict(zip(("development", "selection", "validation", "fresh"), stages)),
                     "deterioration_dev_to_last": (known[-1] - known[0]) if len(known) > 1 else None,
                     "parameters": m.get("parameters")}
    return out


def scaling(models: dict[str, dict], key: str = "fresh") -> dict:
    names = [n for n in models if models[n].get(key) is not None and models[n].get("parameters")]
    return {"spearman_parameters_vs_score": spearman([models[n]["parameters"] for n in names],
                                                     [models[n][key] for n in names]),
            "spearman_compute_vs_score": spearman([models[n].get("compute_s") or np.nan for n in names],
                                                  [models[n][key] for n in names]),
            "models": len(names), "status": "EXPLORATORY"}


def compute_value(models: dict[str, dict], baseline: str) -> dict:
    base = models[baseline]
    out = {}
    for name, m in models.items():
        if name == baseline:
            continue
        gain = (base.get("fresh") - m["fresh"]) if base.get("fresh") is not None and m.get("fresh") is not None else None
        extra = (m.get("compute_s") or 0) - (base.get("compute_s") or 0)
        out[name] = {"extra_compute_s": extra, "fresh_realism_gain": gain,
                     "changed_conclusion": m.get("changed_conclusion"),
                     "verdict": ("expensive complexity did not matter" if gain is not None and gain <= 0 else
                                 "gain" if gain is not None else "NOT_AVAILABLE")}
    return out


def failure_catalogue(evidence: dict[str, list[str]]) -> dict:
    from ..protocol.taxonomy import FAILURE_MODES
    return {mode: {"experiments": evidence.get(mode, []), "observed": bool(evidence.get(mode))} for mode in FAILURE_MODES}


def decision_benchmark(edges: dict, *, margin: float, mde: float | None, worst: dict, history: dict | None,
                       regime_world: str | None = None, queue_width_bps: float | None = None) -> dict:
    """One row per policy pair, ending in the certification status."""
    rows = {}
    for pair, e in edges.items():
        per = e["per_world"]
        signs = [w["direction"] for w in per.values()]
        n = len(signs)
        hist = (history or {}).get(pair)
        hist_dirs = None
        if hist:
            hist_dirs = {m: hist[m].get("direction", 0) if isinstance(hist[m], dict) else
                         (0 if hist[m] is None else int(np.sign(hist[m]))) for m in ("conservative", "optimistic") if m in hist}
        evidence = Evidence(registered=True, estimate=e["pooled_mean_bps"], ci=tuple(e["ci"]), margin=margin, mde=mde,
                            opposite_worlds=e["opposite_worlds"],
                            worst_plausible=(worst.get(pair) or {}).get("difference"),
                            regime_signs=[per[regime_world]["direction"]] if regime_world in per else None,
                            history=hist_dirs)
        cert = certify(evidence)
        rows[pair] = {"mean_difference_bps": e["pooled_mean_bps"], "interval": e["ci"], "margin_bps": margin,
                      "mde_bps": mde, "fraction_worlds_negative": signs.count(-1) / n if n else None,
                      "fraction_worlds_positive": signs.count(1) / n if n else None,
                      "fraction_indeterminate": signs.count(0) / n if n else None,
                      "worst_plausible": worst.get(pair), "regime_sensitivity": per.get(regime_world),
                      "queue_sensitivity_bps": queue_width_bps,
                      "historical_conservative": (hist or {}).get("conservative"),
                      "historical_optimistic": (hist or {}).get("optimistic"),
                      "h8_edge": e["edge"], "final_status": cert["status"], "certification": cert}
    counts = {}
    for r in rows.values():
        counts[r["final_status"]] = counts.get(r["final_status"], 0) + 1
    return {"pairs": rows, "status_counts": counts}
