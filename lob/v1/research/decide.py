"""Mechanical hypothesis decisions under the frozen v1 statistics (M6, M130).

Input per hypothesis: date-cluster values of its registered unit value (one value per calendar date). The
registered rule (``configs/v1/protocol.json`` ``statistics``):

* the decision quantity is ``value - threshold`` (superiority), ``value + margin`` against 0 (non-inferiority),
  ``value - level`` against the threshold (agreement), or both one-sided TOST tests (equivalence);
* primary: one-sided cluster-t (CR1 SE, t on G - 1 df) p-value; Holm within the family at alpha 0.05;
* secondary: exact cluster sign-flip p-value, compared with the same Holm-adjusted level;
* ESTABLISHED = primary Holm rejection AND sign-flip p <= the hypothesis's Holm step level, with G >= min_clusters;
* FAILED = the upper 95% cluster-t bound of the decision quantity is below its threshold;
* INCONCLUSIVE otherwise; NOT_EVALUABLE when G < min_clusters;
* transfer hypotheses decide each carried conclusion (Holm over conclusions); ``transfer`` needs every conclusion,
  ``transfer_any`` at least one;
* resolution of a detail the frozen text leaves open, fixed before any v1 pilot outcome existed: in a family that
  mixes single-endpoint and transfer members, single-endpoint members use Holm with m = the full family size
  (the conservative reading); transfer members use Holm over their own conclusions.
"""
from __future__ import annotations


import numpy as np

from .. import statistics as ST

MIN_CLUSTERS = 8
ALPHA = 0.05


def quantity(h: dict, values: dict[str, float]) -> tuple[dict[str, float], float]:
    """(shifted cluster values, threshold the shifted values are tested against)."""
    t = h["test"]
    if t == "noninferiority":
        return {k: v + float(h["margin"]) for k, v in values.items()}, 0.0
    if t == "agreement":
        return {k: v - float(h["level"]) for k, v in values.items()}, float(h["threshold"])
    return dict(values), float(h["threshold"] if h["threshold"] is not None else 0.0)


def test_one(values: dict[str, float], threshold: float, *, seed: int = 0) -> dict:
    keys = sorted(values)
    x = [values[k] for k in keys]
    g = len(keys)
    if g < MIN_CLUSTERS:
        return {"clusters": g, "status": "NOT_EVALUABLE", "reason": f"{g} < {MIN_CLUSTERS} date clusters"}
    ct = ST.cluster_t(x, keys, threshold=threshold)
    sf = ST.sign_flip(x, keys, threshold=threshold, seed=seed)
    upper_one = ct.estimate + ST.t_ppf(0.975, g - 1) * ct.se
    return {"clusters": g, "estimate": ct.estimate, "se": ct.se, "ci_low": ct.ci_low, "ci_high": ct.ci_high,
            "threshold": threshold, "p_cluster_t": ct.p_one_sided, "p_sign_flip": sf["p_one_sided"],
            "sign_flip_floor": sf["p_floor"], "failed": bool(upper_one < threshold)}


def equivalence(values: dict[str, float], margin: float, *, seed: int = 0) -> dict:
    keys = sorted(values)
    x = np.asarray([values[k] for k in keys], float)
    g = len(keys)
    if g < MIN_CLUSTERS:
        return {"clusters": g, "status": "NOT_EVALUABLE"}
    lo = ST.cluster_t(-x, keys, threshold=-margin)       # H0: mean >= margin  ->  -mean <= -margin
    hi = ST.cluster_t(x, keys, threshold=-margin)        # H0: mean <= -margin
    sf_lo = ST.sign_flip(list(-x), keys, threshold=-margin, seed=seed)
    sf_hi = ST.sign_flip(list(x), keys, threshold=-margin, seed=seed)
    p = max(lo.p_one_sided, hi.p_one_sided)
    return {"clusters": g, "estimate": float(x.mean()), "ci_low": hi.ci_low, "ci_high": hi.ci_high, "margin": margin,
            "p_cluster_t": p, "p_sign_flip": max(sf_lo["p_one_sided"], sf_hi["p_one_sided"]),
            "sign_flip_floor": sf_lo["p_floor"], "failed": bool(abs(x.mean()) > margin and p > 0.5)}


def holm_levels(pvals: dict[str, float], alpha: float = ALPHA, m: int | None = None) -> dict[str, float]:
    """The Holm step level each hypothesis is compared with: alpha / (m - rank), m >= number of p-values."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m = max(len(order), m or 0)
    return {k: alpha / (m - i) for i, k in enumerate(order)}


def holm_adjusted(pvals: dict[str, float], m: int | None = None) -> dict[str, float]:
    order = sorted(pvals, key=lambda k: pvals[k])
    m = max(len(order), m or 0)
    out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


def family(results: dict[str, dict], alpha: float = ALPHA, m: int | None = None) -> dict[str, dict]:
    """Holm over the tested members with family size ``m`` (default: the number tested); statuses + step levels."""
    tested = {k: r for k, r in results.items() if "p_cluster_t" in r}
    ps = {k: r["p_cluster_t"] for k, r in tested.items()}
    adj = holm_adjusted(ps, m) if tested else {}
    levels = holm_levels(ps, alpha, m) if tested else {}
    out = {}
    for k, r in results.items():
        if "p_cluster_t" not in r:
            out[k] = {**r, "status": r.get("status", "NOT_EVALUABLE")}
            continue
        est = adj[k] <= alpha and r["p_sign_flip"] <= levels[k]
        status = "ESTABLISHED" if est else ("FAILED" if r["failed"] else "INCONCLUSIVE")
        out[k] = {**r, "p_holm": adj[k], "holm_step_level": levels[k], "status": status}
    return out


def decide_transfer(conclusion_values: dict[str, dict[str, float]], thresholds: dict[str, float], *, any_: bool,
                    seed: int = 0) -> dict:
    per = {c: test_one(v, thresholds[c], seed=seed) for c, v in conclusion_values.items()}
    fam = family(per)
    n_est = sum(r["status"] == "ESTABLISHED" for r in fam.values())
    if any(r.get("status") == "NOT_EVALUABLE" for r in fam.values()) and n_est == 0:
        status = "NOT_EVALUABLE"
    elif (n_est >= 1) if any_ else (n_est == len(fam)):
        status = "ESTABLISHED"
    elif all(r["status"] == "FAILED" for r in fam.values()):
        status = "FAILED"
    else:
        status = "INCONCLUSIVE"
    return {"conclusions": fam, "status": status, "rule": "transfer_any" if any_ else "transfer",
            "established_conclusions": sorted(c for c, r in fam.items() if r["status"] == "ESTABLISHED")}


def decide_family(hypotheses: dict[str, dict], cluster_values: dict[str, dict], *, seed: int = 0) -> dict:
    """``cluster_values[h]``: date -> value (or, for transfer tests, conclusion -> date -> value)."""
    single, out = {}, {}
    for hid, h in hypotheses.items():
        vals = cluster_values.get(hid)
        if vals is None:
            out[hid] = {"status": "NOT_EVALUABLE", "reason": "no confirmatory values"}
            continue
        if h["test"] in ("transfer", "transfer_any"):
            thr = {c: float(h["thresholds"][c]) for c in vals}
            out[hid] = decide_transfer(vals, thr, any_=h["test"] == "transfer_any", seed=seed)
            continue
        if h["test"] == "equivalence":
            single[hid] = equivalence(vals, float(h["margin"]), seed=seed)
            continue
        shifted, thr = quantity(h, vals)
        single[hid] = test_one(shifted, thr, seed=seed)
    # registered resolution (fixed before any pilot outcome): single-endpoint members are Holm-adjusted with the FULL
    # family size m (transfer members included, so the adjustment is never smaller than registered); transfer
    # members are decided by Holm over their own conclusions
    out.update(family(single, m=len(hypotheses)))
    return out


__all__ = ["MIN_CLUSTERS", "quantity", "test_one", "equivalence", "family", "decide_transfer", "decide_family",
           "holm_levels"]
