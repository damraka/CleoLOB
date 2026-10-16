"""Registered confirmatory decision rules (statistics section of the frozen protocol).

Unit values are oriented so that positive favours the registered direction. Inference is a paired percentile
bootstrap over units (4000 resamples, seeds from the analysis namespace); the one-sided bootstrap p-value of
H0: mean <= 0 feeds Holm over the promoted members of each registered family. Rules (hypotheses-candidates.json):

* superiority: ESTABLISHED iff Holm rejects (the adjusted one-sided lower bound is > 0) and the estimate >= the
  practical threshold; FAILED iff the two-sided 95% upper bound is < 0; otherwise NOT_ESTABLISHED;
* noninferiority (margin m): the same test on (value + m): ESTABLISHED iff the adjusted lower bound is > -m;
  FAILED iff the upper bound is < -m;
* agreement (level L, threshold t): the test on (agreement - L): ESTABLISHED iff Holm rejects and the estimate
  >= L + t; FAILED iff the upper bound of (agreement - L) is < 0;
* transfer: carried conclusions on the target pool, each with its own rule on its conservative (worst-assumption)
  bound, Holm over the carried conclusions; ESTABLISHED iff at least one meets its rule; FAILED iff every carried
  conclusion is determinate in the opposite direction (the upper bound of its most favourable-assumption value is
  < 0); NOT_AVAILABLE iff nothing was carried or no target unit is evaluable.
"""
from __future__ import annotations

import math

import numpy as np

from ..v08.statistics import holm, paired_bootstrap

N_BOOT = 4000
ALPHA = 0.05


def clean(values) -> np.ndarray:
    a = np.asarray([v for v in values if v is not None], float)
    return a[np.isfinite(a)]


def one_sided_p(values, *, seed: int, n_boot: int = N_BOOT) -> float:
    """Bootstrap p-value of H0: mean <= 0 (centred resampling; small-sample +1 correction)."""
    d = clean(values)
    if len(d) < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    centred = d - d.mean()
    boots = centred[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float(min(1.0, (np.sum(boots >= d.mean()) + 1) / (n_boot + 1)))


def interval(values, *, seed: int, alpha: float = ALPHA) -> dict | None:
    d = clean(values)
    if len(d) < 2:
        return None
    return paired_bootstrap(d, alpha=alpha, n_boot=N_BOOT, seed=seed).to_dict()


def test_values(test: str, values, *, threshold: float, level: float = 0.0) -> np.ndarray:
    """Shift unit values to the quantity whose mean is tested against zero."""
    d = clean(values)
    if test == "noninferiority":
        return d + threshold
    if test == "agreement":
        return d - level
    return d


def evaluate(test: str, values, *, threshold: float, level: float = 0.0, seed: int) -> dict:
    """Per-hypothesis statistics before multiplicity adjustment."""
    raw = clean(values)
    shifted = test_values(test, raw, threshold=threshold, level=level)
    if len(shifted) < 2:
        return {"status": "NOT_EVALUABLE", "units": int(len(raw)), "reason": "fewer than two evaluable units"}
    iv = interval(shifted, seed=seed)
    return {"units": int(len(raw)), "estimate": float(raw.mean()), "tested_mean": float(shifted.mean()),
            "interval_95": iv, "p_one_sided": one_sided_p(shifted, seed=seed + 1), "threshold": threshold,
            "level": level, "test": test}


def decide(stats: dict, *, rejected: bool) -> str:
    if stats.get("status") == "NOT_EVALUABLE":
        return "NOT_EVALUABLE"
    test, est, upper = stats["test"], stats["estimate"], stats["interval_95"]["ci_high"]
    if test == "noninferiority":
        if rejected:
            return "ESTABLISHED"
        return "FAILED" if upper < 0 else "NOT_ESTABLISHED"         # upper bound of (value + margin) < 0
    if test == "agreement":
        if rejected and est >= stats["level"] + stats["threshold"]:
            return "ESTABLISHED"
        return "FAILED" if upper < 0 else "NOT_ESTABLISHED"
    if rejected and est >= stats["threshold"]:
        return "ESTABLISHED"
    return "FAILED" if upper < 0 else "NOT_ESTABLISHED"


def family(stats: dict[str, dict], *, alpha: float = ALPHA, correction: str = "holm") -> dict[str, dict]:
    """Holm over the evaluable members of one family; returns stats with ``holm_rejected`` and ``status``."""
    keys = [k for k, s in stats.items() if s.get("status") != "NOT_EVALUABLE"]
    p = [stats[k]["p_one_sided"] for k in keys]
    if correction != "holm":
        raise ValueError("confirmatory families use Holm")
    rej = dict(zip(keys, holm(p, alpha) if p else []))
    out = {}
    for k, s in stats.items():
        r = bool(rej.get(k, False))
        out[k] = {**s, "holm_rejected": r, "family_size": len(keys), "status": decide(s, rejected=r)}
    return out


def holm_adjusted(p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    m = len(p)
    order = np.argsort(p)
    adj = np.zeros(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def conclusion_stats(units: dict, threshold: float, *, seed: int) -> dict:
    """A carried conclusion on one pool: conservative-bound test plus the favourable-assumption upper bound."""
    lower = clean([u["lower"] for u in units.values()])
    upper = clean([u["upper"] for u in units.values()])
    if len(lower) < 2:
        return {"status": "NOT_EVALUABLE", "units": int(len(lower)), "reason": "fewer than two evaluable units"}
    s = evaluate("superiority", lower, threshold=threshold, seed=seed)
    up = interval(upper, seed=seed + 2)
    s["optimistic_estimate"] = float(upper.mean())
    s["optimistic_interval_95"] = up
    s["opposite_determinate"] = bool(up is not None and up["ci_high"] < 0)
    s["identified_fraction"] = float(np.mean([u["status"] == "IDENTIFIED" and u["lower"] > 0 for u in units.values()]))
    return s


def transfer(per_conclusion: dict[str, dict], *, alpha: float = ALPHA) -> dict:
    """Registered transfer rule over carried conclusions (Holm across them)."""
    if not per_conclusion:
        return {"status": "NOT_AVAILABLE", "reason": "no conclusion was carried", "conclusions": {}}
    keys = [k for k, s in per_conclusion.items() if s.get("status") != "NOT_EVALUABLE"]
    if not keys:
        return {"status": "NOT_AVAILABLE", "reason": "no evaluable target units", "conclusions": per_conclusion}
    p = [per_conclusion[k]["p_one_sided"] for k in keys]
    rej = dict(zip(keys, holm(p, alpha)))
    adj = dict(zip(keys, holm_adjusted(p)))
    out = {}
    for k, s in per_conclusion.items():
        if k not in rej:
            out[k] = {**s, "meets_rule": False}
            continue
        meets = bool(rej[k] and s["estimate"] >= s["threshold"])
        out[k] = {**s, "holm_rejected": bool(rej[k]), "holm_adjusted_p": adj[k], "meets_rule": meets,
                  "status": "ESTABLISHED" if meets else ("FAILED" if s["opposite_determinate"] else "NOT_ESTABLISHED")}
    if any(v["meets_rule"] for v in out.values()):
        status = "ESTABLISHED"
    elif all(v.get("opposite_determinate") for v in out.values()):
        status = "FAILED"
    else:
        status = "NOT_ESTABLISHED"
    meeting = [adj[k] for k in keys if out[k]["meets_rule"]]
    return {"status": status, "conclusions": out, "p_family": float(min(meeting)) if meeting else
            float(min(adj.values())), "carried": sorted(per_conclusion)}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)
