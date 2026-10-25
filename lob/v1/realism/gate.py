"""Capability-specific realism acceptance (M133) under the frozen v1 gate.

Reference band of a metric: the 5%-95% range of its values over the calibrate-pool days (at least 8 real days),
widened on both sides by the bootstrap SE of the metric on the test day's real windows. Per capability:

* ``ACCEPTABLE`` - every metric of the capability lies inside its band and the real-vs-generated classifier AUC is
  <= 0.65;
* ``PARTIALLY_ACCEPTABLE`` - every metric within twice the band half-width of the band centre;
* ``REJECTED`` - otherwise (a metric that cannot be estimated on generated data counts as outside).

The decision is fail-closed: a REJECTED capability never supports a claim that needs it (``twins.permissions``).
Distribution-only gating (``marginal_only=True``) applies the same rule to the ``marginal`` metrics alone; it is
the comparison arm of H19 and H20, never a substitute for the registered gate.
"""
from __future__ import annotations

import math

import numpy as np

from .metrics import CAPABILITIES, KIND

AUC_MAX = 0.65
EXECUTION = ("spread_dynamics", "depth_dynamics", "queue_survival", "fill_prediction", "impact")


def bands(reference_days: list[dict], test_se: dict[str, float] | None = None) -> dict[str, tuple[float, float]]:
    if len(reference_days) < 8:
        raise ValueError(f"the registered reference needs >= 8 real days, got {len(reference_days)}")
    out = {}
    for m in KIND:
        x = np.asarray([d[m] for d in reference_days], float)
        x = x[np.isfinite(x)]
        if len(x) < 8:
            out[m] = (math.nan, math.nan)
            continue
        lo, hi = np.percentile(x, [5, 95])
        w = (test_se or {}).get(m, 0.0)
        w = 0.0 if not np.isfinite(w) else w
        out[m] = (float(lo - w), float(hi + w))
    return out


def metric_status(value: float, band: tuple[float, float]) -> str:
    lo, hi = band
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "NOT_ESTIMABLE"
    if not np.isfinite(value):
        return "OUTSIDE"
    if lo <= value <= hi:
        return "INSIDE"
    c, h = (lo + hi) / 2, max((hi - lo) / 2, 1e-12)
    return "NEAR" if abs(value - c) <= 2 * h else "OUTSIDE"


def decide(gen_metrics: dict[str, float], band: dict[str, tuple[float, float]], *, auc: float | None = None,
           marginal_only: bool = False) -> dict[str, dict]:
    out = {}
    for cap, ms in CAPABILITIES.items():
        use = [m for m in ms if not marginal_only or KIND[m] == "marginal"]
        if not use:
            out[cap] = {"status": "NOT_ASSESSED", "metrics": {}}
            continue
        st = {m: metric_status(gen_metrics.get(m, math.nan), band[m]) for m in use}
        known = [s for s in st.values() if s != "NOT_ESTIMABLE"]
        if not known:
            status = "NOT_ASSESSED"
        elif all(s == "INSIDE" for s in known) and (auc is None or not np.isfinite(auc) or auc <= AUC_MAX):
            status = "ACCEPTABLE"
        elif all(s in ("INSIDE", "NEAR") for s in known):
            status = "PARTIALLY_ACCEPTABLE"
        else:
            status = "REJECTED"
        out[cap] = {"status": status, "metrics": st}
    return out


def global_decision(per_generator: dict[str, dict], capabilities=EXECUTION) -> dict:
    acc = [g for g, d in per_generator.items() if all(d[c]["status"] == "ACCEPTABLE" for c in capabilities)]
    part = [g for g, d in per_generator.items()
            if all(d[c]["status"] in ("ACCEPTABLE", "PARTIALLY_ACCEPTABLE") for c in capabilities)]
    return {"status": "ACCEPTABLE_MODEL_SET" if acc else "PARTIAL_MODEL_SET" if part else "NO_ACCEPTABLE_MODEL",
            "acceptable": acc, "partial": part, "capabilities": list(capabilities)}


__all__ = ["bands", "decide", "metric_status", "global_decision", "AUC_MAX", "EXECUTION"]
