"""Impact kernels (M105), state-dependent impact with uncertainty bounds (M106) and cross-instrument impact (M107).

All estimates come from observational order flow: they are **not** causal identifications (information and flow are
confounded); each result carries that label and ``causal.estimands`` marks them ``NOT_IDENTIFIED``.

* ``fit_kernel``: the non-parametric propagator G(l) of the v0.9 ridge regression (r_t = sum_l G(l) v_{t-l}/V) per
  day, then the registered parametric decomposition of its cumulative response
  C(l) = temporary * 1[l = 0] + transient * (1 - exp(-l / tau)) ... fitted as
  C(l) = a + b (1 - exp(-l / tau)), with ``temporary`` = C(0) = a, ``persistent`` = C(L) and ``transient`` = the
  part that decays (peak minus final), tau by grid search; day-block bootstrap intervals over days.
* ``state_impact``: the v0.9 boosted state-dependent impact model fitted on pooled burst features with a day-block
  bootstrap ensemble; partial dependence of the 5 s impact on each feature with 90% bounds.
* ``cross``: response of instrument B's mid to instrument A's bursts (v0.9 definition) per day with a day-block
  bootstrap interval and the same-instrument response for scale.
"""
from __future__ import annotations

import numpy as np

from ...v09.impact import models as IM

LABEL = "observational association; ASSUMPTION_DEPENDENT, not a causal impact identification"


def _param_fit(cum: np.ndarray) -> dict:
    lags = np.arange(len(cum), dtype=float)
    best = None
    for tau in np.geomspace(0.5, 60.0, 40):
        basis = np.column_stack([np.ones_like(lags), 1.0 - np.exp(-lags / tau)])
        coef, *_ = np.linalg.lstsq(basis, cum, rcond=None)
        err = float(((basis @ coef - cum) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, tau, coef)
    err, tau, (a, b) = best
    persistent = float(cum[-1])
    peak = float(np.max(cum))
    return {"temporary_bps": float(a), "tau_s": float(tau), "rise_bps": float(b), "persistent_bps": persistent,
            "transient_bps": float(peak - persistent), "fit_sse": err}


def fit_kernel(caches: list[dict], *, lags: int = 30, boot: int = 200, seed: int = 0) -> dict:
    per_day = [np.asarray(IM.kernel(c, lags=lags)["cumulative"]) for c in caches]
    mean = np.mean(per_day, axis=0)
    point = _param_fit(mean)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(boot if len(per_day) > 1 else 0):
        pick = rng.integers(0, len(per_day), len(per_day))
        draws.append(_param_fit(np.mean([per_day[i] for i in pick], axis=0)))
    intervals = {k: [float(np.percentile([d[k] for d in draws], 5)), float(np.percentile([d[k] for d in draws], 95))]
                 for k in point} if draws else {}
    return {"cumulative": mean.tolist(), "parametric": point, "intervals_90": intervals, "days": len(per_day),
            "label": LABEL}


def state_impact(caches: list[dict], *, members: int = 10, seed: int = 0, trees: int = 80) -> dict:
    data = []
    for c in caches:
        from ...v09.generators import hybrid as Hy
        bt = Hy.burst_targets(c)
        data.append((bt["X"], bt["Y"][:, 1]))
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(members):
        pick = rng.integers(0, len(data), len(data))
        X = np.vstack([data[i][0] for i in pick])
        y = np.concatenate([data[i][1] for i in pick])
        models.append(IM.StateImpact(trees=trees).fit(X, y))
    Xall = np.vstack([d[0] for d in data])
    pd = {}
    for j in range(Xall.shape[1]):
        grid = np.quantile(Xall[:, j], [0.1, 0.3, 0.5, 0.7, 0.9])
        curves = []
        for m in models:
            vals = []
            for g in grid:
                Xg = Xall[:2000].copy()
                Xg[:, j] = g
                vals.append(float(m.predict(Xg).mean()))
            curves.append(vals)
        C = np.asarray(curves)
        pd[f"feature_{j}"] = {"grid": grid.tolist(), "mean": C.mean(0).tolist(),
                              "lo90": np.percentile(C, 5, axis=0).tolist(), "hi90": np.percentile(C, 95, axis=0).tolist()}
    return {"partial_dependence": pd, "members": members, "bursts": int(len(Xall)), "label": LABEL,
            "models": models}


def cross(target_caches: list[dict], source_caches: list[dict], *, horizon: int = 10, boot: int = 500,
          seed: int = 0) -> dict:
    per = [IM.cross_impact(t, s, horizon) for t, s in zip(target_caches, source_caches)]
    vals = np.asarray([p["mean_bps"] for p in per if p], float)
    own = np.asarray([IM.cross_impact(s, s, horizon)["mean_bps"] for s in source_caches if IM.bursts(s)["sec"].size])
    rng = np.random.default_rng(seed)
    bs = [rng.choice(vals, len(vals)).mean() for _ in range(boot)] if len(vals) > 1 else []
    return {"cross_mean_bps": float(vals.mean()) if len(vals) else None,
            "interval_90": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))] if bs else None,
            "own_mean_bps": float(own.mean()) if len(own) else None, "days": int(len(vals)), "horizon_s": horizon,
            "label": LABEL + "; common information across instruments is not separated"}


__all__ = ["fit_kernel", "state_impact", "cross", "LABEL"]
