"""Unified calibration objective and API (M90), realism-aware calibration (M91), hierarchical temporal and
cross-market calibration (M94), calibration half-life and drift recovery (M98), and decision-focused calibration
with uncertainty decomposition (M99).

``calibrate(model_factory, space, objective, train, test)`` is the one entry point:

* ``model_factory(theta) -> generator`` builds a candidate; ``space`` is a box (``inference.sbi.Prior``);
* ``objective`` is one of ``likelihood`` (mean held-out per-bin log-likelihood of the training frames, higher is
  better), ``realism`` (registered gate distance: the mean over metrics of |m_gen - band centre| / band half-width,
  the band from the *training* days only - M91), or ``decision`` (a user scorer, e.g. F2 regret of the policy the
  candidate world ranks first - M99);
* the search is a seeded random search followed by a local refinement (coordinate shrink), budget explicit;
* train and test datasets are disjoint by construction (``sdk.CalibrationResult`` refuses overlap); the result
  reports the objective on both.

``pool_parameters`` (M94) applies the v0.8 DerSimonian-Laird partial pooling over days within instruments and over
instruments. ``half_life`` (M98) measures how fast a calibration degrades with the lag between calibration and
evaluation days and whether recalibration recovers it. ``decompose`` (M99) splits the variance of a decision
quantity into parameter, model-class and intrinsic (seed) components.
"""
from __future__ import annotations

import math

import numpy as np

from ...v08.inference import hierarchical as HI
from ..sdk import CalibrationResult

OBJECTIVES = ("likelihood", "realism", "decision")


def realism_distance(gen_metrics: dict, band: dict) -> float:
    vals = []
    for m, (lo, hi) in band.items():
        g = gen_metrics.get(m)
        if g is None or not (np.isfinite(lo) and np.isfinite(hi)) or not np.isfinite(g):
            continue
        c, h = (lo + hi) / 2, max((hi - lo) / 2, 1e-12)
        vals.append(min(abs(g - c) / h, 10.0))
    return float(np.mean(vals)) if vals else math.inf


def calibrate(model_factory, prior, objective: str, *, scorer, train: tuple[str, ...], test: tuple[str, ...],
              budget: int = 40, refine: int = 20, seed: int = 0, test_scorer=None) -> CalibrationResult:
    """Search ``prior`` for the parameter maximizing ``scorer(model)`` (``realism``/``decision`` are minimized)."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    sign = 1.0 if objective == "likelihood" else -1.0
    rng = np.random.default_rng(seed)
    history = []

    def evaluate(theta):
        v = float(scorer(model_factory(dict(zip(prior.names, theta)))))
        history.append((list(map(float, theta)), v))
        return sign * v

    best_u, best = None, -math.inf
    for u in rng.random((budget, prior.dim)):
        val = evaluate(prior.from_unit(u))
        if val > best:
            best_u, best = u, val
    step = 0.15
    for _ in range(refine):
        cand = np.clip(best_u + step * rng.standard_normal(prior.dim), 0, 1)
        val = evaluate(prior.from_unit(cand))
        if val > best:
            best_u, best = cand, val
        else:
            step *= 0.85
    theta = dict(zip(prior.names, map(float, prior.from_unit(best_u))))
    diag = {"evaluations": len(history), "train_objective": sign * best, "history": history[-10:]}
    if test_scorer is not None:
        diag["test_objective"] = float(test_scorer(model_factory(theta)))
    return CalibrationResult(model=objective, parameters=theta, objective=float(sign * best), train=tuple(train),
                             test=tuple(test), diagnostics=diag)


def pool_parameters(estimates: dict[str, dict[str, tuple[float, float]]]) -> dict:
    """``estimates[instrument][day] = (mean, variance)`` of one parameter -> two-level partial pooling."""
    per_instrument, out = {}, {"instruments": {}}
    for inst, days in estimates.items():
        m = np.asarray([v[0] for v in days.values()])
        v = np.asarray([v[1] for v in days.values()])
        pooled = HI.dersimonian_laird(m, v)
        sm, sv = HI.shrink(m, v, pooled)
        per_instrument[inst] = (pooled["mu"], pooled.get("mu_se", 0.0) ** 2)
        out["instruments"][inst] = {"pooled": pooled, "days": dict(zip(days, map(float, sm))),
                                    "day_variance": dict(zip(days, map(float, sv)))}
    m = np.asarray([v[0] for v in per_instrument.values()])
    v = np.asarray([v[1] for v in per_instrument.values()])
    out["global"] = HI.dersimonian_laird(m, v)
    return out


def half_life(errors_by_lag: dict[int, list[float]], *, baseline_lag: int = 0) -> dict:
    """Lag (days) at which the mean calibration error first doubles relative to the baseline lag."""
    lags = sorted(errors_by_lag)
    mean = {k: float(np.mean(errors_by_lag[k])) for k in lags}
    base = mean[baseline_lag]
    hl = next((k for k in lags if k > baseline_lag and mean[k] >= 2 * base), None)
    return {"mean_error_by_lag": mean, "half_life_days": hl, "baseline": base}


def recovery(before: list[float], after_recalibration: list[float]) -> dict:
    b, a = np.asarray(before, float), np.asarray(after_recalibration, float)
    return {"error_before": float(b.mean()), "error_after": float(a.mean()),
            "recovered_fraction": float(1 - a.mean() / b.mean()) if b.mean() > 0 else None}


def decompose(values: dict[str, dict[str, list[float]]]) -> dict:
    """``values[model_class][parameter_draw] = [seed outcomes]`` -> variance shares (law of total variance)."""
    allv = np.asarray([x for mc in values.values() for d in mc.values() for x in d], float)
    total = float(allv.var())
    class_means = {c: np.mean([x for d in v.values() for x in d]) for c, v in values.items()}
    w_class = {c: sum(len(d) for d in v.values()) / len(allv) for c, v in values.items()}
    grand = float(allv.mean())
    between_class = float(sum(w_class[c] * (class_means[c] - grand) ** 2 for c in values))
    draw_var, seed_var = 0.0, 0.0
    for c, v in values.items():
        dm = {k: np.mean(d) for k, d in v.items()}
        draw_var += sum(len(d) / len(allv) * (dm[k] - class_means[c]) ** 2 for k, d in v.items())
        seed_var += sum(len(d) / len(allv) * float(np.var(d)) for d in v.values())
    shares = {k: (x / total if total > 0 else 0.0) for k, x in
              (("model_class", between_class), ("parameter", draw_var), ("intrinsic", seed_var))}
    return {"total_variance": total, "components": {"model_class": between_class, "parameter": draw_var,
                                                    "intrinsic": seed_var}, "shares": shares}


__all__ = ["OBJECTIVES", "calibrate", "realism_distance", "pool_parameters", "half_life", "recovery", "decompose"]
