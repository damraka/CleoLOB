"""Positive, negative and falsification controls (M131).

Controls run through the same decision rule as the hypotheses, so they test the *pipeline*, not just the data:

* ``negative_self``: a policy compared with itself under identical inputs must give an effect of exactly zero on
  every unit (any non-zero value is a pipeline asymmetry);
* ``negative_placebo``: the treatment/control labels of each unit are swapped at random (sign flips by unit); the
  registered test must not establish an effect more often than alpha (checked over repeated permutations);
* ``positive_injection``: a known effect is added to the unit values; the rule must establish it at the planned power;
* ``falsification``: an effect the theory says must be absent (e.g. a policy pair that differs only in a parameter
  the episode never uses) must not be established.

Each returns a record with ``passed``; a failed control blocks the confirmatory analysis that depends on it.
"""
from __future__ import annotations

import numpy as np

from . import decide as DE


def negative_self(unit_values: dict[str, float], *, tol: float = 1e-12) -> dict:
    worst = max((abs(v) for v in unit_values.values()), default=0.0)
    return {"control": "negative_self", "max_abs_effect": float(worst), "passed": bool(worst <= tol)}


def negative_placebo(cluster_values: dict[str, float], *, threshold: float = 0.0, reps: int = 200, alpha: float = 0.05,
                     seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    keys = sorted(cluster_values)
    x = np.asarray([cluster_values[k] for k in keys], float)
    x = x - x.mean()                                  # placebo world: centred, then randomly signed
    hits = 0
    for r in range(reps):
        flip = rng.choice([-1.0, 1.0], len(x))
        res = DE.test_one(dict(zip(keys, flip * x)), threshold, seed=r)
        if "p_cluster_t" in res and res["p_cluster_t"] <= alpha and res["p_sign_flip"] <= alpha:
            hits += 1
    rate = hits / reps
    return {"control": "negative_placebo", "false_positive_rate": rate, "reps": reps,
            "passed": bool(rate <= alpha + 2 * np.sqrt(alpha * (1 - alpha) / reps))}


def positive_injection(cluster_values: dict[str, float], effect: float, *, threshold: float = 0.0,
                       min_power: float = 0.8, reps: int = 100, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    keys = sorted(cluster_values)
    x = np.asarray([cluster_values[k] for k in keys], float)
    x = x - x.mean()
    hits = 0
    for r in range(reps):
        boot = rng.choice(x, len(x)) + effect
        res = DE.test_one(dict(zip(keys, boot)), threshold, seed=r)
        if "p_cluster_t" in res and res["p_cluster_t"] <= 0.05 and res["p_sign_flip"] <= 0.05:
            hits += 1
    power = hits / reps
    return {"control": "positive_injection", "injected": effect, "detected_rate": power, "passed": bool(power >= min_power)}


def falsification(cluster_values: dict[str, float], *, threshold: float = 0.0, seed: int = 0) -> dict:
    res = DE.test_one(cluster_values, threshold, seed=seed)
    est = "p_cluster_t" in res and res["p_cluster_t"] <= 0.05 and res["p_sign_flip"] <= 0.05
    return {"control": "falsification", "result": res, "passed": bool(not est)}


__all__ = ["negative_self", "negative_placebo", "positive_injection", "falsification"]
