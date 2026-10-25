"""Queue survival and fill-hazard models (M108); post-fill markout and adverse-selection prediction (M109).

``fill_hazard`` fits the registered survival model set (Kaplan-Meier, Cox PH, Weibull AFT, boosted hazard; v0.9
``queue.survival``) on the back-of-queue clearance events (``queue.clearance.events``) of training days, selects by
inner-validation integrated Brier score, and returns P(fill by h) curves. On L2 data the queue ahead is only
bounded, so ``fill_interval`` evaluates the selected model at the optimistic and the pessimistic queue size
(``data.reconstruction.observability.l2_queue_bounds``) and reports the interval, never a point fill probability.

``markouts`` computes, for every hypothetical passive fill (back-of-queue clearance event), the signed mid move after
the fill at registered horizons (positive = favourable to the resting order); ``AdverseSelection`` is a logistic
classifier of P(markout at 5 s < 0 | pre-fill features), reported with its held-out log loss, Brier score and
calibration slope.
"""
from __future__ import annotations

import math

import numpy as np

from ...v09.queue import clearance as C, survival as S

HORIZONS = (1.0, 5.0, 30.0)


def fill_hazard(train_caches: list[dict], *, seed: int = 0, cap: int = 60_000) -> dict:
    ev = [C.events(c) for c in train_caches]
    X = np.vstack([e["X"] for e in ev])
    T = np.concatenate([e["T"] for e in ev])
    E = np.concatenate([e["E"] for e in ev])
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(T), min(cap, len(T)), replace=False))
    X, T, E = X[keep], T[keep], E[keep]
    cut = int(0.75 * len(T))
    models = {"kaplan_meier": S.KaplanMeier(), "cox": S.CoxPH(seed=seed), "weibull_aft": S.WeibullAFT(),
              "boosted_hazard": S.BoostedHazard(seed=seed)}
    ibs = {n: S.integrated_brier(m.fit(X[:cut], T[:cut], E[:cut]).predict(X[cut:]), T[cut:], E[cut:])
           for n, m in models.items()}
    best = min(ibs, key=ibs.get)
    final = {"kaplan_meier": S.KaplanMeier(), "cox": S.CoxPH(seed=seed), "weibull_aft": S.WeibullAFT(),
             "boosted_hazard": S.BoostedHazard(seed=seed)}[best].fit(X, T, E)
    return {"selected": best, "inner_ibs": ibs, "model": final, "grid": S.GRID}


def p_fill(model, X: np.ndarray, horizon: float) -> np.ndarray:
    Sx = model.predict(np.asarray(X, float))
    j = int(np.searchsorted(S.GRID, horizon, side="right") - 1)
    return 1.0 - Sx[:, max(j, 0)]


def fill_interval(model, X: np.ndarray, queue_bounds: tuple[np.ndarray, np.ndarray], horizon: float) -> dict:
    """P(fill by horizon) at the optimistic (smaller) and pessimistic (larger) queue ahead (feature 0 = log1p q)."""
    lo_q, hi_q = (np.asarray(b, float) for b in queue_bounds)
    Xo, Xp = np.array(X, float), np.array(X, float)
    Xo[:, 0], Xp[:, 0] = np.log1p(lo_q), np.log1p(hi_q)
    a, b = p_fill(model, Xo, horizon), p_fill(model, Xp, horizon)
    return {"p_fill_optimistic": a, "p_fill_pessimistic": b, "width": a - b,
            "label": "partially identified: L2 bounds the queue ahead, it does not observe it"}


def markouts(cache: dict, *, horizons=(1.0, 5.0, 30.0)) -> dict:
    ev = C.events(cache)
    t_fill = ev["t"] + ev["T"]
    ok = ev["E"]
    mid = np.asarray(cache["mid"], float)
    out = {"X": ev["X"][ok], "side": ev["side"][ok]}
    b0 = np.clip((t_fill[ok] * 10).astype(int), 0, len(mid) - 1)
    for h in horizons:
        b1 = np.clip(((t_fill[ok] + h) * 10).astype(int), 0, len(mid) - 1)
        # resting bid (side +1) fills when price falls: favourable if the mid then rises
        out[f"markout_{h:g}s_bps"] = ev["side"][ok] * (mid[b1] - mid[b0]) / np.maximum(mid[b0], 1e-9) * 1e4
    return out


class AdverseSelection:
    def __init__(self, *, l2: float = 1e-2, iters: int = 400) -> None:
        self.l2, self.iters = l2, iters

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        w = np.zeros(Z.shape[1])
        for _ in range(self.iters):                       # Newton (IRLS) with ridge
            p = 1 / (1 + np.exp(-Z @ w))
            g = Z.T @ (p - y) + self.l2 * np.r_[0, w[1:]]
            H = (Z * (p * (1 - p))[:, None]).T @ Z + self.l2 * np.eye(len(w))
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        self.w = w
        return self

    def predict(self, X):
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        return 1 / (1 + np.exp(-Z @ self.w))

    def evaluate(self, X, y) -> dict:
        p = np.clip(self.predict(X), 1e-9, 1 - 1e-9)
        logit = np.log(p / (1 - p))
        A = np.column_stack([np.ones(len(p)), logit])
        slope = None
        try:
            w = np.zeros(2)
            for _ in range(50):
                q = 1 / (1 + np.exp(-A @ w))
                H = (A * (q * (1 - q))[:, None]).T @ A + 1e-9 * np.eye(2)
                w -= np.linalg.solve(H, A.T @ (q - y))
            slope = float(w[1])
        except np.linalg.LinAlgError:
            pass
        base = float(np.clip(y.mean(), 1e-9, 1 - 1e-9))
        return {"log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                "base_log_loss": float(-(y * math.log(base) + (1 - y) * math.log(1 - base)).mean()),
                "brier": float(((p - y) ** 2).mean()), "calibration_slope": slope, "n": int(len(y))}


__all__ = ["fill_hazard", "p_fill", "fill_interval", "markouts", "AdverseSelection", "HORIZONS"]
