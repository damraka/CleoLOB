"""Contextual selection, meta-policy, disagreement, abstention v3 and the contextual fallback ladder (M83, M84).

Window context (observed at the window start): spread, five-level imbalance (signed by the mandate side), log touch
depth, 60 s mid volatility (bps), 30 s signed flow (signed by side, per mean second volume), hour of day (sin, cos).

* ``CostModels`` - three contextual cost models per policy (ridge regression, gradient-boosted trees, k-nearest
  neighbours with k = 25) fitted on training windows; the meta-policy (mixture of experts) picks the policy with the
  lowest ensemble-mean predicted cost.
* Disagreement (H18) of a window = standard deviation across the three models of the selected policy's predicted
  cost plus the standard deviation of the per-model argmin costs (registered composite of model and policy
  disagreement).
* Abstention v3 (H19): abstain - choose the registered conservative policy (TWAP) - when the disagreement exceeds the
  75th percentile of the training windows' disagreement. No assumption that abstention helps.
* Contextual fallback ladder (H20, protocol ``policies.fallback_ladder``): low uncertainty -> selected optimized
  policy; moderate -> liquidity_pov; high -> twap; completion danger -> completion-safe aggressive controller.
"""
from __future__ import annotations

import math

import numpy as np

from ..impact.models import StateImpact

CONTEXT = ("spread", "imbalance_signed", "log_touch", "vol60", "flow30", "hour_sin", "hour_cos")


def window_context(cache: dict, window_start: float, side: int) -> np.ndarray:
    b = int(window_start * 10)
    st = cache["state"][b]
    mid = cache["mid"][max(0, b - 600):b + 1:10].astype(float)
    vol = float(np.std(np.diff(mid)) / max(mid[-1], 1e-9) * 1e4) if len(mid) > 3 else 0.0
    t, q, s = cache["trade_t"], cache["trade_q"], cache["trade_s"]
    a, z = np.searchsorted(t, [window_start - 30, window_start])
    a6 = np.searchsorted(t, window_start - 600)
    mean_sec = max(float(np.abs(q[a6:z] * s[a6:z]).sum()) / 600.0, 1e-9)
    flow = float((q[a:z] * s[a:z]).sum()) / 30.0 / mean_sec * side
    hour = window_start / 3600.0
    return np.asarray([float(st[0]), side * float(st[1]), math.log1p(float(cache["depth"][b, 0, 0] +
                                                                            cache["depth"][b, 1, 0])),
                       vol, flow, math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24)])


class _Ridge:
    def fit(self, X, y, lam=1.0):
        self.mu, self.sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1.0)
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        self.w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd]) @ self.w


class _KNN:
    def fit(self, X, y, k=25):
        self.mu, self.sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1.0)
        self.X, self.y, self.k = (X - self.mu) / self.sd, y, k
        return self

    def predict(self, X):
        Z = (X - self.mu) / self.sd
        d = ((Z[:, None] - self.X[None]) ** 2).sum(-1)
        idx = np.argsort(d, 1)[:, : min(self.k, len(self.y))]
        return self.y[idx].mean(1)


class CostModels:
    KINDS = ("ridge", "boosted", "knn")

    def fit(self, X: np.ndarray, costs: dict[str, np.ndarray]):
        self.policies = sorted(costs)
        self.models = {}
        for p in self.policies:
            y = np.asarray(costs[p], float)
            ok = np.isfinite(y)
            self.models[p] = {"ridge": _Ridge().fit(X[ok], y[ok]),
                              "boosted": StateImpact(trees=60, lr=0.1).fit(X[ok], y[ok]),
                              "knn": _KNN().fit(X[ok], y[ok])}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """[n, policies, 3] predicted costs."""
        return np.stack([np.column_stack([self.models[p][k].predict(X) for k in self.KINDS])
                         for p in self.policies], axis=1)


def select(pred: np.ndarray, policies: list[str]) -> tuple[list[str], np.ndarray]:
    """Meta-policy choice (ensemble-mean argmin) and the registered disagreement per window."""
    mean = pred.mean(2)
    choice = mean.argmin(1)
    sel_sd = pred[np.arange(len(pred)), choice].std(1)
    per_model_min = pred.min(1)
    disagreement = sel_sd + per_model_min.std(1)
    return [policies[i] for i in choice], disagreement


def abstain(disagreement: np.ndarray, threshold: float) -> np.ndarray:
    return disagreement > threshold


def fallback_policy(uncertainty: float, completion_danger: bool, *, low: float, high: float,
                    selected: str) -> str:
    if completion_danger:
        return "sweep"
    if uncertainty <= low:
        return selected
    if uncertainty <= high:
        return "liquidity_pov"
    return "twap"
