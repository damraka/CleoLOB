"""Calibration surrogates and active calibration (workstreams 10, 11).

* ``GaussianProcess`` — ARD squared-exponential kernel plus noise, hyperparameters chosen by
  maximizing the log marginal likelihood over a seeded random search (numpy only; scipy is not a
  dependency). Predictive mean and standard deviation.
* ``RegressionForest`` — bagged variance-reduction regression trees; the spread across trees is a
  heuristic (not calibrated) uncertainty.
* ``active_search`` — batched lower-confidence-bound acquisition (mean - kappa * sd) over a seeded
  candidate pool, versus ``random_search`` at the same evaluation budget.

Surrogates only screen candidates; every reported objective comes from simulation. Emulator error
is measured on held-out simulated candidates.
"""
from __future__ import annotations

import numpy as np


class GaussianProcess:
    def __init__(self, *, restarts: int = 48, seed: int = 0) -> None:
        self.restarts, self.seed = restarts, seed

    @staticmethod
    def _kernel(a: np.ndarray, b: np.ndarray, lengths: np.ndarray, amp: float) -> np.ndarray:
        d = ((a[:, None, :] - b[None, :, :]) / lengths) ** 2
        return amp * np.exp(-0.5 * d.sum(-1))

    def _nll(self, x, y, lengths, amp, noise) -> float:
        k = self._kernel(x, x, lengths, amp) + (noise + 1e-8) * np.eye(len(x))
        try:
            chol = np.linalg.cholesky(k)
        except np.linalg.LinAlgError:
            return np.inf
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
        return float(0.5 * y @ alpha + np.log(np.diag(chol)).sum())

    def fit(self, x: np.ndarray, y: np.ndarray) -> GaussianProcess:
        self.x = np.asarray(x, float)
        self.mean_y, self.sd_y = float(np.mean(y)), float(np.std(y) or 1.0)
        yn = (np.asarray(y, float) - self.mean_y) / self.sd_y
        rng = np.random.default_rng(self.seed)
        best = None
        for r in range(self.restarts):
            lengths = np.exp(rng.uniform(np.log(0.05), np.log(3.0), x.shape[1])) if r else np.full(x.shape[1], 0.5)
            amp, noise = float(np.exp(rng.uniform(-1, 1))), float(np.exp(rng.uniform(-6, -1)))
            value = self._nll(self.x, yn, lengths, amp, noise)
            if best is None or value < best[0]:
                best = (value, lengths, amp, noise)
        self.nll, self.lengths, self.amp, self.noise = best
        k = self._kernel(self.x, self.x, self.lengths, self.amp) + (self.noise + 1e-8) * np.eye(len(self.x))
        self.chol = np.linalg.cholesky(k)
        self.alpha = np.linalg.solve(self.chol.T, np.linalg.solve(self.chol, yn))
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ks = self._kernel(np.asarray(x, float), self.x, self.lengths, self.amp)
        mean = ks @ self.alpha
        v = np.linalg.solve(self.chol, ks.T)
        var = np.maximum(self.amp - (v ** 2).sum(0), 1e-12) + self.noise
        return mean * self.sd_y + self.mean_y, np.sqrt(var) * self.sd_y


class RegressionForest:
    def __init__(self, trees: int = 48, depth: int = 6, min_leaf: int = 5, seed: int = 0) -> None:
        self.trees, self.depth, self.min_leaf, self.seed = trees, depth, min_leaf, seed

    def _grow(self, x, y, depth, rng):
        if depth == 0 or len(y) < 2 * self.min_leaf:
            return float(y.mean())
        best = None
        for j in rng.choice(x.shape[1], max(1, int(np.sqrt(x.shape[1]))), replace=False):
            for t in np.unique(np.quantile(x[:, j], np.linspace(0.1, 0.9, 9))):
                left = x[:, j] <= t
                if left.sum() < self.min_leaf or (~left).sum() < self.min_leaf:
                    continue
                sse = y[left].var() * left.sum() + y[~left].var() * (~left).sum()
                if best is None or sse < best[0]:
                    best = (sse, j, t, left)
        if best is None:
            return float(y.mean())
        _, j, t, left = best
        return (j, t, self._grow(x[left], y[left], depth - 1, rng), self._grow(x[~left], y[~left], depth - 1, rng))

    def fit(self, x, y) -> RegressionForest:
        rng = np.random.default_rng(self.seed)
        x, y = np.asarray(x, float), np.asarray(y, float)
        self.members = [self._grow(x[idx], y[idx], self.depth, rng)
                        for idx in (rng.integers(0, len(y), len(y)) for _ in range(self.trees))]
        return self

    @staticmethod
    def _one(node, row):
        while isinstance(node, tuple):
            j, t, left, right = node
            node = left if row[j] <= t else right
        return node

    def predict(self, x) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, float)
        draws = np.asarray([[self._one(m, row) for row in x] for m in self.members])
        return draws.mean(0), draws.std(0)


def emulator_report(model, x_train, y_train, x_test, y_test, *, top: float = 0.05) -> dict:
    model.fit(x_train, y_train)
    mean, sd = model.predict(x_test)
    err = mean - y_test
    covered = float(np.mean(np.abs(err) <= 1.645 * sd))
    k = max(1, int(round(top * len(y_test))))
    true_top = set(np.argsort(y_test)[:k])
    screened = np.argsort(mean)
    needed = next((i + 1 for i in range(len(screened)) if true_top <= set(screened[: i + 1])), len(screened))
    recall_at_k = len(true_top & set(screened[:k])) / k
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "rank_correlation": float(np.corrcoef(np.argsort(np.argsort(mean)), np.argsort(np.argsort(y_test)))[0, 1]),
            "interval90_coverage": covered, "recall_of_true_top": recall_at_k,
            "simulations_to_recover_all_true_top": int(needed), "test_size": int(len(y_test)),
            "simulation_saving_fraction": float(1 - needed / len(y_test))}


def random_search(objective, budget: int, dim: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.random((budget, dim))
    y = np.asarray(objective(x), float)
    return {"x": x, "y": y, "best": float(np.nanmin(y)), "best_x": x[int(np.nanargmin(y))]}


def active_search(objective, budget: int, dim: int, seed: int, *, initial: int = 16, batch: int = 8,
                  pool: int = 2000, kappa: float = 1.0) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.random((initial, dim))
    y = np.asarray(objective(x), float)
    while len(y) < budget:
        finite = np.isfinite(y)
        gp = GaussianProcess(restarts=24, seed=int(rng.integers(1 << 31))).fit(x[finite], y[finite])
        candidates = rng.random((pool, dim))
        mean, sd = gp.predict(candidates)
        chosen = candidates[np.argsort(mean - kappa * sd)[: min(batch, budget - len(y))]]
        x = np.vstack([x, chosen])
        y = np.r_[y, np.asarray(objective(chosen), float)]
    return {"x": x, "y": y, "best": float(np.nanmin(y)), "best_x": x[int(np.nanargmin(y))]}
