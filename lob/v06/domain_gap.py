"""M5/M7: leakage-resistant real-versus-synthetic two-sample tests (domain gap) and OOD support.

Windows of 60 s are summarized by the sealed measurement operator
(``lob.v06.observables.measure``) into scale-free features from ``FEATURES``
only. No timestamp, clock-of-day, price level, dataset/file identifier, seed,
simulator parameter, regime label or row order can enter: ``feature_matrix``
refuses any other column. Splits are chronological for history (first half
train, 10-window purge, second half test) and grouped by seed for simulation.

Classifiers are small numpy implementations so the core release needs no extra
dependency: L2-regularized logistic regression (primary), a depth-3 tree and a
64-tree depth-4 random forest (secondary). A strong discriminator is evidence of
a detectable domain gap; a weak one is NOT evidence that the simulator is right.
"""
from __future__ import annotations

import math

import numpy as np

from .inference import counts, interval
from .observables import measure
from .tape import BookTape

WINDOW_S = 60.0
FEATURES = ("spread_mean", "spread_sd", "log_depth_l1", "log_depth5", "log_depth10", "abs_imbalance_mean",
            "imbalance_sd", "concentration_mean", "abs_microprice_dev", "return_sd", "nonzero_return_fraction",
            "abs_return_window", "log1p_trades", "mean_log_trade_size", "max_log_trade_size", "log1p_book_changes",
            "log1p_additions", "log1p_cancellations", "mean_level_gap", "mean_log_add_size")
FORBIDDEN = ("time", "timestamp", "clock", "hour", "price", "dataset", "file", "seed", "parameter", "regime", "index",
             "order", "label", "world")


def window_features(tape: BookTape, window_s: float = WINDOW_S, *, min_valid: float = 0.8) -> np.ndarray:
    """One row per complete window with at least ``min_valid`` valid samples; columns follow FEATURES."""
    rows = []
    for block in tape.blocks(window_s):
        if not len(block.t) or block.valid.mean() < min_valid:
            continue
        raw = measure(block)
        rows.append(_row(raw, block))
    return np.asarray(rows, float).reshape(-1, len(FEATURES))


def _mean(x: np.ndarray, transform=None) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if transform is not None:
        x = transform(x[x > 0])
    return float(x.mean()) if len(x) else 0.0


def _row(raw: dict, block: BookTape) -> list[float]:
    r1 = raw["r1_bps"]
    sizes = raw["trade_size"]
    first, last = np.flatnonzero(block.valid)[[0, -1]]
    window_return = math.log(block.mid[last] / block.mid[first]) * 1e4
    return [_mean(raw["spread_bps"]), float(np.std(raw["spread_bps"])) if len(raw["spread_bps"]) else 0.0,
            _mean(raw["depth_l1"], np.log), _mean(raw["depth5"], np.log), _mean(raw["depth10"], np.log),
            _mean(raw["abs_imbalance5"]), float(np.std(raw["imbalance5"])) if len(raw["imbalance5"]) else 0.0,
            _mean(raw["concentration"]), _mean(np.abs(raw["microprice_dev_bps"])),
            float(np.std(r1)) if len(r1) else 0.0, float(np.mean(r1 != 0)) if len(r1) else 0.0, abs(window_return),
            math.log1p(len(sizes)), _mean(sizes, np.log), float(np.log(sizes.max())) if len(sizes) else 0.0,
            math.log1p(float(np.sum(raw["change_count_10s"]))), math.log1p(float(np.sum(raw["add_count_10s"]))),
            math.log1p(float(np.sum(raw["cancel_count_10s"]))), _mean(raw["level_gap_ticks"]),
            _mean(raw["add_size"], np.log)]


def feature_matrix(columns: dict[str, np.ndarray]) -> np.ndarray:
    """Assemble features by name; refuses anything outside the sealed whitelist (leakage guard)."""
    names = list(columns)
    for name in names:
        lowered = name.lower()
        if name not in FEATURES or any(word in lowered for word in FORBIDDEN):
            raise ValueError(f"feature {name!r} is not in the sealed leakage-resistant whitelist")
    return np.column_stack([np.asarray(columns[n], float) for n in FEATURES if n in columns])


# ----------------------------------------------------------------------------- splits


def chronological_split(n: int, *, purge: int = 10) -> tuple[np.ndarray, np.ndarray]:
    half = n // 2
    return np.arange(0, half), np.arange(min(n, half + purge), n)


def balance(x: np.ndarray, groups: np.ndarray, size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= size:
        return x, groups
    pick = np.sort(rng.choice(len(x), size, replace=False))
    return x[pick], groups[pick]


# ----------------------------------------------------------------------------- classifiers


class Standardizer:
    def __init__(self, x: np.ndarray) -> None:
        self.mean = x.mean(0)
        self.sd = np.where(x.std(0) > 1e-12, x.std(0), 1.0)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.sd


class Logistic:
    """L2-regularized logistic regression by Newton iterations (intercept unpenalized)."""

    def __init__(self, penalty: float = 1.0, iterations: int = 50) -> None:
        self.penalty, self.iterations = penalty, iterations

    def fit(self, x: np.ndarray, y: np.ndarray) -> Logistic:
        self.scale = Standardizer(x)
        z = np.column_stack([np.ones(len(x)), self.scale(x)])
        w = np.zeros(z.shape[1])
        ridge = np.eye(z.shape[1]) * self.penalty
        ridge[0, 0] = 0.0
        for _ in range(self.iterations):
            p = 1 / (1 + np.exp(-np.clip(z @ w, -35, 35)))
            gradient = z.T @ (p - y) + ridge @ w
            hessian = (z * (p * (1 - p))[:, None]).T @ z + ridge + 1e-9 * np.eye(len(w))
            step = np.linalg.solve(hessian, gradient)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.w = w
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        z = np.column_stack([np.ones(len(x)), self.scale(x)])
        return 1 / (1 + np.exp(-np.clip(z @ self.w, -35, 35)))

    @property
    def coefficients(self) -> np.ndarray:
        return self.w[1:]


class Tree:
    """Gini decision tree with thresholds at feature deciles; leaves predict the class-1 fraction."""

    def __init__(self, depth: int = 3, min_leaf: int = 20, features: int | None = None, rng=None) -> None:
        self.depth, self.min_leaf, self.features, self.rng = depth, min_leaf, features, rng

    def fit(self, x: np.ndarray, y: np.ndarray) -> Tree:
        self.root = self._grow(x, y, self.depth)
        return self

    def _grow(self, x, y, depth):
        leaf = {"value": float(y.mean()) if len(y) else 0.5}
        if depth == 0 or len(y) < 2 * self.min_leaf or y.min() == y.max():
            return leaf
        columns = np.arange(x.shape[1])
        if self.features is not None:
            columns = self.rng.choice(columns, self.features, replace=False)
        best, parent = None, 1 - (y.mean() ** 2 + (1 - y.mean()) ** 2)
        for j in columns:
            for threshold in np.unique(np.quantile(x[:, j], np.linspace(0.1, 0.9, 9))):
                left = x[:, j] <= threshold
                nl, nr = left.sum(), (~left).sum()
                if nl < self.min_leaf or nr < self.min_leaf:
                    continue
                gini = sum(n / len(y) * (1 - (m ** 2 + (1 - m) ** 2))
                           for n, m in ((nl, y[left].mean()), (nr, y[~left].mean())))
                if best is None or gini < best[0] - 1e-12:
                    best = (gini, j, threshold, left)
        if best is None or best[0] >= parent:
            return leaf
        _, j, threshold, left = best
        return {"feature": int(j), "threshold": float(threshold), "left": self._grow(x[left], y[left], depth - 1),
                "right": self._grow(x[~left], y[~left], depth - 1)}

    def score(self, x: np.ndarray) -> np.ndarray:
        out = np.empty(len(x))
        for i, row in enumerate(x):
            node = self.root
            while "value" not in node:
                node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
            out[i] = node["value"]
        return out


class Forest:
    def __init__(self, trees: int = 64, depth: int = 4, seed: int = 0) -> None:
        self.trees, self.depth, self.seed = trees, depth, seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> Forest:
        rng = np.random.default_rng(self.seed)
        k = max(1, int(round(math.sqrt(x.shape[1]))))
        self.members = []
        for _ in range(self.trees):
            pick = rng.integers(0, len(y), len(y))
            self.members.append(Tree(self.depth, 10, k, rng).fit(x[pick], y[pick]))
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.mean([m.score(x) for m in self.members], axis=0)


# ----------------------------------------------------------------------------- metrics


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney statistic with average ranks for ties."""
    from .inference import ranks
    y = np.asarray(y, bool)
    positives, negatives = y.sum(), (~y).sum()
    if positives == 0 or negatives == 0:
        return math.nan
    r = ranks(scores)
    return float((r[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def balanced_accuracy(scores: np.ndarray, y: np.ndarray) -> float:
    predicted = scores >= 0.5
    y = np.asarray(y, bool)
    return float((np.mean(predicted[y]) + np.mean(~predicted[~y])) / 2)


def auc_interval(scores: np.ndarray, y: np.ndarray, real_blocks: np.ndarray, sim_groups: np.ndarray, *,
                 alpha: float, samples: int, seed: int) -> dict:
    """Block bootstrap of the test set: real windows in contiguous blocks, simulated windows by seed."""
    rng = np.random.default_rng(seed)
    real_idx = np.flatnonzero(y == 1)
    sim_idx = np.flatnonzero(y == 0)
    real_units = [real_idx[real_blocks == b] for b in np.unique(real_blocks)]
    sim_units = [sim_idx[sim_groups == g] for g in np.unique(sim_groups)]
    draws = np.empty(samples)
    for b in range(samples):
        ru = counts(len(real_units), rng)
        su = counts(len(sim_units), rng)
        pick = np.concatenate([np.repeat(u, int(c)) for u, c in zip(real_units, ru) if c] +
                              [np.repeat(u, int(c)) for u, c in zip(sim_units, su) if c])
        draws[b] = auc(scores[pick], y[pick])
    low, high = interval(draws, 2 * alpha)
    one_sided_lower = float(np.nanquantile(draws, alpha))
    return {"ci_low": low, "ci_high": high, "one_sided_lower": one_sided_lower, "alpha": alpha, "samples": samples,
            "real_blocks": len(real_units), "simulated_groups": len(sim_units)}


def permutation_importance(model, x: np.ndarray, y: np.ndarray, *, seed: int, repeats: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    base = auc(model.score(x), y)
    out = {}
    for j, name in enumerate(FEATURES[: x.shape[1]]):
        drops = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[:, j] = shuffled[rng.permutation(len(x)), j]
            drops.append(base - auc(model.score(shuffled), y))
        out[name] = float(np.mean(drops))
    return out


def two_sample_test(real: np.ndarray, sim_by_seed: list[np.ndarray], *, seed: int, alpha: float = 0.05,
                    samples: int = 2000, block: int = 10) -> dict:
    """Train on early real / first-half seeds, test on late real / second-half seeds; all classifiers."""
    if len(real) < 60 or len(sim_by_seed) < 2:
        return {"status": "NOT_AVAILABLE", "reason": "too few real windows or simulated seeds"}
    rng = np.random.default_rng(seed)
    train_r, test_r = chronological_split(len(real))
    half = len(sim_by_seed) // 2
    # Resampling unit for simulated windows: contiguous blocks of ``block`` windows within a seed (the same
    # unit as history); whole-seed resampling with few seeds was liberal in null calibration (8% at 5%).
    sim_train = np.vstack(sim_by_seed[:half])
    sim_train_groups = np.concatenate([i * 100_000 + np.arange(len(s)) // block for i, s in enumerate(sim_by_seed[:half])])
    sim_test = np.vstack(sim_by_seed[half:])
    sim_test_groups = np.concatenate([i * 100_000 + np.arange(len(s)) // block for i, s in enumerate(sim_by_seed[half:])])
    sim_train, sim_train_groups = balance(sim_train, sim_train_groups, len(train_r), rng)
    sim_test, sim_test_groups = balance(sim_test, sim_test_groups, len(test_r), rng)
    x_train = np.vstack([real[train_r], sim_train])
    y_train = np.r_[np.ones(len(train_r)), np.zeros(len(sim_train))]
    x_test = np.vstack([real[test_r], sim_test])
    y_test = np.r_[np.ones(len(test_r)), np.zeros(len(sim_test))]
    real_blocks = np.arange(len(test_r)) // block
    results = {}
    for name, model in (("logistic", Logistic()), ("tree", Tree(3)), ("forest", Forest(64, 4, seed))):
        model.fit(x_train, y_train)
        scores = model.score(x_test)
        entry = {"auc": auc(scores, y_test), "balanced_accuracy": balanced_accuracy(scores, y_test),
                 "brier": float(np.mean((scores - y_test) ** 2)),
                 **auc_interval(scores, y_test, real_blocks, sim_test_groups, alpha=alpha, samples=samples,
                                seed=seed + 1),
                 "permutation_importance": permutation_importance(model, x_test, y_test, seed=seed + 2)}
        if name == "logistic":
            entry["standardized_coefficients"] = dict(zip(FEATURES, model.coefficients.tolist()))
        results[name] = entry
    return {"status": "AVAILABLE", "classifiers": results, "train": {"real": len(train_r), "simulated": len(sim_train)},
            "test": {"real": len(test_r), "simulated": len(sim_test)}, "features": list(FEATURES),
            "interpretation": "AUC near 0.5 is not evidence of realism; AUC above 0.5 indicates a detectable gap."}


def gap_status(result: dict, classifier: str = "logistic") -> str:
    if result.get("status") != "AVAILABLE":
        return "NOT_AVAILABLE"
    lower = result["classifiers"][classifier]["one_sided_lower"]
    return "ESTABLISHED" if math.isfinite(lower) and lower > 0.5 else "NOT_ESTABLISHED"


# ----------------------------------------------------------------------------- support (OOD)


def knn_distance(query: np.ndarray, reference: np.ndarray, k: int = 5) -> np.ndarray:
    out = np.empty(len(query))
    for start in range(0, len(query), 512):
        chunk = query[start:start + 512]
        d = np.sqrt(((chunk[:, None, :] - reference[None, :, :]) ** 2).sum(-1))
        out[start:start + 512] = np.sort(d, axis=1)[:, min(k, reference.shape[0]) - 1]
    return out


def support(history: np.ndarray, sim_by_seed: list[np.ndarray], scaler: Standardizer, *, k: int = 5,
            quantile: float = 0.99, no_claim_fraction: float = 0.5) -> dict:
    """Out-of-support fraction of historical windows relative to the simulated ensemble's windows."""
    if len(sim_by_seed) < 2 or not len(history):
        return {"status": "NOT_AVAILABLE"}
    sims = [scaler(s) for s in sim_by_seed]
    loo = np.concatenate([knn_distance(s, np.vstack([o for j, o in enumerate(sims) if j != i]), k)
                          for i, s in enumerate(sims)])
    threshold = float(np.quantile(loo, quantile))
    distances = knn_distance(scaler(history), np.vstack(sims), k)
    fraction = float(np.mean(distances > threshold))
    return {"status": "AVAILABLE", "threshold": threshold, "out_of_support_fraction": fraction,
            "median_distance": float(np.median(distances)), "windows": int(len(history)),
            "label": "NO_CLAIM" if fraction > no_claim_fraction else "SUPPORTED_ENOUGH",
            "rule": f"{k}-NN distance > {quantile:.0%} of leave-one-seed-out simulated distances"}
