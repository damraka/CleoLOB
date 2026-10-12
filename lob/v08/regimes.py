"""Regime detection framework (M5).

Frames are cut into blocks (default 30 s); each block gets a feature vector
(``BLOCK_FEATURES``): realized volatility, mean log touch depth, mean spread,
order-flow imbalance and event activity. Labelers are fitted on development
blocks only and then applied unchanged:

* ``threshold`` — activity terciles; ``volatility`` — volatility terciles;
  ``liquidity`` — depth terciles; ``spread`` — tight (<= 1.5 ticks) vs wide;
  ``ofi`` — order-flow imbalance sell / neutral / buy;
* ``kmeans`` — k-means (k = 3, seeded k-means++) on standardized features;
* ``changepoint`` — segments from binary segmentation of activity, each segment
  labelled by its activity tercile.

``compare`` scores every labeler on evaluation blocks: persistence (mean run
length), separation (between/within variance of activity), next-block
predictive R^2 of activity and volatility, and label-distribution shift versus
development. A shuffled-label negative control is always reported.
"""
from __future__ import annotations

import math

import numpy as np

BLOCK_FEATURES = ("volatility", "log_depth", "spread", "ofi", "activity")
METHODS = ("threshold", "volatility", "liquidity", "spread", "ofi", "kmeans", "changepoint")


def block_features(frame, block: int = 300) -> np.ndarray:
    rows = []
    for start in range(0, len(frame) - block + 1, block):
        part = frame.slice(start, start + block)
        v = part.valid
        if v.mean() < 0.9:
            rows.append([np.nan] * len(BLOCK_FEATURES))
            continue
        mid = part.mid.astype(float)
        vol = float(np.std(np.diff(mid))) * math.sqrt(10)
        depth = float(np.mean(np.log(np.maximum(part.state[v, 3] + part.state[v, 4], 1e-6))))
        spread = float(part.state[v, 0].mean())
        buy, sell = part.volume[:, 0].sum(), part.volume[:, 1].sum()
        ofi = float((buy - sell) / (buy + sell)) if buy + sell > 0 else 0.0
        activity = float(part.counts[v].sum(1).mean())
        rows.append([vol, depth, spread, ofi, activity])
    return np.asarray(rows, float).reshape(-1, len(BLOCK_FEATURES))


def _terciles(x: np.ndarray) -> list[float]:
    x = x[np.isfinite(x)]
    return np.quantile(x, [1 / 3, 2 / 3]).tolist() if len(x) else [0.0, 0.0]


def _kmeans(x: np.ndarray, k: int, seed: int, iterations: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min([((x - c) ** 2).sum(1) for c in centres], axis=0)
        centres.append(x[rng.choice(len(x), p=d / d.sum())] if d.sum() > 0 else x[rng.integers(len(x))])
    c = np.asarray(centres)
    for _ in range(iterations):
        labels = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(2), 1)
        new = np.asarray([x[labels == j].mean(0) if np.any(labels == j) else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    order = np.argsort(c[:, -1])            # sort clusters by standardized activity for stable labels
    return c[order]


def binary_segmentation(x: np.ndarray, *, penalty: float | None = None, min_size: int = 4,
                        max_changes: int = 20) -> list[int]:
    """Change points (indices) for a mean shift by recursive binary segmentation with a BIC-type penalty."""
    x = np.asarray(x, float)
    sd = float(np.std(x)) or 1.0
    penalty = penalty if penalty is not None else 2.0 * math.log(max(len(x), 2)) * sd * sd
    changes: list[int] = []

    def split(lo: int, hi: int) -> None:
        if len(changes) >= max_changes or hi - lo < 2 * min_size:
            return
        seg = x[lo:hi]
        total = ((seg - seg.mean()) ** 2).sum()
        best, at = 0.0, None
        csum, csq = np.cumsum(seg), np.cumsum(seg * seg)
        n = len(seg)
        for k in range(min_size, n - min_size + 1):
            left = csq[k - 1] - csum[k - 1] ** 2 / k
            right = (csq[-1] - csq[k - 1]) - (csum[-1] - csum[k - 1]) ** 2 / (n - k)
            gain = total - left - right
            if gain > best:
                best, at = gain, k
        if at is not None and best > penalty:
            changes.append(lo + at)
            split(lo, lo + at)
            split(lo + at, hi)

    split(0, len(x))
    return sorted(changes)


class Labeler:
    def __init__(self, method: str, *, seed: int = 0) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown regime method {method!r}")
        self.method, self.seed = method, seed
        self.params: dict = {}

    def fit(self, features: np.ndarray) -> Labeler:
        f = features[np.all(np.isfinite(features), 1)]
        if len(f) < 6:
            raise ValueError("need at least six valid development blocks")
        col = {"threshold": 4, "volatility": 0, "liquidity": 1, "changepoint": 4}
        if self.method in col:
            self.params["edges"] = _terciles(f[:, col[self.method]])
        elif self.method == "spread":
            self.params["edges"] = [1.5]
        elif self.method == "ofi":
            self.params["edges"] = [-0.2, 0.2]
        elif self.method == "kmeans":
            mean, sd = f.mean(0), f.std(0) + 1e-9
            self.params.update(mean=mean.tolist(), sd=sd.tolist(),
                               centres=_kmeans((f - mean) / sd, 3, self.seed).tolist())
        return self

    def label(self, features: np.ndarray) -> np.ndarray:
        out = np.full(len(features), -1)
        ok = np.all(np.isfinite(features), 1)
        if not ok.any():
            return out
        f = features[ok]
        if self.method == "kmeans":
            z = (f - np.asarray(self.params["mean"])) / np.asarray(self.params["sd"])
            c = np.asarray(self.params["centres"])
            out[ok] = np.argmin(((z[:, None, :] - c[None]) ** 2).sum(2), 1)
            return out
        if self.method == "changepoint":
            changes = binary_segmentation(f[:, 4])
            bounds = [0, *changes, len(f)]
            labels = np.zeros(len(f), int)
            for a, b in zip(bounds[:-1], bounds[1:]):
                labels[a:b] = int(np.searchsorted(self.params["edges"], f[a:b, 4].mean(), side="right"))
            out[ok] = labels
            return out
        col = {"threshold": 4, "volatility": 0, "liquidity": 1, "spread": 2, "ofi": 3}[self.method]
        out[ok] = np.searchsorted(self.params["edges"], f[:, col], side="right")
        return out

    def to_dict(self) -> dict:
        return {"method": self.method, "seed": self.seed, "params": self.params}


def _persistence(labels: np.ndarray) -> float:
    labels = labels[labels >= 0]
    if len(labels) < 2:
        return 0.0
    runs = 1 + int(np.sum(labels[1:] != labels[:-1]))
    return len(labels) / runs


def _separation(labels: np.ndarray, y: np.ndarray) -> float:
    ok = (labels >= 0) & np.isfinite(y)
    labels, y = labels[ok], y[ok]
    groups = [y[labels == g] for g in np.unique(labels)]
    if len(groups) < 2:
        return 0.0
    between = sum(len(g) * (g.mean() - y.mean()) ** 2 for g in groups)
    within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    return float(between / within) if within > 0 else math.inf


def _next_r2(labels: np.ndarray, y: np.ndarray) -> float:
    """R^2 of next-block ``y`` predicted by the current label's mean (fit on the same sequence)."""
    cur, nxt = labels[:-1], y[1:]
    ok = (cur >= 0) & np.isfinite(nxt)
    cur, nxt = cur[ok], nxt[ok]
    if len(nxt) < 3 or np.var(nxt) == 0:
        return 0.0
    pred = np.asarray([nxt[cur == c].mean() for c in cur])
    return float(1 - ((nxt - pred) ** 2).sum() / ((nxt - nxt.mean()) ** 2).sum())


def _shift(dev: np.ndarray, ev: np.ndarray, k: int) -> float:
    p = np.bincount(dev[dev >= 0], minlength=k) / max((dev >= 0).sum(), 1)
    q = np.bincount(ev[ev >= 0], minlength=k) / max((ev >= 0).sum(), 1)
    return float(0.5 * np.abs(p - q).sum())


def compare(dev_features: np.ndarray, eval_features: np.ndarray, *, seed: int = 0) -> dict:
    out = {}
    rng = np.random.default_rng(seed)
    for method in METHODS:
        lab = Labeler(method, seed=seed).fit(dev_features)
        dev_labels, ev_labels = lab.label(dev_features), lab.label(eval_features)
        shuffled = ev_labels.copy()
        valid = shuffled >= 0
        shuffled[valid] = rng.permutation(shuffled[valid])
        k = int(max(dev_labels.max(), ev_labels.max(), 0)) + 1
        out[method] = {
            "labeler": lab.to_dict(), "persistence_blocks": _persistence(ev_labels),
            "separation_activity": _separation(ev_labels, eval_features[:, 4]),
            "next_block_r2_activity": _next_r2(ev_labels, eval_features[:, 4]),
            "next_block_r2_volatility": _next_r2(ev_labels, eval_features[:, 0]),
            "negative_control_shuffled_r2_activity": _next_r2(shuffled, eval_features[:, 4]),
            "distribution_shift_tv": _shift(dev_labels, ev_labels, k),
            "label_counts": np.bincount(ev_labels[ev_labels >= 0], minlength=k).tolist(),
            "labels_fitted_on": "development blocks only"}
    return out
