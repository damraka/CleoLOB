"""Realism v2 metrics (workstreams 14-17, 24): paired domain-gap and support contrasts, MMD, precision/recall.

All window features are the sealed v0.6 whitelist (``lob.v06.domain_gap.FEATURES``)
on 60 s windows; real windows are split chronologically (first half train,
10-window purge, second half test) and simulated windows by seed halves.

* ``auc_contrast`` (H2) — two primary logistic classifiers trained on the *same*
  real training windows against a family's and the baseline's simulated
  training windows; both are scored on the *same* real test windows. The
  difference AUC(family) - AUC(baseline) is bootstrapped with shared resamples of
  real 10-window blocks and independent resamples of each model's simulated
  10-window blocks. Negative = the family is harder to distinguish from history.
* ``coverage_contrast`` (H3) — support coverage (1 - v0.6 out-of-support fraction,
  each model with its own leave-one-seed-out threshold) of the same historical
  windows; bootstrap over real blocks (shared) and simulated seeds (per model).
* ``mmd_rbf``, ``precision_recall`` (k-NN manifolds), ``roc_curve``,
  ``pr_curve``, ``calibration_curve`` and per-timescale features are descriptive.

A weak discriminator or a high coverage is never evidence that a generator is
correct; these are relative statements about detectability on registered data.
"""
from __future__ import annotations

import math

import numpy as np

from ...v06.domain_gap import Logistic, Standardizer, auc, chronological_split, knn_distance, window_features
from ...v06.inference import counts

BLOCK = 10


def _groups(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = np.vstack(arrays)
    g = np.concatenate([i * 100_000 + np.arange(len(a)) // BLOCK for i, a in enumerate(arrays)])
    return x, g


def _resample(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    units = [np.flatnonzero(groups == u) for u in np.unique(groups)]
    c = counts(len(units), rng)
    return np.concatenate([np.repeat(u, int(k)) for u, k in zip(units, c) if k])


def _classifier_scores(real: np.ndarray, sims: list[np.ndarray], train_r, test_r, rng) -> dict:
    half = max(1, len(sims) // 2)
    sim_train, _ = _groups(sims[:half])
    sim_test, sim_groups = _groups(sims[half:])
    take = rng.choice(len(sim_train), min(len(sim_train), len(train_r)), replace=False)
    x = np.vstack([real[train_r], sim_train[take]])
    y = np.r_[np.ones(len(train_r)), np.zeros(len(take))]
    model = Logistic().fit(x, y)
    return {"real": model.score(real[test_r]), "sim": model.score(sim_test), "sim_groups": sim_groups}


def auc_contrast(real: np.ndarray, family: list[np.ndarray], baseline: list[np.ndarray], *, alpha: float,
                 samples: int, seed: int) -> dict:
    if len(real) < 60 or len(family) < 2 or len(baseline) < 2:
        return {"status": "NOT_AVAILABLE", "reason": "too few real windows or simulated seeds"}
    rng = np.random.default_rng(seed)
    train_r, test_r = chronological_split(len(real))
    f = _classifier_scores(real, family, train_r, test_r, rng)
    b = _classifier_scores(real, baseline, train_r, test_r, rng)
    real_blocks = np.arange(len(test_r)) // BLOCK

    def value(ri, fi, bi):
        af = auc(np.r_[f["real"][ri], f["sim"][fi]], np.r_[np.ones(len(ri)), np.zeros(len(fi))])
        ab = auc(np.r_[b["real"][ri], b["sim"][bi]], np.r_[np.ones(len(ri)), np.zeros(len(bi))])
        return af, ab
    all_r, all_f, all_b = np.arange(len(test_r)), np.arange(len(f["sim"])), np.arange(len(b["sim"]))
    auc_f, auc_b = value(all_r, all_f, all_b)
    draws = np.empty((samples, 2))
    for s in range(samples):
        draws[s] = value(_resample(real_blocks, rng), _resample(f["sim_groups"], rng), _resample(b["sim_groups"], rng))
    diff = draws[:, 0] - draws[:, 1]
    low, high = float(np.nanquantile(diff, alpha / 2)), float(np.nanquantile(diff, 1 - alpha / 2))
    return {"status": "AVAILABLE", "auc_family": auc_f, "auc_baseline": auc_b, "difference": auc_f - auc_b,
            "ci_low": low, "ci_high": high, "alpha": alpha, "samples": samples,
            "auc_family_ci": [float(np.nanquantile(draws[:, 0], alpha / 2)), float(np.nanquantile(draws[:, 0], 1 - alpha / 2))],
            "auc_baseline_ci": [float(np.nanquantile(draws[:, 1], alpha / 2)), float(np.nanquantile(draws[:, 1], 1 - alpha / 2))],
            "test": {"real": int(len(test_r)), "family": int(len(all_f)), "baseline": int(len(all_b))},
            "scores": {"real_family": f["real"].tolist(), "sim_family": f["sim"].tolist(),
                       "real_baseline": b["real"].tolist(), "sim_baseline": b["sim"].tolist()}}


def h2_status(result: dict) -> tuple[str, list[str]]:
    if result.get("status") != "AVAILABLE":
        return "NOT_AVAILABLE", []
    qualifiers = []
    if result["auc_family_ci"][0] >= 0.99 and result["auc_baseline_ci"][0] >= 0.99:
        return "NOT_ESTABLISHED", ["VACUOUS"]
    if result["ci_high"] < 0:
        return "ESTABLISHED", qualifiers
    if result["ci_low"] > 0:
        return "FAILED", qualifiers
    return "NOT_ESTABLISHED", qualifiers


# ----------------------------------------------------------------------------- support coverage (H3)


class _Support:
    """Precomputed standardized distances for repeated seed-resampled support coverage."""

    def __init__(self, history: np.ndarray, sims: list[np.ndarray], scaler: Standardizer, k: int = 5) -> None:
        self.k = k
        self.sims = [scaler(s) for s in sims]
        self.h = scaler(history)
        self.offsets = np.cumsum([0] + [len(s) for s in self.sims])
        allsim = np.vstack(self.sims)
        self.d_hist = np.sqrt(((self.h[:, None, :] - allsim[None, :, :]) ** 2).sum(-1))
        self.d_sim = np.sqrt(((allsim[:, None, :] - allsim[None, :, :]) ** 2).sum(-1))

    def coverage(self, seeds: np.ndarray, rows: np.ndarray) -> float:
        cols = np.concatenate([np.arange(self.offsets[s], self.offsets[s + 1]) for s in seeds])
        loo = []
        for i, s in enumerate(seeds):
            own = np.arange(self.offsets[s], self.offsets[s + 1])
            others = np.concatenate([np.arange(self.offsets[t], self.offsets[t + 1]) for j, t in enumerate(seeds)
                                     if j != i and t != s] or [cols])
            k = min(self.k, len(others))
            loo.append(np.partition(self.d_sim[np.ix_(own, others)], k - 1, axis=1)[:, k - 1])
        threshold = float(np.quantile(np.concatenate(loo), 0.99))
        k = min(self.k, len(cols))
        dist = np.partition(self.d_hist[np.ix_(rows, cols)], k - 1, axis=1)[:, k - 1]
        return float(np.mean(dist <= threshold))


def coverage_contrast(history: np.ndarray, family: list[np.ndarray], baseline: list[np.ndarray], *,
                      alpha: float, samples: int, seed: int) -> dict:
    if len(history) < 60 or len(family) < 2 or len(baseline) < 2:
        return {"status": "NOT_AVAILABLE"}
    rng = np.random.default_rng(seed)
    f = _Support(history, family, Standardizer(np.vstack(family)))
    b = _Support(history, baseline, Standardizer(np.vstack(baseline)))
    rows = np.arange(len(history))
    blocks = rows // BLOCK
    cf = f.coverage(np.arange(len(family)), rows)
    cb = b.coverage(np.arange(len(baseline)), rows)
    draws = np.empty((samples, 2))
    for s in range(samples):
        r = _resample(blocks, rng)
        draws[s] = (f.coverage(rng.integers(0, len(family), len(family)), r),
                    b.coverage(rng.integers(0, len(baseline), len(baseline)), r))
    diff = draws[:, 0] - draws[:, 1]
    return {"status": "AVAILABLE", "coverage_family": cf, "coverage_baseline": cb, "difference": cf - cb,
            "ci_low": float(np.quantile(diff, alpha / 2)), "ci_high": float(np.quantile(diff, 1 - alpha / 2)),
            "alpha": alpha, "samples": samples, "windows": int(len(history))}


def h3_status(result: dict, margin: float = 0.05) -> tuple[str, list[str]]:
    if result.get("status") != "AVAILABLE":
        return "NOT_AVAILABLE", []
    qualifiers = ["VACUOUS"] if max(result["coverage_family"], result["coverage_baseline"]) < 0.01 else []
    if result["ci_low"] > 0 and result["difference"] >= margin:
        return "ESTABLISHED", qualifiers
    if result["ci_high"] < 0:
        return "FAILED", qualifiers
    return "NOT_ESTABLISHED", qualifiers


# ----------------------------------------------------------------------------- descriptive metrics


def mmd_rbf(x: np.ndarray, y: np.ndarray, *, max_rows: int = 1500, seed: int = 0) -> dict:
    """Unbiased squared MMD with an RBF kernel (median-heuristic bandwidth) on jointly standardized windows."""
    rng = np.random.default_rng(seed)
    x = x[rng.choice(len(x), min(len(x), max_rows), replace=False)]
    y = y[rng.choice(len(y), min(len(y), max_rows), replace=False)]
    s = Standardizer(np.vstack([x, y]))
    x, y = s(x), s(y)
    z = np.vstack([x, y])
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    bandwidth = float(np.median(d2[np.triu_indices(len(z), 1)]))
    k = np.exp(-d2 / max(bandwidth, 1e-12))
    n, m = len(x), len(y)
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    value = ((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
             - 2 * kxy.mean())
    return {"mmd2": float(value), "bandwidth": bandwidth, "n_real": n, "n_sim": m}


def precision_recall(real: np.ndarray, sim: np.ndarray, *, k: int = 5, max_rows: int = 3000, seed: int = 0) -> dict:
    """k-NN manifold precision (sim inside real support) and recall (real inside sim support)."""
    rng = np.random.default_rng(seed)
    real = real[rng.choice(len(real), min(len(real), max_rows), replace=False)]
    sim = sim[rng.choice(len(sim), min(len(sim), max_rows), replace=False)]
    s = Standardizer(np.vstack([real, sim]))
    real, sim = s(real), s(sim)
    r_radius = knn_distance(real, real, k + 1)
    s_radius = knn_distance(sim, sim, k + 1)

    def inside(query, ref, radius):
        out = np.zeros(len(query), bool)
        for start in range(0, len(query), 512):
            d = np.sqrt(((query[start:start + 512, None, :] - ref[None, :, :]) ** 2).sum(-1))
            out[start:start + 512] = np.any(d <= radius[None, :], axis=1)
        return out
    return {"precision": float(inside(sim, real, r_radius).mean()), "recall": float(inside(real, sim, s_radius).mean()),
            "k": k}


def roc_curve(scores_real: np.ndarray, scores_sim: np.ndarray, points: int = 51) -> dict:
    thresholds = np.quantile(np.r_[scores_real, scores_sim], np.linspace(0, 1, points))
    tpr = [float(np.mean(scores_real >= t)) for t in thresholds]
    fpr = [float(np.mean(scores_sim >= t)) for t in thresholds]
    return {"threshold": thresholds.tolist(), "tpr": tpr, "fpr": fpr}


def pr_curve(scores_real: np.ndarray, scores_sim: np.ndarray, points: int = 51) -> dict:
    thresholds = np.quantile(np.r_[scores_real, scores_sim], np.linspace(0, 1, points))
    precision, recall = [], []
    for t in thresholds:
        tp, fp = np.sum(scores_real >= t), np.sum(scores_sim >= t)
        precision.append(float(tp / (tp + fp)) if tp + fp else 1.0)
        recall.append(float(tp / len(scores_real)))
    return {"threshold": thresholds.tolist(), "precision": precision, "recall": recall}


def calibration_curve(scores_real: np.ndarray, scores_sim: np.ndarray, bins: int = 10) -> dict:
    s = np.r_[scores_real, scores_sim]
    y = np.r_[np.ones(len(scores_real)), np.zeros(len(scores_sim))]
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi if hi < 1 else s <= hi)
        if m.any():
            rows.append({"bin": [float(lo), float(hi)], "mean_score": float(s[m].mean()), "real_fraction": float(y[m].mean()),
                         "n": int(m.sum())})
    return {"bins": rows, "brier": float(np.mean((s - y) ** 2))}


TIMESCALES = (10.0, 60.0, 300.0)


def multiscale_features(tape, scales: tuple[float, ...] = TIMESCALES) -> dict:
    """The v0.6 window features at several window lengths (descriptive multi-timescale realism)."""
    return {str(int(s)): window_features(tape, s) for s in scales}


def scale_auc(real: np.ndarray, sims: list[np.ndarray], *, seed: int) -> dict:
    """Descriptive held-out logistic AUC at one timescale (chronological real split, seed-half simulated split)."""
    if len(real) < 60 or len(sims) < 2:
        return {"status": "NOT_AVAILABLE", "reason": "too few windows or simulated units"}
    rng = np.random.default_rng(seed)
    train_r, test_r = chronological_split(len(real))
    s = _classifier_scores(real, sims, train_r, test_r, rng)
    value = auc(np.r_[s["real"], s["sim"]], np.r_[np.ones(len(s["real"])), np.zeros(len(s["sim"]))])
    return {"status": "AVAILABLE", "auc": value, "real_windows": int(len(real)), "simulated_windows": int(len(s["sim"]))}


def finite_or_none(value) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


# ----------------------------------------------------------------------------- domain-gap attribution (16, 17)

from ...v06.domain_gap import FEATURES  # noqa: E402

FEATURE_FAMILIES = {
    "spread": ("spread_mean", "spread_sd", "abs_microprice_dev", "mean_level_gap"),
    "depth": ("log_depth_l1", "log_depth5", "log_depth10", "concentration_mean"),
    "imbalance": ("abs_imbalance_mean", "imbalance_sd"),
    "returns_volatility": ("return_sd", "nonzero_return_fraction", "abs_return_window"),
    "event_rate": ("log1p_trades", "log1p_book_changes", "log1p_additions", "log1p_cancellations"),
    "sizes": ("mean_log_trade_size", "max_log_trade_size", "mean_log_add_size"),
}


def _auc_subset(real, sims, columns, train_r, test_r, seed):
    rng = np.random.default_rng(seed)
    s = _classifier_scores(real[:, columns], [x[:, columns] for x in sims], train_r, test_r, rng)
    return auc(np.r_[s["real"], s["sim"]], np.r_[np.ones(len(s["real"])), np.zeros(len(s["sim"]))]), s


def discriminator_outputs(real: np.ndarray, sims: list[np.ndarray], *, seed: int) -> dict:
    """Per-window scores, labels, split and fold ids of the primary classifier (stored for ROC/PR/calibration)."""
    train_r, test_r = chronological_split(len(real))
    value, s = _auc_subset(real, sims, list(range(len(FEATURES))), train_r, test_r, seed)
    half = max(1, len(sims) // 2)
    sim_seed_ids = np.concatenate([np.full(len(x), i) for i, x in enumerate(sims[half:])])
    rows = ([{"label": 1, "split": "test", "fold": 0, "unit": int(i), "score": float(v)}
             for i, v in zip(test_r, s["real"])] +
            [{"label": 0, "split": "test", "fold": 0, "unit": int(u), "score": float(v)}
             for u, v in zip(sim_seed_ids, s["sim"])])
    return {"auc": value, "rows": rows}


def reproduce_auc(rows: list[dict]) -> float:
    scores = np.asarray([r["score"] for r in rows])
    labels = np.asarray([r["label"] for r in rows])
    return auc(scores, labels)


def attribution(real: np.ndarray, sims: list[np.ndarray], *, seed: int, samples: int = 200) -> dict:
    """Feature-family ablations: AUC with each family excluded and with only that family; subgroup AUCs."""
    if len(real) < 60 or len(sims) < 2:
        return {"status": "NOT_AVAILABLE"}
    train_r, test_r = chronological_split(len(real))
    full, base = _auc_subset(real, sims, list(range(len(FEATURES))), train_r, test_r, seed)
    out = {"auc_all": full, "families": {}}
    blocks = np.arange(len(test_r)) // BLOCK
    rng = np.random.default_rng(seed + 1)
    for name, cols in FEATURE_FAMILIES.items():
        idx = [FEATURES.index(c) for c in cols]
        rest = [i for i in range(len(FEATURES)) if i not in idx]
        excluded, s_ex = _auc_subset(real, sims, rest, train_r, test_r, seed)
        only, _ = _auc_subset(real, sims, idx, train_r, test_r, seed)
        drops = []
        for _ in range(samples):
            r = _resample(blocks, rng)
            si = rng.integers(0, len(base["sim"]), len(base["sim"]))
            y = np.r_[np.ones(len(r)), np.zeros(len(si))]
            drops.append(auc(np.r_[base["real"][r], base["sim"][si]], y)
                         - auc(np.r_[s_ex["real"][r], s_ex["sim"][si]], y))
        out["families"][name] = {"auc_excluded": excluded, "auc_only": only, "drop": full - excluded,
                                 "drop_ci": [float(np.quantile(drops, 0.025)), float(np.quantile(drops, 0.975))]}
    out["ranking_by_only_auc"] = sorted(out["families"], key=lambda k: -out["families"][k]["auc_only"])
    spread = real[test_r, FEATURES.index("spread_mean")]
    cuts = np.quantile(spread, [1 / 3, 2 / 3])
    groups = np.searchsorted(cuts, spread)
    out["subgroups_by_spread_tercile"] = {
        str(g): auc(np.r_[base["real"][groups == g], base["sim"]],
                    np.r_[np.ones(int((groups == g).sum())), np.zeros(len(base["sim"]))]) for g in range(3)}
    out["interpretation"] = "associational attribution of detectability; not a causal account"
    return out
