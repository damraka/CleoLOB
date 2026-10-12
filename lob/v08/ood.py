"""Out-of-distribution detection (M19).

Every detector maps a feature vector (window summaries, ``SUMMARY_NAMES``) to a
score where larger means more unusual. Detectors are fitted on in-distribution
reference vectors only; thresholds are set on a separate in-distribution
calibration split at the registered false-positive rate.

* ``support`` — fraction of coordinates outside the reference range (+/- margin);
* ``mahalanobis`` — shrinkage covariance (``lam`` towards the diagonal);
* ``knn`` — mean standardized distance to the ``k`` nearest reference vectors;
* ``ensemble`` — disagreement (standard deviation) of predictions from several
  models (passed in, e.g. per-family predicted costs);
* ``surprise`` — posterior predictive surprise (negative log predictive density
  per bin, standardized by the reference distribution of the same quantity);
* ``conformal`` — conformal p-value of the Mahalanobis nonconformity score
  (``1 - p``, so larger is more unusual; valid under exchangeability of the
  reference and test points).
"""
from __future__ import annotations

import numpy as np

DETECTORS = ("support", "mahalanobis", "knn", "ensemble", "surprise", "conformal")


class Reference:
    def __init__(self, x: np.ndarray, *, lam: float = 0.1, margin: float = 0.05, k: int = 5) -> None:
        x = np.asarray(x, float)
        if len(x) < 5:
            raise ValueError("need at least five reference vectors")
        self.x, self.k, self.margin = x, min(k, len(x) - 1), margin
        self.mean = x.mean(0)
        self.sd = x.std(0) + 1e-9
        z = (x - self.mean) / self.sd
        cov = np.cov(z.T) if x.shape[1] > 1 else np.atleast_2d(np.var(z))
        self.cov_inv = np.linalg.pinv((1 - lam) * cov + lam * np.eye(cov.shape[0]))
        self.lo, self.hi = x.min(0), x.max(0)
        self.ref_maha = self.mahalanobis(x)

    def support(self, x) -> np.ndarray:
        x = np.atleast_2d(x)
        pad = self.margin * (self.hi - self.lo + 1e-9)
        return np.mean((x < self.lo - pad) | (x > self.hi + pad), 1)

    def mahalanobis(self, x) -> np.ndarray:
        z = (np.atleast_2d(x) - self.mean) / self.sd
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", z, self.cov_inv, z), 0.0))

    def knn(self, x) -> np.ndarray:
        z = (np.atleast_2d(x) - self.mean) / self.sd
        r = (self.x - self.mean) / self.sd
        d = np.sqrt(((z[:, None, :] - r[None]) ** 2).sum(2))
        return np.sort(d, 1)[:, : self.k].mean(1)

    def conformal(self, x) -> np.ndarray:
        s = self.mahalanobis(x)
        ref = self.ref_maha
        p = (1 + (ref[None, :] >= s[:, None]).sum(1)) / (len(ref) + 1)
        return 1 - p

    def score(self, method: str, x) -> np.ndarray:
        if method not in ("support", "mahalanobis", "knn", "conformal"):
            raise ValueError(f"{method} needs model predictions; use ensemble_score/surprise_score")
        return getattr(self, method)(x)


def ensemble_score(predictions: np.ndarray) -> np.ndarray:
    """Rows: cases; columns: models. Disagreement = across-model standard deviation."""
    return np.asarray(predictions, float).std(1)


def surprise_score(nll: np.ndarray, reference_nll: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference_nll, float)
    return (np.asarray(nll, float) - ref.mean()) / (ref.std() + 1e-9)


def threshold(calibration_scores: np.ndarray, fpr: float) -> float:
    """Score threshold with at most ``fpr`` of in-distribution calibration scores above it."""
    s = np.sort(np.asarray(calibration_scores, float))
    k = int(np.ceil((1 - fpr) * (len(s) + 1))) - 1
    return float(s[min(max(k, 0), len(s) - 1)])


def evaluate(scores_id: np.ndarray, scores_shift: dict[str, np.ndarray], thr: float) -> dict:
    from .statistics import wilson
    id_flags = np.asarray(scores_id) > thr
    out = {"threshold": thr, "false_positive_rate": float(id_flags.mean()),
           "fpr_ci": list(wilson(int(id_flags.sum()), len(id_flags))), "n_id": int(len(id_flags)), "shifts": {}}
    for name, s in scores_shift.items():
        flags = np.asarray(s) > thr
        out["shifts"][name] = {"detection_rate": float(flags.mean()), "n": int(len(flags)),
                               "ci": list(wilson(int(flags.sum()), len(flags)))}
    return out
