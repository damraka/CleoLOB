"""Market-state ontology (M54), regime discovery (M55) and structural-break / online drift detection (M56).

Ontology: every 5-minute block of a day is labelled on three registered axes from its features (thresholds are
quantiles of the calibrate pool, passed in as ``cuts``):

* volatility: ``calm`` / ``normal`` / ``volatile`` (60 s realized volatility terciles);
* liquidity: ``deep`` / ``thin`` (median log touch depth above / below the median);
* activity: ``quiet`` / ``active`` (trade count above / below the median);

the label is the tuple, e.g. ``volatile|thin|active``. Discovery: a Gaussian mixture (diagonal, EM, seeded
k-means++ initialization) on standardized block features with BIC model selection over K in 2..6, so data-driven
regimes can be compared with the ontology. Drift: CUSUM (two-sided, standardized) and Bayesian online change-point
detection (Adams & MacKay 2007, Gaussian with unknown mean, constant hazard) on any feature series.
"""
from __future__ import annotations

import math

import numpy as np

BLOCK_S = 300
AXES = ("volatility", "liquidity", "activity")


def block_features(features: dict[str, np.ndarray], *, interval_s: float = 1.0) -> np.ndarray:
    k = int(BLOCK_S / interval_s)
    n = len(features["spread"]) // k
    out = []
    for b in range(n):
        sl = slice(b * k, (b + 1) * k)
        v = np.asarray(features["valid"][sl], bool)
        if v.mean() < 0.9:
            out.append([np.nan] * 4)
            continue
        out.append([float(np.nanmean(features["rv_60s"][sl][v])), float(np.median(features["log_touch_depth"][sl][v])),
                     float(features["trade_count"][sl][v].sum()), float(features["spread"][sl][v].mean())])
    return np.asarray(out)


def cuts_from(blocks: list[np.ndarray]) -> dict:
    X = np.vstack(blocks)
    X = X[np.isfinite(X).all(1)]
    return {"volatility": np.quantile(X[:, 0], [1 / 3, 2 / 3]).tolist(), "liquidity": float(np.median(X[:, 1])),
            "activity": float(np.median(X[:, 2]))}


def label(blocks: np.ndarray, cuts: dict) -> list[str | None]:
    out = []
    for r in blocks:
        if not np.isfinite(r).all():
            out.append(None)
            continue
        vol = ("calm", "normal", "volatile")[int(np.searchsorted(cuts["volatility"], r[0]))]
        liq = "deep" if r[1] >= cuts["liquidity"] else "thin"
        act = "active" if r[2] >= cuts["activity"] else "quiet"
        out.append(f"{vol}|{liq}|{act}")
    return out


def majority(labels: list[str | None], axis: str = "volatility") -> str | None:
    i = AXES.index(axis)
    vals = [x.split("|")[i] for x in labels if x]
    if not vals:
        return None
    u, c = np.unique(vals, return_counts=True)
    return str(u[np.argmax(c)])


class GaussianMixture:
    def __init__(self, k: int, *, seed: int = 0, iters: int = 200) -> None:
        self.k, self.seed, self.iters = k, seed, iters

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        c = [X[rng.integers(n)]]
        for _ in range(1, self.k):                         # k-means++ seeding
            dist = np.min([((X - m) ** 2).sum(1) for m in c], axis=0)
            c.append(X[rng.choice(n, p=dist / dist.sum())])
        self.mu = np.asarray(c)
        self.var = np.tile(X.var(0) + 1e-6, (self.k, 1))
        self.w = np.full(self.k, 1 / self.k)
        prev = -math.inf
        for _ in range(self.iters):
            r, ll = self._resp(X)
            nk = r.sum(0) + 1e-12
            self.w = nk / n
            self.mu = (r.T @ X) / nk[:, None]
            self.var = (r.T @ X ** 2) / nk[:, None] - self.mu ** 2 + 1e-6
            if ll - prev < 1e-8 * abs(ll):
                break
            prev = ll
        self.loglik = ll
        return self

    def _logp(self, X):
        return (-0.5 * (((X[:, None, :] - self.mu[None]) ** 2) / self.var[None]
                        + np.log(2 * math.pi * self.var[None])).sum(-1) + np.log(self.w)[None])

    def _resp(self, X):
        lp = self._logp(X)
        m = lp.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(lp - m).sum(1))
        return np.exp(lp - lse[:, None]), float(lse.sum())

    def predict(self, X):
        return np.argmax(self._logp(X), 1)

    def bic(self, X) -> float:
        p = self.k * (2 * X.shape[1]) + self.k - 1
        return -2 * self._resp(X)[1] + p * math.log(len(X))


def discover(X: np.ndarray, *, ks=range(2, 7), seed: int = 0) -> dict:
    X = X[np.isfinite(X).all(1)]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    fits = {k: GaussianMixture(k, seed=seed).fit(Z) for k in ks if len(Z) > 5 * k}
    if not fits:
        return {"k": None, "bic": {}}
    bic = {k: f.bic(Z) for k, f in fits.items()}
    best = min(bic, key=bic.get)
    return {"k": best, "bic": bic, "model": fits[best], "standardize": (mu, sd),
            "labels": fits[best].predict(Z).tolist()}


def cusum(x: np.ndarray, *, threshold: float = 5.0, drift: float = 0.5, warmup: int = 50) -> list[int]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) <= warmup:
        return []
    mu, sd = x[:warmup].mean(), x[:warmup].std() + 1e-12
    z = (x - mu) / sd
    gp = gn = 0.0
    alarms = []
    for i in range(warmup, len(z)):
        gp = max(0.0, gp + z[i] - drift)
        gn = max(0.0, gn - z[i] - drift)
        if gp > threshold or gn > threshold:
            alarms.append(i)
            gp = gn = 0.0
    return alarms


def bocpd(x: np.ndarray, *, hazard: float = 1 / 250, mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0,
          beta0: float = 1.0, min_run: int = 20, drop_to: int = 5) -> dict:
    """Bayesian online change-point detection (Normal-Gamma conjugate, Student-t predictive, constant hazard).

    Returns the MAP run-length path and the change points: times where the MAP run length collapses to
    <= ``drop_to`` after having exceeded ``min_run``. Inputs are standardized by the first 50 observations.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) > 50:
        x = (x - x[:50].mean()) / (x[:50].std() + 1e-12)
    lg = np.vectorize(math.lgamma)
    R = np.array([1.0])
    mu, kappa, alpha, beta = (np.array([v]) for v in (mu0, kappa0, alpha0, beta0))
    cps, rl = [], []
    for t, xt in enumerate(x):
        nu = 2 * alpha
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        z = (xt - mu) / scale
        logpred = lg((nu + 1) / 2) - lg(nu / 2) - 0.5 * np.log(nu * math.pi) - np.log(scale)             - (nu + 1) / 2 * np.log1p(z * z / nu)
        pred = np.exp(logpred)
        growth = R * pred * (1 - hazard)
        cp = float((R * pred * hazard).sum())
        R = np.r_[cp, growth]
        R = R / R.sum()
        beta_new = beta + kappa * (xt - mu) ** 2 / (2 * (kappa + 1))      # uses the pre-update mean
        mu = np.r_[mu0, (kappa * mu + xt) / (kappa + 1)]
        beta = np.r_[beta0, beta_new]
        kappa = np.r_[kappa0, kappa + 1]
        alpha = np.r_[alpha0, alpha + 0.5]
        if len(R) > 2000:                                                 # truncate negligible long runs
            R, mu, kappa, alpha, beta = (a[:2000] for a in (R, mu, kappa, alpha, beta))
            R = R / R.sum()
        rl.append(int(np.argmax(R)))
        if len(rl) > 1 and rl[-1] <= drop_to and max(rl[max(0, len(rl) - 10):-1]) >= min_run                 and (not cps or t - cps[-1] > min_run):
            cps.append(t)
    return {"run_length": rl, "change_points": cps}


__all__ = ["block_features", "cuts_from", "label", "majority", "GaussianMixture", "discover", "cusum", "bocpd",
           "AXES", "BLOCK_S"]
