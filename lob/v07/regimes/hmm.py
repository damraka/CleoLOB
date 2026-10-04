"""Latent-regime models (workstream 21): Gaussian hidden Markov models on block features.

``GaussianHMM(k).fit(x)`` runs Baum-Welch with diagonal covariances from a seeded k-means++
start; ``score`` is the log-likelihood (forward algorithm, log space) and ``bic`` the Bayesian
information criterion, so predictive utility can be weighed against complexity (k = 1 is the
no-regime baseline). Latent states are statistical clusters of market observables; they are
never interpreted as participant intent.
"""
from __future__ import annotations

import numpy as np


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


class GaussianHMM:
    def __init__(self, k: int, *, iterations: int = 100, seed: int = 0, floor: float = 1e-3) -> None:
        self.k, self.iterations, self.seed, self.floor = k, iterations, seed, floor

    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - self.means[None]
        return -0.5 * np.sum(diff ** 2 / self.vars[None] + np.log(2 * np.pi * self.vars[None]), axis=2)

    def _forward_backward(self, x):
        e = self._log_emission(x)
        n = len(x)
        la = np.log(self.trans)
        alpha = np.empty((n, self.k))
        alpha[0] = np.log(self.start) + e[0]
        for t in range(1, n):
            alpha[t] = e[t] + _logsumexp(alpha[t - 1][:, None] + la, axis=0)
        beta = np.zeros((n, self.k))
        for t in range(n - 2, -1, -1):
            beta[t] = _logsumexp(la + (e[t + 1] + beta[t + 1])[None, :], axis=1)
        loglik = float(_logsumexp(alpha[-1], axis=0))
        gamma = np.exp(alpha + beta - loglik)
        xi = np.exp(alpha[:-1, :, None] + la[None] + (e[1:] + beta[1:])[:, None, :] - loglik)
        return loglik, gamma, xi

    def fit(self, x: np.ndarray) -> GaussianHMM:
        x = np.asarray(x, float)
        rng = np.random.default_rng(self.seed)
        centres = [x[rng.integers(len(x))]]
        for _ in range(1, self.k):
            d = np.min([np.sum((x - c) ** 2, axis=1) for c in centres], axis=0)
            centres.append(x[rng.choice(len(x), p=d / d.sum())])
        self.means = np.asarray(centres)
        self.vars = np.tile(np.maximum(x.var(0), self.floor), (self.k, 1))
        self.start = np.full(self.k, 1 / self.k)
        self.trans = np.full((self.k, self.k), 0.1 / max(self.k - 1, 1)) + np.eye(self.k) * (0.9 - 0.1 / max(self.k - 1, 1))
        if self.k == 1:
            self.trans = np.ones((1, 1))
        previous = -np.inf
        for _ in range(self.iterations):
            loglik, gamma, xi = self._forward_backward(x)
            weights = gamma.sum(0) + 1e-12
            self.start = gamma[0] / gamma[0].sum()
            self.trans = xi.sum(0) / xi.sum(0).sum(1, keepdims=True)
            self.means = (gamma.T @ x) / weights[:, None]
            self.vars = np.maximum((gamma.T @ x ** 2) / weights[:, None] - self.means ** 2, self.floor)
            if loglik - previous < 1e-6:
                break
            previous = loglik
        self.loglik = loglik
        order = np.argsort(self.means[:, 0])          # deterministic labelling: by the first feature
        self.means, self.vars, self.start = self.means[order], self.vars[order], self.start[order]
        self.trans = self.trans[np.ix_(order, order)]
        return self

    def score(self, x: np.ndarray) -> float:
        return self._forward_backward(np.asarray(x, float))[0]

    def states(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self._forward_backward(np.asarray(x, float))[1], axis=1)

    @property
    def parameters(self) -> int:
        d = self.means.shape[1]
        return self.k * 2 * d + self.k * (self.k - 1) + (self.k - 1)

    def bic(self, x: np.ndarray) -> float:
        return -2 * self.score(x) + self.parameters * np.log(len(x))


def sample(means, sds, trans, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = len(means)
    s = np.empty(n, int)
    s[0] = rng.integers(k)
    for t in range(1, n):
        s[t] = rng.choice(k, p=trans[s[t - 1]])
    x = np.asarray(means)[s] + rng.normal(size=(n, len(means[0]))) * np.asarray(sds)[s]
    return x, s
