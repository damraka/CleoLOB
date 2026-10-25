"""Marked Hawkes process family for aggressive-order streams (M72).

All members score the same events with the same per-event negative log-likelihood of (inter-arrival time, mark)
given the history, as in the v0.9 point-process module (``lob.v09.generators.point_process``), so they are
directly comparable in nats per event:

* ``MarkedPoisson`` - constant intensity per mark (v0.9, reused unchanged);
* ``UnmarkedHawkes`` - one self-exciting intensity for all orders, lambda(t) = mu + a sum_j beta exp(-beta(t - t_j)),
  and marks drawn independently from their empirical distribution: lambda_m(t) = p_m lambda(t). This is the H21
  baseline (excitation that ignores which kind of order arrived);
* ``MarkedHawkes`` - mark-dependent excitation alpha[m, k] (v0.9, reused unchanged).

``UnmarkedHawkes`` reuses the v0.9 exact blockwise excitation recursion and is fitted by Adam on (log mu, log a)
for each registered decay beta in {0.5, 2, 10}; the best log-likelihood wins.
"""
from __future__ import annotations

import math

import numpy as np

from ....v09.generators import point_process as PP

N_MARKS = PP.N_MARKS
MarkedPoisson = PP.MarkedPoisson
MarkedHawkes = PP.MarkedHawkes


def _excite(t: np.ndarray, beta: float) -> np.ndarray:
    """R_i = sum_{j < i} beta exp(-beta (t_i - t_j)) (exact, via the v0.9 blockwise recursion with one mark)."""
    return PP.MarkedHawkes.excitation(t, np.zeros(len(t), dtype=int), beta)[:, 0]


class UnmarkedHawkes:
    name, family = "unmarked_hawkes", "classical"

    def __init__(self, iters: int = 300, lr: float = 0.05, max_events: int = 60_000) -> None:
        self.iters, self.lr, self.max_events = iters, lr, max_events

    def _ll_grad(self, theta, beta, segs):
        mu, a = math.exp(theta[0]), math.exp(theta[1])
        ll, g_mu, g_a, count = 0.0, 0.0, 0.0, 0
        for t, _m in segs:
            R = _excite(t, beta)
            lam = mu + a * R[1:]
            lam = np.maximum(lam, 1e-12)
            dur = float(t[-1] - t[0])
            tail = 1.0 - np.exp(-beta * (t[-1] - t[:-1]))
            ll += float(np.log(lam).sum() - mu * dur - a * tail.sum())
            g_mu += float((1.0 / lam).sum() - dur)
            g_a += float((R[1:] / lam).sum() - tail.sum())
            count += len(t) - 1
        count = max(count, 1)
        return ll / count, np.array([g_mu * mu, g_a * a]) / count

    def fit(self, segs):
        keep, used = [], 0
        for t, m in segs:
            if used >= self.max_events:
                break
            keep.append((t, m))
            used += len(t)
        n = np.bincount(np.concatenate([m for _, m in keep]), minlength=N_MARKS).astype(float)
        self.p = np.maximum(n / n.sum(), 1e-9)
        T = sum(float(t[-1] - t[0]) for t, _ in keep)
        rate = max(n.sum() / max(T, 1e-9), 1e-6)
        best = None
        for beta in (0.5, 2.0, 10.0):
            theta = np.array([math.log(rate * 0.5), math.log(0.3)])
            m1, v1 = np.zeros(2), np.zeros(2)
            for it in range(1, self.iters + 1):
                _, g = self._ll_grad(theta, beta, keep)
                m1 = 0.9 * m1 + 0.1 * g
                v1 = 0.999 * v1 + 0.001 * g * g
                theta = theta + self.lr * (m1 / (1 - 0.9 ** it)) / (np.sqrt(v1 / (1 - 0.999 ** it)) + 1e-8)
            ll, _ = self._ll_grad(theta, beta, keep)
            if best is None or ll > best[0]:
                best = (ll, theta.copy(), beta)
        _, theta, beta = best
        self.mu, self.a, self.beta = math.exp(theta[0]), math.exp(theta[1]), beta
        self.fit_info = {"events_used": used, "beta": beta, "branching_ratio": self.a}
        return self

    def nll(self, segs) -> np.ndarray:
        out = []
        for t, m in segs:
            R = _excite(t, self.beta)
            dt = np.diff(t)
            lam = np.maximum(self.mu + self.a * R[1:], 1e-12)
            # compensator between t_{i-1} and t_i: mu dt + a (R_{i-1} + beta) (1 - e^{-beta dt}) / beta
            comp = self.mu * dt + self.a * (R[:-1] + self.beta) * (1 - np.exp(-self.beta * dt)) / self.beta
            out.append(-(np.log(lam) + np.log(self.p[m[1:]]) - comp))
        return np.concatenate(out) if out else np.zeros(0)

    def simulate(self, horizon: float, seed: int, *, max_events: int = 1_000_000) -> tuple[np.ndarray, np.ndarray]:
        """Ogata thinning on [0, horizon); returns (times, marks)."""
        rng = np.random.default_rng(seed)
        t, R, times = 0.0, 0.0, []
        while len(times) < max_events:
            lam_bar = self.mu + self.a * (R + self.beta)
            w = rng.exponential(1.0 / lam_bar)
            R = R * math.exp(-self.beta * w)
            t += w
            if t >= horizon:
                break
            if rng.random() * lam_bar <= self.mu + self.a * R:
                times.append(t)
                R += self.beta
        times = np.asarray(times)
        return times, rng.choice(N_MARKS, size=len(times), p=self.p / self.p.sum())


FAMILY = {"marked_poisson": MarkedPoisson, "unmarked_hawkes": UnmarkedHawkes, "marked_hawkes": MarkedHawkes}

__all__ = ["MarkedPoisson", "UnmarkedHawkes", "MarkedHawkes", "FAMILY", "N_MARKS"]
