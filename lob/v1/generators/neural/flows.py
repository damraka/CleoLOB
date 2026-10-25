"""Conditional density models of order marks: a normalizing flow (M86) and conditional flow matching (M87).

Target x = (log aggressive-order size, log inter-arrival time) of the next aggressive order; context c = (spread
ticks, 5-level imbalance, log1p trailing 1 s order count, side) at the previous order. Both models learn p(x | c).

* ``ConditionalFlow`` (RealNVP): K affine coupling layers alternating the two coordinates, scale and shift from an
  MLP of (the other coordinate, c); exact log-likelihood by change of variables, exact sampling by inversion.
* ``FlowMatching``: a velocity field v(x_t, t, c) trained with the conditional flow-matching objective on the linear
  path x_t = (1 - t) x0 + t x1, x0 ~ N(0, I); samples by Euler integration of dx/dt = v over 64 steps. Its
  likelihood is not computed (labelled); it is scored by the energy distance of samples to held-out data.

A classical baseline (``GaussianConditional``: linear-Gaussian regression of x on c) gives the comparison point.
"""
from __future__ import annotations

import math

import numpy as np

from .base import NeuralModel, fingerprint, mlp, torch


def mark_data(cache: dict) -> tuple[np.ndarray, np.ndarray]:
    from ....v09.generators import point_process as PP
    ev = PP.aggressive_events(cache)
    t, side, size = ev["t"], ev["side"], ev["size"]
    if len(t) < 10:
        return np.zeros((0, 2), np.float32), np.zeros((0, 4), np.float32)
    dt = np.diff(t)
    keep = dt > 0
    b = np.clip(np.floor(t[:-1] * 10).astype(int), 0, len(cache["state"]) - 1)
    st = np.asarray(cache["state"], float)[b]
    ok = np.asarray(cache["valid"], bool)[b] & keep
    cnt = np.searchsorted(t, t[:-1]) - np.searchsorted(t, t[:-1] - 1.0)
    x = np.column_stack([np.log(np.maximum(size[1:], 1e-9)), np.log(np.where(keep, dt, 1.0))])
    c = np.column_stack([st[:, 0], st[:, 1], np.log1p(cnt), side[:-1]])
    return x[ok].astype(np.float32), c[ok].astype(np.float32)


class _Std:
    def standardize(self, x, c):
        self.mx, self.sx = x.mean(0), x.std(0) + 1e-6
        self.mc, self.sc = c.mean(0), c.std(0) + 1e-6

    def zx(self, x):
        return (x - self.mx) / self.sx

    def zc(self, c):
        return (c - self.mc) / self.sc

    def extra_state(self):
        return {k: getattr(self, k).tolist() for k in ("mx", "sx", "mc", "sc")}

    def load_extra(self, e):
        for k in ("mx", "sx", "mc", "sc"):
            setattr(self, k, np.asarray(e[k], np.float32))


class ConditionalFlow(_Std, NeuralModel):
    name = "conditional_flow"

    def __init__(self, *, layers: int = 6, hidden: int = 48, steps: int = 2000, batch: int = 512, lr: float = 2e-3,
                 seed: int = 0, device: str = "cpu") -> None:
        NeuralModel.__init__(self, seed=seed, device=device, layers=layers, hidden=hidden, steps=steps, batch=batch,
                             lr=lr)

    def build(self):
        T = torch()
        self.net = T.nn.ModuleList([mlp([1 + 4, self.hp["hidden"], self.hp["hidden"], 2])
                                    for _ in range(self.hp["layers"])])

    def _forward(self, x, c):
        """x -> z, log|det dz/dx|."""
        T = torch()
        logdet = T.zeros(len(x))
        cols = [x[:, 0], x[:, 1]]
        for k, layer in enumerate(self.net):
            i, j = (0, 1) if k % 2 == 0 else (1, 0)
            st = layer(T.cat([cols[j][:, None], c], -1))
            s, t = T.tanh(st[:, 0]), st[:, 1]
            cols[i] = cols[i] * T.exp(s) + t
            logdet = logdet + s
        return T.stack(cols, -1), logdet

    def _logp(self, x, c):
        z, ld = self._forward(x, c)
        return -0.5 * (z ** 2).sum(-1) - math.log(2 * math.pi) + ld

    def fit(self, x: np.ndarray, c: np.ndarray):
        T = torch()
        self.standardize(x, c)
        X, C = T.as_tensor(self.zx(x).astype(np.float32)), T.as_tensor(self.zc(c).astype(np.float32))
        self.data_fingerprint = fingerprint(x, c)
        self.build()
        self._train(lambda b: -self._logp(X[b], C[b]).mean(), len(X), steps=self.hp["steps"],
                    batch=self.hp["batch"], lr=self.hp["lr"])
        return self

    def nll(self, x: np.ndarray, c: np.ndarray) -> float:
        """Exact NLL per order in the original units (nats)."""
        T = torch()
        with T.no_grad():
            lp = self._logp(T.as_tensor(self.zx(x).astype(np.float32)), T.as_tensor(self.zc(c).astype(np.float32)))
        return float(-lp.mean() + np.log(self.sx).sum())

    def sample(self, c: np.ndarray, seed: int) -> np.ndarray:
        T = torch()
        g = T.Generator().manual_seed(seed)
        C = T.as_tensor(self.zc(c).astype(np.float32))
        z = T.randn(len(C), 2, generator=g)
        cols = [z[:, 0], z[:, 1]]
        with T.no_grad():
            for k in reversed(range(len(self.net))):
                i, j = (0, 1) if k % 2 == 0 else (1, 0)
                st = self.net[k](T.cat([cols[j][:, None], C], -1))
                s, t = T.tanh(st[:, 0]), st[:, 1]
                cols[i] = (cols[i] - t) * T.exp(-s)
        return T.stack(cols, -1).numpy() * self.sx + self.mx


class FlowMatching(_Std, NeuralModel):
    name = "flow_matching"

    def __init__(self, *, hidden: int = 64, steps: int = 3000, batch: int = 512, lr: float = 2e-3, ode_steps: int = 64,
                 seed: int = 0, device: str = "cpu") -> None:
        NeuralModel.__init__(self, seed=seed, device=device, hidden=hidden, steps=steps, batch=batch, lr=lr,
                             ode_steps=ode_steps)

    def build(self):
        h = self.hp["hidden"]
        self.net = mlp([2 + 1 + 4, h, h, h, 2])

    def fit(self, x: np.ndarray, c: np.ndarray):
        T = torch()
        self.standardize(x, c)
        X1, C = T.as_tensor(self.zx(x).astype(np.float32)), T.as_tensor(self.zc(c).astype(np.float32))
        self.data_fingerprint = fingerprint(x, c)
        self.build()
        T.manual_seed(self.seed)

        def loss(b):
            x1, cc = X1[b], C[b]
            x0 = T.randn_like(x1)
            t = T.rand(len(b), 1)
            xt = (1 - t) * x0 + t * x1
            v = self.net(T.cat([xt, t, cc], -1))
            return ((v - (x1 - x0)) ** 2).sum(-1).mean()

        self._train(loss, len(X1), steps=self.hp["steps"], batch=self.hp["batch"], lr=self.hp["lr"])
        return self

    def sample(self, c: np.ndarray, seed: int) -> np.ndarray:
        T = torch()
        g = T.Generator().manual_seed(seed)
        C = T.as_tensor(self.zc(c).astype(np.float32))
        x = T.randn(len(C), 2, generator=g)
        n = self.hp["ode_steps"]
        with T.no_grad():
            for k in range(n):
                t = T.full((len(C), 1), k / n)
                x = x + self.net(T.cat([x, t, C], -1)) / n
        return x.numpy() * self.sx + self.mx


class GaussianConditional:
    """Classical baseline: x | c ~ N(A [1, c], Sigma) by least squares."""
    name = "gaussian_conditional"

    def fit(self, x, c):
        A = np.column_stack([np.ones(len(c)), c])
        self.B, *_ = np.linalg.lstsq(A, x, rcond=None)
        r = x - A @ self.B
        self.S = np.cov(r.T) + 1e-9 * np.eye(x.shape[1])
        return self

    def nll(self, x, c) -> float:
        r = x - np.column_stack([np.ones(len(c)), c]) @ self.B
        inv, (_, logdet) = np.linalg.inv(self.S), np.linalg.slogdet(self.S)
        return float(0.5 * (np.einsum("ij,jk,ik->i", r, inv, r) + logdet + x.shape[1] * math.log(2 * math.pi)).mean())

    def sample(self, c, seed):
        rng = np.random.default_rng(seed)
        return np.column_stack([np.ones(len(c)), c]) @ self.B + rng.multivariate_normal(np.zeros(len(self.S)), self.S,
                                                                                      size=len(c))


def energy_distance(a: np.ndarray, b: np.ndarray, *, n: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), min(n, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(n, len(b)), replace=False)]

    def md(x, y):
        return float(np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)).mean())
    return 2 * md(a, b) - md(a, a) - md(b, b)


__all__ = ["ConditionalFlow", "FlowMatching", "GaussianConditional", "mark_data", "energy_distance"]
