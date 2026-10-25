"""Neural state-space market dynamics (M84) and a neural stochastic differential equation (M85).

``StateSpace`` (deep Markov model): latent z_t in R^d with transition z_t = z_{t-1} + f(z_{t-1}) + s_z * eps and a
Gaussian emission x_t ~ N(g(z_t), diag(s_x^2)) of per-second market features (log1p trade counts per side, spread,
imbalance, 1 s log return in bps). Inference network: a GRU over the observed past gives q(z_t | x_<=t); training
maximizes the ELBO with the reparameterization trick. ``nll`` reports the negative ELBO per step (an upper bound on
the NLL, labelled as such). ``sample`` rolls the prior transition and emission forward.

``NeuralSDE``: observed state y = (1 s mid return in bps, log realized volatility over 60 s), dynamics
dy = mu(y) dt + sigma(y) dW with MLP drift and (softplus) diagonal diffusion, fitted by the exact Gaussian
likelihood of the Euler-Maruyama transition on the observed 1 s grid. It is the registered "comparable stochastic
latent model" for M85: volatility is a filtered state, not a fully latent one (the label says so). ``sample`` uses
Euler-Maruyama with the same step.
"""
from __future__ import annotations

import math

import numpy as np

from .base import NeuralModel, fingerprint, mlp, torch

LOG2PI = math.log(2 * math.pi)


def second_features(cache: dict, starts, *, warmup: int = 30, horizon: int = 300) -> list[np.ndarray]:
    """Per-second (log1p buys, log1p sells, spread ticks, imbalance, return bps) windows of a v1 day cache."""
    out = []
    mid = np.asarray(cache["mid"], float)
    counts = np.asarray(cache["counts"], float)
    state = np.asarray(cache["state"], float)
    valid = np.asarray(cache["valid"], bool)
    for s in starts:
        a, b = (s - warmup) * 10, (s + horizon) * 10
        if not valid[a:b].all():
            continue
        m = mid[a:b + 1:10]
        ret = np.diff(m) / np.maximum(m[:-1], 1e-9) * 1e4
        c = counts[a:b].reshape(-1, 10, 6).sum(1)
        st = state[a:b:10]
        out.append(np.column_stack([np.log1p(c[:, 0]), np.log1p(c[:, 1]), st[:, 0], st[:, 1],
                                    np.clip(ret, -50, 50)]).astype(np.float32))
    return out


class StateSpace(NeuralModel):
    name = "state_space"

    def __init__(self, *, latent: int = 4, hidden: int = 32, dims: int = 5, steps: int = 1200, batch: int = 32,
                 seq: int = 60, lr: float = 3e-3, seed: int = 0, device: str = "cpu") -> None:
        super().__init__(seed=seed, device=device, latent=latent, hidden=hidden, dims=dims, steps=steps, batch=batch,
                         seq=seq, lr=lr)
        self.mu = np.zeros(dims, np.float32)
        self.sd = np.ones(dims, np.float32)

    def extra_state(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}

    def load_extra(self, e):
        self.mu, self.sd = np.asarray(e["mu"], np.float32), np.asarray(e["sd"], np.float32)

    def build(self):
        T = torch()
        d, h, k = self.hp["latent"], self.hp["hidden"], self.hp["dims"]

        class Net(T.nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = T.nn.GRU(k, h, batch_first=True)
                self.q = T.nn.Linear(h + d, 2 * d)
                self.f = mlp([d, h, d], act="tanh")
                self.g = mlp([d, h, k])
                self.log_sz = T.nn.Parameter(T.full((d,), -1.0))
                self.log_sx = T.nn.Parameter(T.zeros(k))

        self.net = Net()

    def _elbo(self, x):
        T = torch()
        n = self.net
        B, L, _ = x.shape
        d = self.hp["latent"]
        hs, _ = n.rnn(x)
        z = T.zeros(B, d)
        total = 0.0
        sz, sx = T.exp(n.log_sz), T.exp(n.log_sx)
        for t in range(L):
            prior_mu = z + n.f(z)
            stats = n.q(T.cat([hs[:, t], z], -1))
            qmu, qls = stats[:, :d], T.clamp(stats[:, d:], -6, 2)
            qs = T.exp(qls)
            z = qmu + qs * T.randn_like(qs)
            kl = (T.log(sz) - qls + (qs ** 2 + (qmu - prior_mu) ** 2) / (2 * sz ** 2) - 0.5).sum(-1)
            rec = -0.5 * (((x[:, t] - n.g(z)) / sx) ** 2 + 2 * T.log(sx) + LOG2PI).sum(-1)
            total = total + (rec - kl)
        return total / L

    def fit(self, windows: list[np.ndarray]):
        T = torch()
        allx = np.concatenate(windows)
        self.mu, self.sd = allx.mean(0), allx.std(0) + 1e-3
        L = self.hp["seq"]
        chunks = [((w[i:i + L] - self.mu) / self.sd) for w in windows for i in range(0, len(w) - L + 1, L // 2)]
        X = T.as_tensor(np.stack(chunks).astype(np.float32))
        self.data_fingerprint = fingerprint(allx)
        self.build()
        T.manual_seed(self.seed)
        self._train(lambda b: -self._elbo(X[b]).mean(), len(X), steps=self.hp["steps"], batch=self.hp["batch"],
                    lr=self.hp["lr"])
        return self

    def nll(self, windows: list[np.ndarray]) -> float:
        """Negative ELBO per step in standardized units plus the log-Jacobian of the standardization."""
        T = torch()
        L = self.hp["seq"]
        chunks = [((w[i:i + L] - self.mu) / self.sd) for w in windows for i in range(0, len(w) - L + 1, L)]
        T.manual_seed(self.seed)
        with T.no_grad():
            v = -self._elbo(T.as_tensor(np.stack(chunks).astype(np.float32))).mean().item()
        return float(v + np.log(self.sd).sum())

    def sample(self, steps: int, seed: int) -> np.ndarray:
        T = torch()
        T.manual_seed(seed)
        n, d = self.net, self.hp["latent"]
        z = T.zeros(1, d)
        out = []
        with T.no_grad():
            for _ in range(steps):
                z = z + n.f(z) + T.exp(n.log_sz) * T.randn(1, d)
                x = n.g(z) + T.exp(n.log_sx) * T.randn(1, self.hp["dims"])
                out.append(x[0].numpy())
        return np.asarray(out) * self.sd + self.mu


def sde_states(cache: dict, starts, *, horizon: int = 300, warmup: int = 90) -> list[np.ndarray]:
    mid = np.asarray(cache["mid"], float)
    valid = np.asarray(cache["valid"], bool)
    out = []
    for s in starts:
        a, b = (s - warmup) * 10, (s + horizon) * 10
        if not valid[a:b].all():
            continue
        m = mid[a:b + 1:10]
        r = np.diff(m) / np.maximum(m[:-1], 1e-9) * 1e4
        rv = np.sqrt(np.convolve(r ** 2, np.ones(60) / 60, "valid"))
        y = np.column_stack([r[59:], np.log(np.maximum(rv, 1e-3))])
        out.append(y.astype(np.float32))
    return out


class NeuralSDE(NeuralModel):
    name = "neural_sde"

    def __init__(self, *, hidden: int = 32, dt: float = 1.0, steps: int = 1500, batch: int = 512, lr: float = 3e-3,
                 seed: int = 0, device: str = "cpu") -> None:
        super().__init__(seed=seed, device=device, hidden=hidden, dt=dt, steps=steps, batch=batch, lr=lr)

    def build(self):
        T = torch()
        h = self.hp["hidden"]

        class Net(T.nn.Module):
            def __init__(self):
                super().__init__()
                self.drift = mlp([2, h, h, 2])
                self.diff = mlp([2, h, h, 2])

        self.net = Net()

    def _ll(self, y0, y1):
        T = torch()
        dt = self.hp["dt"]
        mu = self.net.drift(y0)
        s = T.nn.functional.softplus(self.net.diff(y0)) + 1e-3
        z = (y1 - y0 - mu * dt) / (s * math.sqrt(dt))
        return -0.5 * (z ** 2 + 2 * T.log(s * math.sqrt(dt)) + LOG2PI).sum(-1)

    def fit(self, paths: list[np.ndarray]):
        T = torch()
        Y0 = np.concatenate([p[:-1] for p in paths])
        Y1 = np.concatenate([p[1:] for p in paths])
        self.data_fingerprint = fingerprint(Y0)
        self.build()
        a, b = T.as_tensor(Y0), T.as_tensor(Y1)
        self._train(lambda i: -self._ll(a[i], b[i]).mean(), len(Y0), steps=self.hp["steps"], batch=self.hp["batch"],
                    lr=self.hp["lr"])
        return self

    def nll(self, paths: list[np.ndarray]) -> float:
        """Exact Euler-transition NLL per step (nats)."""
        T = torch()
        a = T.as_tensor(np.concatenate([p[:-1] for p in paths]))
        b = T.as_tensor(np.concatenate([p[1:] for p in paths]))
        with T.no_grad():
            return float(-self._ll(a, b).mean())

    def sample(self, y0: np.ndarray, steps: int, seed: int) -> np.ndarray:
        T = torch()
        rng = np.random.default_rng(seed)
        y = T.as_tensor(np.asarray(y0, np.float32))[None]
        dt = self.hp["dt"]
        out = [y[0].numpy()]
        with T.no_grad():
            for _ in range(steps):
                mu = self.net.drift(y)
                s = T.nn.functional.softplus(self.net.diff(y)) + 1e-3
                y = y + mu * dt + s * math.sqrt(dt) * T.as_tensor(rng.standard_normal((1, 2)).astype(np.float32))
                out.append(y[0].numpy())
        return np.asarray(out)


def gaussian_ar1_nll(paths: list[np.ndarray]) -> float:
    """Classical baseline for the SDE: per-dimension Gaussian AR(1) fitted by least squares (in-sample NLL)."""
    Y0 = np.concatenate([p[:-1] for p in paths])
    Y1 = np.concatenate([p[1:] for p in paths])
    tot = 0.0
    for k in range(Y0.shape[1]):
        A = np.column_stack([np.ones(len(Y0)), Y0[:, k]])
        beta, *_ = np.linalg.lstsq(A, Y1[:, k], rcond=None)
        res = Y1[:, k] - A @ beta
        s2 = max(res.var(), 1e-9)
        tot += 0.5 * (np.log(2 * np.pi * s2) + 1)
    return float(tot)


__all__ = ["StateSpace", "NeuralSDE", "second_features", "sde_states", "gaussian_ar1_nll"]
