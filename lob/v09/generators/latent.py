"""Mid/spread path generators for path-level realism (M47; H4).

Each generator is fitted on real training paths ([n, 601, 2]: mid half-ticks relative to start, spread ticks) and
samples new 60 s paths with a seed. Mid increments are integers in half-ticks; spreads are positive integers.

Classical:
* ``IIDBootstrap`` - increments and spreads drawn independently from the training distribution;
* ``RegimeSwitching`` - two-state Markov chain (quiet / active by the 1 s activity median) with state-specific
  increment and spread distributions and a fitted transition matrix.

Latent dynamics:
* ``LatentActivitySSM`` - latent AR(1) log-activity h_t: P(increment != 0) = sigmoid(h_t), signed magnitudes drawn
  from the training nonzero increments; (phi, sigma, mean) fitted by matching the lag-1..10 autocorrelation of the
  move indicator and its mean (method of simulated moments on a registered grid);
* ``NeuralLatentAR`` - GRU over the past 50 bins (increment class, spread) producing a categorical over the increment
  classes {<=-2, -1, 0, +1, >=+2} and the spread change; trained by maximum likelihood with early stopping; sampled
  autoregressively. A discrete-time neural latent dynamics model.

``NEURAL_SDE_STATUS``: a continuous-time Neural SDE/ODE was not implemented - integer tick paths on a 100 ms grid make
a diffusion likelihood misspecified, and a defensible jump-SDE was outside the registered scope (BLOCKED, reported).
"""
from __future__ import annotations

import copy

import numpy as np

NEURAL_SDE_STATUS = {"status": "BLOCKED",
                     "reason": "continuous-time Neural SDE/ODE likelihoods are misspecified for integer tick paths on a "
                               "100 ms grid; the discrete-time NeuralLatentAR is the implemented neural latent model"}
CLASSES = np.asarray([-2, -1, 0, 1, 2])


def _increments(paths):
    return np.diff(paths[:, :, 0], axis=1).astype(int), paths[:, 1:, 1].astype(int)


class IIDBootstrap:
    name, family = "iid_bootstrap", "classical"

    def fit(self, paths):
        inc, spr = _increments(paths)
        self.inc, self.spr = inc.ravel(), spr.ravel()
        return self

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        L = 600
        inc = rng.choice(self.inc, (n, L))
        spr = rng.choice(self.spr, (n, L))
        return np.stack([np.c_[np.zeros(n), np.cumsum(inc, 1)], np.c_[spr[:, :1], spr]], -1).astype(float)


class RegimeSwitching:
    name, family = "regime_switching", "classical"

    def fit(self, paths):
        inc, spr = _increments(paths)
        act = np.abs(inc).reshape(len(inc), -1, 10).sum(2)            # moves per second
        med = float(np.median(act))
        state = np.repeat((act > med).astype(int), 10, axis=1)
        self.dist = {s: (inc[state == s], spr[state == s]) for s in (0, 1)}
        P = np.ones((2, 2))
        for row in state:
            np.add.at(P, (row[:-1], row[1:]), 1)
        self.P = P / P.sum(1, keepdims=True)
        self.p0 = float((state[:, 0] == 1).mean())
        return self

    def sample(self, n, seed):
        rng = np.random.default_rng(seed)
        L = 600
        out = np.zeros((n, L + 1, 2))
        for i in range(n):
            s = int(rng.random() < self.p0)
            m, row_inc, row_spr = 0, [], []
            for _ in range(L):
                inc, spr = self.dist[s]
                j = rng.integers(len(inc)) if len(inc) else 0
                row_inc.append(inc[j] if len(inc) else 0)
                row_spr.append(spr[j] if len(spr) else 1)
                s = int(rng.random() < self.P[s, 1])
            out[i, 1:, 0] = np.cumsum(row_inc)
            out[i, :, 1] = np.r_[row_spr[0], row_spr]
            del m
        return out


def _acf(x, lags):
    x = x - x.mean()
    v = (x * x).mean() or 1e-12
    return np.asarray([(x[:-k] * x[k:]).mean() / v for k in lags])


class LatentActivitySSM:
    name, family = "latent_activity_ssm", "latent"

    def fit(self, paths, seed: int = 0):
        inc, spr = _increments(paths)
        move = (inc != 0).astype(float)
        self.nonzero = inc[inc != 0] if (inc != 0).any() else np.asarray([1, -1])
        self.spr = spr.ravel()
        target = np.mean([_acf(r, range(1, 11)) for r in move if r.std() > 0], axis=0) if move.std() > 0 \
            else np.zeros(10)
        rate = float(move.mean())
        best = None
        for phi in (0.9, 0.95, 0.98, 0.99, 0.995):
            for sigma in (0.1, 0.2, 0.4, 0.8):
                mean = np.log(rate / max(1 - rate, 1e-6)) if 0 < rate < 1 else -5.0
                sim = self._simulate(phi, sigma, mean, 64, 600, seed)
                ac = np.mean([_acf(r, range(1, 11)) for r in sim if r.std() > 0], axis=0) if sim.std() > 0 \
                    else np.zeros(10)
                err = float(((ac - target) ** 2).sum() + 10 * (sim.mean() - rate) ** 2)
                if best is None or err < best[0]:
                    best = (err, phi, sigma, mean)
        _, self.phi, self.sigma, self.mean = best
        return self

    @staticmethod
    def _simulate(phi, sigma, mean, n, L, seed):
        rng = np.random.default_rng(seed)
        h = np.full(n, mean)
        out = np.zeros((n, L))
        for k in range(L):
            h = mean + phi * (h - mean) + sigma * rng.standard_normal(n)
            out[:, k] = rng.random(n) < 1 / (1 + np.exp(-h))
        return out

    def sample(self, n, seed):
        rng = np.random.default_rng(seed + 1)
        move = self._simulate(self.phi, self.sigma, self.mean, n, 600, seed)
        inc = np.where(move > 0, rng.choice(self.nonzero, move.shape), 0)
        spr = rng.choice(self.spr, (n, 600))
        return np.stack([np.c_[np.zeros(n), np.cumsum(inc, 1)], np.c_[spr[:, :1], spr]], -1).astype(float)


class NeuralLatentAR:
    name, family = "neural_latent_ar", "latent"

    def __init__(self, hidden: int = 32, steps: int = 1500, history: int = 50, seed: int = 0) -> None:
        self.hidden, self.steps, self.history, self.seed = hidden, steps, history, seed

    @staticmethod
    def _cls(inc):
        return np.clip(inc, -2, 2) + 2

    def fit(self, paths):
        import torch
        from ..runtime import torch_threads
        torch_threads()
        torch.manual_seed(self.seed)
        inc, spr = _increments(paths)
        self.spread_values = np.unique(spr)
        self.pos = {int(v): i for i, v in enumerate(self.spread_values)}
        c = self._cls(inc)
        s = np.vectorize(self.pos.get)(spr)
        S = len(self.spread_values)
        self.rnn = torch.nn.GRU(5 + S, self.hidden, batch_first=True)
        self.head = torch.nn.Linear(self.hidden, 5 + S)
        self.S = S
        params = [*self.rnn.parameters(), *self.head.parameters()]
        opt = torch.optim.Adam(params, lr=3e-3, weight_decay=1e-4)
        X = np.concatenate([np.eye(5)[c], np.eye(S)[s]], -1).astype(np.float32)
        rng = np.random.default_rng(self.seed)
        n_val = max(1, len(X) // 7)
        val, tr = np.arange(n_val), np.arange(n_val, len(X))
        Xt = torch.tensor(X)
        ct, st = torch.tensor(c), torch.tensor(s)
        best, best_state, bad = float("inf"), None, 0

        def loss_of(rows):
            out = self.head(self.rnn(Xt[rows, :-1])[0])
            return (torch.nn.functional.cross_entropy(out[..., :5].reshape(-1, 5), ct[rows, 1:].reshape(-1))
                    + torch.nn.functional.cross_entropy(out[..., 5:].reshape(-1, S), st[rows, 1:].reshape(-1)))

        for step in range(1, self.steps + 1):
            rows = rng.choice(tr, min(32, len(tr)), replace=False)
            loss = loss_of(rows)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            if step % 100 == 0:
                with torch.no_grad():
                    v = float(loss_of(val))
                if v < best - 1e-4:
                    best, bad, best_state = v, 0, [copy.deepcopy(p.detach()) for p in params]
                else:
                    bad += 1
                    if bad >= 3:
                        break
        if best_state is not None:
            with torch.no_grad():
                for p, b in zip(params, best_state):
                    p.copy_(b)
        self.first = (c[:, 0], s[:, 0])
        return self

    def sample(self, n, seed):
        import torch
        g = torch.Generator().manual_seed(seed)
        rng = np.random.default_rng(seed)
        pick = rng.integers(0, len(self.first[0]), n)
        c0, s0 = torch.tensor(self.first[0][pick]), torch.tensor(self.first[1][pick])
        x = torch.cat([torch.nn.functional.one_hot(c0, 5), torch.nn.functional.one_hot(s0, self.S)], -1).float()
        h = None
        incs, sprs = [], []
        with torch.no_grad():
            for _ in range(600):
                o, h = self.rnn(x.unsqueeze(1), h)
                logits = self.head(o[:, -1])
                c = torch.multinomial(torch.softmax(logits[:, :5], -1), 1, generator=g).squeeze(1)
                s = torch.multinomial(torch.softmax(logits[:, 5:], -1), 1, generator=g).squeeze(1)
                incs.append(CLASSES[c.numpy()])
                sprs.append(self.spread_values[s.numpy()])
                x = torch.cat([torch.nn.functional.one_hot(c, 5), torch.nn.functional.one_hot(s, self.S)], -1).float()
        inc = np.stack(incs, 1)
        spr = np.stack(sprs, 1)
        return np.stack([np.c_[np.zeros(n), np.cumsum(inc, 1)], np.c_[spr[:, :1], spr]], -1).astype(float)


CLASSICAL = {"iid_bootstrap": IIDBootstrap, "regime_switching": RegimeSwitching}
LATENT = {"latent_activity_ssm": LatentActivitySSM, "neural_latent_ar": NeuralLatentAR}
