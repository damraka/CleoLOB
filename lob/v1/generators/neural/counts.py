"""Neural event-count generators that drive a full limit order book (M82 causal autoregressive, M83 recurrent).

Both models predict the next 100 ms bin's six event counts (market buy/sell, add bid/ask, cancel bid/ask) from the
last ``history`` bins of (log1p counts, book state features) with an independent negative-binomial head per mark
(mean ``softplus``, dispersion ``softplus`` + 1e-3). They implement the v0.8 ``MarketGenerator`` interface
(``initial_state``, ``expected``, ``draw``, ``update``, ``loglik``), so ``MarketSimV1`` turns them into F5 neural
generative markets that can host execution episodes: the network decides *how many* events arrive given the
simulated book state, the corrected book model decides *where* they land.

* ``AutoregressiveCounts`` (M82): causal dilated temporal convolution (WaveNet-style, receptive field = history);
* ``RecurrentCounts`` (M83): state-conditioned GRU baseline.

Sampling draws negative-binomial counts with the simulator's per-bin numpy generator (Gamma-Poisson mixture), so
a seeded simulation is reproducible. Held-out scoring is the per-bin negative log-likelihood (nats per bin).
"""
from __future__ import annotations

from collections import deque
import math

import numpy as np

from ....v08.market.frame import EventFrame
from ....v08.market.sim import BookModel
from .base import NeuralModel, fingerprint, mlp, torch

K = 6
STATE = (0, 1)        # scale-free book state only: spread (ticks, capped at 10) and 5-level imbalance
F = len(STATE)


def _state_cols(s: np.ndarray) -> np.ndarray:
    """Registered conditioning: absolute depths differ in scale between real and simulated books, so a model
    conditioned on them leaves its training distribution as soon as it drives a simulated book (observed on the
    development day: mean simulated spread 156 ticks); only scale-free features are used."""
    return np.asarray(s, float)[..., list(STATE)]


def _inputs(frame: EventFrame) -> np.ndarray:
    c = np.log1p(np.asarray(frame.counts, float))
    return np.concatenate([c, _state_cols(frame.state)], axis=1).astype(np.float32)


def nb_logpmf(y, mu, r):
    T = torch()
    return (T.lgamma(y + r) - T.lgamma(r) - T.lgamma(y + 1) + r * T.log(r / (r + mu)) + y * T.log(mu / (r + mu)))


class _CountModel(NeuralModel):
    name = "_counts"

    def __init__(self, *, history: int = 32, hidden: int = 48, steps: int = 1500, batch: int = 256,
                 lr: float = 2e-3, seed: int = 0, device: str = "cpu") -> None:
        super().__init__(seed=seed, device=device, history=history, hidden=hidden, steps=steps, batch=batch, lr=lr)
        self.mu_x = np.zeros(K + F, np.float32)
        self.sd_x = np.ones(K + F, np.float32)
        self.book = BookModel()
        self.fit_info: dict = {}

    capabilities = frozenset({"count_process", "neural"})

    def extra_state(self) -> dict:
        return {"mu_x": self.mu_x.tolist(), "sd_x": self.sd_x.tolist()}

    def load_extra(self, extra: dict) -> None:
        self.mu_x = np.asarray(extra["mu_x"], np.float32)
        self.sd_x = np.asarray(extra["sd_x"], np.float32)

    def _windows(self, frames: list[EventFrame]):
        H = self.hp["history"]
        X, Y = [], []
        for fr in frames:
            z = (_inputs(fr) - self.mu_x) / self.sd_x
            y = np.asarray(fr.counts, np.float32)
            v = np.asarray(fr.valid, bool)
            for i in range(H, len(fr)):
                if v[i - H:i + 1].all():
                    X.append(i)
            Y.append((z, y))
        return Y, X

    def fit(self, frames, *_):
        T = torch()
        frames = frames if isinstance(frames, list) else [frames]
        allx = np.concatenate([_inputs(f)[f.valid] for f in frames])
        self.mu_x, self.sd_x = allx.mean(0), allx.std(0) + 1e-3
        H = self.hp["history"]
        seqs, items = [], []
        for k, fr in enumerate(frames):
            z = ((_inputs(fr) - self.mu_x) / self.sd_x).astype(np.float32)
            y = np.asarray(fr.counts, np.float32)
            v = np.asarray(fr.valid, bool)
            seqs.append((T.as_tensor(z), T.as_tensor(y)))
            ok = np.flatnonzero(np.convolve(~v, np.ones(H + 1), "full")[H:len(v)] == 0)
            items += [(k, i) for i in ok if i >= H]
        self.data_fingerprint = fingerprint(allx)
        self.build()
        items = np.asarray(items)

        def loss(b):
            sel = items[b]
            Xb = T.stack([seqs[k][0][i - H:i] for k, i in sel])
            Yb = T.stack([seqs[k][1][i] for k, i in sel])
            mu, r = self._head(Xb)
            return -nb_logpmf(Yb, mu, r).sum(1).mean()

        self._train(loss, len(items), steps=self.hp["steps"], batch=self.hp["batch"], lr=self.hp["lr"])
        self.fit_info = {"windows": int(len(items)), "history": H, "val_history": self.history[-3:]}
        return self

    def _head(self, X):
        T = torch()
        out = self.net(X)
        mu = T.nn.functional.softplus(out[:, :K]) + 1e-4
        r = T.nn.functional.softplus(out[:, K:]) + 1e-3
        return mu, r

    # ------------------------------------------------------------------ v0.8 MarketGenerator interface
    def initial_state(self, simulate: bool = False):
        z0 = ((np.zeros(K + F, np.float32) - self.mu_x) / self.sd_x).astype(np.float32)
        return deque([z0] * self.hp["history"], maxlen=self.hp["history"])

    def _params(self, state):
        T = torch()
        with T.no_grad():
            X = T.as_tensor(np.stack(state)[None])
            mu, r = self._head(X)
        return mu[0].numpy().astype(float), r[0].numpy().astype(float)

    def expected(self, state, x):
        return self._params(state)[0]

    def draw(self, state, x, rng):
        mu, r = self._params(state)
        lam = rng.gamma(r, mu / r)
        return rng.poisson(np.clip(lam, 0.0, 60.0))

    def update(self, state, realized, x, own=None):
        frame_like = np.concatenate([np.log1p(np.asarray(realized, float)), _state_cols(x)])
        state.append(((frame_like - self.mu_x) / self.sd_x).astype(np.float32))
        return state

    def nll(self, frame: EventFrame) -> np.ndarray:
        """Per-valid-bin negative log-likelihood (nats per bin), filtering on the realized history."""
        T = torch()
        H = self.hp["history"]
        z = ((_inputs(frame) - self.mu_x) / self.sd_x).astype(np.float32)
        y = np.asarray(frame.counts, np.float32)
        v = np.asarray(frame.valid, bool)
        idx = [i for i in range(H, len(frame)) if v[i - H:i + 1].all()]
        out = []
        with T.no_grad():
            for s in range(0, len(idx), 2048):
                b = idx[s:s + 2048]
                X = T.as_tensor(np.stack([z[i - H:i] for i in b]))
                mu, r = self._head(X)
                out.append((-nb_logpmf(T.as_tensor(y[b]), mu, r).sum(1)).numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def loglik(self, frame: EventFrame) -> np.ndarray:
        out = np.full(len(frame), np.nan)
        H = self.hp["history"]
        v = np.asarray(frame.valid, bool)
        idx = [i for i in range(H, len(frame)) if v[i - H:i + 1].all()]
        if idx:
            out[idx] = -self.nll(frame)
        return out


class _TCN:
    @staticmethod
    def build(inp: int, hidden: int, history: int):
        T = torch()

        class Net(T.nn.Module):
            def __init__(self):
                super().__init__()
                dil, layers, ch = 1, [], inp
                while dil < history:
                    layers.append(T.nn.Conv1d(ch, hidden, 2, dilation=dil))
                    ch, dil = hidden, dil * 2
                self.convs = T.nn.ModuleList(layers)
                self.out = mlp([hidden, hidden, 2 * K])

            def forward(self, x):                    # x: (B, H, C) -> causal conv over time
                h = x.transpose(1, 2)
                for conv in self.convs:
                    pad = conv.dilation[0]
                    h = T.nn.functional.silu(conv(T.nn.functional.pad(h, (pad, 0))))
                return self.out(h[:, :, -1])

        return Net()


class AutoregressiveCounts(_CountModel):
    name = "autoregressive_counts"

    def build(self) -> None:
        self.net = _TCN.build(K + F, self.hp["hidden"], self.hp["history"])


class RecurrentCounts(_CountModel):
    name = "recurrent_counts"

    def build(self) -> None:
        T = torch()
        hidden = self.hp["hidden"]

        class Net(T.nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = T.nn.GRU(K + F, hidden, batch_first=True)
                self.out = mlp([hidden, hidden, 2 * K])

            def forward(self, x):
                h, _ = self.gru(x)
                return self.out(h[:, -1])

        self.net = Net()


def mean_nll(model, frames) -> float:
    v = np.concatenate([model.nll(f) for f in frames])
    return float(v.mean()) if len(v) else math.nan


__all__ = ["AutoregressiveCounts", "RecurrentCounts", "nb_logpmf", "mean_nll"]
