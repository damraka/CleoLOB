"""Bin-level conditional event models, continual learning and contrastive representations (M38-M43, M46).

Data (registered): the 100 ms frames inside the registered windows (warmup start to horizon end). Per bin the input is
the canonical LOB state encoding (M38): log1p of the six event counts and the six state features (spread ticks,
imbalance5, log L1 depth, log1p bid q0, log1p ask q0, |ret10|) standardized with training-day statistics. The
conditional event target of the *next* bin is one of 7 classes: no event, or the mark with the largest count (ties:
lowest mark index) - market buy, market sell, add bid, add ask, cancel bid, cancel ask.

Models (all report the held-out per-bin NLL of the next-bin class):

* ``MultinomialLR`` (classical, M44 baseline) - current state, last-bin counts and the 10-bin count sums;
* ``SeqClassifier('gru'|'lstm'|'transformer')`` (M46) - 32-bin history; Adam with weight decay and early stopping on
  15 % held-out sequences;
* continual variants (``naive``, ``replay``, ``ewc``) for H16;
* ``ContrastiveEncoder`` (M39-M41): a GRU over 100-bin windows trained with InfoNCE between two augmented views
  (feature noise, amplitude scaling, random crop), and linear probes for the 10 s-ahead mid direction (M43).
"""
from __future__ import annotations

import copy

import numpy as np

from ..data.daycache import BIN_MS, HORIZON_S, WARMUP_S, WINDOW_STARTS_S

HISTORY = 32
N_CLASSES = 7
WINDOW = 100          # bins (10 s) for representations


def frame_features(cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(features per bin [n, 12], next-bin class [n], window id [n], mid [n]) over the registered windows."""
    rows, cls, wid, mids = [], [], [], []
    for w, s in enumerate(WINDOW_STARTS_S):
        a, b = int((s - WARMUP_S) * 1000 // BIN_MS), int((s + HORIZON_S) * 1000 // BIN_MS)
        c = cache["counts"][a:b + 1].astype(float)
        st = cache["state"][a:b + 1].astype(float)
        st = np.column_stack([st[:, 0], st[:, 1], st[:, 2], np.log1p(st[:, 3]), np.log1p(st[:, 4]), st[:, 5]])
        x = np.column_stack([np.log1p(c), st])
        nxt = c[1:]
        y = np.where(nxt.sum(1) > 0, 1 + np.argmax(nxt, 1), 0)
        ok = cache["valid"][a:b]
        rows.append(x[:-1][ok])
        cls.append(y[ok])
        wid.append(np.full(int(ok.sum()), w))
        mids.append(cache["mid"][a:b][ok])
    return np.vstack(rows), np.concatenate(cls), np.concatenate(wid), np.concatenate(mids)


def standardizer(X: np.ndarray):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return mu, sd


def windows_of(X: np.ndarray, wid: np.ndarray, length: int) -> np.ndarray:
    """Indices i such that X[i-length+1..i] lies inside one window (for sequence inputs)."""
    same = np.zeros(len(wid), bool)
    same[length - 1:] = wid[length - 1:] == wid[:len(wid) - length + 1]
    return np.flatnonzero(same)


class MultinomialLR:
    name = "multinomial_lr"

    def __init__(self, l2: float = 1e-3, iters: int = 300, lr: float = 0.05) -> None:
        self.l2, self.iters, self.lr = l2, iters, lr

    @staticmethod
    def design(X, wid):
        """Intercept, current inputs and the 10-bin rolling sums of the count inputs (within each window)."""
        cs = np.cumsum(X[:, :6], 0)
        prev = np.vstack([np.zeros((10, 6)), cs[:-10]])
        last10 = cs - prev
        start = np.r_[0, np.flatnonzero(np.diff(wid) != 0) + 1]
        for s in start:                      # never sum across a window boundary
            base = cs[s - 1] if s > 0 else np.zeros(6)
            last10[s:s + 10] = cs[s:s + 10] - base
        return np.column_stack([np.ones(len(X)), X, last10])

    def fit(self, Z, y):
        self.W = np.zeros((Z.shape[1], N_CLASSES))
        m, v = np.zeros_like(self.W), np.zeros_like(self.W)
        onehot = np.eye(N_CLASSES)[y]
        for it in range(1, self.iters + 1):
            P = self._p(Z)
            g = Z.T @ (P - onehot) / len(y) + self.l2 * self.W
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            self.W -= self.lr * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
        return self

    def _p(self, Z):
        logits = Z @ self.W
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(1, keepdims=True)

    def nll(self, Z, y) -> np.ndarray:
        return -np.log(np.maximum(self._p(Z)[np.arange(len(y)), y], 1e-12))


def torch_index(a: np.ndarray):
    import torch
    return torch.as_tensor(a, dtype=torch.long)


class SeqClassifier:
    def __init__(self, kind: str = "gru", hidden: int = 32, steps: int = 2000, batch: int = 256, seed: int = 0,
                 n_classes: int = N_CLASSES, history: int = HISTORY) -> None:
        if kind not in ("gru", "lstm", "transformer"):
            raise ValueError(kind)
        self.kind, self.hidden, self.steps, self.batch, self.seed = kind, hidden, steps, batch, seed
        self.n_classes, self.history = n_classes, history
        self.name = kind

    def _build(self, n_in: int):
        import torch
        from ..runtime import torch_threads
        torch_threads()
        torch.manual_seed(self.seed)
        if self.kind == "transformer":
            self.proj = torch.nn.Linear(n_in, self.hidden)
            layer = torch.nn.TransformerEncoderLayer(self.hidden, 4, 2 * self.hidden, dropout=0.0, batch_first=True)
            self.enc = torch.nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.mods = [self.proj, self.enc]
        else:
            cls = torch.nn.GRU if self.kind == "gru" else torch.nn.LSTM
            self.rnn = cls(n_in, self.hidden, batch_first=True)
            self.mods = [self.rnn]
        self.head = torch.nn.Linear(self.hidden, self.n_classes)
        self.mods.append(self.head)

    def params(self):
        return [p for m in self.mods for p in m.parameters()]

    def _forward(self, xb):
        import torch
        if self.kind == "transformer":
            z = self.proj(xb)
            n = z.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), 1)
            h = self.enc(z, mask=mask)
        else:
            h, _ = self.rnn(xb)
        return self.head(h[:, -1])

    def _batch(self, X, idx):
        """Index array [batch, history] of the input rows ending at each ``idx``."""
        offs = np.arange(-self.history + 1, 1)
        return torch_index(np.asarray(idx)[:, None] + offs[None, :])

    def fit(self, X, y, idx, *, init: bool = True, steps: int | None = None, replay=None, ewc=None):
        """Train on sequences ending at ``idx``; returns self. ``replay`` = (X, y, idx) mixed 50 %; ``ewc`` = (fisher,
        anchor, lambda)."""
        import torch
        if init:
            self._build(X.shape[1])
            self.opt = torch.optim.Adam(self.params(), lr=3e-3, weight_decay=1e-4)
        rng = np.random.default_rng(self.seed + (0 if init else 7))
        perm = rng.permutation(idx)
        n_val = max(1, int(0.15 * len(perm)))
        val, tr = perm[:n_val], perm[n_val:]
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        best, best_state, bad = float("inf"), None, 0
        total = steps or self.steps
        for step in range(1, total + 1):
            b = rng.choice(tr, min(self.batch, len(tr)), replace=False)
            xb, yb = Xt[self._batch(X, b)], yt[b]
            loss = torch.nn.functional.cross_entropy(self._forward(xb), yb)
            if replay is not None:
                Xr, yr, ir = replay
                rb = rng.choice(ir, min(self.batch, len(ir)), replace=False)
                loss = 0.5 * loss + 0.5 * torch.nn.functional.cross_entropy(
                    self._forward(torch.tensor(Xr, dtype=torch.float32)[self._batch(Xr, rb)]),
                    torch.tensor(yr[rb], dtype=torch.long))
            if ewc is not None:
                fisher, anchor, lam = ewc
                loss = loss + lam * sum(((p - a) ** 2 * f).sum() for p, a, f in zip(self.params(), anchor, fisher))
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params(), 5.0)
            self.opt.step()
            if step % 200 == 0 or step == total:
                v = float(self.nll(X, y, val).mean())
                if v < best - 1e-4:
                    best, bad = v, 0
                    best_state = [copy.deepcopy(p.detach()) for p in self.params()]
                else:
                    bad += 1
                    if bad >= 3:
                        break
        if best_state is not None:
            with torch.no_grad():
                for p, s in zip(self.params(), best_state):
                    p.copy_(s)
        return self

    def nll(self, X, y, idx) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32)
            for a in range(0, len(idx), 4096):
                b = idx[a:a + 4096]
                logp = torch.log_softmax(self._forward(Xt[self._batch(X, b)]), -1).numpy()
                out.append(-logp[np.arange(len(b)), y[b]])
        return np.concatenate(out) if out else np.zeros(0)

    def fisher(self, X, y, idx, n: int = 2048):
        import torch
        rng = np.random.default_rng(self.seed + 11)
        b = rng.choice(idx, min(n, len(idx)), replace=False)
        Xt = torch.tensor(X, dtype=torch.float32)
        loss = torch.nn.functional.cross_entropy(self._forward(Xt[self._batch(X, b)]), torch.tensor(y[b]))
        self.opt.zero_grad()
        loss.backward()
        return [p.grad.detach() ** 2 + 1e-8 for p in self.params()], [p.detach().clone() for p in self.params()]


# ----------------------------------------------------------------------------- representations (M39-M43)


def direction_labels(mid: np.ndarray, wid: np.ndarray, horizon: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """10 s-ahead mid direction (0 down, 1 flat, 2 up) at bins that have a full window behind and ahead."""
    idx = [i for i in range(WINDOW - 1, len(mid) - horizon)
           if wid[i] == wid[i - WINDOW + 1] == wid[i + horizon]]
    idx = np.asarray(idx, int)
    d = mid[idx + horizon] - mid[idx]
    return idx, np.where(d > 0, 2, np.where(d < 0, 0, 1))


def hand_features(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Hand-engineered window summaries (baseline representation): window means of the 12 inputs, last state,
    signed count imbalance and mid-change proxies."""
    out = []
    for i in idx:
        w = X[i - WINDOW + 1:i + 1]
        out.append(np.r_[w.mean(0), w[-1], w[:, 0].sum() - w[:, 1].sum(), w[:, 2].sum() - w[:, 3].sum()])
    return np.asarray(out)


class ContrastiveEncoder:
    name = "contrastive_gru"

    def __init__(self, dim: int = 16, hidden: int = 32, steps: int = 1500, batch: int = 128, tau: float = 0.2,
                 seed: int = 0) -> None:
        self.dim, self.hidden, self.steps, self.batch, self.tau, self.seed = dim, hidden, steps, batch, tau, seed

    def _augment(self, xb, g):
        import torch
        noise = 0.1 * torch.randn(xb.shape, generator=g)
        scale = 1.0 + 0.1 * torch.randn((xb.shape[0], 1, xb.shape[2]), generator=g)
        crop = int(torch.randint(0, 20, (1,), generator=g))
        out = (xb + noise) * scale
        return out[:, crop:]

    def fit(self, X, idx):
        import torch
        from ..runtime import torch_threads
        torch_threads()
        torch.manual_seed(self.seed)
        self.rnn = torch.nn.GRU(X.shape[1], self.hidden, batch_first=True)
        self.proj = torch.nn.Linear(self.hidden, self.dim)
        params = [*self.rnn.parameters(), *self.proj.parameters()]
        opt = torch.optim.Adam(params, lr=3e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Xt = torch.tensor(X, dtype=torch.float32)
        offs = np.arange(-WINDOW + 1, 1)
        for _ in range(self.steps):
            b = rng.choice(idx, min(self.batch, len(idx)), replace=False)
            xb = Xt[b[:, None] + offs[None, :]]
            z1 = torch.nn.functional.normalize(self.proj(self.rnn(self._augment(xb, g))[0][:, -1]), dim=-1)
            z2 = torch.nn.functional.normalize(self.proj(self.rnn(self._augment(xb, g))[0][:, -1]), dim=-1)
            logits = z1 @ z2.T / self.tau
            target = torch.arange(len(b))
            loss = 0.5 * (torch.nn.functional.cross_entropy(logits, target)
                          + torch.nn.functional.cross_entropy(logits.T, target))
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    def embed(self, X, idx) -> np.ndarray:
        import torch
        offs = np.arange(-WINDOW + 1, 1)
        out = []
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32)
            for a in range(0, len(idx), 2048):
                b = idx[a:a + 2048]
                h = self.rnn(Xt[b[:, None] + offs[None, :]])[0][:, -1]
                out.append(h.numpy())
        return np.vstack(out) if out else np.zeros((0, self.hidden))


def probe_nll(train_Z, train_y, test_Z, test_y, classes: int = 3) -> np.ndarray:
    """Linear probe (multinomial logistic regression) log loss on test rows."""
    mu, sd = standardizer(train_Z)
    tz, sz = (train_Z - mu) / sd, (test_Z - mu) / sd
    m = MultinomialLR(l2=1e-2, iters=400)
    m.W = None
    Z1, Z2 = np.column_stack([np.ones(len(tz)), tz]), np.column_stack([np.ones(len(sz)), sz])
    m.W = np.zeros((Z1.shape[1], classes))
    mm, vv = np.zeros_like(m.W), np.zeros_like(m.W)
    onehot = np.eye(classes)[train_y]
    for it in range(1, m.iters + 1):
        logits = Z1 @ m.W
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(1, keepdims=True)
        gr = Z1.T @ (P - onehot) / len(train_y) + m.l2 * m.W
        mm = 0.9 * mm + 0.1 * gr
        vv = 0.999 * vv + 0.001 * gr * gr
        m.W -= 0.05 * (mm / (1 - 0.9 ** it)) / (np.sqrt(vv / (1 - 0.999 ** it)) + 1e-8)
    logits = Z2 @ m.W
    logits -= logits.max(1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(1, keepdims=True)
    return -np.log(np.maximum(P[np.arange(len(test_y)), test_y], 1e-12))

