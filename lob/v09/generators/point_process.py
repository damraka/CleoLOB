"""Marked temporal point processes of aggressive orders (M45; H2, H5).

Events: prints with the same timestamp and aggressor side are one aggressive order (a sweep); its size is the summed
quantity. Marks: side (buy/sell) x size class (registered tertiles of the training days' event sizes) = 6 marks.
Inputs per event: time (s), mark. Every model returns the per-event negative log-likelihood of (inter-arrival time,
mark) given the history, so models are compared on the same events in nats per event.

Models:

* ``MarkedPoisson`` (classical) - constant intensity per mark;
* ``MarkedHawkes`` (classical) - exponential kernel, lambda_m(t) = mu_m + sum_k alpha_{m,k} sum_{j: m_j=k} beta
  exp(-beta (t - t_j)), exact likelihood with the O(n) recursion, fitted by Adam on log-parameters;
* ``RMTPP`` (neural) - GRU over (mark embedding, log inter-arrival); intensity lambda(t) = exp(v.h + w (t - t_j) + b)
  with a softmax mark head (Du et al., 2016);
* ``TransformerTPP`` (neural) - causal Transformer encoder over the same inputs with the RMTPP intensity form.

Neural models are trained on fixed-length chunks of the training days (CPU, seeded) and evaluated by streaming over
held-out events with the hidden state carried across chunks.
"""
from __future__ import annotations

import math

import numpy as np

N_MARKS = 6


def aggressive_events(cache: dict, *, windows_only: bool = False) -> dict:
    t, s, q = cache["trade_t"], cache["trade_s"].astype(int), cache["trade_q"].astype(float)
    key_change = np.r_[True, (np.diff(t) != 0) | (np.diff(s) != 0)]
    gid = np.cumsum(key_change) - 1
    times = t[key_change]
    sides = s[key_change]
    sizes = np.bincount(gid, weights=q)
    if windows_only:
        from ..data.daycache import HORIZON_S, WINDOW_STARTS_S, WARMUP_S
        keep = np.zeros(len(times), bool)
        for w in WINDOW_STARTS_S:
            keep |= (times >= w - WARMUP_S) & (times < w + HORIZON_S)
        times, sides, sizes = times[keep], sides[keep], sizes[keep]
    return {"t": times, "side": sides, "size": sizes}


def size_cuts(events: list[dict]) -> np.ndarray:
    allsz = np.concatenate([e["size"] for e in events])
    return np.quantile(allsz, [1 / 3, 2 / 3])


def marks(ev: dict, cuts: np.ndarray) -> np.ndarray:
    cls = np.searchsorted(cuts, ev["size"], side="right")
    return (np.where(ev["side"] > 0, 0, 3) + np.clip(cls, 0, 2)).astype(int)


def segments(ev: dict, cuts: np.ndarray, gap: float = 60.0, span: float = 3600.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a day into contiguous segments at gaps > ``gap`` seconds and at every ``span``-second boundary."""
    t, m = ev["t"], marks(ev, cuts)
    if not len(t):
        return []
    block = np.floor(t / span)
    cut = np.flatnonzero((np.diff(t) > gap) | (np.diff(block) != 0)) + 1
    return [(t[a:b], m[a:b]) for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)]) if b - a >= 3]


class MarkedPoisson:
    name, family = "marked_poisson", "classical"

    def fit(self, segs):
        n = np.zeros(N_MARKS)
        T = 0.0
        for t, m in segs:
            n += np.bincount(m, minlength=N_MARKS)
            T += float(t[-1] - t[0])
        self.lam = np.maximum(n / max(T, 1e-9), 1e-9)
        return self

    def nll(self, segs) -> np.ndarray:
        out = []
        L = self.lam.sum()
        for t, m in segs:
            dt = np.diff(t)
            out.append(-(np.log(self.lam[m[1:]]) - L * dt))
        return np.concatenate(out) if out else np.zeros(0)


class MarkedHawkes:
    name, family = "marked_hawkes", "classical"

    def __init__(self, iters: int = 300, lr: float = 0.05, max_events: int = 60_000) -> None:
        self.iters, self.lr, self.max_events = iters, lr, max_events

    @staticmethod
    def excitation(t: np.ndarray, m: np.ndarray, beta: float, block: int = 128) -> np.ndarray:
        """R[i, k] = sum_{j < i, m_j = k} beta exp(-beta (t_i - t_j)), exact, blockwise (no overflow)."""
        n = len(t)
        R = np.zeros((n, N_MARKS))
        carry = np.zeros(N_MARKS)          # excitation just after the previous block's last event, at its time
        t_carry = t[0]
        for s in range(0, n, block):
            e = min(n, s + block)
            tb, mb = t[s:e], m[s:e]
            decay_in = np.exp(-beta * (tb - t_carry))
            D = np.exp(-beta * np.clip(tb[:, None] - tb[None, :], 0.0, None))
            D = np.tril(D, -1)
            onehot = np.zeros((e - s, N_MARKS))
            onehot[np.arange(e - s), mb] = beta
            R[s:e] = decay_in[:, None] * carry[None, :] + D @ onehot
            carry = R[e - 1] + onehot[-1]
            t_carry = tb[-1]
        return R

    @classmethod
    def _terms(cls, params, t, m):
        mu, alpha, beta = params
        Rs = cls.excitation(t, m, beta)
        lam_ev = mu[m] + (alpha[m] * Rs).sum(1)
        dt = np.diff(t)
        after = Rs[:-1].copy()
        after[np.arange(len(dt)), m[:-1]] += beta
        comp = float(mu.sum() * dt.sum() + ((alpha.sum(0)[None, :] * after).sum(1) *
                                             (1 - np.exp(-beta * dt)) / beta).sum())
        return lam_ev, comp, Rs

    def fit(self, segs):
        total = sum(len(t) for t, _ in segs)
        keep, used = [], 0
        for t, m in segs:
            if used >= self.max_events:
                break
            keep.append((t, m))
            used += len(t)
        T = sum(float(t[-1] - t[0]) for t, _ in keep)
        n = np.bincount(np.concatenate([m for _, m in keep]), minlength=N_MARKS)
        mu0 = np.maximum(n / max(T, 1e-9), 1e-6)
        best = None
        for beta in (0.5, 2.0, 10.0):
            theta = np.r_[np.log(mu0 * 0.5), np.full(N_MARKS * N_MARKS, math.log(0.05))]
            m1, v1 = np.zeros_like(theta), np.zeros_like(theta)
            for it in range(1, self.iters + 1):
                g, ll = self._grad(theta, beta, keep)
                m1 = 0.9 * m1 + 0.1 * g
                v1 = 0.999 * v1 + 0.001 * g * g
                theta = theta + self.lr * (m1 / (1 - 0.9 ** it)) / (np.sqrt(v1 / (1 - 0.999 ** it)) + 1e-8)
            _, ll = self._grad(theta, beta, keep, value_only=True)
            if best is None or ll > best[0]:
                best = (ll, theta.copy(), beta)
        _, theta, beta = best
        self.mu = np.exp(theta[:N_MARKS])
        self.alpha = np.exp(theta[N_MARKS:]).reshape(N_MARKS, N_MARKS)
        self.beta = beta
        self.fit_info = {"events_used": used, "events_available": total, "beta": beta,
                         "branching_ratio": float(np.max(np.abs(np.linalg.eigvals(self.alpha))))}
        return self

    def _grad(self, theta, beta, segs, value_only=False):
        mu = np.exp(theta[:N_MARKS])
        alpha = np.exp(theta[N_MARKS:]).reshape(N_MARKS, N_MARKS)
        g_mu = np.zeros(N_MARKS)
        g_al = np.zeros((N_MARKS, N_MARKS))
        ll = 0.0
        count = 0
        for t, m in segs:
            lam, comp, Rs = self._terms((mu, alpha, beta), t, m)
            ll += float(np.log(np.maximum(lam[1:], 1e-12)).sum() - comp)
            count += len(t) - 1
            if value_only:
                continue
            inv = 1.0 / np.maximum(lam[1:], 1e-12)
            mi = m[1:]
            np.add.at(g_mu, mi, inv)
            np.add.at(g_al, mi, inv[:, None] * Rs[1:])
            dur = float(t[-1] - t[0])
            g_mu -= dur
            # compensator gradient wrt alpha[:,k]: integral of the k-excitation = sum_j (1 - e^{-beta (T - t_j)})
            tail = 1.0 - np.exp(-beta * (t[-1] - t))
            g_al -= np.bincount(m, weights=tail, minlength=N_MARKS)[None, :]
        if value_only:
            return None, ll / max(count, 1)
        g = np.r_[g_mu * mu, (g_al * alpha).ravel()] / max(count, 1)
        return g, ll / max(count, 1)

    def nll(self, segs) -> np.ndarray:
        out = []
        for t, m in segs:
            lam_ev, _, Rs = self._terms((self.mu, self.alpha, self.beta), t, m)
            dt = np.diff(t)
            # per-event compensator increments between t_{i-1} and t_i
            comp = self.mu.sum() * dt + (self.alpha.sum(0)[None, :] * Rs[:-1]).sum(1) * \
                (1 - np.exp(-self.beta * dt)) / self.beta
            Rprev = Rs[:-1].copy()
            Rprev[np.arange(len(dt)), m[:-1]] += self.beta
            comp = self.mu.sum() * dt + (self.alpha.sum(0)[None, :] * Rprev).sum(1) * \
                (1 - np.exp(-self.beta * dt)) / self.beta
            out.append(-(np.log(np.maximum(lam_ev[1:], 1e-12)) - comp))
        return np.concatenate(out) if out else np.zeros(0)


class _NeuralTPP:
    """Neural intensity lambda(t) = exp(a + w (t - t_j)) + c: a decaying burst (w < 0) over a floor c > 0.

    a, w and c are read from the hidden state after event j; the compensator over (t_j, t_j + d] is
    (exp(a + w d) - exp(a)) / w + c d. An increasing hazard (w > 0) is excluded: it explodes over long gaps.
    Marks: softmax head.
    """
    family = "neural"

    def __init__(self, hidden: int = 32, steps: int = 1500, chunk: int = 128, batch: int = 32, seed: int = 0) -> None:
        self.hidden, self.steps, self.chunk, self.batch, self.seed = hidden, steps, chunk, batch, seed

    @staticmethod
    def _features(t, m):
        dt = np.r_[0.0, np.diff(t)]
        return np.log1p(dt * 10.0)[:, None].astype(np.float32), np.asarray(m, np.int64),             np.r_[np.diff(t), 0.0].astype(np.float32)

    def _build(self):
        import torch
        self.emb = torch.nn.Embedding(N_MARKS, 8)
        self.head = torch.nn.Linear(self.hidden, 3)          # a, w, b
        self.head_mark = torch.nn.Linear(self.hidden, N_MARKS)

    def _params(self):
        return [*self.encoder_params(), *self.emb.parameters(), *self.head.parameters(),
                *self.head_mark.parameters()]

    def _nll_terms(self, h, next_dt, next_mark):
        import torch
        out = self.head(h)
        a, w_raw, b = torch.clamp(out[..., 0], max=20.0), out[..., 1], out[..., 2]
        w = -torch.nn.functional.softplus(w_raw) - 1e-3                 # decaying burst: w < 0
        c = torch.nn.functional.softplus(b) + 1e-6                       # baseline floor
        burst = torch.exp(a + w * next_dt)
        lam = burst + c
        comp = (burst - torch.exp(a)) / w + c * next_dt
        logp_mark = torch.log_softmax(self.head_mark(h), -1).gather(-1, next_mark.unsqueeze(-1)).squeeze(-1)
        return -(torch.log(lam + 1e-12) - comp + logp_mark)

    def fit(self, segs):
        """Adam with weight decay on half-overlapping chunks; early stopping on 15% held-out chunks (patience 3)."""
        import copy
        import torch
        torch.manual_seed(self.seed)
        self._build()
        self.build_encoder()
        opt = torch.optim.Adam(self._params(), lr=3e-3, weight_decay=1e-4)
        L = self.chunk + 1
        X, M, N = [], [], []
        for t, m in segs:
            x, mk, nxt = self._features(t, m)
            for a in range(0, len(t) - L + 1, max(1, self.chunk // 2)):
                X.append(x[a:a + L])
                M.append(mk[a:a + L])
                N.append(nxt[a:a + L])
        if len(X) < 4:
            raise ValueError("not enough training events for the registered chunking")
        X, M, N = (torch.tensor(np.stack(v)) for v in (X, M, N))
        perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(self.seed))
        n_val = max(1, int(0.15 * len(X)))
        val, tr = perm[:n_val], perm[n_val:]
        g = torch.Generator().manual_seed(self.seed + 1)
        best, best_state, bad = float("inf"), None, 0
        self.fit_info = {"chunks": int(len(X)), "validation_chunks": int(n_val)}
        for step in range(1, self.steps + 1):
            idx = tr[torch.randint(0, len(tr), (min(self.batch, len(tr)),), generator=g)]
            loss = self._nll_terms(self.encode(X[idx, :-1], M[idx, :-1]), N[idx, :-1], M[idx, 1:]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._params(), 5.0)
            opt.step()
            if step % 100 == 0 or step == self.steps:
                with torch.no_grad():
                    v = float(self._nll_terms(self.encode(X[val, :-1], M[val, :-1]), N[val, :-1], M[val, 1:]).mean())
                if v < best - 1e-4:
                    best, bad = v, 0
                    best_state = [copy.deepcopy(p.detach()) for p in self._params()]
                else:
                    bad += 1
                    if bad >= 3:
                        break
        if best_state is not None:
            with torch.no_grad():
                for p, b in zip(self._params(), best_state):
                    p.copy_(b)
        self.fit_info.update({"best_validation_nll": best, "stopped_at_step": step})
        return self

    def nll(self, segs) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for t, m in segs:
                x, mk, nxt = (torch.tensor(v) for v in self._features(t, m))
                vals = []
                for a in range(0, len(t) - 1, self.chunk):
                    b = min(len(t), a + self.chunk + 1)
                    lo = max(0, a - self.context)
                    h = self.encode(x[None, lo:b - 1], mk[None, lo:b - 1])[0]
                    v = self._nll_terms(h, nxt[lo:b - 1], mk[lo + 1:b]).numpy()
                    vals.append(v[a - lo:])
                out.append(np.concatenate(vals)[: len(t) - 1])
        return np.concatenate(out) if out else np.zeros(0)


class RMTPP(_NeuralTPP):
    name = "rmtpp"
    context = 64

    def build_encoder(self):
        import torch
        self.rnn = torch.nn.GRU(9, self.hidden, batch_first=True)

    def encoder_params(self):
        return list(self.rnn.parameters()) if hasattr(self, "rnn") else []

    def encode(self, x, mk):
        import torch
        h, _ = self.rnn(torch.cat([x, self.emb(mk)], -1))
        return h


class TransformerTPP(_NeuralTPP):
    name = "transformer_tpp"
    context = 64

    def build_encoder(self):
        import torch
        self.proj = torch.nn.Linear(9, self.hidden)
        layer = torch.nn.TransformerEncoderLayer(self.hidden, 4, 2 * self.hidden, dropout=0.0, batch_first=True)
        self.enc = torch.nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)

    def encoder_params(self):
        return [*self.proj.parameters(), *self.enc.parameters()] if hasattr(self, "enc") else []

    def encode(self, x, mk):
        import torch
        z = self.proj(torch.cat([x, self.emb(mk)], -1))
        n = z.shape[1]
        mask = torch.triu(torch.full((n, n), float("-inf")), 1)
        return self.enc(z, mask=mask)


CLASSICAL = {"marked_poisson": MarkedPoisson, "marked_hawkes": MarkedHawkes}
NEURAL = {"rmtpp": RMTPP, "transformer_tpp": TransformerTPP}
