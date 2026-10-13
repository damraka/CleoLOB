"""Queue-survival models and survival metrics (M67, M68).

Models (fit on training units, predict survival curves S(t | x) on the registered grid):

* ``KaplanMeier`` - marginal baseline;
* ``CoxPH`` - proportional hazards, Breslow ties, Newton-Raphson with a small ridge;
* ``WeibullAFT`` - accelerated failure time, censored maximum likelihood (Adam on the exact log-likelihood);
* ``BoostedHazard`` - discrete-time hazard on the grid intervals, gradient-boosted depth-2 histogram trees on the
  logistic loss (person-period expansion);
* ``NeuralHazard`` - discrete-time hazard MLP (PyTorch, CPU, seeded; optional extra ``neural``).

Metrics: time-dependent Brier score with inverse-probability-of-censoring weights (Graf et al.), integrated Brier
score over [0, 30] s (trapezoid on the grid), Harrell's C-index (exact, O(n log n) via a Fenwick tree on the
predicted 5 s risk), and calibration (expected calibration error of P(T <= 5 s) over deciles).
"""
from __future__ import annotations

import math

import numpy as np

GRID = np.asarray([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])


def _standardize(X: np.ndarray, stats=None):
    if stats is None:
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        stats = (mu, sd)
    return (X - stats[0]) / stats[1], stats


def km_curve(T: np.ndarray, E: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    order = np.argsort(T, kind="stable")
    t, e = T[order], E[order].astype(float)
    uniq, idx = np.unique(t, return_index=True)
    deaths = np.add.reduceat(e, idx)
    at_risk = len(t) - idx
    surv = np.cumprod(1.0 - deaths / at_risk)
    pos = np.searchsorted(uniq, grid, side="right") - 1
    return np.where(pos >= 0, surv[np.clip(pos, 0, None)], 1.0)


class KaplanMeier:
    name = "kaplan_meier"

    def fit(self, X, T, E):
        self.curve = km_curve(T, E)
        return self

    def predict(self, X) -> np.ndarray:
        return np.tile(self.curve, (len(X), 1))


class CoxPH:
    name = "cox"

    def __init__(self, ridge: float = 1.0, iters: int = 50, max_rows: int = 20_000, seed: int = 0) -> None:
        self.ridge, self.iters, self.max_rows, self.seed = ridge, iters, max_rows, seed

    def fit(self, X, T, E):
        X, T, E = np.asarray(X, float), np.asarray(T, float), np.asarray(E, bool)
        if len(T) > self.max_rows:                      # exact Newton steps on a seeded subsample
            keep = np.sort(np.random.default_rng(self.seed).choice(len(T), self.max_rows, replace=False))
            X, T, E = X[keep], T[keep], E[keep]
        Z, self.stats = _standardize(X)
        order = np.argsort(-T, kind="stable")          # descending time: risk sets are prefixes
        Zd, ed = Z[order], E[order].astype(float)
        beta = np.zeros(Z.shape[1])
        ev = ed > 0
        def loglik(bb):
            eta = np.clip(Zd @ bb, -30, 30)
            return float((ed * (eta - np.log(np.cumsum(np.exp(eta))))).sum() - 0.5 * self.ridge * bb @ bb)

        current = loglik(beta)
        for _ in range(self.iters):
            w = np.exp(np.clip(Zd @ beta, -30, 30))
            s0 = np.cumsum(w)
            s1 = np.cumsum(w[:, None] * Zd, axis=0)
            s2 = np.cumsum(w[:, None, None] * (Zd[:, :, None] * Zd[:, None, :]), axis=0)
            zbar = s1 / s0[:, None]
            grad = (ed[:, None] * (Zd - zbar)).sum(0) - self.ridge * beta
            hess = (s2[ev] / s0[ev, None, None] - zbar[ev, :, None] * zbar[ev, None, :]).sum(0)
            step = np.linalg.solve(hess + self.ridge * np.eye(len(beta)), grad)
            scale = 1.0
            while scale > 1e-4:                         # damped Newton: never accept a decrease
                trial = beta + scale * step
                value = loglik(trial)
                if value >= current - 1e-9:
                    break
                scale *= 0.5
            beta, current = trial, value
            if np.abs(scale * step).max() < 1e-6:
                break
        self.beta = beta
        asc = np.argsort(T, kind="stable")
        ta, ea = T[asc], E[asc].astype(float)
        wa = np.exp(np.clip(Z[asc] @ beta, -30, 30))
        risk = np.cumsum(wa[::-1])[::-1]               # sum of w over t_j >= t_i
        cum = np.cumsum(ea / np.maximum(risk, 1e-300))
        pos = np.searchsorted(ta, GRID, side="right") - 1
        self.H0 = np.where(pos >= 0, cum[np.clip(pos, 0, None)], 0.0)
        return self

    def predict(self, X) -> np.ndarray:
        Z, _ = _standardize(np.asarray(X, float), self.stats)
        r = np.exp(np.clip(Z @ self.beta, -30, 30))
        return np.exp(-np.outer(r, self.H0))


class WeibullAFT:
    name = "weibull_aft"

    def __init__(self, iters: int = 400, lr: float = 0.05) -> None:
        self.iters, self.lr = iters, lr

    def fit(self, X, T, E):
        Z, self.stats = _standardize(np.asarray(X, float))
        Z1 = np.column_stack([np.ones(len(Z)), Z])
        lt, e = np.log(T), E.astype(float)
        theta = np.zeros(Z1.shape[1] + 1)                # coefficients for log scale, then log shape
        theta[0] = float(np.median(lt))
        m, v = np.zeros_like(theta), np.zeros_like(theta)
        for it in range(1, self.iters + 1):
            g = self._grad(theta, Z1, lt, e)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            theta += self.lr * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
        self.theta = theta
        return self

    @staticmethod
    def _grad(theta, Z1, lt, e):
        gamma, logk = theta[:-1], theta[-1]
        k = math.exp(logk)
        u = np.clip(k * (lt - Z1 @ gamma), -50, 50)
        eu = np.exp(u)
        # log-lik per obs: e*(log k + u - lt) - exp(u)
        d_u = e - eu
        g_gamma = -(k * d_u) @ Z1 / len(lt)
        g_logk = float((e + d_u * u).mean())
        return np.r_[g_gamma, g_logk]

    def predict(self, X) -> np.ndarray:
        Z, _ = _standardize(np.asarray(X, float), self.stats)
        Z1 = np.column_stack([np.ones(len(Z)), Z])
        k = math.exp(self.theta[-1])
        scale = Z1 @ self.theta[:-1]
        u = np.clip(k * (np.log(GRID)[None, :] - scale[:, None]), -50, 50)
        return np.exp(-np.exp(u))


def person_period(X, T, E, grid=GRID):
    """Discrete-time expansion: one row per (sample, interval at risk); label 1 in the event interval."""
    edges = np.r_[0.0, grid]
    k_event = np.searchsorted(grid, T, side="left")          # interval index containing T
    n_rows = np.minimum(k_event + 1, len(grid))
    idx = np.repeat(np.arange(len(T)), n_rows)
    k = np.concatenate([np.arange(r) for r in n_rows]) if len(n_rows) else np.zeros(0, int)
    y = ((k == k_event[idx]) & E[idx]).astype(float)
    feats = np.column_stack([X[idx], np.log(edges[k + 1])])
    return feats, y, idx, k


class _Tree:
    """Depth-2 regression tree on pre-binned features (histogram splits)."""

    def fit(self, B, g, h, n_bins, lam=1.0, min_leaf=50.0):
        self.nodes = {}
        self._split(B, g, h, np.arange(len(g)), 0, 0, n_bins, lam, min_leaf)
        return self

    def _split(self, B, g, h, rows, node, depth, n_bins, lam, min_leaf):
        G, Hs = g[rows].sum(), h[rows].sum()
        if depth == 2 or len(rows) < 2 * min_leaf:
            self.nodes[node] = ("leaf", -G / (Hs + lam))
            return
        best = (0.0, None, None)
        base = G * G / (Hs + lam)
        for f in range(B.shape[1]):
            gb = np.bincount(B[rows, f], weights=g[rows], minlength=n_bins)
            hb = np.bincount(B[rows, f], weights=h[rows], minlength=n_bins)
            gl, hl = np.cumsum(gb)[:-1], np.cumsum(hb)[:-1]
            gr, hr = G - gl, Hs - hl
            ok = (hl >= min_leaf * 0.01) & (hr >= min_leaf * 0.01)
            gain = np.where(ok, gl * gl / (hl + lam) + gr * gr / (hr + lam) - base, -np.inf)
            j = int(np.argmax(gain))
            if gain[j] > best[0]:
                best = (float(gain[j]), f, j)
        if best[1] is None:
            self.nodes[node] = ("leaf", -G / (Hs + lam))
            return
        _, f, j = best
        self.nodes[node] = ("split", f, j)
        left = rows[B[rows, f] <= j]
        right = rows[B[rows, f] > j]
        self._split(B, g, h, left, 2 * node + 1, depth + 1, n_bins, lam, min_leaf)
        self._split(B, g, h, right, 2 * node + 2, depth + 1, n_bins, lam, min_leaf)

    def predict(self, B):
        out = np.zeros(len(B))
        node = np.zeros(len(B), int)
        for _ in range(3):
            for nid in np.unique(node[node >= 0]):
                kind = self.nodes[nid]
                sel = node == nid
                if kind[0] == "leaf":
                    out[sel] = kind[1]
                    node[sel] = -1
                else:
                    go_left = B[sel, kind[1]] <= kind[2]
                    node[np.flatnonzero(sel)[go_left]] = 2 * nid + 1
                    node[np.flatnonzero(sel)[~go_left]] = 2 * nid + 2
            if (node < 0).all():
                break
        return out


class BoostedHazard:
    name = "boosted_hazard"

    def __init__(self, trees: int = 120, lr: float = 0.1, n_bins: int = 32, seed: int = 0) -> None:
        self.trees, self.lr, self.n_bins, self.seed = trees, lr, n_bins, seed

    def _bin(self, F):
        return np.stack([np.clip(np.searchsorted(self.cuts[j], F[:, j], side="right"), 0, self.n_bins - 1)
                         for j in range(F.shape[1])], axis=1).astype(np.int64)

    def fit(self, X, T, E):
        F, y, _, _ = person_period(np.asarray(X, float), T, E)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.cuts = [np.unique(np.quantile(F[:, j], qs)) for j in range(F.shape[1])]
        B = self._bin(F)
        p0 = min(max(y.mean(), 1e-4), 1 - 1e-4)
        self.base = math.log(p0 / (1 - p0))
        f = np.full(len(y), self.base)
        self.ensemble = []
        for _ in range(self.trees):
            p = 1 / (1 + np.exp(-f))
            g, h = p - y, p * (1 - p)
            tree = _Tree().fit(B, g, h, self.n_bins)
            f += self.lr * tree.predict(B)
            self.ensemble.append(tree)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        edges = np.r_[0.0, GRID]
        haz = np.zeros((len(X), len(GRID)))
        for k in range(len(GRID)):
            F = np.column_stack([X, np.full(len(X), math.log(edges[k + 1]))])
            B = self._bin(F)
            f = self.base + self.lr * sum(t.predict(B) for t in self.ensemble)
            haz[:, k] = 1 / (1 + np.exp(-f))
        return np.cumprod(1 - haz, axis=1)


class NeuralHazard:
    name = "neural_hazard"

    def __init__(self, hidden: int = 32, steps: int = 1500, batch: int = 512, seed: int = 0) -> None:
        self.hidden, self.steps, self.batch, self.seed = hidden, steps, batch, seed

    def fit(self, X, T, E):
        import torch
        torch.manual_seed(self.seed)
        Z, self.stats = _standardize(np.asarray(X, float))
        k_event = np.searchsorted(GRID, T, side="left")
        at_risk = (np.arange(len(GRID))[None, :] <= k_event[:, None]).astype(np.float32)
        label = ((np.arange(len(GRID))[None, :] == k_event[:, None]) & E[:, None]).astype(np.float32)
        self.net = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], self.hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(self.hidden, len(GRID)))
        marginal = np.clip(label.sum(0) / np.maximum(at_risk.sum(0), 1.0), 1e-4, 1 - 1e-4)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor(np.log(marginal / (1 - marginal)), dtype=torch.float32))
            self.net[-1].weight.mul_(0.1)
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        xt, rt, lt = (torch.tensor(a, dtype=torch.float32) for a in (Z, at_risk, label))
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.steps):
            b = torch.randint(0, len(xt), (min(self.batch, len(xt)),), generator=g)
            logits = self.net(xt[b])
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, lt[b], reduction="none")
                    * rt[b]).sum() / rt[b].sum().clamp_min(1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    def predict(self, X) -> np.ndarray:
        import torch
        Z, _ = _standardize(np.asarray(X, float), self.stats)
        with torch.no_grad():
            haz = torch.sigmoid(self.net(torch.tensor(Z, dtype=torch.float32))).numpy().astype(float)
        return np.cumprod(1 - haz, axis=1)


# ============================================================================= metrics


def _censor_km(T, E):
    return lambda t: np.maximum(np.interp(t, *_km_points(T, ~E)), 1e-3)


def _km_points(T, E):
    order = np.argsort(T, kind="stable")
    t, e = T[order], E[order].astype(float)
    uniq, idx = np.unique(t, return_index=True)
    deaths = np.add.reduceat(e, idx)
    at_risk = len(t) - idx
    surv = np.cumprod(1.0 - deaths / at_risk)
    return np.r_[0.0, uniq], np.r_[1.0, surv]


def brier(S: np.ndarray, T: np.ndarray, E: np.ndarray, grid: np.ndarray = GRID) -> np.ndarray:
    """IPCW time-dependent Brier score at each grid time."""
    G = _censor_km(T, E)
    out = np.zeros(len(grid))
    g_T = G(T)
    for j, tau in enumerate(grid):
        died = (T <= tau) & E
        alive = T > tau
        w = np.where(died, 1.0 / g_T, np.where(alive, 1.0 / G(np.full(len(T), tau)), 0.0))
        out[j] = float((w * (np.where(alive, 1.0, 0.0) - S[:, j]) ** 2).mean())
    return out


def integrated_brier(S, T, E, grid: np.ndarray = GRID) -> float:
    b = brier(S, T, E, grid)
    x = np.r_[0.0, grid]
    y = np.r_[0.0, b]
    return float(np.trapezoid(y, x) / grid[-1]) if hasattr(np, "trapezoid") else float(np.trapz(y, x) / grid[-1])


def c_index(risk: np.ndarray, T: np.ndarray, E: np.ndarray) -> float:
    """Harrell's C: P(risk_i > risk_j | T_i < T_j, E_i), exact in O(n log n)."""
    order = np.argsort(T, kind="stable")
    r, t, e = risk[order], T[order], E[order]
    ranks = np.searchsorted(np.unique(r), r)          # 0-based dense ranks
    m = int(ranks.max()) + 1 if len(ranks) else 1
    tree = np.zeros(m + 1)

    def add(i):
        i += 1
        while i <= m:
            tree[i] += 1
            i += i & -i

    def prefix(i):              # count of inserted ranks <= i
        s, i = 0.0, i + 1
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    concordant = ties = total = 0.0
    n = len(t)
    j = n - 1
    inserted = 0
    # traverse from the largest time: insert later times, then score earlier events against them
    for i in range(n - 1, -1, -1):
        while j > i and t[j] > t[i]:
            add(ranks[j])
            inserted += 1
            j -= 1
        if e[i] and inserted:
            less = prefix(ranks[i] - 1) if ranks[i] > 0 else 0.0
            eq = prefix(ranks[i]) - less
            concordant += less
            ties += eq
            total += inserted
    return float((concordant + 0.5 * ties) / total) if total else float("nan")


def calibration_error(S: np.ndarray, T: np.ndarray, E: np.ndarray, at: float = 5.0, bins: int = 10) -> float:
    j = int(np.searchsorted(GRID, at))
    p = 1.0 - S[:, j]
    known = (T > at) | E
    y = ((T <= at) & E).astype(float)
    p, y = p[known], y[known]
    if not len(p):
        return float("nan")
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    err = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            err += sel.mean() * abs(p[sel].mean() - y[sel].mean())
    return float(err)


def evaluate(S, T, E) -> dict:
    j5 = int(np.searchsorted(GRID, 5.0))
    return {"ibs": integrated_brier(S, T, E), "brier": dict(zip((f"{g:g}s" for g in GRID), brier(S, T, E).tolist())),
            "c_index": c_index(1.0 - S[:, j5], T, E), "ece_5s": calibration_error(S, T, E)}


MODELS = {"kaplan_meier": KaplanMeier, "cox": CoxPH, "weibull_aft": WeibullAFT, "boosted_hazard": BoostedHazard,
          "neural_hazard": NeuralHazard}
