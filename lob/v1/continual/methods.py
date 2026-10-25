"""Continual learning comparison framework (M127), meta-learning for rapid regime adaptation (M128) and learned
representation transfer with model-based planning (M129).

* ``compare_continual``: trains a model on a chronological task sequence with each registered method (``naive``
  fine-tuning, ``replay`` buffer, ``ewc``) - the unchanged v0.9 ``SeqClassifier`` - and reports per method the
  forgetting on every earlier task, the forward transfer to the next task and the final average NLL.
* ``Reptile``: first-order meta-learning (Nichol et al. 2018) of a small classifier over regime tasks; ``adapt``
  performs k gradient steps on a few examples of a new regime. ``few_shot_vs_refit`` compares the adapted model with a
  model trained from scratch on the same few examples and with a full refit on all data.
* ``Representation``: a small autoencoder of state features learned on source data; ``transfer_probe`` fits a linear
  probe on target data in the frozen latent space vs on raw features; ``plan`` is a one-step model-based planner in
  latent space (learned linear dynamics per action, greedy action that minimizes predicted cost).
"""
from __future__ import annotations

import numpy as np

from ..generators.neural.base import mlp, torch


def compare_continual(tasks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], *, seed: int = 0,
                      steps: int = 300, methods=("naive", "replay", "ewc"), ewc_lambda: float = 50.0) -> dict:
    """``tasks[k] = (X, y, train_idx, val_idx)`` in chronological order (v0.9 sequence-feature layout)."""
    from ...v09.representations import sequence as S
    out = {}
    for method in methods:
        model = S.SeqClassifier("gru", seed=seed, steps=steps)
        own, anchor = {}, None
        forward = {}
        for k, (X, y, tr, val) in enumerate(tasks):
            if k > 0:
                forward[k] = float(model.nll(X, y, val).mean())
            replay = None
            if method == "replay" and k > 0:
                Xr = np.vstack([tasks[j][0] for j in range(k)])
                yr = np.concatenate([tasks[j][1] for j in range(k)])
                offs = np.cumsum([0] + [len(tasks[j][0]) for j in range(k)])
                ir = np.concatenate([tasks[j][2] + offs[j] for j in range(k)])
                replay = (Xr, yr, ir)
            ewc = (anchor[0], anchor[1], ewc_lambda) if method == "ewc" and anchor is not None else None
            model.fit(X, y, tr, init=(k == 0), steps=steps, replay=replay, ewc=ewc)
            own[k] = float(model.nll(X, y, val).mean())
            if method == "ewc":
                fisher, params = model.fisher(X, y, tr)
                anchor = (fisher, params) if anchor is None else ([0.5 * (a + b) for a, b in zip(anchor[0], fisher)],
                                                                  params)
        last = {k: float(model.nll(X, y, val).mean()) for k, (X, y, tr, val) in enumerate(tasks)}
        out[method] = {"forgetting": {k: last[k] - own[k] for k in range(len(tasks) - 1)}, "after_own": own,
                       "after_last": last, "forward_nll": forward,
                       "mean_final_nll": float(np.mean(list(last.values())))}
    return out


class Reptile:
    def __init__(self, n_in: int, n_classes: int, *, hidden: int = 32, inner_steps: int = 5, inner_lr: float = 0.05,
                 meta_lr: float = 0.3, seed: int = 0) -> None:
        T = torch()
        T.manual_seed(seed)
        self.net = mlp([n_in, hidden, hidden, n_classes])
        self.inner_steps, self.inner_lr, self.meta_lr = inner_steps, inner_lr, meta_lr
        self.rng = np.random.default_rng(seed)

    def _sgd(self, net, X, y, steps):
        T = torch()
        opt = T.optim.SGD(net.parameters(), lr=self.inner_lr)
        Xt, yt = T.as_tensor(X, dtype=T.float32), T.as_tensor(y)
        for _ in range(steps):
            loss = T.nn.functional.cross_entropy(net(Xt), yt)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def meta_fit(self, tasks: list[tuple[np.ndarray, np.ndarray]], iterations: int = 200, shots: int = 32):
        import copy
        T = torch()
        for _ in range(iterations):
            X, y = tasks[int(self.rng.integers(len(tasks)))]
            idx = self.rng.choice(len(X), min(shots, len(X)), replace=False)
            fast = copy.deepcopy(self.net)
            self._sgd(fast, X[idx], y[idx], self.inner_steps)
            with T.no_grad():
                for p, q in zip(self.net.parameters(), fast.parameters()):
                    p += self.meta_lr * (q - p)
        return self

    def adapt(self, X, y, steps: int | None = None):
        import copy
        fast = copy.deepcopy(self.net)
        self._sgd(fast, X, y, steps or self.inner_steps)
        return fast


def nll_of(net, X, y) -> float:
    T = torch()
    with T.no_grad():
        return float(T.nn.functional.cross_entropy(net(T.as_tensor(X, dtype=T.float32)), T.as_tensor(y)))


def few_shot_vs_refit(meta: Reptile, new_task, *, shots: int = 16, seed: int = 0) -> dict:
    X, y = new_task
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), shots, replace=False)
    test = np.setdiff1d(np.arange(len(X)), idx)
    adapted = meta.adapt(X[idx], y[idx], steps=10)
    scratch = Reptile(X.shape[1], int(y.max()) + 1, seed=seed + 1)
    scratch_net = scratch.adapt(X[idx], y[idx], steps=10)
    return {"adapted_nll": nll_of(adapted, X[test], y[test]), "scratch_nll": nll_of(scratch_net, X[test], y[test]),
            "shots": shots}


class Representation:
    def __init__(self, n_in: int, *, latent: int = 3, hidden: int = 32, steps: int = 800, seed: int = 0) -> None:
        T = torch()
        T.manual_seed(seed)
        self.enc, self.dec = mlp([n_in, hidden, latent]), mlp([latent, hidden, n_in])
        self.steps, self.seed = steps, seed

    def fit(self, X: np.ndarray):
        T = torch()
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = T.as_tensor(((X - self.mu) / self.sd).astype(np.float32))
        opt = T.optim.Adam([*self.enc.parameters(), *self.dec.parameters()], lr=3e-3)
        rng = np.random.default_rng(self.seed)
        for _ in range(self.steps):
            b = Z[rng.integers(0, len(Z), min(256, len(Z)))]
            loss = ((self.dec(self.enc(b)) - b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return self

    def encode(self, X):
        T = torch()
        with T.no_grad():
            return self.enc(T.as_tensor(((X - self.mu) / self.sd).astype(np.float32))).numpy()


def transfer_probe(rep: Representation, Xt: np.ndarray, yt: np.ndarray, *, train_frac: float = 0.3) -> dict:
    n = int(len(Xt) * train_frac)

    def ridge(F, y, Ft):
        A = np.column_stack([np.ones(len(F)), F])
        w = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ y)
        return np.column_stack([np.ones(len(Ft)), Ft]) @ w
    Z = rep.encode(Xt)
    mse_lat = float(((ridge(Z[:n], yt[:n], Z[n:]) - yt[n:]) ** 2).mean())
    mse_raw = float(((ridge(Xt[:n], yt[:n], Xt[n:]) - yt[n:]) ** 2).mean())
    return {"latent_probe_mse": mse_lat, "raw_probe_mse": mse_raw}


def fit_latent_dynamics(Z, A, Z_next, n_actions: int) -> dict:
    out = {}
    for a in range(n_actions):
        m = A == a
        if m.sum() < Z.shape[1] + 2:
            continue
        X = np.column_stack([np.ones(m.sum()), Z[m]])
        out[a] = np.linalg.lstsq(X, Z_next[m], rcond=None)[0]
    return out


def plan(z: np.ndarray, dynamics: dict, cost) -> int:
    """Greedy one-step model-based action: argmin over actions of cost(predicted next latent)."""
    best, arg = np.inf, 0
    for a, W in dynamics.items():
        c = float(cost(np.r_[1.0, z] @ W))
        if c < best:
            best, arg = c, a
    return arg


__all__ = ["compare_continual", "Reptile", "few_shot_vs_refit", "Representation", "transfer_probe",
           "fit_latent_dynamics", "plan", "nll_of"]
