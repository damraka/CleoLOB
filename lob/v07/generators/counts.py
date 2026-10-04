"""Count models for the event-driven engine: G1 (state-conditioned Hawkes), G3 (conditional resampling), G4 (GRU).

All three are fitted on development bundles only (``lob.v07.generators.events``)
and share one interface: ``initial()``, ``update(state, realized, features)``
and ``sample(state, features, rng)`` returning six event counts for the next
100 ms bin.

* G1 ``HawkesCounts`` — a discretized, state-conditioned *nonlinear* Hawkes
  process: per mark, ``log E[N] = b + sum_j a_j log1p(E_fast_j) + sum_j c_j
  log1p(E_slow_j) + d . s`` with exponentially decayed past counts of every mark
  (decays 2/s and 0.2/s) and standardized state ``s``; Poisson counts. Fitted by
  ridge-penalized Newton (IRLS) Poisson regression under a stationarity
  constraint: each mark's total excitation elasticity is at most 0.9 (the
  unconstrained fit is superlinear and explodes in closed loop; both fits'
  deviance is reported).
* G3 ``ConditionalCounts`` — nonparametric: resample a historical count vector
  from the cell (spread bucket x imbalance tercile x previous-bin activity class)
  of the current state.
* G4 ``GRUCounts`` — a GRU (hidden 32) over the previous 50 bins of
  ``[log1p counts, standardized state]`` predicting Poisson log-rates; trained
  with torch on CPU, simulated with an exact numpy forward pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .events import MARKS

BIN_S = 0.1
DECAYS = (2.0, 0.2)
MAX_ELASTICITY = 0.9   # total excitation elasticity per mark (< 1: sublinear, closed-loop stationary)
RIDGE = 1e-3


def _standardize(state: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (state - mean) / sd


def guard_bounds(counts: np.ndarray, state: np.ndarray, valid: np.ndarray) -> dict:
    """Stability guard fixed on development data: state range and the 99.9th-percentile count per mark.

    Log-linear excitation can be superlinear in past counts; without a bound a simulated path can
    explode. Rates are capped and state features clipped to what development data covers.
    """
    return {"state_min": state[valid].min(0), "state_max": state[valid].max(0),
            "rate_cap": np.maximum(np.quantile(counts[valid], 0.999, axis=0), 1.0)}


# ----------------------------------------------------------------------------- G1


def excitation_design(counts: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """log1p of exponentially decayed past counts (fast then slow) known at the start of each bin."""
    n, k = counts.shape
    out = np.zeros((n, 2 * k))
    factors = [np.exp(-d * BIN_S) for d in DECAYS]
    e = [np.zeros(k), np.zeros(k)]
    previous = np.zeros(k)
    for i in range(n):
        for j, f in enumerate(factors):
            e[j] = f * (e[j] + previous)
        out[i, :k], out[i, k:] = np.log1p(e[0]), np.log1p(e[1])
        previous = counts[i] if valid[i] else np.zeros(k)
    return out


def poisson_irls(x: np.ndarray, y: np.ndarray, *, ridge: float = RIDGE, iterations: int = 50,
                 offset: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    offset = np.zeros(len(y)) if offset is None else offset
    beta = np.zeros(x.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-6))
    penalty = ridge * len(y) * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    converged = False
    for it in range(iterations):
        eta = np.clip(x @ beta + offset, -20, 6)
        mu = np.exp(eta)
        grad = x.T @ (y - mu) - penalty @ beta
        hess = (x * mu[:, None]).T @ x + penalty
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-7:
            converged = True
            break
    mu = np.exp(np.clip(x @ beta + offset, -20, 6))
    deviance = 2 * np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0) - (y - mu))
    null_mu = y.mean()
    null = 2 * np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / null_mu), 0) - (y - null_mu))
    return beta, {"iterations": it + 1, "converged": converged, "deviance_explained": float(1 - deviance / null)}


@dataclass
class HawkesCounts:
    family: str = "G1_state_hawkes"
    coef: np.ndarray = field(default_factory=lambda: np.zeros((6, 16)))
    state_mean: np.ndarray = field(default_factory=lambda: np.zeros(3))
    state_sd: np.ndarray = field(default_factory=lambda: np.ones(3))
    fit_info: dict = field(default_factory=dict)
    guard: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, counts: np.ndarray, state: np.ndarray, valid: np.ndarray, *,
            max_elasticity: float = MAX_ELASTICITY) -> HawkesCounts:
        mean, sd = state[valid].mean(0), state[valid].std(0) + 1e-9
        x = np.hstack([np.ones((len(counts), 1)), excitation_design(counts, valid),
                       _standardize(state, mean, sd)])[valid]
        excite = slice(1, 1 + 2 * len(MARKS))
        free = np.r_[0, np.arange(1 + 2 * len(MARKS), x.shape[1])]
        coef, info = [], {}
        for m, name in enumerate(MARKS):
            y = counts[valid, m].astype(float)
            beta, diagnostics = poisson_irls(x, y)
            elasticity = float(beta[excite].sum())
            diagnostics["unconstrained_elasticity"] = elasticity
            if elasticity > max_elasticity:
                # Stationarity constraint: shrink the excitation block, then refit intercept and state effects.
                beta[excite] *= max_elasticity / elasticity
                refit, constrained = poisson_irls(x[:, free], y, offset=x[:, excite] @ beta[excite])
                beta[free] = refit
                diagnostics["constrained_deviance_explained"] = constrained["deviance_explained"]
            diagnostics["elasticity"] = float(beta[excite].sum())
            coef.append(beta)
            info[name] = diagnostics
        return cls(coef=np.asarray(coef), state_mean=mean, state_sd=sd, fit_info=info,
                   guard=guard_bounds(counts, state, valid))

    def initial(self):
        return [np.zeros(6), np.zeros(6)]

    def update(self, mstate, realized, features):
        return [np.exp(-d * BIN_S) * (e + realized) for d, e in zip(DECAYS, mstate)]

    def rates(self, mstate, features) -> np.ndarray:
        s = np.clip(features, self.guard["state_min"], self.guard["state_max"])
        x = np.r_[1.0, np.log1p(mstate[0]), np.log1p(mstate[1]), _standardize(s, self.state_mean, self.state_sd)]
        return np.minimum(np.exp(np.clip(self.coef @ x, -20, 4)), self.guard["rate_cap"])

    def sample(self, mstate, features, rng):
        return rng.poisson(self.rates(mstate, features))

    def describe(self) -> dict:
        return {"family": self.family, "parameters": int(self.coef.size), "decays_per_s": list(DECAYS),
                "fit": self.fit_info, "rate_cap_per_bin": np.asarray(self.guard["rate_cap"]).tolist()}


# ----------------------------------------------------------------------------- G3


ACTIVITY_EDGES = (0, 1, 3, 6, 11)


@dataclass
class ConditionalCounts:
    family: str = "G3_conditional_ar"
    cells: dict = field(default_factory=dict)
    imbalance_cuts: tuple = (-0.3, 0.3)
    marginal: np.ndarray = field(default_factory=lambda: np.zeros((1, 6), dtype=np.int16))

    def key(self, features: np.ndarray, previous_total: int) -> tuple:
        spread = int(min(max(round(features[0]), 1), 3))
        imbalance = int(np.searchsorted(self.imbalance_cuts, features[1]))
        activity = int(np.searchsorted(ACTIVITY_EDGES, previous_total, side="right") - 1)
        return spread, imbalance, activity

    @classmethod
    def fit(cls, counts: np.ndarray, state: np.ndarray, valid: np.ndarray) -> ConditionalCounts:
        cuts = tuple(float(q) for q in np.quantile(state[valid, 1], [1 / 3, 2 / 3]))
        model = cls(imbalance_cuts=cuts)
        previous = np.r_[0, counts[:-1].sum(1)]
        previous_valid = np.r_[False, valid[:-1]]
        groups: dict[tuple, list[int]] = {}
        for i in np.flatnonzero(valid & previous_valid):
            groups.setdefault(model.key(state[i], int(previous[i])), []).append(i)
        model.cells = {k: counts[v].astype(np.int16) for k, v in groups.items()}
        model.marginal = counts[valid].astype(np.int16)
        return model

    def initial(self):
        return 0

    def update(self, mstate, realized, features):
        return int(np.sum(realized))

    def sample(self, mstate, features, rng):
        rows = self.cells.get(self.key(features, mstate))
        if rows is None or len(rows) == 0:
            rows = self.marginal
        return rows[rng.integers(len(rows))]

    def describe(self) -> dict:
        return {"family": self.family, "cells": len(self.cells), "rows": int(sum(len(v) for v in self.cells.values())),
                "parameters_effective": "nonparametric (stored historical rows)"}


# ----------------------------------------------------------------------------- G4


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class GRUCounts:
    family: str = "G4_neural_temporal"
    weights: dict = field(default_factory=dict)
    state_mean: np.ndarray = field(default_factory=lambda: np.zeros(3))
    state_sd: np.ndarray = field(default_factory=lambda: np.ones(3))
    hidden: int = 32
    fit_info: dict = field(default_factory=dict)
    guard: dict = field(default_factory=dict)

    def initial(self):
        return np.zeros(self.hidden)

    def _input(self, realized, features):
        s = np.clip(features, self.guard["state_min"], self.guard["state_max"])
        return np.r_[np.log1p(np.minimum(realized, 4 * self.guard["rate_cap"])), _standardize(s, self.state_mean,
                                                                                                self.state_sd)]

    def update(self, mstate, realized, features):
        w = self.weights
        x = self._input(realized, features)
        gi = w["w_ih"] @ x + w["b_ih"]
        gh = w["w_hh"] @ mstate + w["b_hh"]
        h = self.hidden
        r = _sigmoid(gi[:h] + gh[:h])
        z = _sigmoid(gi[h:2 * h] + gh[h:2 * h])
        n = np.tanh(gi[2 * h:] + r * gh[2 * h:])
        return (1 - z) * n + z * mstate

    def rates(self, mstate) -> np.ndarray:
        return np.minimum(np.exp(np.clip(self.weights["w_out"] @ mstate + self.weights["b_out"], -20, 4)),
                          self.guard["rate_cap"])

    def sample(self, mstate, features, rng):
        return rng.poisson(self.rates(mstate))

    @classmethod
    def fit(cls, counts: np.ndarray, state: np.ndarray, valid: np.ndarray, *, seed: int, sequences: int = 24000,
            length: int = 50, epochs: int = 4, hidden: int = 32, threads: int = 4) -> GRUCounts:
        import torch
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        mean, sd = state[valid].mean(0), state[valid].std(0) + 1e-9
        inputs = np.hstack([np.log1p(counts), _standardize(state, mean, sd)]).astype(np.float32)
        inputs[~valid] = 0.0
        rng = np.random.default_rng(seed)
        # Input at bin i is (counts of bin i-1, state at start of bin i); target is counts of bin i.
        x_all = np.vstack([np.zeros((1, inputs.shape[1]), np.float32), inputs[:-1]])
        x_all[:, 6:] = inputs[:, 6:]
        starts = rng.integers(0, len(counts) - length - 1, sequences)
        keep = np.asarray([valid[s:s + length].all() for s in starts])
        starts = starts[keep]
        gru = torch.nn.GRU(inputs.shape[1], hidden, batch_first=True)
        head = torch.nn.Linear(hidden, 6)
        with torch.no_grad():
            head.bias.copy_(torch.log(torch.tensor(counts[valid].mean(0) + 1e-3, dtype=torch.float32)))
        optimiser = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=3e-3)
        losses = []
        for epoch in range(epochs):
            order = rng.permutation(len(starts))
            total = 0.0
            for b in range(0, len(order), 256):
                idx = starts[order[b:b + 256]]
                xb = torch.tensor(np.stack([x_all[s:s + length] for s in idx]))
                yb = torch.tensor(np.stack([counts[s:s + length] for s in idx]).astype(np.float32))
                out, _ = gru(xb)
                log_rate = head(out)
                loss = torch.nn.functional.poisson_nll_loss(log_rate, yb, log_input=True)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                total += float(loss.detach()) * len(idx)
            losses.append(total / len(starts))
        state_dict = {k: v.detach().numpy().astype(float) for k, v in gru.state_dict().items()}
        weights = {"w_ih": state_dict["weight_ih_l0"], "w_hh": state_dict["weight_hh_l0"],
                   "b_ih": state_dict["bias_ih_l0"], "b_hh": state_dict["bias_hh_l0"],
                   "w_out": head.weight.detach().numpy().astype(float), "b_out": head.bias.detach().numpy().astype(float)}
        return cls(weights=weights, state_mean=mean, state_sd=sd, hidden=hidden, guard=guard_bounds(counts, state, valid),
                   fit_info={"sequences": int(len(starts)), "length": length, "epochs": epochs,
                             "train_loss_by_epoch": losses, "seed": seed})

    def describe(self) -> dict:
        return {"family": self.family, "hidden": self.hidden,
                "parameters": int(sum(v.size for v in self.weights.values())), "fit": self.fit_info}
