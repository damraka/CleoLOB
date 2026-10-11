"""The six registered v0.8 generator families (M8).

* ``poisson`` — homogeneous Poisson counts (the existing baseline family, now with
  tied totals and side shares so SBI can work on three rates);
* ``hawkes`` — discretized multivariate linear Hawkes counts with an exponential
  kernel: ``lambda_t = mu + alpha E_t``, ``E_t = sum_k f^k N_{t-k}``, ``f = exp(-beta dt)``.
  ``alpha`` is stored as a shape (spectral radius of its branching matrix = 1)
  times a scalar ``branching``; stationarity requires ``branching < 1`` and the
  fit caps it at 0.95 (recorded);
* ``queue_reactive`` — rates depend on the relevant touch queue and the spread:
  ``log lambda_m = log base_m + gamma_m log(q / qref) + log(spread_mult_m) 1[spread >= 2]``;
* ``regime_switching`` — hidden Markov regimes with Poisson emissions (Baum-Welch fit,
  forward-filter predictive scoring);
* ``state_dependent`` — log-linear Poisson GLM on standardized book features;
* ``mixture`` — a mixture of fitted components with fixed-share switching. With
  ``switch_prob = 0`` it is a path-level mixture (model uncertainty); scoring is
  the sequential Bayesian mixture predictive, which is also the dynamic model
  averaging rule of ``lob.v08.model_averaging``.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from ...v07.generators.counts import poisson_irls
from ..market.frame import BIN_S, EventFrame
from .base import K, MarketGenerator, poisson_logpmf, register

TOUCH_OF_MARK = (1, 0, 0, 1, 0, 1)   # index into (bid_q0, ask_q0): market_buy hits the ask, ...


def _valid_counts(frame: EventFrame) -> np.ndarray:
    return frame.counts[frame.valid].astype(float)


# ----------------------------------------------------------------------------- Poisson


@register
class PoissonGenerator(MarketGenerator):
    name = "poisson"
    version = "1"
    capabilities = frozenset({"count_process", "closed_form_loglik"})

    @classmethod
    def default_params(cls) -> dict:
        return {"market": 1.0, "add": 4.0, "cancel": 3.8, "market_buy_share": 0.5, "add_bid_share": 0.5,
                "cancel_bid_share": 0.5}

    def _refresh(self) -> None:
        p = self.params
        self._mu = np.asarray([p["market"] * p["market_buy_share"], p["market"] * (1 - p["market_buy_share"]),
                               p["add"] * p["add_bid_share"], p["add"] * (1 - p["add_bid_share"]),
                               p["cancel"] * p["cancel_bid_share"], p["cancel"] * (1 - p["cancel_bid_share"])])

    def __init__(self, params=None, book=None) -> None:
        super().__init__(params, book)
        self._refresh()

    def expected(self, state, x) -> np.ndarray:
        return self._mu

    def loglik(self, frame: EventFrame) -> np.ndarray:
        out = np.full(len(frame), np.nan)
        v = frame.valid
        out[v] = poisson_logpmf(frame.counts[v], self._mu[None, :]).sum(1)
        return out

    def fit(self, frame: EventFrame) -> PoissonGenerator:
        c = _valid_counts(frame).mean(0)
        share = lambda a, b: float(a / (a + b)) if a + b > 0 else 0.5  # noqa: E731
        params = {"market": float(c[0] + c[1]), "add": float(c[2] + c[3]), "cancel": float(c[4] + c[5]),
                  "market_buy_share": share(c[0], c[1]), "add_bid_share": share(c[2], c[3]),
                  "cancel_bid_share": share(c[4], c[5])}
        g = PoissonGenerator(params, self.book)
        g.fit_info = {"method": "maximum likelihood (bin means)", "bins": int(frame.valid.sum())}
        return g

    def parameter_space(self) -> dict:
        return {"market": (0.05, 20.0), "add": (0.2, 60.0), "cancel": (0.2, 60.0)}


# ----------------------------------------------------------------------------- Hawkes


def _decayed(counts: np.ndarray, valid: np.ndarray, f: float) -> np.ndarray:
    """E_t = sum_{k>=1} f^k N_{t-k} known at the start of bin t."""
    n = len(counts)
    out = np.zeros((n, counts.shape[1]))
    e = np.zeros(counts.shape[1])
    for i in range(n):
        out[i] = e
        e = f * (e + (counts[i] if valid[i] else 0.0))
    return out


def _spectral_radius(m: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(m)))) if np.any(m) else 0.0


@register
class HawkesGenerator(MarketGenerator):
    name = "hawkes"
    version = "1"
    capabilities = frozenset({"count_process", "self_exciting", "filtered_loglik"})
    BETAS = (0.5, 1.0, 2.0, 5.0, 10.0)
    MAX_BRANCHING = 0.95

    @classmethod
    def default_params(cls) -> dict:
        shape = np.full((K, K), 0.02)
        np.fill_diagonal(shape, 0.12)
        shape[0, 0] = shape[1, 1] = 0.3
        return {"mu": [0.25, 0.25, 1.4, 1.4, 1.3, 1.3], "shape": (shape / _spectral_radius(shape)).tolist(),
                "branching": 0.5, "beta": 2.0}

    def __init__(self, params=None, book=None) -> None:
        super().__init__(params, book)
        self._refresh()

    def _refresh(self) -> None:
        p = self.params
        f = math.exp(-float(p["beta"]) * BIN_S)
        self._f = f
        self._mu = np.asarray(p["mu"], float)
        # branching matrix G = alpha f / (1 - f) = branching * shape  =>  alpha = branching * shape (1 - f) / f
        self._alpha = float(p["branching"]) * np.asarray(p["shape"], float) * (1 - f) / f

    def initial_state(self, simulate: bool = False):
        return np.zeros(K)

    def expected(self, state, x) -> np.ndarray:
        return self._mu + self._alpha @ state

    def update(self, state, realized, x, own=None):
        return self._f * (state + realized)

    def fit(self, frame: EventFrame, *, iterations: int = 300) -> HawkesGenerator:
        y = frame.counts.astype(float)
        v = frame.valid
        best = None
        for beta in self.BETAS:
            f = math.exp(-beta * BIN_S)
            e = _decayed(y, v, f)
            x = np.hstack([np.ones((len(y), 1)), e])[v]
            coef = np.zeros((K, K + 1))
            ll = 0.0
            for m in range(K):
                target = y[v, m]
                theta = np.r_[max(target.mean(), 1e-3), np.full(K, 0.01)]
                colsum = x.sum(0) + 1e-12
                for _ in range(iterations):   # EM / multiplicative updates for a nonnegative linear Poisson model
                    lam = np.maximum(x @ theta, 1e-12)
                    theta = theta * (x.T @ (target / lam)) / colsum
                lam = np.maximum(x @ theta, 1e-12)
                ll += float(np.sum(poisson_logpmf(target, lam)))
                coef[m] = theta
            if best is None or ll > best[0]:
                best = (ll, beta, coef)
        ll, beta, coef = best
        f = math.exp(-beta * BIN_S)
        g = coef[:, 1:] * f / (1 - f)
        rho = _spectral_radius(g)
        capped = rho > self.MAX_BRANCHING
        shape = g / rho if rho > 0 else np.eye(K)
        params = {"mu": coef[:, 0].tolist(), "shape": shape.tolist(),
                  "branching": float(min(rho, self.MAX_BRANCHING)), "beta": float(beta)}
        out = HawkesGenerator(params, self.book)
        out.fit_info = {"method": "EM for nonnegative linear Poisson regression per mark; beta by likelihood grid",
                        "beta_grid": list(self.BETAS), "unconstrained_branching": rho,
                        "branching_capped": bool(capped), "loglik_at_fit": ll, "bins": int(v.sum())}
        return out

    def parameter_space(self) -> dict:
        return {"branching": (0.0, 0.95), "beta": (0.3, 12.0)}


# ----------------------------------------------------------------------------- queue-reactive


@register
class QueueReactiveGenerator(MarketGenerator):
    name = "queue_reactive"
    version = "1"
    capabilities = frozenset({"count_process", "queue_reactive", "state_dependent", "closed_form_loglik"})

    @classmethod
    def default_params(cls) -> dict:
        return {"base": [0.5, 0.5, 2.2, 2.2, 1.8, 1.8], "gamma": [-0.4, -0.4, -0.3, -0.3, 0.5, 0.5],
                "spread_mult": [0.6, 0.6, 1.8, 1.8, 0.8, 0.8], "qref": 12.0}

    def __init__(self, params=None, book=None) -> None:
        super().__init__(params, book)
        self._refresh()

    def _refresh(self) -> None:
        p = self.params
        self._logbase = np.log(np.maximum(np.asarray(p["base"], float), 1e-9))
        self._gamma = np.asarray(p["gamma"], float)
        self._logsm = np.log(np.maximum(np.asarray(p["spread_mult"], float), 1e-9))
        self._qref = float(p["qref"])

    def expected(self, state, x) -> np.ndarray:
        q = np.asarray([x[3], x[4]])[list(TOUCH_OF_MARK)]
        wide = 1.0 if x[0] >= 1.5 else 0.0
        return np.exp(np.clip(self._logbase + self._gamma * np.log(np.maximum(q, 1.0) / self._qref)
                              + self._logsm * wide, -12, 4))

    def loglik(self, frame: EventFrame) -> np.ndarray:
        out = np.full(len(frame), np.nan)
        v = frame.valid
        s = frame.state[v]
        q = np.maximum(s[:, [3, 4]][:, list(TOUCH_OF_MARK)], 1.0)
        wide = (s[:, 0] >= 1.5).astype(float)[:, None]
        mu = np.exp(np.clip(self._logbase + self._gamma * np.log(q / self._qref) + self._logsm * wide, -12, 4))
        out[v] = poisson_logpmf(frame.counts[v], mu).sum(1)
        return out

    def fit(self, frame: EventFrame) -> QueueReactiveGenerator:
        v = frame.valid
        s = frame.state[v]
        q = np.maximum(s[:, [3, 4]], 1.0)
        qref = float(np.median(q))
        wide = (s[:, 0] >= 1.5).astype(float)
        base, gamma, mult, info = [], [], [], {}
        for m in range(K):
            x = np.column_stack([np.ones(len(s)), np.log(q[:, TOUCH_OF_MARK[m]] / qref), wide])
            if wide.min() == wide.max():
                x = x[:, :2]
            coef, diag = poisson_irls(x, frame.counts[v, m].astype(float))
            base.append(float(np.exp(coef[0])))
            gamma.append(float(np.clip(coef[1], -3.0, 3.0)))
            mult.append(float(np.exp(np.clip(coef[2], -4, 4))) if len(coef) > 2 else 1.0)
            info[str(m)] = diag
        out = QueueReactiveGenerator({"base": base, "gamma": gamma, "spread_mult": mult, "qref": qref}, self.book)
        out.fit_info = {"method": "Poisson GLM (IRLS) per mark on log touch queue and wide-spread indicator",
                        "per_mark": info, "bins": int(v.sum())}
        return out

    def parameter_space(self) -> dict:
        return {f"gamma.{m}": (-2.0, 2.0) for m in range(K)}


# ----------------------------------------------------------------------------- regime switching


@register
class RegimeSwitchingGenerator(MarketGenerator):
    name = "regime_switching"
    version = "1"
    capabilities = frozenset({"count_process", "latent_regimes", "filtered_loglik"})

    @classmethod
    def default_params(cls) -> dict:
        return {"rates": [[0.25, 0.25, 1.6, 1.6, 1.5, 1.5], [1.2, 1.2, 5.0, 5.0, 4.8, 4.8]],
                "trans": [[0.995, 0.005], [0.02, 0.98]], "init": [0.8, 0.2]}

    def __init__(self, params=None, book=None) -> None:
        super().__init__(params, book)
        self._refresh()

    def _refresh(self) -> None:
        p = self.params
        self._rates = np.asarray(p["rates"], float)
        self._trans = np.asarray(p["trans"], float)
        self._trans = self._trans / self._trans.sum(1, keepdims=True)
        self._init = np.asarray(p["init"], float) / np.sum(p["init"])

    @property
    def n_regimes(self) -> int:
        return len(self._rates)

    def initial_state(self, simulate: bool = False):
        return {"regime": None, "probs": self._init.copy(), "simulate": simulate}

    def expected(self, state, x) -> np.ndarray:
        if state["simulate"] and state["regime"] is not None:
            return self._rates[state["regime"]]
        return state["probs"] @ self._rates

    def draw(self, state, x, rng) -> np.ndarray:
        """Simulation: the first bin draws the initial regime, later bins first make one Markov transition."""
        if state["regime"] is None:
            state["regime"] = int(rng.choice(self.n_regimes, p=self._init))
        else:
            state["regime"] = int(rng.choice(self.n_regimes, p=self._trans[state["regime"]]))
        return rng.poisson(self._rates[state["regime"]])

    def update(self, state, realized, x, own=None):
        if state["simulate"]:
            return state
        like = np.exp(poisson_logpmf(realized[None, :], self._rates).sum(1))
        post = state["probs"] * like
        post = post / post.sum() if post.sum() > 0 else self._init.copy()
        state["probs"] = post @ self._trans
        return state

    def loglik(self, frame: EventFrame) -> np.ndarray:
        probs = self._init.copy()
        out = np.full(len(frame), np.nan)
        logem = poisson_logpmf(frame.counts[:, None, :], self._rates[None, :, :]).sum(2)   # (n, R)
        for i in range(len(frame)):
            if frame.valid[i]:
                top = logem[i].max()
                w = probs * np.exp(logem[i] - top)
                out[i] = float(math.log(max(w.sum(), 1e-300)) + top)
                probs = (w / w.sum()) @ self._trans if w.sum() > 0 else self._init.copy()
            else:
                probs = probs @ self._trans
        return out

    def filtered(self, frame: EventFrame) -> np.ndarray:
        """Filtered regime probabilities at each bin start (used by regime-aware calibration and selection)."""
        probs = self._init.copy()
        out = np.zeros((len(frame), self.n_regimes))
        logem = poisson_logpmf(frame.counts[:, None, :], self._rates[None, :, :]).sum(2)
        for i in range(len(frame)):
            out[i] = probs
            if frame.valid[i]:
                w = probs * np.exp(logem[i] - logem[i].max())
                probs = (w / w.sum()) @ self._trans if w.sum() > 0 else self._init.copy()
        return out

    def fit(self, frame: EventFrame, *, n_regimes: int | None = None, iterations: int = 60) -> RegimeSwitchingGenerator:
        y = _valid_counts(frame)
        r = n_regimes or self.n_regimes
        activity = y.sum(1)
        edges = np.quantile(activity, np.linspace(0, 1, r + 1)[1:-1])
        labels = np.searchsorted(edges, activity, side="right")
        rates = np.asarray([y[labels == k].mean(0) if np.any(labels == k) else y.mean(0) for k in range(r)]) + 1e-3
        trans = np.full((r, r), 0.02 / max(r - 1, 1))
        np.fill_diagonal(trans, 0.98)
        init = np.full(r, 1.0 / r)
        previous = -np.inf
        ll = previous
        for it in range(iterations):
            loge = poisson_logpmf(y[:, None, :], rates[None, :, :]).sum(2)
            top = loge.max(1, keepdims=True)
            em = np.exp(loge - top)
            n = len(y)
            alpha = np.zeros((n, r))
            scale = np.zeros(n)
            a = init * em[0]
            scale[0] = a.sum()
            alpha[0] = a / scale[0]
            for t in range(1, n):
                a = (alpha[t - 1] @ trans) * em[t]
                scale[t] = a.sum() if a.sum() > 0 else 1e-300
                alpha[t] = a / scale[t]
            beta_ = np.ones((n, r))
            for t in range(n - 2, -1, -1):
                b = trans @ (em[t + 1] * beta_[t + 1])
                beta_[t] = b / scale[t + 1]
            gamma = alpha * beta_
            gamma /= gamma.sum(1, keepdims=True)
            xi = np.zeros((r, r))
            for t in range(n - 1):
                x = alpha[t][:, None] * trans * (em[t + 1] * beta_[t + 1])[None, :] / scale[t + 1]
                xi += x / max(x.sum(), 1e-300)
            init = gamma[0] + 1e-6
            init /= init.sum()
            trans = xi + 1e-6
            trans /= trans.sum(1, keepdims=True)
            rates = (gamma.T @ y) / gamma.sum(0)[:, None] + 1e-4
            ll = float(np.sum(np.log(np.maximum(scale, 1e-300))) + top.sum())
            if abs(ll - previous) < 1e-6 * max(1.0, abs(ll)):
                break
            previous = ll
        order = np.argsort(rates.sum(1))
        rates, trans, init = rates[order], trans[np.ix_(order, order)], init[order]
        out = RegimeSwitchingGenerator({"rates": rates.tolist(), "trans": trans.tolist(), "init": init.tolist()},
                                       self.book)
        out.fit_info = {"method": "Baum-Welch (scaled) with Poisson emissions", "iterations": it + 1,
                        "loglik": ll, "regimes": r, "bins": int(len(y))}
        return out

    def parameter_space(self) -> dict:
        return {"trans.0.1": (0.0005, 0.1), "trans.1.0": (0.001, 0.2)}

    def with_params(self, theta: dict) -> RegimeSwitchingGenerator:
        new = super().with_params(theta)
        trans = np.asarray(new.params["trans"], float)
        for i in range(len(trans)):
            off = trans[i].sum() - trans[i, i]
            trans[i, i] = max(1.0 - off, 1e-6)
        new.params["trans"] = trans.tolist()
        new._refresh()
        return new


# ----------------------------------------------------------------------------- state dependent

STATE_COLUMNS = (0, 1, 2, 5)   # spread, imbalance, log L1 depth, |10-bin return|


@register
class StateDependentGenerator(MarketGenerator):
    name = "state_dependent"
    version = "1"
    capabilities = frozenset({"count_process", "state_dependent", "closed_form_loglik"})

    @classmethod
    def default_params(cls) -> dict:
        coef = np.zeros((K, 1 + len(STATE_COLUMNS)))
        coef[:, 0] = np.log([0.5, 0.5, 2.0, 2.0, 1.9, 1.9])
        coef[0, 2], coef[1, 2] = 0.3, -0.3            # imbalance tilts market flow
        coef[2:4, 1] = 0.4                            # wide spreads attract additions
        coef[4:6, 3] = 0.3                            # deep books see more cancellations
        return {"coef": coef.tolist(), "mean": [1.5, 0.0, 3.0, 0.5], "sd": [0.8, 0.4, 0.8, 0.7]}

    def __init__(self, params=None, book=None) -> None:
        super().__init__(params, book)
        self._refresh()

    def _refresh(self) -> None:
        p = self.params
        self._coef = np.asarray(p["coef"], float)
        self._mean = np.asarray(p["mean"], float)
        self._sd = np.maximum(np.asarray(p["sd"], float), 1e-6)

    def _z(self, x: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(x)[..., list(STATE_COLUMNS)] - self._mean) / self._sd, -4.0, 4.0)

    def expected(self, state, x) -> np.ndarray:
        return np.exp(np.clip(self._coef @ np.r_[1.0, self._z(x)], -12, 4))

    def loglik(self, frame: EventFrame) -> np.ndarray:
        out = np.full(len(frame), np.nan)
        v = frame.valid
        z = np.column_stack([np.ones(v.sum()), self._z(frame.state[v])])
        out[v] = poisson_logpmf(frame.counts[v], np.exp(np.clip(z @ self._coef.T, -12, 4))).sum(1)
        return out

    def fit(self, frame: EventFrame) -> StateDependentGenerator:
        v = frame.valid
        s = frame.state[v][:, list(STATE_COLUMNS)]
        mean, sd = s.mean(0), s.std(0) + 1e-6
        z = np.column_stack([np.ones(len(s)), np.clip((s - mean) / sd, -4, 4)])
        coef, info = [], {}
        for m in range(K):
            beta, diag = poisson_irls(z, frame.counts[v, m].astype(float), ridge=1e-3)
            coef.append(beta.tolist())
            info[str(m)] = diag
        out = StateDependentGenerator({"coef": coef, "mean": mean.tolist(), "sd": sd.tolist()}, self.book)
        out.fit_info = {"method": "ridge Poisson GLM (IRLS) per mark", "per_mark": info, "bins": int(v.sum())}
        return out

    def parameter_space(self) -> dict:
        return {f"coef.{m}.0": (-4.0, 3.0) for m in range(K)}


# ----------------------------------------------------------------------------- mixture


@register
class MixtureGenerator(MarketGenerator):
    name = "mixture"
    version = "1"
    capabilities = frozenset({"count_process", "mixture", "filtered_loglik"})

    @classmethod
    def default_params(cls) -> dict:
        return {"weights": [0.5, 0.5], "switch_prob": 0.0,
                "components": [PoissonGenerator().to_dict(), HawkesGenerator().to_dict()]}

    def __init__(self, params=None, book=None) -> None:
        super().__init__(params, book)
        self._refresh()

    def _refresh(self) -> None:
        self._components = [MarketGenerator.from_dict(c) for c in self.params["components"]]
        w = np.asarray(self.params["weights"], float)
        self._w = w / w.sum()
        self._switch = float(self.params["switch_prob"])

    @classmethod
    def of(cls, components: list[MarketGenerator], weights=None, *, switch_prob: float = 0.0, book=None):
        weights = list(weights) if weights is not None else [1.0] * len(components)
        return cls({"weights": weights, "switch_prob": switch_prob, "components": [c.to_dict() for c in components]},
                   book or components[0].book)

    def initial_state(self, simulate: bool = False):
        return {"k": None, "w": self._w.copy(), "states": [c.initial_state(simulate) for c in self._components],
                "simulate": simulate}

    def expected(self, state, x) -> np.ndarray:
        if state["simulate"] and state["k"] is not None:
            return self._components[state["k"]].expected(state["states"][state["k"]], x)
        return sum(w * c.expected(s, x) for w, c, s in zip(state["w"], self._components, state["states"]))

    def draw(self, state, x, rng) -> np.ndarray:
        if state["k"] is None or (self._switch > 0 and rng.random() < self._switch):
            state["k"] = int(rng.choice(len(self._components), p=self._w))
        k = state["k"]
        return self._components[k].draw(state["states"][k], x, rng)

    def update(self, state, realized, x, own=None):
        if not state["simulate"]:
            like = np.asarray([float(np.exp(np.sum(poisson_logpmf(realized, c.expected(s, x)))))
                               for c, s in zip(self._components, state["states"])])
            w = state["w"] * like
            w = w / w.sum() if w.sum() > 0 else self._w.copy()
            state["w"] = (1 - self._switch) * w + self._switch * self._w
        state["states"] = [c.update(s, realized, x, own) for c, s in zip(self._components, state["states"])]
        return state

    def component_logliks(self, frame: EventFrame) -> np.ndarray:
        return np.column_stack([c.loglik(frame) for c in self._components])

    def loglik(self, frame: EventFrame) -> np.ndarray:
        """Sequential Bayesian (fixed-share) mixture predictive; components filter on their own state."""
        comp = self.component_logliks(frame)
        return fixed_share(comp, self._w, self._switch)[0]

    def fit(self, frame: EventFrame) -> MixtureGenerator:
        """Refit each component, then weights by in-sample posterior model probability (equal priors)."""
        components = [c.fit(frame) for c in self._components]
        ll = np.asarray([np.nansum(c.loglik(frame)) for c in components])
        w = np.exp(ll - ll.max())
        out = MixtureGenerator.of(components, (w / w.sum()).tolist(), switch_prob=self._switch, book=self.book)
        out.fit_info = {"method": "component MLE + posterior model probabilities", "component_loglik": ll.tolist()}
        return out

    def parameter_space(self) -> dict:
        return {"switch_prob": (0.0, 0.05)}


def fixed_share(component_ll: np.ndarray, prior: np.ndarray, switch: float) -> tuple[np.ndarray, np.ndarray]:
    """Mixture predictive log-likelihood per bin and the weight path (n, k) before each bin."""
    w = np.asarray(prior, float).copy()
    n = len(component_ll)
    out = np.full(n, np.nan)
    path = np.zeros((n, len(w)))
    for i in range(n):
        path[i] = w
        row = component_ll[i]
        if not np.all(np.isfinite(row)):
            continue
        top = row.max()
        like = w * np.exp(row - top)
        total = like.sum()
        out[i] = float(math.log(total) + top)
        w = (1 - switch) * like / total + switch * np.asarray(prior)
    return out, path


def copy_generator(g: MarketGenerator) -> MarketGenerator:
    return deepcopy(g)
