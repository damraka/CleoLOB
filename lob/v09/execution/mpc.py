"""MPC v2 (M80): fee-aware receding-horizon execution with stochastic, CVaR, min-max, DRO, chance and adaptive
objectives.

At every decision the controller re-plans over the remaining horizon from the *observed* state only: remaining
quantity, observed book shape (contiguous depth), online volatility and passive-fill-rate estimates and the time
left. A plan is (market child ``qm``, passive child ``qp``) for the current step with the remainder spread evenly
over the remaining steps. Its cost in ticks per mandate under one scenario ``(sigma x s, depth x d, fill x f)`` is

    immediate walk of qm on the depth scaled by d + taker fee on qm
  + passive: fill f*qp at the own touch (gains half the spread, pays the maker fee)
  + continuation: the expected remainder as even market children (walk + taker fee)
  + risk: lambda sigma^2 R^2 T_left / 3
  + completion: penalty on expected unfilled quantity at the deadline

Fees are converted to ticks with the observed tick size in bps (v0.8 MPC ignored fees; on the registered
instruments the maker/taker fee gap is ~100x the half spread, so v0.9 must not).

Objectives over ``K`` registered seeded scenarios: ``stochastic`` (mean), ``cvar`` (CVaR at 0.9), ``minmax``
(worst), ``dro`` (KL-ball worst-case expectation via the dual ``min_l l log E exp(c/l) + l rho``), ``chance``
(mean subject to P(scenario completion) >= 0.95, else the most completing plan) and ``adaptive`` (stochastic with
volatility / fill-rate estimates under exponential forgetting).
"""
from __future__ import annotations

import math

import numpy as np

from ...v08.execution.policies import realized_sigma
from ...v08.execution.simulator import Observation, cancel_all, market
from .policies import _Base, register, rest

OBJECTIVES = ("stochastic", "cvar", "minmax", "dro", "chance", "adaptive")


class MPC2(_Base):
    name, label, family = "mpc_stochastic", "MPC v2 (expected cost)", "mpc_v2"
    objective = "stochastic"
    defaults = {"risk_aversion": 1e-4, "completion_weight": 4.0, "scenarios": 32, "taker_bps": 5.5,
                "maker_bps": 2.0, "cvar_level": 0.9, "dro_radius": 0.1, "chance_level": 0.95, "forgetting": 0.8,
                "participation_limit": None}

    def reset(self, ctx):
        super().reset(ctx)
        self.fill_obs: list[float] = []
        self.last_posted = 0
        self.last_filled = 0
        self.sigma_est: float | None = None
        seed = int(getattr(ctx, "context", {}).get("seed", 0)) if ctx is not None else 0
        self.rng = np.random.default_rng(np.random.SeedSequence([9, 80, seed]))

    def _fill_rate(self, obs: Observation) -> float:
        if self.last_posted > 0:
            got = max(0, obs.filled - self.last_filled - getattr(self, "_market_sent", 0))
            self.fill_obs.append(min(1.0, got / self.last_posted))
        if self.objective == "adaptive" and self.fill_obs:
            w = self.params["forgetting"] ** np.arange(len(self.fill_obs))[::-1]
            return float((0.3 * 2 + (w * np.asarray(self.fill_obs)).sum()) / (2 + w.sum()))
        return float((0.3 * 3 + sum(self.fill_obs)) / (3 + len(self.fill_obs)))

    def _sigma(self, obs: Observation) -> float:
        s = realized_sigma(obs, window=300)
        if self.objective == "adaptive":
            f = self.params["forgetting"]
            self.sigma_est = s if self.sigma_est is None else f * self.sigma_est + (1 - f) * s
            return self.sigma_est
        return s

    def scenarios(self) -> np.ndarray:
        k = int(self.params["scenarios"])
        s = np.exp(self.rng.normal(0.0, 0.5, k))
        d = np.exp(self.rng.normal(0.0, 0.5, k))
        f = self.rng.beta(2.0, 2.0, k) * 2.0
        return np.column_stack([s, d, f])

    def decide(self, obs: Observation) -> list:
        p = self.params
        fill_rate = self._fill_rate(obs)
        left = obs.remaining
        self.last_filled = obs.filled
        self.last_posted, self._market_sent = 0, 0
        if left <= 0:
            return [cancel_all()] if obs.resting else []
        steps_left = max(1, obs.steps - obs.step)
        if steps_left == 1:
            self._market_sent = left
            return ([cancel_all()] if obs.resting else []) + [market(left)]
        sigma = self._sigma(obs)
        dt = obs.horizon / obs.steps
        tick_bps = max(obs.tick_bps, 1e-9)
        fee_t, fee_m = p["taker_bps"] / tick_bps, p["maker_bps"] / tick_bps
        slice_ = max(1, int(math.ceil(left / steps_left)))
        cap = None
        if p["participation_limit"] is not None:
            cap = max(1, int(p["participation_limit"] * max(obs.volume_since_last, 1.0)))
        plans = []
        for qm in sorted({0, max(1, slice_ // 2), slice_, 2 * slice_, left}):
            for qp in (0, slice_, left):
                qm_, qp_ = min(qm, left), min(qp, max(0, left - min(qm, left)))
                if cap is not None and qm_ > cap and qm_ != left:
                    continue
                plans.append((qm_, qp_))
        plans = sorted(set(plans))
        sc = self.scenarios()
        costs, completes = self._costs(obs, plans, left, steps_left, sigma, fill_rate, dt, fee_t, fee_m, sc)
        score = self._objective(costs, completes)
        qm, qp = plans[int(np.argmin(score))]
        self.last_posted, self._market_sent = qp, qm
        out = rest(obs, qp) if qp > 0 else ([cancel_all()] if obs.resting else [])
        if qm > 0:
            out = out + [market(qm)]
        return out

    def _objective(self, costs: np.ndarray, completes: np.ndarray) -> np.ndarray:
        p = self.params
        if self.objective in ("stochastic", "adaptive"):
            return costs.mean(1)
        if self.objective == "minmax":
            return costs.max(1)
        if self.objective == "cvar":
            k = max(1, int(math.ceil((1 - p["cvar_level"]) * costs.shape[1])))
            return np.sort(costs, axis=1)[:, -k:].mean(1)
        if self.objective == "dro":
            out = []
            for row in costs:
                c = row - row.max()
                best = math.inf
                for lam in np.geomspace(1e-3, 1e4, 60) * (abs(row).mean() + 1e-9):
                    v = lam * math.log(np.mean(np.exp(c / lam))) + row.max() + lam * p["dro_radius"]
                    best = min(best, v)
                out.append(best)
            return np.asarray(out)
        if self.objective == "chance":
            prob = completes.mean(1)
            ok = prob >= p["chance_level"]
            mean = costs.mean(1)
            return np.where(ok, mean, 1e18 - prob * 1e17)
        raise ValueError(self.objective)

    @staticmethod
    def _walk_curve(obs: Observation) -> tuple[np.ndarray, np.ndarray, float]:
        """Cumulative (quantity, cost-over-mid in ticks) breakpoints of the observed book; slope beyond the book."""
        depth = np.maximum(np.asarray(obs.ask_depth if obs.side > 0 else obs.bid_depth, float), 1.0)
        base = obs.ask if obs.side > 0 else obs.bid
        dist = np.abs(base + obs.side * np.arange(len(depth)) - obs.mid)
        q = np.r_[0.0, np.cumsum(depth)]
        k = np.r_[0.0, np.cumsum(depth * dist)]
        beyond = float(dist[-1] + 1.0) if len(dist) else 1.0
        return q, k, beyond

    @staticmethod
    def _walk_cost(curve, qty: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Walk cost of ``qty`` on the depth scaled by ``d``: d * F(qty / d) (exact for piecewise-linear F)."""
        q, k, beyond = curve
        y = np.asarray(qty, float) / d
        inside = np.interp(np.minimum(y, q[-1]), q, k)
        return d * (inside + np.maximum(y - q[-1], 0.0) * beyond)

    def _costs(self, obs, plans, left, steps_left, sigma, fill_rate, dt, fee_t, fee_m, sc):
        p = self.params
        curve = self._walk_curve(obs)
        s, d, f = sc[:, 0], sc[:, 1], np.minimum(1.0, fill_rate * sc[:, 2])
        half = 0.5 * obs.spread
        rest_steps = steps_left - 1
        t_left = rest_steps * dt
        rows, comp = [], []
        for qm, qp in plans:
            immediate = self._walk_cost(curve, np.full_like(d, qm), d) + fee_t * qm
            passive = f * qp * (-half + fee_m)
            expected_left = np.maximum(0.0, left - qm - f * qp)
            per = expected_left / rest_steps
            cont = rest_steps * (self._walk_cost(curve, per, d) + fee_t * per)
            risk = p["risk_aversion"] * (sigma * s) ** 2 * expected_left ** 2 * t_left / 3.0
            unfilled = (1.0 - f) * qp if steps_left <= 2 else np.zeros_like(f)
            rows.append(immediate + passive + cont + risk + p["completion_weight"] * (half + fee_t) * unfilled)
            comp.append((expected_left <= 0) | (steps_left > 2))
        return np.asarray(rows), np.asarray(comp)


@register
class MPCStochastic(MPC2):
    name, label, objective = "mpc_stochastic", "MPC v2 (expected cost)", "stochastic"


@register
class MPCCVaR(MPC2):
    name, label, objective = "mpc_cvar", "MPC v2 (CVaR 0.9)", "cvar"


@register
class MPCMinMax(MPC2):
    name, label, objective = "mpc_minmax", "MPC v2 (min-max)", "minmax"


@register
class MPCDRO(MPC2):
    name, label, objective = "mpc_dro", "MPC v2 (KL-DRO)", "dro"


@register
class MPCChance(MPC2):
    name, label, objective = "mpc_chance", "MPC v2 (chance-constrained completion)", "chance"


@register
class MPCAdaptive(MPC2):
    name, label, objective = "mpc_adaptive", "MPC v2 (adaptive estimates)", "adaptive"


@register
class StaticOptimal(_Base):
    """Almgren-Chriss schedule with a registered urgency, executed with market children (static optimizer)."""
    name, label = "static_opt", "Static optimizer (AC schedule)"
    defaults = {"kappa_t": 1.0}

    def decide(self, obs: Observation) -> list:
        from ...v08.execution.policies import ac_fraction
        return [market(self.to_schedule(obs, ac_fraction(self.tau_next(obs), self.params["kappa_t"])))]
