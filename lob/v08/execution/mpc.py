"""Receding-horizon (model predictive) execution control (M25).

At every decision the controller re-plans over the remaining horizon from the
*observed* state: remaining inventory (known fills, in-flight and resting
orders), observed book shape, an online volatility estimate, an online passive
fill-rate estimate and the time left. It chooses the current child (market part
and passive part) from a small grid by minimizing

    immediate market cost (walk of the observed book)
  - expected passive saving (half spread plus rebate, times an estimated fill probability)
  + continuation cost (the rest spread evenly over the remaining steps at the observed book shape)
  + risk penalty  lambda sigma^2 R'^2 T_left / 3
  + completion pressure  phi E[unfilled at the deadline] (only passive plans can leave it unfilled)

subject to a participation limit (when set) and ``market + passive <= remaining``.
``robust=True`` evaluates every plan under a registered scenario set (volatility
x0.5/x1/x2, depth x0.5/x1, fill rate x0.5/x1) and minimizes the worst case
(robust MPC). Estimates are computed from observed data only; the controller
never reads the exchange-truth book.
"""
from __future__ import annotations

import math

import numpy as np

from .policies import Policy, PolicyRequirements, realized_sigma, register
from .simulator import Observation, cancel_all, limit, market

SCENARIOS = tuple((s, d, f) for s in (0.5, 1.0, 2.0) for d in (0.5, 1.0) for f in (0.5, 1.0))


@register
class MPC(Policy):
    name, label, family = "mpc", "MPC (receding horizon)", "mpc"
    defaults = {"risk_aversion": 1e-4, "completion_weight": 2.0, "participation_limit": None, "robust": False,
                "rebate_ticks": 0.0}
    requirements = PolicyRequirements(observations=frozenset({"touch", "depth", "volume", "mids", "own_fills"}),
                                      market_capabilities=frozenset({"aggregate_depth", "trade_prints"}),
                                      completion_behaviour="constrained")

    def reset(self, ctx):
        super().reset(ctx)
        self.fill_obs: list[float] = []
        self.last_posted = 0
        self.last_filled = 0

    def _fill_rate(self, obs: Observation) -> float:
        """Online fraction of posted passive quantity filled within one step (prior 0.3, shrinkage)."""
        if self.last_posted > 0:
            got = max(0, obs.filled - self.last_filled - getattr(self, "_market_sent", 0))
            self.fill_obs.append(min(1.0, got / self.last_posted))
        n = len(self.fill_obs)
        return float((0.3 * 3 + sum(self.fill_obs)) / (3 + n))

    def decide(self, obs: Observation) -> list:
        p = self.params
        fill_rate = self._fill_rate(obs)
        actions = [cancel_all()] if obs.resting else []
        left = obs.remaining - obs.context.get("in_flight", 0)
        self.last_filled = obs.filled
        self.last_posted, self._market_sent = 0, 0
        if left <= 0:
            return actions
        steps_left = max(1, obs.steps - obs.step)
        if steps_left == 1:
            self._market_sent = left
            return actions + [market(left)]
        mid = obs.mid
        sigma = realized_sigma(obs)
        dt = obs.horizon / obs.steps
        slice_ = max(1, int(math.ceil(left / steps_left)))
        cap = None
        if p["participation_limit"] is not None:
            cap = max(1, int(p["participation_limit"] * max(obs.volume_since_last, 1.0)))
        candidates = []
        for qm in sorted({0, max(1, slice_ // 2), slice_, 2 * slice_, left}):
            for qp in (0, slice_):
                qm_, qp_ = min(qm, left), min(qp, max(0, left - min(qm, left)))
                if cap is not None and qm_ + qp_ > cap and qm_ + qp_ > 0 and qm_ != left:
                    continue
                candidates.append((qm_, qp_))
        scenarios = SCENARIOS if p["robust"] else ((1.0, 1.0, 1.0),)
        best = None
        for qm, qp in candidates:
            worst = -math.inf
            for s_mult, d_mult, f_mult in scenarios:
                cost = self._plan_cost(obs, qm, qp, left, steps_left, mid, sigma * s_mult, d_mult,
                                       min(1.0, fill_rate * f_mult), dt)
                worst = max(worst, cost)
            if best is None or worst < best[0] - 1e-12:
                best = (worst, qm, qp)
        _, qm, qp = best
        self.last_posted, self._market_sent = qp, qm
        out = list(actions)
        if qp > 0:
            out.append(limit(qp))
        if qm > 0:
            out.append(market(qm))
        return out

    def _impact(self, obs: Observation, qty: int, mid: float, depth_mult: float) -> float:
        """Market-order cost over the mid (ticks), on the observed book with depth scaled by ``depth_mult``."""
        if qty <= 0:
            return 0.0
        depth = obs.ask_depth if obs.side > 0 else obs.bid_depth
        base = obs.ask if obs.side > 0 else obs.bid
        rem, cost, i = qty, 0.0, 0
        while rem > 0:
            size = max(1.0, (depth[i] if i < len(depth) else np.mean(depth)) * depth_mult)
            take = min(rem, size)
            cost += take * (abs(base + obs.side * i - mid))
            rem -= take
            i += 1
        return cost

    def _plan_cost(self, obs, qm, qp, left, steps_left, mid, sigma, depth_mult, fill_rate, dt) -> float:
        p = self.params
        immediate = self._impact(obs, qm, mid, depth_mult)
        half = 0.5 * obs.spread
        passive_saving = fill_rate * qp * (half + p["rebate_ticks"])
        expected_left = left - qm - fill_rate * qp
        rest_steps = steps_left - 1
        per = expected_left / rest_steps
        continuation = rest_steps * self._impact(obs, int(math.ceil(per)), mid, depth_mult) * (per / max(math.ceil(per), 1))
        t_left = rest_steps * dt
        risk = p["risk_aversion"] * sigma * sigma * expected_left * expected_left * t_left / 3.0
        unfilled = (1.0 - fill_rate) * qp if steps_left <= 2 else 0.0
        return immediate - passive_saving + continuation + risk + p["completion_weight"] * half * unfilled


@register
class RobustMPC(MPC):
    name, label = "mpc_robust", "Robust MPC (worst case over registered scenarios)"
    defaults = {**MPC.defaults, "robust": True}
