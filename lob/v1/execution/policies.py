"""v1 execution policies (M110-M118), registered in the shared policy registry so that the same policy object runs
in F2 bounded replay (``lob.v09.replay.engine``), the F4/F5 simulator (``lob.v08.execution.simulator``) and the F6
ecology (``lob.v1.execution.episode``).

Baseline suite (M110, existing implementations reused unchanged): ``twap``, ``vwap``, ``pov``, ``ac`` (Almgren-Chriss),
``sweep``, ``static_opt``; adaptive (M111/M112): ``adaptive_twap``, ``liquidity_pov``, ``imbalance_passive``,
``toxicity_passive``, ``passive_deadline``; MPC v2 (M113-M116): ``mpc_stochastic``, ``mpc_chance``, ``mpc_cvar``,
``mpc_dro``, ``mpc_minmax``, ``mpc_adaptive``.

New in v1:

* ``completion_aware`` (M111) - passive at the touch while the projected completion (filled + observed passive fill
  rate x remaining passive opportunities) covers the remainder with a registered safety margin; otherwise the
  deficit is sent as market children spread over the remaining steps. Completion feasibility is assessed before
  cost: the policy never trades off the deadline for price;
* ``imbalance_adaptive`` (M112) - aggressiveness as a function of the side-signed 5-level imbalance and the schedule
  gap: favourable imbalance -> rest passively, adverse -> catch up with market children;
* ``mpc_lexicographic`` (M117) - the MPC v2 planner with a lexicographic objective over plans: (1) maximize the
  scenario completion probability up to the chance level, (2) among those, keep plans whose CVaR90 is within
  ``tail_tolerance`` of the best CVaR, (3) minimize the mean cost.
"""
from __future__ import annotations

import math

import numpy as np

from ...v08.execution.simulator import Observation, cancel_all, market
from ...v09.execution import mpc as M9, policies as P9

BASELINES = ("twap", "vwap", "pov", "ac", "sweep", "static_opt")
ADAPTIVE = ("adaptive_twap", "liquidity_pov", "imbalance_passive", "toxicity_passive", "passive_deadline",
            "completion_aware", "imbalance_adaptive")
MPC = ("mpc_stochastic", "mpc_chance", "mpc_cvar", "mpc_dro", "mpc_minmax", "mpc_adaptive", "mpc_lexicographic")


@P9.register
class CompletionAware(P9._Base):
    name, label, family = "completion_aware", "Completion-aware adaptive", "v1"
    defaults = {"safety": 1.25, "prior_fill_rate": 0.3, "prior_weight": 3.0}

    def reset(self, ctx):
        super().reset(ctx)
        self.posted, self.filled_before, self.obs_rates = 0, 0, []

    def decide(self, obs: Observation) -> list:
        left = obs.remaining
        if left <= 0:
            return [cancel_all()] if obs.resting else []
        if self.posted > 0:
            self.obs_rates.append(min(1.0, max(0, obs.filled - self.filled_before) / self.posted))
        p = self.params
        rate = (p["prior_fill_rate"] * p["prior_weight"] + sum(self.obs_rates)) / (p["prior_weight"] + len(self.obs_rates))
        steps_left = max(1, obs.steps - obs.step)
        # projected passive completion over the remaining steps (each step fills ``rate`` of what rests)
        projected = left * (1.0 - (1.0 - rate) ** (steps_left - 1))
        deficit = left - projected / p["safety"]
        mkt = int(math.ceil(max(0.0, deficit) / steps_left)) if steps_left > 1 else left
        mkt = min(mkt, left)
        passive = left - mkt
        self.filled_before, self.posted = obs.filled, passive
        if steps_left <= 1 or passive <= 0:
            return ([cancel_all()] if obs.resting else []) + ([market(mkt)] if mkt > 0 else [])
        # resting orders keep queue priority (``rest`` re-pegs only when the touch moved)
        return ([market(mkt)] if mkt > 0 else []) + P9.rest(obs, passive)


@P9.register
class ImbalanceAdaptive(P9._Base):
    name, label, family = "imbalance_adaptive", "Imbalance-adaptive aggressiveness", "v1"
    defaults = {"favourable": 0.2, "adverse": -0.2}

    def decide(self, obs: Observation) -> list:
        left = obs.remaining
        if left <= 0:
            return [cancel_all()] if obs.resting else []
        b, a = float(sum(obs.bid_depth[:5])), float(sum(obs.ask_depth[:5]))
        imb = obs.side * (b - a) / max(a + b, 1e-9)          # > 0: the book leans in the mandate's favour
        steps_left = max(1, obs.steps - obs.step)
        gap = (obs.step / obs.steps) - obs.filled / max(obs.target, 1)
        drop = [cancel_all()] if obs.resting else []
        if steps_left <= 1:
            return drop + [market(left)]
        if imb <= self.params["adverse"] or gap > 0.15:
            return drop + [market(int(math.ceil(left / steps_left)))]
        if imb >= self.params["favourable"]:
            return P9.rest(obs, left)
        half = int(math.ceil(0.5 * left / steps_left))
        return [market(half)] + P9.rest(obs, left - half)


@P9.register
class MPCLexicographic(M9.MPC2):
    name, label, objective = "mpc_lexicographic", "MPC v3 (lexicographic completion > tail > mean)", "lexicographic"
    defaults = {**M9.MPC2.defaults, "tail_tolerance": 0.05}

    def _objective(self, costs: np.ndarray, completes: np.ndarray) -> np.ndarray:
        p = self.params
        prob = completes.mean(1)
        level = min(p["chance_level"], prob.max())
        feasible = prob >= level - 1e-12
        k = max(1, int(math.ceil((1 - p["cvar_level"]) * costs.shape[1])))
        cvar = np.sort(costs, axis=1)[:, -k:].mean(1)
        best_tail = cvar[feasible].min()
        keep = feasible & (cvar <= best_tail + p["tail_tolerance"] * max(abs(best_tail), 1e-9))
        mean = costs.mean(1)
        return np.where(keep, mean, 1e18 - prob * 1e17)


def make(name: str, **params):
    return P9.make(name, **params)


def catalogue() -> dict:
    return {"baselines": list(BASELINES), "adaptive": list(ADAPTIVE), "mpc": list(MPC),
            "registered": sorted(P9.REGISTRY)}


__all__ = ["CompletionAware", "ImbalanceAdaptive", "MPCLexicographic", "make", "catalogue", "BASELINES",
           "ADAPTIVE", "MPC"]
