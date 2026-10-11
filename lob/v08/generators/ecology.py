"""Stylized synthetic market ecology (M9).

An ``EcologyGenerator`` sums the event-rate contributions of stylized background
agents. The agents are modelling devices with transparent rules; they are
**not** behavioural identities of real participants and nothing in v0.8 infers
who trades in real data.

* ``market_maker`` — quotes at the touch, more when the spread is wide; withdraws
  (cancels) from the side the imbalance leans against;
* ``liquidity_taker`` — symmetric market orders;
* ``noise`` — small symmetric additions, cancellations and market orders;
* ``directional`` — market orders tilted by a persistent AR(1) bias (seed-addressable);
* ``cancellation_heavy`` — high, nearly balanced addition and cancellation rates;
* ``replenishment`` — refills a thin touch (rate grows as the touch queue falls
  below its target).

Ecology stress tests scale an agent population (``scaled``).
"""
from __future__ import annotations

import math

import numpy as np

from ..market.frame import EventFrame
from .base import K, MarketGenerator, register

AGENTS = ("market_maker", "liquidity_taker", "noise", "directional", "cancellation_heavy", "replenishment")


def _agent_rates(name: str, p: dict, x: np.ndarray, bias: float) -> np.ndarray:
    spread, imbalance, _, bq, aq, _ = x
    r = np.zeros(K)
    a = float(p.get("intensity", 0.0))
    if a <= 0:
        return r
    if name == "market_maker":
        quote = a * (1.0 + 0.8 * max(spread - 1.0, 0.0))
        r[2] = r[3] = quote
        r[4] = a * 0.9 * (1.0 + max(-imbalance, 0.0))       # bid-light book: makers pull bids
        r[5] = a * 0.9 * (1.0 + max(imbalance, 0.0))
    elif name == "liquidity_taker":
        r[0] = r[1] = 0.5 * a
    elif name == "noise":
        r[:] = a * np.asarray([0.1, 0.1, 0.45, 0.45, 0.4, 0.4])
    elif name == "directional":
        r[0] = a * 0.5 * (1.0 + bias)
        r[1] = a * 0.5 * (1.0 - bias)
    elif name == "cancellation_heavy":
        r[2] = r[3] = 0.5 * a
        r[4] = r[5] = 0.49 * a
    elif name == "replenishment":
        target = float(p.get("target", 10.0))
        r[2] = a * max(0.0, 1.0 - bq / target)
        r[3] = a * max(0.0, 1.0 - aq / target)
    return r


@register
class EcologyGenerator(MarketGenerator):
    name = "ecology"
    version = "1"
    capabilities = frozenset({"count_process", "agent_ecology", "state_dependent"})

    @classmethod
    def default_params(cls) -> dict:
        return {"agents": {"market_maker": {"intensity": 1.6}, "liquidity_taker": {"intensity": 0.7},
                           "noise": {"intensity": 1.0}, "directional": {"intensity": 0.2, "persistence": 0.995,
                                                                       "volatility": 0.05},
                           "cancellation_heavy": {"intensity": 1.2}, "replenishment": {"intensity": 0.8,
                                                                                      "target": 10.0}}}

    def __init__(self, params=None, book=None) -> None:
        super().__init__(params, book)
        unknown = set(self.params["agents"]) - set(AGENTS)
        if unknown:
            raise ValueError(f"unknown ecology agents {sorted(unknown)}")

    def initial_state(self, simulate: bool = False):
        return {"bias": 0.0, "simulate": simulate}

    def expected(self, state, x) -> np.ndarray:
        bias = state["bias"]
        return sum((_agent_rates(n, p, np.asarray(x, float), bias) for n, p in self.params["agents"].items()),
                   np.zeros(K))

    def draw(self, state, x, rng) -> np.ndarray:
        d = self.params["agents"].get("directional")
        if d and state["simulate"]:
            rho, vol = float(d.get("persistence", 0.995)), float(d.get("volatility", 0.05))
            state["bias"] = float(np.clip(rho * state["bias"] + vol * rng.standard_normal(), -0.95, 0.95))
        return rng.poisson(np.clip(self.expected(state, x), 0.0, 60.0))

    def fit(self, frame: EventFrame) -> EcologyGenerator:
        """Method of moments: one scale per agent group so mean market/add/cancel rates match the frame."""
        v = frame.valid
        target = frame.counts[v].mean(0)
        model = np.mean([self.expected({"bias": 0.0}, x) for x in frame.state[v][:: max(1, v.sum() // 2000)]], 0)
        ratio = lambda a, b: float(a / b) if b > 0 else 1.0  # noqa: E731
        market = ratio(target[:2].sum(), model[:2].sum())
        limit = ratio(target[2:].sum(), model[2:].sum())
        agents = {}
        for n, p in self.params["agents"].items():
            scale = market if n in {"liquidity_taker", "directional"} else limit
            agents[n] = {**p, "intensity": float(p["intensity"] * scale)}
        out = EcologyGenerator({"agents": agents}, self.book)
        out.fit_info = {"method": "method of moments (market and limit-flow scales)", "market_scale": market,
                        "limit_scale": limit}
        return out

    def scaled(self, agent: str, factor: float) -> EcologyGenerator:
        """Ecology stress: multiply one agent population's intensity."""
        if agent not in self.params["agents"]:
            raise KeyError(agent)
        agents = {n: dict(p) for n, p in self.params["agents"].items()}
        agents[agent]["intensity"] = float(agents[agent]["intensity"] * factor)
        return EcologyGenerator({"agents": agents}, self.book)

    def parameter_space(self) -> dict:
        return {f"agents.{a}.intensity": (0.0, 5.0) for a in self.params["agents"]}

    def with_params(self, theta: dict) -> EcologyGenerator:
        return super().with_params(theta)


def describe_agents() -> dict:
    return {n: "stylized synthetic agent (not a real behavioural identity)" for n in AGENTS} | {
        "_note": "rates combine additively; see lob.v08.generators.ecology"}


__all__ = ["EcologyGenerator", "AGENTS", "describe_agents", "math"]
