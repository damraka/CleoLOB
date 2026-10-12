"""Controlled endogenous market response to strategy activity (M10).

``EndogenousGenerator`` wraps any base generator. The strategy's own aggressive
volume (reported by the simulator each bin) is filtered through an impact kernel
(``lob.v08.impact``) into a signed, depth-normalized signal ``I``; rates are then
modified multiplicatively (for net strategy buying, ``I > 0``; selling mirrors):

* temporary depletion: additions on the hit side ``x exp(-depletion I)``;
* transient impact: cancellations on the hit side and additions on the other
  side ``x (1 + impact I)``;
* state-dependent resilience: replenishment of the hit side
  ``x (1 + replenish I_slow (qref / touch)^resilience_exponent)``;
* participation sensitivity: all effects ``x (1 + participation_sensitivity p)``,
  where ``p`` is the strategy's share of recent traded volume.

This is a controlled synthetic mechanism. It makes no causal claim about how
real markets respond to any trader.
"""
from __future__ import annotations

import math

import numpy as np

from ..impact import Kernel
from ..market.frame import BIN_S, EventFrame
from .base import MarketGenerator, register


@register
class EndogenousGenerator(MarketGenerator):
    name = "endogenous"
    version = "1"
    capabilities = frozenset({"count_process", "endogenous_response", "state_dependent"})

    @classmethod
    def default_params(cls) -> dict:
        from .families import PoissonGenerator
        return {"base": PoissonGenerator().to_dict(),
                "kernel": {"kind": "exponential", "params": {"a": 1.0, "tau": 5.0}},
                "slow_kernel": {"kind": "exponential", "params": {"a": 1.0, "tau": 30.0}},
                "depletion": 0.6, "impact": 0.4, "replenish": 0.5, "resilience_exponent": 0.5,
                "participation_sensitivity": 1.0, "qref": 12.0}

    def __init__(self, params=None, book=None) -> None:
        super().__init__(params, book)
        self._refresh()

    def _refresh(self) -> None:
        p = self.params
        self.base_generator = MarketGenerator.from_dict(p["base"])
        self._fast = [(w, math.exp(-r * BIN_S)) for w, r in Kernel(**p["kernel"]).exponentials()]
        self._slow = [(w, math.exp(-r * BIN_S)) for w, r in Kernel(**p["slow_kernel"]).exponentials()]

    @classmethod
    def wrap(cls, base: MarketGenerator, **overrides) -> EndogenousGenerator:
        params = {**cls.default_params(), **overrides, "base": base.to_dict()}
        return cls(params, base.book)

    def initial_state(self, simulate: bool = False):
        return {"base": self.base_generator.initial_state(simulate), "fast": np.zeros(len(self._fast)),
                "slow": np.zeros(len(self._slow)), "own_vol": 0.0, "all_vol": 0.0}

    def signal(self, state) -> tuple[float, float, float]:
        q = float(self.params["qref"])
        fast = sum(w * e for (w, _), e in zip(self._fast, state["fast"])) / q
        slow = sum(w * e for (w, _), e in zip(self._slow, state["slow"])) / q
        part = state["own_vol"] / state["all_vol"] if state["all_vol"] > 0 else 0.0
        return fast, slow, part

    def expected(self, state, x) -> np.ndarray:
        r = np.asarray(self.base_generator.expected(state["base"], x), float).copy()
        fast, slow, part = self.signal(state)
        if fast == 0 and slow == 0:
            return r
        p = self.params
        scale = 1.0 + p["participation_sensitivity"] * part
        sign = 1 if fast >= 0 else -1
        i_fast, i_slow = abs(fast) * scale, abs(slow) * scale
        hit_add, hit_cancel, other_add = (3, 5, 2) if sign > 0 else (2, 4, 3)
        touch = x[4] if sign > 0 else x[3]
        r[hit_add] *= math.exp(-p["depletion"] * i_fast)
        r[hit_cancel] *= 1.0 + p["impact"] * i_fast
        r[other_add] *= 1.0 + p["impact"] * i_fast
        resilience = (p["qref"] / max(float(touch), 1.0)) ** p["resilience_exponent"]
        r[hit_add] *= 1.0 + p["replenish"] * i_slow * resilience
        return r

    def draw(self, state, x, rng) -> np.ndarray:
        """Base draw (keeps the base model's latent dynamics), then exact Poisson thinning or superposition."""
        counts = np.asarray(self.base_generator.draw(state["base"], x, rng), dtype=np.int64)
        base_mean = np.maximum(np.asarray(self.base_generator.expected(state["base"], x), float), 1e-9)
        ratio = self.expected(state, x) / base_mean
        if np.allclose(ratio, 1.0):
            return counts
        thin = rng.binomial(counts, np.clip(ratio, 0.0, 1.0))
        extra = rng.poisson(np.clip(base_mean * np.maximum(ratio - 1.0, 0.0), 0.0, 60.0))
        return (thin + extra).astype(np.int64)

    def update(self, state, realized, x, own=None):
        signed = 0.0 if own is None else float(own[0] - own[1])
        state["fast"] = np.asarray([d * (e + signed) for (_, d), e in zip(self._fast, state["fast"])])
        state["slow"] = np.asarray([d * (e + signed) for (_, d), e in zip(self._slow, state["slow"])])
        own_total = 0.0 if own is None else float(own[0] + own[1])
        decay = math.exp(-BIN_S / 10.0)
        state["own_vol"] = decay * state["own_vol"] + own_total
        state["all_vol"] = decay * state["all_vol"] + float(realized[0] + realized[1]) * self.book.market_size_mean
        state["base"] = self.base_generator.update(state["base"], realized, x, own)
        return state

    def loglik(self, frame: EventFrame) -> np.ndarray:
        # Historical frames carry no strategy flow, so the response terms are inactive: score the base model.
        return self.base_generator.loglik(frame)

    def fit(self, frame: EventFrame) -> EndogenousGenerator:
        base = self.base_generator.fit(frame)
        out = EndogenousGenerator({**self.params, "base": base.to_dict()}, base.book)
        out.fit_info = {"method": "base generator refitted; response parameters are registered settings "
                                  "(not identifiable from data without strategy flow)"}
        return out

    def parameter_space(self) -> dict:
        return {"depletion": (0.0, 3.0), "impact": (0.0, 3.0), "replenish": (0.0, 3.0)}
