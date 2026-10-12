"""Per-mark rate scaling of any base generator (used by structured stress worlds, M28).

``ScaledGenerator`` keeps the base generator's own dynamics (latent regimes,
self-excitation, state dependence) and rescales its per-mark draws exactly:
binomial thinning where a multiplier is below one, independent Poisson
superposition where it is above one.
"""
from __future__ import annotations

import numpy as np

from ..market.frame import EventFrame
from .base import K, MarketGenerator, register


@register
class ScaledGenerator(MarketGenerator):
    name = "scaled"
    version = "1"
    capabilities = frozenset({"count_process", "state_dependent"})

    @classmethod
    def default_params(cls) -> dict:
        from .families import PoissonGenerator
        return {"base": PoissonGenerator().to_dict(), "multipliers": [1.0] * K}

    def __init__(self, params=None, book=None) -> None:
        super().__init__(params, book)
        self._refresh()

    def _refresh(self) -> None:
        self.base_generator = MarketGenerator.from_dict(self.params["base"])
        m = np.asarray(self.params["multipliers"], float)
        if m.shape != (K,) or np.any(m < 0):
            raise ValueError("multipliers must be six nonnegative numbers")
        self._m = m

    @classmethod
    def wrap(cls, base: MarketGenerator, multipliers, book=None) -> ScaledGenerator:
        return cls({"base": base.to_dict(), "multipliers": list(map(float, multipliers))}, book or base.book)

    def initial_state(self, simulate: bool = False):
        return self.base_generator.initial_state(simulate)

    def expected(self, state, x) -> np.ndarray:
        return np.asarray(self.base_generator.expected(state, x), float) * self._m

    def draw(self, state, x, rng) -> np.ndarray:
        counts = np.asarray(self.base_generator.draw(state, x, rng), dtype=np.int64)
        base_mean = np.maximum(np.asarray(self.base_generator.expected(state, x), float), 1e-9)
        thin = rng.binomial(counts, np.clip(self._m, 0.0, 1.0))
        extra = rng.poisson(np.clip(base_mean * np.maximum(self._m - 1.0, 0.0), 0.0, 60.0))
        return (thin + extra).astype(np.int64)

    def update(self, state, realized, x, own=None):
        return self.base_generator.update(state, realized, x, own)

    def loglik(self, frame: EventFrame) -> np.ndarray:
        raise NotImplementedError("scaled stress worlds are simulation devices; they are not scored on data")

    def fit(self, frame: EventFrame) -> ScaledGenerator:
        raise NotImplementedError("scaled stress worlds are not fitted")
