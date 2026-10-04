"""G2: latent two-state regime switching between two v0.6-family parameter sets.

The two parameter sets are the v0.6 volatility-regime-conditioned calibrations
(``results/v06/m12/regime``: one fitted on development high-volatility blocks,
one on low-volatility blocks; their within-model activity regime is switched
off). A continuous-time Markov chain switches between them; its exit rates are
the maximum-likelihood rates of the v0.6 5-minute volatility labels on the
development day. While a state is active every engine read of ``cfg``/``ext``
returns that state's parameters (data-descriptor properties), so clocks adapt at
their next draw. Hawkes thinning bounds drawn just before a switch can be
exceeded briefly; acceptance is then capped at one (a small, documented bias).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ...engine import SimConfig
from ...sim_v2 import ExtendedSimulator, FlowExtensions

HORIZON_S = 200_000.0


class RegimeSwitchingSimulator(ExtendedSimulator):
    def __init__(self, configs: dict[str, SimConfig], extensions: dict[str, FlowExtensions], *,
                 exit_rates: dict[str, float], seed: int) -> None:
        self._configs, self._extensions = configs, extensions
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7_708_002]))
        share_high = exit_rates["low"] / (exit_rates["low"] + exit_rates["high"])
        state = "high" if rng.random() < share_high else "low"
        times, states, t = [0.0], [state], 0.0
        while t < HORIZON_S and len(times) < 1_000_000:
            t += float(rng.exponential(1.0 / exit_rates[state]))
            state = "low" if state == "high" else "high"
            times.append(t)
            states.append(state)
        self._switch_times = np.asarray(times)
        self._switch_states = states
        self._time = 0.0
        super().__init__(configs[states[0]], extensions[states[0]])

    def regime(self, t: float | None = None) -> str:
        t = getattr(self, "t", 0.0) if t is None else t
        return self._switch_states[int(np.searchsorted(self._switch_times, t, side="right")) - 1]

    @property
    def cfg(self) -> SimConfig:
        return self._configs[self.regime()]

    @cfg.setter
    def cfg(self, value) -> None:   # the base constructor assigns cfg once; the regime path decides reads
        pass

    @property
    def ext(self) -> FlowExtensions:
        return self._extensions[self.regime()]

    @ext.setter
    def ext(self, value) -> None:
        pass


@dataclass(frozen=True)
class RegimeSpec:
    family: str
    configs: dict
    extensions: dict
    exit_rates: dict
    meta: dict = field(default_factory=dict)

    def build(self, seed: int) -> RegimeSwitchingSimulator:
        configs = {k: SimConfig(**{**v, "seed": seed, "record_events": False, "check_invariants": False})
                   for k, v in self.configs.items()}
        extensions = {k: FlowExtensions(**{**v, "regime_switch_rate": 0.0, "regime_multiplier": 1.0})
                      for k, v in self.extensions.items()}
        return RegimeSwitchingSimulator(configs, extensions, exit_rates=self.exit_rates, seed=seed)

    def capped(self, seconds: float, events_per_second: float) -> RegimeSpec:
        cap = int(events_per_second * (seconds + 60))
        return replace(self, configs={k: {**v, "max_events": cap} for k, v in self.configs.items()})

    def describe(self) -> dict:
        return {"family": self.family, "exit_rates_per_s": self.exit_rates, "meta": self.meta,
                "parameters": 2 * 14 + 2}


def exit_rates(labels: list[str], block_s: float) -> dict:
    """MLE exit rates of a two-state chain from consecutive block labels (time in state / switches)."""
    time_in = {"high": 0.0, "low": 0.0}
    exits = {"high": 0, "low": 0}
    for a, b in zip(labels, labels[1:]):
        time_in[a] += block_s
        exits[a] += int(a != b)
    if labels:
        time_in[labels[-1]] += block_s
    return {k: (exits[k] + 0.5) / max(time_in[k], block_s) for k in exits}   # +0.5: Jeffreys-style floor
