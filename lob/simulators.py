"""Simulator factory: the v0.4 engine unless calibration-v2 flow extensions are declared."""
from __future__ import annotations

from typing import Any, Mapping

from .engine import ExchangeSimulator, SimConfig


def make_simulator(cfg: SimConfig, flow_extensions: Mapping[str, Any] | None = None,
                   historical: Mapping[str, Any] | None = None) -> ExchangeSimulator:
    """``ExchangeSimulator`` unless extensions are active (``ExtendedSimulator``) or a historical
    episode is supplied (``HistoricalSimulator`` with a declared bounded fill mode)."""
    if historical:
        if flow_extensions:
            raise ValueError("historical replay cannot combine with simulated flow extensions")
        from .historical_sim import HistoricalSimulator
        return HistoricalSimulator(cfg, historical["episode"], historical["fill_mode"])
    if not flow_extensions:
        return ExchangeSimulator(cfg)
    from .sim_v2 import ExtendedSimulator, FlowExtensions
    ext = flow_extensions if isinstance(flow_extensions, FlowExtensions) else FlowExtensions(**dict(flow_extensions))
    return ExtendedSimulator(cfg, ext) if ext.active else ExchangeSimulator(cfg)
