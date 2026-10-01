"""M9/M13: controlled structural interventions on the selected v3 simulator.

Each intervention changes one declared mechanism of the selected world and
leaves everything else fixed. These are synthetic interventions on the
simulator only; they describe how the simulator's conclusions respond and
establish nothing about causal mechanisms in real markets.
"""
from __future__ import annotations

import copy

from .worlds import World

FLOW_BOUNDS = {"regime_multiplier": (1.0, 20.0)}


def _scale_sizes(extensions: dict, key: str, factor: float) -> dict:
    out = dict(extensions)
    out[key] = [max(1.0, float(v) * factor) for v in extensions[key]]
    return out


def _config(world: World, **changes) -> tuple[dict, dict]:
    config = {**world.config, **changes}
    return config, dict(world.extensions)


def interventions(selected: World, *, offset_p_low: float = 0.05) -> list[World]:
    """The 14 preregistered interventions, in protocol order."""
    c, e = selected.config, selected.extensions
    out: list[tuple[str, str, dict, dict]] = []

    def add(name: str, note: str, config: dict, extensions: dict) -> None:
        out.append((name, note, config, extensions))

    for name, field, factor in (("cancel_x0.5", "cancel_rate", 0.5), ("cancel_x2", "cancel_rate", 2.0),
                                ("trade_intensity_x0.5", "market_rate", 0.5), ("trade_intensity_x2", "market_rate", 2.0),
                                ("limit_flow_x0.5", "limit_rate", 0.5), ("limit_flow_x2", "limit_rate", 2.0)):
        add(name, f"{field} x {factor}", {**c, field: c[field] * factor}, dict(e))
    for name, factor in (("depth_x0.5", 0.5), ("depth_x2", 2.0)):
        add(name, f"target_level_vol x {factor}", {**c, "target_level_vol": max(1, int(round(c["target_level_vol"] * factor)))},
            dict(e))
    add("resilience_weak", "resilience x 0.25", {**c, "resilience": c["resilience"] * 0.25}, dict(e))
    add("spread_distortion", f"offset_p -> {offset_p_low}", {**c, "offset_p": offset_p_low}, dict(e))
    persistence = dict(e)
    persistence.update(hawkes_alpha=0.0, regime_switch_rate=0.0, regime_multiplier=1.0)
    add("persistence_removed", "hawkes excitation and regime switching removed", dict(c), persistence)
    add("impact_x2", "market-order size table x 2", dict(c), _scale_sizes(e, "market_size_quantiles", 2.0))
    add("impact_x0.5", "market-order size table x 0.5 (floor one lot)", dict(c),
        _scale_sizes(e, "market_size_quantiles", 0.5))
    regime = dict(e)
    regime["regime_multiplier"] = min(FLOW_BOUNDS["regime_multiplier"][1], 2 * float(e.get("regime_multiplier", 1.0)))
    add("volatility_regime_x2", "regime_multiplier x 2 (capped at 20)", dict(c), regime)
    return [World(name, "intervention", copy.deepcopy(config), copy.deepcopy(ext), note)
            for name, note, config, ext in out]
