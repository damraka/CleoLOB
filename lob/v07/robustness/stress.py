"""Registered synthetic stresses and structural interventions (workstream 40). Stress is not validation.

Each stress transforms a v0.6-family world (``SimulatorSpec``) into a labelled
SYNTHETIC_STRESS world; results describe the simulator only and never enter the
plausible set or any robustness claim. Mandate-level stresses (fee change,
latency) modify the mandate or latency model instead of the market.
"""
from __future__ import annotations

from ...engine import SimConfig
from ...sim_v2 import SimulatorSpec
from ..uncertainty.worlds import WorldManifest

STRESSES = {
    "liquidity_drought": {"config": {"limit_rate": 0.3, "target_level_vol": 0.3}},
    "spread_explosion": {"config": {"offset_p": 0.2}, "extensions": {"inside_spread_prob": 0.0}},
    "volatility_spike": {"config": {"market_rate": 5.0}},
    "event_rate_surge": {"config": {"limit_rate": 3.0, "market_rate": 3.0, "cancel_rate": 3.0}},
    "depth_collapse": {"config": {"target_level_vol": 0.1}},
    "cancellation_surge": {"config": {"cancel_rate": 5.0}},
    "asymmetric_book_shock": {"set_extensions": {"imbalance_beta": 0.9}},
    "regime_transition": {"set_extensions": {"regime_multiplier": 8.0, "regime_switch_rate": 0.05,
                                             "regime_high_share": 0.5}},
    "latency_increase": {"config": {"latency_base": 10.0, "latency_jitter": 10.0}},
    "tick_size_perturbation": {"tick_multiplier": 2},
    "structural_misspecification": {"set_extensions": {"hawkes_alpha": 0.0}, "config": {"resilience": 0.0}},
}
MANDATE_STRESSES = {"fee_change": {"fees": {"maker_bps": 0.0, "taker_bps": 5.0}}}


def apply(spec: SimulatorSpec, name: str) -> SimulatorSpec:
    rule = STRESSES[name]
    config, extensions = dict(spec.config), dict(spec.extensions)
    for key, factor in rule.get("config", {}).items():
        value = config.get(key, getattr(SimConfig(), key)) * factor
        config[key] = int(max(1, round(value))) if key == "target_level_vol" else value
    for key, factor in rule.get("extensions", {}).items():
        extensions[key] = factor
    for key, value in rule.get("set_extensions", {}).items():
        extensions[key] = value
    if "tick_multiplier" in rule:
        m = rule["tick_multiplier"]
        config["tick_size"] = config["tick_size"] * m
        config["initial_mid_ticks"] = max(21, int(round(config["initial_mid_ticks"] / m)))
    if extensions.get("hawkes_alpha", 0.0) >= extensions.get("hawkes_decay", 1.0):
        extensions["hawkes_alpha"] = 0.9 * extensions.get("hawkes_decay", 1.0)
    return SimulatorSpec(config, extensions)


def manifest(name: str, base_id: str) -> WorldManifest:
    return WorldManifest("G0_point", base_id, "SYNTHETIC_STRESS", structural_assumption=name)


def stressed_mandate(mandate: dict, name: str) -> dict:
    return {**mandate, **MANDATE_STRESSES[name]}
