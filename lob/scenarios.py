"""Standard benchmark scenarios.

Each scenario is a full parameter set for ``lob.runner``; ``sim`` holds overrides for
``SimConfig`` fields beyond the UI-exposed ones. Every agent in an evaluation sees the
same scenario and the same seed, so differences are attributable to the agent.
"""
from __future__ import annotations

from typing import Any, Dict

DEFAULT_PARAMS: Dict[str, Any] = {
    "qty": 10_000, "horizon": 60.0, "dt": 0.5, "seed": 42,
    "latency_ms": 10.0, "resilience": 0.8, "market_rate": 6.0,
    "risk_aversion": 1e-6, "model_path": "models/ppo_lob",
    "baseline": "ac", "pov_rate": 0.3, "sim": {},
}

SCENARIOS: Dict[str, Dict[str, Any]] = {
    # A — small order, calm market (the UI default)
    "calm": {},
    # B — large order, thin book: less background liquidity, weaker refill
    "thin": {"qty": 20_000, "market_rate": 5.0, "resilience": 0.4,
             "sim": {"limit_rate": 15.0, "target_level_vol": 150}},
    # C — high-volatility flow: bigger, more frequent market orders
    "volatile": {"market_rate": 12.0, "resilience": 0.5,
                 "sim": {"market_qty_mean": 60.0}},
    # D — high latency: cancels and orders race fills
    "high_latency": {"latency_ms": 40.0},
    # E — stressed liquidity (the UI preset)
    "stressed": {"latency_ms": 20.0, "resilience": 0.3, "market_rate": 14.0},
}

DESCRIPTIONS: Dict[str, str] = {
    "calm": "10k shares / 60 s, default liquidity — the baseline case",
    "thin": "20k shares into a thin, slowly-refilling book — impact dominates",
    "volatile": "large, frequent background market orders — price risk dominates",
    "high_latency": "40 ms mean latency — fills race cancels, queue position degrades",
    "stressed": "thin refill + heavy aggressor flow + 20 ms latency",
}


def scenario_params(name: str, seed: int, **overrides: Any) -> Dict[str, Any]:
    """Full runner parameter dict for a scenario (defaults < scenario < explicit overrides)."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario '{name}' (choose from {', '.join(SCENARIOS)})")
    p: Dict[str, Any] = {**DEFAULT_PARAMS, **SCENARIOS[name], **overrides, "seed": int(seed)}
    p["sim"] = {**DEFAULT_PARAMS["sim"], **SCENARIOS[name].get("sim", {}),
                **overrides.get("sim", {})}
    return p
