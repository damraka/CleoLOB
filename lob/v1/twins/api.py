"""Standardized fidelity API (M60), fidelity wrappers (M61-M67) and claim permissions (M69).

A ``Twin`` exposes one interface at every fidelity::

    twin.capabilities          # what the fidelity can and cannot say (never inherited from a higher fidelity)
    twin.episode(policy, ...)  # an execution episode (fidelities that host execution only)
    twin.observe(...)          # observed or simulated market statistics

Fidelities: F0 observational evidence (statistics of recorded data, no intervention), F1 exact-observation replay
(the recorded book and tape played back; no agent can act), F2 bounded counterfactual replay (interval over the
registered assumption grid), F3 hybrid historical twin (recorded backbone + modelled response), F4 classical
generative market (corrected book), F5 neural generative market (neural counts drive the corrected book), F6
strategic multi-agent market (order-level ecology).

``permitted(fidelity, claim, capabilities)`` is the fail-closed claim gate: e.g. ``queue_position`` needs exact
order identity (an L3 source at F0/F1), ``execution_cost`` needs F2 or higher, ``historical_execution_cost`` is
only ever bounded (F2), any synthetic (F4-F6) capability claim needs that capability ACCEPTABLE in the realism
gate, and no fidelity yields a causal impact claim without an identifying design.
"""
from __future__ import annotations

from dataclasses import dataclass, field

FIDELITIES = ("F0", "F1", "F2", "F3", "F4", "F5", "F6")
CLAIMS = ("observed_statistic", "replayed_path", "execution_cost", "historical_execution_cost", "passive_fill",
          "queue_position", "self_impact", "policy_ranking", "realism", "causal_impact")

CAPABILITIES = {
    "F0": {"execution": False, "counterfactual": False, "self_impact": False, "uncertainty": "sampling"},
    "F1": {"execution": False, "counterfactual": False, "self_impact": False, "uncertainty": "none (observed)"},
    "F2": {"execution": True, "counterfactual": "bounded", "self_impact": "assumption grid", "uncertainty": "interval"},
    "F3": {"execution": True, "counterfactual": "modelled", "self_impact": "fitted response", "uncertainty": "seeds"},
    "F4": {"execution": True, "counterfactual": "simulated", "self_impact": "simulated book", "uncertainty": "worlds"},
    "F5": {"execution": True, "counterfactual": "simulated", "self_impact": "simulated book", "uncertainty": "worlds"},
    "F6": {"execution": True, "counterfactual": "simulated", "self_impact": "endogenous agents", "uncertainty": "worlds"},
}
SYNTHETIC = frozenset({"F4", "F5", "F6"})
CLAIM_CAPABILITY = {"passive_fill": "fill_prediction", "execution_cost": "spread_dynamics",
                    "self_impact": "impact", "realism": "spread_dynamics", "policy_ranking": "spread_dynamics"}


def permitted(fidelity: str, claim: str, capabilities: dict | None = None) -> bool:
    """Fail closed: unknown fidelities/claims raise; a synthetic claim needs the gate's ACCEPTABLE decision."""
    if fidelity not in FIDELITIES or claim not in CLAIMS:
        raise KeyError(f"unknown fidelity {fidelity} or claim {claim}")
    caps = capabilities or {}
    if claim == "causal_impact":
        return bool(caps.get("identifying_design"))
    if claim == "observed_statistic":
        return fidelity in ("F0", "F1")
    if claim == "replayed_path":
        return fidelity == "F1"
    if claim == "queue_position":
        return fidelity in ("F0", "F1") and caps.get("queue_observability") == "exact_order_id"
    if claim == "historical_execution_cost":
        return fidelity == "F2"
    if not CAPABILITIES[fidelity]["execution"]:
        return False
    if fidelity in SYNTHETIC:
        gate = caps.get("gate", {})
        need = CLAIM_CAPABILITY.get(claim)
        return need is not None and gate.get(need) == "ACCEPTABLE"
    return claim in ("execution_cost", "passive_fill", "self_impact", "policy_ranking")


@dataclass
class Twin:
    fidelity: str
    name: str
    runner: object = None
    gate: dict = field(default_factory=dict)
    source_capability: dict = field(default_factory=dict)

    @property
    def capabilities(self) -> dict:
        return {**CAPABILITIES[self.fidelity], "gate": dict(self.gate), **self.source_capability}

    def permits(self, claim: str) -> bool:
        return permitted(self.fidelity, claim, self.capabilities)

    def episode(self, policy, **kw) -> dict:
        if not CAPABILITIES[self.fidelity]["execution"]:
            raise PermissionError(f"{self.fidelity} hosts no execution (observation only)")
        return self.runner(policy, **kw)


def f0(cache: dict) -> dict:
    """F0 observational evidence: statistics of the recorded day (no intervention)."""
    import numpy as np
    v = np.asarray(cache["valid"], bool)
    sp = (np.asarray(cache["ask_px"]) - np.asarray(cache["bid_px"]))[v]
    return {"fidelity": "F0", "valid_fraction": float(v.mean()), "mean_spread_ticks": float(sp.mean()),
            "trades": int(len(cache["trade_t"])), "label": "observed (no counterfactual)"}


def f1_replay(cache: dict, start_s: float, seconds: float):
    """F1 exact-observation replay: the recorded touch/depth path and prints of a window, played back unchanged."""
    import numpy as np
    a, b = int(start_s * 10), int((start_s + seconds) * 10)
    t = np.asarray(cache["trade_t"])
    lo, hi = np.searchsorted(t, [start_s, start_s + seconds])
    return {"fidelity": "F1", "bid_px": np.asarray(cache["bid_px"][a:b]), "ask_px": np.asarray(cache["ask_px"][a:b]),
            "depth": np.asarray(cache["depth"][a:b]), "valid": np.asarray(cache["valid"][a:b]),
            "trades": {k: np.asarray(cache[k][lo:hi]) for k in ("trade_t", "trade_px", "trade_q", "trade_s")}}


def f2_twin(cache: dict, *, notional_usd: float) -> Twin:
    from ...v09.replay import engine as E

    def run(policy, *, window: int, assumptions=None):
        grid = assumptions or E.grid()
        book = E.DayBook(cache)
        rows = [E.episode(book, policy, window=window, notional_usd=notional_usd, assumption=a) for a in grid]
        ok = [r for r in rows if r["status"] == "OK"]
        if not ok:
            return {"fidelity": "F2", "status": "NOT_EVALUABLE"}
        vals = [r["is_net_bps"] for r in ok]
        return {"fidelity": "F2", "status": "OK", "is_net_bps_lower": min(vals), "is_net_bps_upper": max(vals),
                "assumptions": len(ok), "label": "BOUNDED, HYPOTHETICAL, ASSUMPTION_DEPENDENT"}
    return Twin("F2", "bounded-replay", run)


def f3_twin(cache: dict, *, notional_usd: float, impact_model, persistent: float) -> Twin:
    from ...v09 import twin as T9

    def run(policy, *, window: int, seed: int = 0):
        return T9.hybrid_episode(cache, policy, window=window, notional=notional_usd, impact_model=impact_model,
                                 persistent=persistent, seed=seed)
    return Twin("F3", "hybrid-twin", run)


def f4_twin(generator, *, gate: dict | None = None, name: str = "classical") -> Twin:
    from .classical import synthetic_episode

    def run(policy, *, qty: int, seed: int = 0, side: int = 1):
        return {**synthetic_episode(generator, policy, qty=qty, seed=seed, side=side), "fidelity": "F4"}
    return Twin("F4", name, run, gate or {})


def f5_twin(neural_counts, *, gate: dict | None = None) -> Twin:
    """F5: a neural count generator (``generators.neural.counts``) driving the corrected book in the F4 simulator."""
    from .classical import synthetic_episode

    def run(policy, *, qty: int, seed: int = 0, side: int = 1):
        return {**synthetic_episode(neural_counts, policy, qty=qty, seed=seed, side=side), "fidelity": "F5"}
    return Twin("F5", getattr(neural_counts, "name", "neural"), run, gate or {})


def f6_twin(*, agents=None, gate: dict | None = None) -> Twin:
    from ..execution.episode import f6_episode

    def run(policy, *, qty: int, seed: int = 0, side: int = 1, horizon_s: float = 300.0):
        return f6_episode(policy, qty=qty, seed=seed, side=side, horizon_s=horizon_s, agents=agents)
    return Twin("F6", "ecology", run, gate or {})


__all__ = ["FIDELITIES", "CLAIMS", "CAPABILITIES", "permitted", "Twin", "f0", "f1_replay", "f2_twin", "f3_twin",
           "f4_twin", "f5_twin", "f6_twin"]
