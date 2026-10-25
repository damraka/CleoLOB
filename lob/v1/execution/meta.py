"""Meta-policy, mixture of experts and contextual fallback (M118); adversarial stress and failure envelopes (M119).

``MetaPolicy`` wraps the v0.9 contextual cost models (ridge, boosted trees, k-NN; ``lob.v09.execution.selector``):
fitted on (window context, realized cost per policy) of training windows, it selects per window the policy with the
lowest ensemble predicted cost (mixture of experts), measures disagreement, and applies the contextual fallback
ladder (low uncertainty -> selected policy; moderate -> ``liquidity_pov``; high -> ``twap``; completion danger ->
an aggressive completion-safe controller). Thresholds are quantiles of the training disagreement, fixed at fit.

``stress`` evaluates a policy over a registered adversarial grid in the F6 ecology - liquidity withdrawal (market
makers removed), toxic flow (momentum population scaled), latency spikes and thin books - and returns the failure
envelope: worst cost, worst completion, and for a policy pair whether the ranking flips anywhere in the grid.
"""
from __future__ import annotations

import numpy as np

from ...v09.execution import selector as SEL
from ..ecology import agents as AG

LADDER = {"low": "selected", "moderate": "liquidity_pov", "high": "twap", "completion_danger": "sweep"}


class MetaPolicy:
    def __init__(self, policies: list[str], *, moderate_q: float = 0.5, high_q: float = 0.85) -> None:
        self.policies, self.moderate_q, self.high_q = list(policies), moderate_q, high_q

    def fit(self, X: np.ndarray, costs: dict[str, np.ndarray]) -> MetaPolicy:
        self.models = SEL.CostModels().fit(np.asarray(X), {p: np.asarray(costs[p]) for p in self.policies})
        _, dis = SEL.select(self.models.predict(np.asarray(X)), self.policies)
        self.t_mod, self.t_high = float(np.quantile(dis, self.moderate_q)), float(np.quantile(dis, self.high_q))
        self.best_global = min(self.policies, key=lambda p: float(np.mean(costs[p])))
        return self

    def choose(self, X: np.ndarray, *, completion_risk: np.ndarray | None = None) -> dict:
        choice, dis = SEL.select(self.models.predict(np.asarray(X)), self.policies)
        out, level = [], []
        for i, (c, d) in enumerate(zip(choice, dis)):
            if completion_risk is not None and completion_risk[i]:
                out.append(LADDER["completion_danger"])
                level.append("completion_danger")
            elif d <= self.t_mod:
                out.append(c)
                level.append("low")
            elif d <= self.t_high:
                out.append(LADDER["moderate"])
                level.append("moderate")
            else:
                out.append(LADDER["high"])
                level.append("high")
        return {"mixture_choice": list(choice), "fallback_choice": out, "uncertainty_level": level,
                "disagreement": [float(x) for x in dis]}


STRESS_GRID = {
    "baseline": {},
    "liquidity_withdrawal": {"drop": ("mm1", "mm2")},
    "toxic_flow": {"scale": {"mom": 4.0}},
    "thin_book": {"lp_thin": 1, "scale": {"mm1": 0.5, "mm2": 0.5}},
    "latency_spike": {"entry_ns": 50_000_000},
}


def _population(spec: dict) -> list:
    pop = []
    for a in AG.default_ecology():
        if a.name in spec.get("drop", ()):
            continue
        a.rate *= spec.get("scale", {}).get(a.name, 1.0)
        if a.name == "lp" and "lp_thin" in spec:
            a.params = {**a.params, "thin": spec["lp_thin"]}
        pop.append(a)
    return pop


def stress(policy_factory, *, qty: int = 120, horizon_s: float = 120.0, seeds=(0, 1), grid=None,
           compare_factory=None) -> dict:
    from .episode import f6_episode
    grid = grid or STRESS_GRID
    rows = {}
    for name, spec in grid.items():
        vals, comps, diffs = [], [], []
        for s in seeds:
            r = f6_episode(policy_factory(), qty=qty, horizon_s=horizon_s, seed=s, agents=_population(spec))
            if r["status"] != "OK":
                continue
            vals.append(r["is_net_bps"])
            comps.append(r["completion"])
            if compare_factory is not None:
                r2 = f6_episode(compare_factory(), qty=qty, horizon_s=horizon_s, seed=s, agents=_population(spec))
                if r2["status"] == "OK":
                    diffs.append(r2["is_net_bps"] - r["is_net_bps"])
        rows[name] = {"mean_is_bps": float(np.mean(vals)) if vals else None,
                      "worst_is_bps": float(np.max(vals)) if vals else None,
                      "min_completion": float(np.min(comps)) if comps else None,
                      "advantage_vs_compare": float(np.mean(diffs)) if diffs else None}
    ok = [r for r in rows.values() if r["worst_is_bps"] is not None]
    signs = {np.sign(r["advantage_vs_compare"]) for r in rows.values() if r["advantage_vs_compare"]}
    return {"grid": rows, "envelope": {
        "worst_cost_bps": max(r["worst_is_bps"] for r in ok) if ok else None,
        "worst_scenario": max(rows, key=lambda k: rows[k]["worst_is_bps"] or -1e9) if ok else None,
        "min_completion": min(r["min_completion"] for r in ok) if ok else None,
        "ranking_flips": bool(len(signs) > 1)},
        "label": "F6 simulated stress of assumed participant behaviour; a failure envelope, not a forecast"}


__all__ = ["MetaPolicy", "LADDER", "STRESS_GRID", "stress"]
