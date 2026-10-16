"""Causal-estimand registry, DAG identification checks, sensitivity analysis and partial identification (M75-M78).

Every causal question is registered as an ``Estimand`` (treatment, outcome, confounders, DAG, assumptions,
estimator, sensitivity analysis). ``identify`` applies the back-door criterion on the registered DAG: the effect is
IDENTIFIED only if the registered adjustment set blocks every back-door path and contains no descendant of the
treatment; with an unobserved node on a back-door path it is NOT_IDENTIFIED; effects that are identified only under
unverifiable assumptions (e.g. no unobserved information shocks) are ASSUMPTION_DEPENDENT.

``sensitivity`` reports, for a linear-regression effect estimate, the partial R^2 an unobserved confounder would need
with both treatment and outcome to explain the estimate away (the robustness value of Cinelli & Hazlett, 2020).
``manski`` gives assumption-free bounds for a bounded outcome under missingness (partial identification).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math


@dataclass(frozen=True)
class Estimand:
    name: str
    treatment: str
    outcome: str
    confounders: tuple[str, ...]
    unobserved: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    adjustment: tuple[str, ...]
    assumptions: tuple[str, ...]
    estimator: str
    sensitivity: str = "robustness value (partial R^2) of an unobserved confounder"
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _parents(edges, node):
    return {a for a, b in edges if b == node}


def _children(edges, node):
    return {b for a, b in edges if a == node}


def descendants(edges, node) -> set[str]:
    out, stack = set(), [node]
    while stack:
        for c in _children(edges, stack.pop()):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def backdoor_paths(edges, treatment, outcome) -> list[list[str]]:
    """Undirected paths from treatment to outcome that start with an edge into the treatment."""
    nbrs: dict[str, set[str]] = {}
    for a, b in edges:
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)
    paths = []

    def walk(path):
        node = path[-1]
        if node == outcome:
            paths.append(list(path))
            return
        for n in nbrs.get(node, ()):
            if n not in path:
                walk(path + [n])

    for p in _parents(edges, treatment):
        walk([treatment, p])
    return paths


def _blocked(path, edges, adjust: set[str]) -> bool:
    for i in range(1, len(path) - 1):
        a, m, b = path[i - 1], path[i], path[i + 1]
        collider = (a, m) in edges and (b, m) in edges
        if collider:
            if m not in adjust and not (descendants(edges, m) & adjust):
                return True
        elif m in adjust:
            return True
    return False


def identify(e: Estimand) -> dict:
    edges = set(e.edges)
    adjust = set(e.adjustment)
    bad = adjust & descendants(edges, e.treatment)
    paths = backdoor_paths(edges, e.treatment, e.outcome)
    open_paths = [p for p in paths if not _blocked(p, edges, adjust)]
    if bad:
        status, why = "NOT_IDENTIFIED", f"adjustment set contains descendants of the treatment: {sorted(bad)}"
    elif open_paths and any(set(p) & set(e.unobserved) for p in open_paths):
        status, why = "NOT_IDENTIFIED", "an open back-door path runs through an unobserved node"
    elif open_paths:
        status, why = "NOT_IDENTIFIED", "an open back-door path remains"
    elif e.assumptions:
        status, why = "ASSUMPTION_DEPENDENT", "identified only under the registered untestable assumptions"
    else:
        status, why = "IDENTIFIED", "every back-door path is blocked"
    return {"estimand": e.name, "status": status, "reason": why, "backdoor_paths": paths,
            "open_paths": open_paths, "assumptions": list(e.assumptions)}


def robustness_value(t_stat: float, dof: int, q: float = 1.0) -> float:
    """Partial R^2 (with treatment and outcome) of a confounder that would reduce the estimate by 100q %."""
    if dof <= 0:
        return float("nan")
    f = q * abs(t_stat) / math.sqrt(dof)
    return 0.5 * (math.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)


def manski(mean_observed: float, observed_fraction: float, lo: float, hi: float) -> tuple[float, float]:
    """Worst-case bounds on a mean when a fraction of outcomes is missing (outcome bounded in [lo, hi])."""
    p = observed_fraction
    return p * mean_observed + (1 - p) * lo, p * mean_observed + (1 - p) * hi


REGISTRY = {
    "flow_impact": Estimand(
        name="flow_impact", treatment="aggressive_flow", outcome="mid_move_10s",
        confounders=("spread", "depth", "imbalance", "recent_volatility", "recent_flow"),
        unobserved=("information",), edges=(("information", "aggressive_flow"), ("information", "mid_move_10s"),
                                            ("state", "aggressive_flow"), ("state", "mid_move_10s"),
                                            ("aggressive_flow", "mid_move_10s")),
        adjustment=("state",), assumptions=(), estimator="state-adjusted regression (StateImpact)",
        notes={"why": "private information drives both flow and later prices; public data cannot block it"}),
    "own_order_impact_bounded": Estimand(
        name="own_order_impact_bounded", treatment="hypothetical_child_order", outcome="execution_cost",
        confounders=(), unobserved=(), edges=(("hypothetical_child_order", "execution_cost"),),
        adjustment=(), assumptions=("other participants do not react to the hypothetical order beyond the registered "
                                    "impact rule",), estimator="bounded counterfactual replay over the assumption grid"),
    "fallback_effect_simulated": Estimand(
        name="fallback_effect_simulated", treatment="fallback_policy", outcome="catastrophic_cost",
        confounders=("world",), unobserved=(), edges=(("world", "fallback_policy"), ("world", "catastrophic_cost"),
                                                      ("fallback_policy", "catastrophic_cost")),
        adjustment=("world",), assumptions=(), estimator="paired simulation on common random numbers"),
}


def registry_report() -> dict:
    return {name: {**e.to_dict(), "identification": identify(e)} for name, e in REGISTRY.items()}
