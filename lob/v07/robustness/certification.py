"""Decision certification, abstention, model-dependence attribution and the ranking graph (workstreams 36-39).

``certify(evidence)`` applies the registered gates in order. A policy conclusion is
certified (``ROBUST_ACROSS_MODEL_UNCERTAINTY``) only if every gate passes:

1. registration   — the comparison belongs to a registered confirmatory family (else INVALID);
2. statistics     — the multiplicity-adjusted interval excludes zero (else NOT_ESTABLISHED, or
                    INCONCLUSIVE with an abstention message when the design is underpowered);
3. materiality    — |estimate| >= the registered margin (else NOT_ESTABLISHED);
4. model          — no plausible world is determinate in the opposite direction and the worst
                    plausible world keeps the sign (else MODEL_DEPENDENT, naming the worlds);
5. seeds          — training-seed means agree in sign (learned policies only; else MODEL_DEPENDENT);
6. regime         — regime-conditioned worlds agree in sign (else MODEL_DEPENDENT);
7. history        — both bounded historical fill modes agree in sign and none is determinate
                    in the opposite direction (else ASSUMPTION_DEPENDENT; NOT_AVAILABLE history
                    leaves the conclusion INCONCLUSIVE: simulation-only evidence is not certified).

Abstention is first-class: ``abstain`` returns "We cannot support a conclusion" with the reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations

GATES = ("registration", "statistics", "materiality", "model", "seeds", "regime", "history")
ABSTAIN = "We cannot support a conclusion"


@dataclass
class Evidence:
    registered: bool
    estimate: float | None
    ci: tuple[float, float] | None
    margin: float
    mde: float | None = None
    opposite_worlds: list = field(default_factory=list)
    worst_plausible: float | None = None          # difference in the most adverse plausible world
    seed_signs: list | None = None                # per training seed: -1/0/+1 (None: not a learned policy)
    regime_signs: list | None = None              # per regime model
    history: dict | None = None                   # {"conservative": -1/0/+1, "optimistic": ...}; None = NOT_AVAILABLE


def abstain(reason: str, **details) -> dict:
    return {"status": "INCONCLUSIVE", "certified": False, "message": f"{ABSTAIN}: {reason}", **details}


def certify(e: Evidence) -> dict:
    trail = {}
    if not e.registered:
        return {"status": "INVALID", "certified": False, "failed_gate": "registration",
                "message": "unregistered confirmatory comparison", "gates": {"registration": False}}
    trail["registration"] = True
    if e.ci is None or e.estimate is None:
        return {**abstain("no interval"), "failed_gate": "statistics", "gates": trail}
    low, high = e.ci
    sign = -1 if high < 0 else 1 if low > 0 else 0
    trail["statistics"] = sign != 0
    if not sign:
        if e.mde is not None and e.mde > e.margin:
            return {**abstain(f"underpowered: MDE {e.mde:.3g} exceeds the margin {e.margin:.3g}"),
                    "failed_gate": "statistics", "gates": trail}
        status = "NOT_ESTABLISHED"
        equivalent = -e.margin < low and high < e.margin
        return {"status": status, "certified": False, "failed_gate": "statistics", "gates": trail,
                "equivalence": "EQUIVALENT_WITHIN_MARGIN" if equivalent else "NOT_ESTABLISHED"}
    trail["materiality"] = abs(e.estimate) >= e.margin
    if not trail["materiality"]:
        return {"status": "NOT_ESTABLISHED", "certified": False, "failed_gate": "materiality", "gates": trail}
    adverse = e.worst_plausible is not None and (e.worst_plausible * sign) <= 0
    trail["model"] = not e.opposite_worlds and e.worst_plausible is not None and not adverse
    if not trail["model"]:
        return {"status": "MODEL_DEPENDENT", "certified": False, "failed_gate": "model", "gates": trail,
                "responsible_worlds": list(e.opposite_worlds) or (["worst-plausible"] if adverse else []),
                "message": "direction changes across registered plausible models" if (e.opposite_worlds or adverse)
                else "no adversarial search result"}
    if e.seed_signs is not None:
        trail["seeds"] = all(s == sign for s in e.seed_signs)
        if not trail["seeds"]:
            return {"status": "MODEL_DEPENDENT", "certified": False, "failed_gate": "seeds", "gates": trail,
                    "message": "training seeds disagree in sign"}
    if e.regime_signs is not None:
        trail["regime"] = all(s in (0, sign) for s in e.regime_signs) and any(s == sign for s in e.regime_signs)
        if not trail["regime"]:
            return {"status": "MODEL_DEPENDENT", "certified": False, "failed_gate": "regime", "gates": trail,
                    "message": "regime-conditioned worlds disagree"}
    if e.history is None:
        return {**abstain("historical execution bounds are NOT_AVAILABLE; simulation-only evidence is not certified"),
                "failed_gate": "history", "gates": trail}
    modes = list(e.history.values())
    trail["history"] = all(m == sign for m in modes)
    if not trail["history"]:
        reversed_ = any(m == -sign for m in modes)
        return {"status": "ASSUMPTION_DEPENDENT", "certified": False, "failed_gate": "history", "gates": trail,
                "message": "historical fill bounds " + ("contradict" if reversed_ else "do not both support") +
                           " the simulated direction"}
    return {"status": "ROBUST_ACROSS_MODEL_UNCERTAINTY", "certified": True, "failed_gate": None, "gates": trail,
            "direction": sign}


# ----------------------------------------------------------------------------- ranking graph


def ranking_graph(edges: dict) -> dict:
    """Directed graph: an arc a -> b means 'a robustly cheaper than b'. Reports edges and robust cycles."""
    nodes = sorted({x for k in edges for x in k.split("|")})
    arcs = []
    for key, value in edges.items():
        a, b = key.split("|")
        edge = value["edge"] if isinstance(value, dict) else value
        if edge == "ROBUSTLY_BETTER":
            arcs.append((a, b))
        elif edge == "ROBUSTLY_WORSE":
            arcs.append((b, a))
    adjacency = {n: {b for a, b in arcs if a == n} for n in nodes}
    cycles = [list(c) for c in permutations(nodes, 3) if c[0] == min(c)
              and c[1] in adjacency[c[0]] and c[2] in adjacency[c[1]] and c[0] in adjacency[c[2]]]
    consistent = all(not (a in adjacency[b] and b in adjacency[a]) for a, b in arcs)
    return {"nodes": nodes, "robust_arcs": arcs, "edge_types": {k: (v["edge"] if isinstance(v, dict) else v)
                                                                  for k, v in edges.items()},
            "three_cycles": cycles, "antisymmetric": consistent,
            "open_transitive_triples": sum(1 for a, b in arcs for c in adjacency[b] if c not in adjacency[a] and c != a)}
