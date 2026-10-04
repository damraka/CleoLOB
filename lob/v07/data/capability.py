"""Capability-aware validation of study requests (workstream 85).

A study declares the observations it needs; each dataset's capability level
decides whether it can run. A missing capability yields a NOT_AVAILABLE status
with the reason, never a silently weakened analysis and never invented fields.
"""
from __future__ import annotations

from ...capabilities import Capability, CapabilityError
from ..protocol import core as pr
from .schema import venue_spec

# What each v0.7 analysis needs from a dataset.
REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "realism": ("aggregate_depth", "trade_prints"),
    "domain_gap": ("aggregate_depth", "trade_prints"),
    "bounded_replay": ("aggregate_depth", "trade_prints"),
    "queue_position": ("exact_fifo_position", "quantity_ahead"),
    "order_survival": ("order_identity",),
    "exact_passive_fill": ("observed_order_fill", "exact_fifo_position"),
    "hidden_liquidity": ("hidden_order_quantity",),
}


def check(protocol: dict, dataset_id: str, analysis: str) -> dict:
    """``{"status": "AVAILABLE"}`` or ``{"status": "NOT_AVAILABLE", "reason": ...}`` for one dataset/analysis."""
    if analysis not in REQUIREMENTS:
        raise ValueError(f"unknown analysis {analysis!r}; register its requirements first")
    declaration = pr.dataset_declaration(protocol, dataset_id)
    contract = venue_spec(declaration["venue"], declaration["instrument"]).contract
    missing = []
    for name in REQUIREMENTS[analysis]:
        try:
            if not contract.supports(Capability(name)):
                missing.append(name)
        except (ValueError, CapabilityError):
            missing.append(name)
    if missing:
        return {"status": "NOT_AVAILABLE", "dataset": dataset_id, "analysis": analysis, "missing": missing,
                "reason": f"{declaration['capability_level']} does not provide {missing}"}
    return {"status": "AVAILABLE", "dataset": dataset_id, "analysis": analysis}


def matrix(protocol: dict) -> dict:
    """Capability matrix over every declared dataset and registered analysis (reported in docs)."""
    return {d["id"]: {a: check(protocol, d["id"], a)["status"] for a in REQUIREMENTS} for d in protocol["datasets"]}
