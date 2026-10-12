"""Dataset registry v3 and the capability matrix (M2).

Every v0.8 dataset declares provider, instrument, venue, time range, access
class, role, freshness (ledger-derived), its capability matrix and the claims it
may support. ``require`` fails closed: a study that needs a capability the
dataset lacks (e.g. queue identity on aggregate L2) is refused, never silently
downgraded. Synthetic worlds have their own capability row (full ground truth,
labelled SIMULATED).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..preregistration import ProtocolError
from . import protocol as pr

CAPABILITIES = ("l1", "l2", "l3", "mbo", "trades", "queue_identity", "exchange_timestamp", "receive_timestamp",
                "hidden_liquidity_observable", "historical_counterfactual")
CLAIMS = ("aggregate_book_statistics", "generator_calibration", "regime_detection", "drift_detection",
          "bounded_counterfactual_execution", "ranking_transfer_bounded", "exact_queue_position",
          "exact_historical_fill", "hidden_liquidity", "simulated_ground_truth")
SYNTHETIC_CAPABILITIES = {"l1": True, "l2": True, "l3": True, "mbo": False, "trades": True, "queue_identity": True,
                          "exchange_timestamp": True, "receive_timestamp": True, "hidden_liquidity_observable": True,
                          "historical_counterfactual": "simulated_truth"}
# what each v0.8 study needs from a real dataset
STUDY_REQUIREMENTS = {
    "validate_market_data": ("l2",),
    "regime_study": ("l2", "trades"),
    "change_point_study": ("l2", "trades"),
    "generator_fit": ("l2", "trades"),
    "inference": ("l2", "trades"),
    "counterfactual_bounds": ("l2", "trades", "receive_timestamp"),
    "ranking_transfer": ("l2", "trades", "receive_timestamp"),
    "exact_queue_study": ("queue_identity", "mbo"),
    "hidden_liquidity_study": ("hidden_liquidity_observable",),
    "exact_fill_replay": ("queue_identity", "l3"),
}


class CapabilityError(ProtocolError):
    """A dataset lacks a capability the study needs (fail closed)."""


def registry(root: Path) -> dict:
    return json.loads((root / pr.CONFIG_DIR / "dataset-registry.json").read_text(encoding="utf-8"))


def capability_matrix(root: Path) -> dict[str, dict]:
    reg = registry(root)
    return {d["id"]: d["capabilities"] for d in reg["datasets"]}


def require(capabilities: dict, study: str, *, dataset: str = "?") -> None:
    if study not in STUDY_REQUIREMENTS:
        raise KeyError(f"unknown study {study!r}")
    missing = [c for c in STUDY_REQUIREMENTS[study] if not capabilities.get(c)]
    if missing:
        raise CapabilityError(f"{dataset}: study {study!r} needs {missing}; NOT_AVAILABLE for this dataset")


def allowed(declaration: dict, claim: str) -> bool:
    if claim not in CLAIMS:
        raise KeyError(f"unknown claim class {claim!r}")
    return claim in declaration.get("allowed_claims", [])


def validate_registry(root: Path) -> list[str]:
    """Registry and protocol must agree; capabilities must be complete; forbidden claims must stay forbidden."""
    issues = []
    reg = registry(root)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    declared = {d["id"]: d for d in protocol["datasets"]}
    for d in reg["datasets"]:
        if d["id"] not in declared:
            issues.append(f"{d['id']}: in registry but not in the protocol")
            continue
        missing = [c for c in CAPABILITIES if c not in d["capabilities"]]
        if missing:
            issues.append(f"{d['id']}: capability matrix incomplete {missing}")
        caps = d["capabilities"]
        claims = set(declared[d["id"]]["allowed_claims"])
        if not caps.get("queue_identity") and "exact_queue_position" in claims:
            issues.append(f"{d['id']}: exact queue claims without queue identity")
        if not (caps.get("queue_identity") and caps.get("l3")) and "exact_historical_fill" in claims:
            issues.append(f"{d['id']}: exact historical fill claims without order identity")
        if not caps.get("hidden_liquidity_observable") and "hidden_liquidity" in claims:
            issues.append(f"{d['id']}: hidden-liquidity claims without observability")
        if "simulated_ground_truth" in claims:
            issues.append(f"{d['id']}: real datasets cannot carry simulated ground truth")
    missing_ids = set(declared) - {d["id"] for d in reg["datasets"]}
    if missing_ids:
        issues.append(f"protocol datasets absent from the registry: {sorted(missing_ids)}")
    return issues


def report(root: Path) -> dict:
    """Registry v3 view: declarations, capability matrix, ledger freshness and stage history."""
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), protocol)
    reg = {d["id"]: d for d in registry(root)["datasets"]}
    rows = []
    for d in protocol["datasets"]:
        r = reg.get(d["id"], {})
        sources: dict[str, str] = {}
        for access in state.accesses.get(d["id"], []):
            sources.update(access.get("source_sha256", {}))
        rows.append({
            "id": d["id"], "provider": d["provider"], "instrument": d["instrument"], "venue": d["venue"],
            "time_range": r.get("time_range", f"{d['date']}T00:00:00Z/{d['date']}T23:59:59.9Z"),
            "access_class": r.get("access_class"), "role": d["role"], "freshness": state.freshness(d["id"]),
            "consumed": state.freshness(d["id"]) == "consumed", "capabilities": r.get("capabilities", {}),
            "studies": {s: _study_status(r.get("capabilities", {}), s) for s in STUDY_REQUIREMENTS},
            "allowed_claims": d["allowed_claims"], "access_stages": state.stages.get(d["id"], []),
            "source_sha256": sources, "acquisition": r.get("acquisition")})
    return {"schema": "cleolob-v08-dataset-registry-3", "datasets": rows,
            "synthetic_world_capabilities": SYNTHETIC_CAPABILITIES, "issues": validate_registry(root)}


def _study_status(caps: dict, study: str) -> str:
    missing = [c for c in STUDY_REQUIREMENTS[study] if not caps.get(c)]
    return "AVAILABLE" if not missing else f"NOT_AVAILABLE (missing: {', '.join(missing)})"
