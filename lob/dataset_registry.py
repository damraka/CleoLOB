"""v0.5 dataset registry (M6): declared identity, static facts and ledger-derived history.

Every field required by the roadmap is present for each registered dataset.
Freshness, consumption, access time and source hashes are derived from the
append-only ledger; a registry file can therefore not make consumed data fresh.
``comparable`` refuses cross-dataset comparisons whose semantics differ.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .experiments.registry import PROJECT_ROOT
from . import preregistration as pr
from .v05_evidence import LEDGER_PATH, PROTOCOL_PATH

FACTS_PATH = "configs/v05/datasets.json"
REQUIRED = ("source", "venue", "instrument", "contract", "date_range", "access_time", "capability_level",
            "tick_size", "lot_size", "timezone", "sequence_semantics", "source_sha256", "normalized_sha256",
            "license", "freshness", "consumed")


def tape_sha256(tape) -> str:
    """Canonical hash of a normalized tape's arrays (float64 little-endian bytes)."""
    digest = hashlib.sha256()
    for name in ("t", "bp", "bq", "ap", "aq", "trade_t", "trade_p", "trade_q", "trade_s"):
        array = np.ascontiguousarray(np.asarray(getattr(tape, name), dtype="<f8"))
        digest.update(name.encode() + str(array.shape).encode() + array.tobytes())
    return digest.hexdigest()


def build(root: Path = PROJECT_ROOT, normalized: dict[str, str] | None = None) -> dict:
    protocol = pr.load_protocol(root / PROTOCOL_PATH)
    facts = json.loads((root / FACTS_PATH).read_text(encoding="utf-8"))
    entries = pr.read_ledger(root / LEDGER_PATH)
    state = pr.replay_ledger(entries, protocol)
    registry = {}
    for declaration in protocol["datasets"]:
        dataset_id = declaration["id"]
        instrument = facts["instruments"][f"{declaration['venue']}:{declaration['instrument']}"]
        provider = facts["providers"][declaration["provider"]]
        accesses = [e for e in entries if e["event"] == "access" and e["payload"].get("dataset_id") == dataset_id]
        sources = {}
        for entry in accesses:
            sources.update(entry["payload"].get("source_sha256", {}))
        registry[dataset_id] = {
            "source": declaration["provider"], "venue": declaration["venue"], "instrument": declaration["instrument"],
            "instrument_type": instrument["instrument_type"], "contract": instrument["contract"],
            "date_range": f"{declaration['date']} ({provider['date_range']})",
            "access_time": accesses[0]["at"] if accesses else None, "capability_level": declaration["capability_level"],
            "tick_size": instrument["tick_size"], "lot_size": instrument["lot_size"], "timezone": provider["timezone"],
            "sequence_semantics": provider["sequence_semantics"], "source_sha256": sources,
            "normalized_sha256": (normalized or {}).get(dataset_id), "license": provider["license"],
            "freshness": state.freshness(dataset_id), "consumed": dataset_id in state.consumed,
            "role": declaration["role"], "identity_sha256": pr.dataset_identity_sha256(declaration, sources)}
    for dataset_id, entry in registry.items():
        missing = [k for k in REQUIRED if k not in entry]
        if missing:
            raise ValueError(f"{dataset_id}: registry missing {missing}")
    return registry


def comparable(a: dict, b: dict, *, pooled: bool = False) -> tuple[bool, list[str]]:
    """Whether results from two datasets may be compared (or pooled) as the same quantity."""
    reasons = []
    if a["capability_level"] != b["capability_level"]:
        reasons.append("different capability levels")
    if a["instrument_type"] != b["instrument_type"]:
        reasons.append("different instrument types (units and microstructure differ)")
    if a["venue"] != b["venue"] and a["sequence_semantics"] != b["sequence_semantics"]:
        reasons.append("different venues with different sequence semantics")
    if pooled and (a["instrument"] != b["instrument"] or a["tick_size"] != b["tick_size"]):
        reasons.append("pooling across instruments or tick sizes is unsupported")
    return (not reasons, reasons)
