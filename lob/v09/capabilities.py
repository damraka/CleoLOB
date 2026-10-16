"""Machine-readable dataset/runtime capability graph and audit (M11, M34).

``audit`` answers, for every candidate hypothesis, whether each capability it requires is provided by a registered
source of its pool (data capabilities), by the runtime (``neural_backend``: PyTorch importable) or by simulation
(``propensities``). It never reads market data; data quality and availability are decided at access time by the
quality engine. Missing capabilities make the candidate ``CAPABILITY_MISSING`` for promotion (fail closed).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from . import protocol as pr

RUNTIME_CAPS = {"neural_backend": lambda: importlib.util.find_spec("torch") is not None}
SIMULATION_CAPS = frozenset({"propensities"})
CAPABILITY_KEYS = {"incremental_l2": "incremental_l2", "trades_with_side": "aggressor_side",
                   "l1_quotes": "l1_quotes", "update_sequence": "sequence_numbers",
                   "queue_clearance_observable": "queue_clearance_observable", "order_ids": "order_ids"}


def pool_capabilities(protocol: dict, pool: str) -> dict[str, dict[str, bool]]:
    """stream (venue:instrument) -> capability -> provided."""
    out: dict[str, dict[str, bool]] = {}
    for d in pr.datasets_by(protocol, pool=pool):
        stream = f"{d['venue']}:{d['instrument']}"
        caps = d["capabilities"]
        out[stream] = {cap: bool(caps.get(key)) for cap, key in CAPABILITY_KEYS.items()}
    return out


def provided(protocol: dict, pool: str | None, capability: str, *, stream_filter=None) -> bool:
    if capability in RUNTIME_CAPS:
        return bool(RUNTIME_CAPS[capability]())
    if capability in SIMULATION_CAPS:
        return True
    if pool is None:
        return False
    streams = pool_capabilities(protocol, pool)
    return any(v.get(capability, False) for s, v in streams.items() if stream_filter is None or stream_filter(s))


def audit(root: Path) -> dict:
    root = Path(root)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    candidates = json.loads((root / pr.CANDIDATES_PATH).read_text(encoding="utf-8"))
    matrix = json.loads((root / pr.CONFIG_DIR / "capability-matrix.json").read_text(encoding="utf-8"))
    rows = {}
    for hid, h in candidates["hypotheses"].items():
        caps = {c: provided(protocol, h["pool"], c) for c in h["required_capabilities"]}
        rows[hid] = {"pool": h["pool"], "required": caps,
                     "status": "AVAILABLE" if all(caps.values()) else "CAPABILITY_MISSING"}
    return {"valid": True, "schema": "cleolob-v09-capability-audit-1",
            "pools": {p: pool_capabilities(protocol, p) for p in protocol["pools"]},
            "runtime": {k: bool(f()) for k, f in RUNTIME_CAPS.items()}, "hypotheses": rows,
            "never_provided": matrix["never_provided"],
            "missing": sorted(h for h, r in rows.items() if r["status"] != "AVAILABLE")}
