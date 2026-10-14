"""verify-v09: protocol, ledger, every sealed run, lineage DAG, retained attempts and (optionally) the bundle.

Verification means byte integrity plus protocol / ledger / dataset / config / result / lineage binding of the sealed
evidence. It is not independent scientific replication.
"""
from __future__ import annotations

from pathlib import Path

from . import artifacts, protocol as pr


def verify_command(args) -> dict:
    root = Path(args.root).resolve()
    out: dict = {"meaning": artifacts.MEANING}
    if getattr(args, "bundle", None):
        from .reporting import verify_bundle
        b = verify_bundle(Path(args.bundle))
        return {**out, "valid": b["valid"], "bundle": b}
    proto = pr.verify_protocol_files(root)
    tree = artifacts.verify_tree(root / artifacts.RESULTS_ROOT, root=root) \
        if (root / artifacts.RESULTS_ROOT).is_dir() else {"valid": True, "runs": 0, "invalid": []}
    dag = artifacts.dag(root / artifacts.RESULTS_ROOT, root=root) \
        if (root / artifacts.RESULTS_ROOT).is_dir() else {"acyclic": True, "dangling_inputs": []}
    _, state = pr.state_of(root)
    retained = [a for a in state.attempts if a.get("directory")]
    missing = [a["directory"] for a in retained if not (root / a["directory"]).exists()]
    issues = list(proto["issues"]) + [f"run does not verify: {r}" for r in tree.get("invalid", [])]
    if not dag["acyclic"]:
        issues.append("lineage DAG has a cycle")
    if dag.get("dangling_inputs"):
        issues.append(f"lineage inputs missing: {[e['from'] for e in dag['dangling_inputs']][:5]}")
    if missing:
        issues.append(f"retained attempt directories missing: {missing[:5]}")
    return {**out, "valid": not issues, "issues": issues, "protocol": {k: proto[k] for k in (
        "valid", "ledger_entries", "ledger_head_sha256", "frozen", "promoted", "confirmatory_closed", "amendments")},
            "runs": tree.get("runs"), "invalid_runs": tree.get("invalid"), "dag_acyclic": dag["acyclic"],
            "attempts": len(state.attempts), "incidents": len(state.incidents)}
