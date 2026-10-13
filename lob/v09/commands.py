"""Implementations of the non-study v0.9 commands (protocol, promotion, registry, audit, verify, report, smoke)."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from . import protocol as pr


def protocol_v09(args) -> dict:
    root = Path(args.root)
    op = args.operation
    if op == "verify":
        return pr.verify_protocol_files(root)
    if op == "status":
        report = pr.verify_protocol_files(root)
        report.pop("guarded_accessed", None)
        return report
    if op == "build":
        from . import registry
        if not args.base_commit:
            raise ValueError("build requires --base-commit")
        return {"valid": True, **registry.build(root, args.base_commit)}
    if op == "init":
        written = pr.initialize_ledger(root)
        return {"valid": True, "entries": len(written), "head": written[-1]["sha256"]}
    from . import runtime
    if op == "fingerprint":
        return {"valid": True, **runtime.fingerprint()}
    if op == "sbom":
        doc = runtime.sbom()
        return {"valid": True, "components": len(doc["components"]), "digest": doc["digest"], "sbom": doc}
    if op == "lock":
        return {"valid": True, "lock": runtime.write_lock(root), "check": runtime.check_lock(root)}
    raise ValueError(f"unknown operation {op}")


def hypothesis_promotion_v09(args) -> dict:
    from .studies import promotion as study
    if args.operation == "run":
        if not args.out:
            raise ValueError("run requires --out")
        return study.run(args)
    if not args.run:
        raise ValueError("freeze requires --run")
    from . import promotion
    entry = promotion.freeze(Path(args.root), Path(args.root) / args.run)
    return {"valid": True, "promote_entry": entry["index"], "confirmatory_sha256": entry["payload"]["confirmatory_sha256"],
            "promoted": entry["payload"]["promoted"]}


def dataset_registry_v09(args) -> dict:
    root = Path(args.root)
    protocol, state = pr.state_of(root)
    registry = json.loads((root / pr.CONFIG_DIR / "dataset-registry.json").read_text(encoding="utf-8"))
    rows = [d for d in registry["datasets"] if args.pool is None or d["pool"] == args.pool]
    for row in rows:
        row["freshness"] = state.freshness(row["dataset_id"])
    return {"valid": True, "schema": registry["schema"], "count": len(rows),
            "pools": {k: {kk: v[kk] for kk in ("role", "start", "end", "count", "backups", "streams")}
                      for k, v in protocol["pools"].items()},
            "datasets": rows if args.pool else [{k: r[k] for k in ("dataset_id", "role", "pool", "slot", "freshness")}
                                                for r in rows]}


def capability_audit_v09(args) -> dict:
    from . import capabilities
    return capabilities.audit(Path(args.root))


def _module(name: str):
    return importlib.import_module(f"lob.v09.{name}")


def report_v09(args) -> dict:
    return _module("reporting").report(args)


def evidence_export_v09(args) -> dict:
    return _module("reporting").export(args)


def claim_audit_v09(args) -> dict:
    return _module("claims").audit_command(args)


def verify_v09(args) -> dict:
    return _module("verify").verify_command(args)


def smoke_v09(args) -> dict:
    return _module("smoke").run(args)
