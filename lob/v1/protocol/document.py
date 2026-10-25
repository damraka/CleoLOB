"""The v1 protocol document, its freeze and numbered amendments, sealed designs and verification (M2).

A ``Workspace`` binds a repository root to its v1 configuration directory (``configs/v1`` by default; tests use
temporary roots). The protocol is a JSON document validated by ``validate``; ``freeze`` records its canonical
SHA-256 in the ledger, after which the file on disk must keep matching the last frozen or amended hash. An
amendment writes the new document, archives the previous one under ``protocol-history/`` and records both hashes,
the affected sections and whether it is post hoc (mandatory after any guarded access).

Referenced configuration files (dataset registry, hypothesis candidates, statistical families, ...) are hashed at
freeze and amendment time; ``verify`` reports any later change. Sealed designs live under ``designs/`` and are
verified the same way.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from . import ledger as L
from .namespaces import GUARDED, NAMESPACES, ROLES

SCHEMA = "cleolob-v1-protocol-1"
STATUSES = ("ESTABLISHED", "FAILED", "NOT_ESTABLISHED", "INCONCLUSIVE", "NOT_AVAILABLE", "NOT_EVALUABLE")
LIFECYCLE = ("CANDIDATE", "FEASIBILITY_PASS", "FEASIBILITY_FAIL", "PROMOTED", "NOT_PROMOTED", "SEALED",
             "CONFIRMED_TERMINAL")
EVIDENCE_LEVELS = {
    "E0": "engineering verification only",
    "E1": "synthetic (F4-F6) only",
    "E2": "multi-model synthetic consensus",
    "E3": "hybrid historical twin (F3)",
    "E4": "observed or reconstructed real-data prediction (F0/F1)",
    "E5": "bounded historical counterfactual, direction identified (F2)",
    "E6": "cross-time real support on fresh data",
    "E7": "cross-instrument real support on fresh data",
    "E8": "cross-venue real support on fresh data",
    "E9": "untouched final vault support",
    "E10": "prospective read-only shadow support",
}
REQUIRED_SECTIONS = ("schema", "protocol_id", "research_question", "statuses", "evidence_levels", "roles",
                     "namespaces", "pools", "promotion_rules", "statistics", "realism_gate", "resources",
                     "amendment_policy", "referenced_configs", "fidelities")


class ProtocolError(RuntimeError):
    pass


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(L.canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(doc: dict) -> list[str]:
    issues = [f"missing section {s}" for s in REQUIRED_SECTIONS if s not in doc]
    if issues:
        return issues
    if doc["schema"] != SCHEMA:
        issues.append(f"schema must be {SCHEMA}")
    if tuple(doc["statuses"]) != STATUSES:
        issues.append("statuses must be the registered terminal statuses")
    if set(doc["evidence_levels"]) != set(EVIDENCE_LEVELS):
        issues.append("evidence levels must be E0..E10")
    if tuple(doc["namespaces"]) != NAMESPACES:
        issues.append("namespaces must be the registered namespaces")
    if tuple(doc["roles"]) != ROLES:
        issues.append("roles must be the registered roles")
    st = doc["statistics"]
    for key in ("independent_unit", "primary_test", "min_clusters", "alpha", "correction"):
        if key not in st:
            issues.append(f"statistics.{key} missing")
    if isinstance(st.get("min_clusters"), int) and st.get("alpha"):
        if 2.0 ** -st["min_clusters"] > float(st.get("strictest_family_alpha", st["alpha"])):
            issues.append("min_clusters cannot reach the strictest family alpha with an exact sign-flip test")
    pr = doc["promotion_rules"]
    for key in ("min_power", "planning_effect", "min_pilot_clusters", "compute_ceiling_hours"):
        if key not in pr:
            issues.append(f"promotion_rules.{key} missing")
    for name, pool in doc["pools"].items():
        if pool.get("role") not in ROLES:
            issues.append(f"pool {name}: unknown role {pool.get('role')}")
        if pool.get("role") in ("confirm", "replicate", "vault") and not pool.get("streams"):
            issues.append(f"pool {name}: guarded pools list their streams")
    return issues


@dataclass(frozen=True)
class Workspace:
    root: Path
    config_dir: str = "configs/v1"
    allow_dirty: bool = False          # tests only; guarded events in the repository require a clean tree

    @property
    def config(self) -> Path:
        return self.root / self.config_dir

    @property
    def protocol_path(self) -> Path:
        return self.config / "protocol.json"

    @property
    def ledger_path(self) -> Path:
        return self.config / "ledger.jsonl"

    @property
    def designs_dir(self) -> Path:
        return self.config / "designs"

    def load(self) -> dict:
        return json.loads(self.protocol_path.read_text(encoding="utf-8"))

    def state(self) -> L.LedgerState:
        return L.replay(L.read(self.ledger_path))

    def append(self, event: str, *, reason: str, payload: dict | None = None, dataset: str | None = None) -> dict:
        return L.append(self.ledger_path, event, reason=reason, payload=payload, dataset=dataset, root=self.root,
                        allow_dirty=self.allow_dirty)

    def referenced_hashes(self, doc: dict) -> dict[str, str]:
        out = {}
        for rel in doc.get("referenced_configs", []):
            p = self.config / rel
            if not p.exists():
                raise ProtocolError(f"referenced config {rel} is missing")
            out[rel] = file_sha256(p)
        return out


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def freeze(ws: Workspace, doc: dict, *, reason: str) -> dict:
    issues = validate(doc)
    if issues:
        raise ProtocolError("; ".join(issues))
    if ws.protocol_path.exists() and ws.state().frozen is not None:
        raise ProtocolError("already frozen; use amend")
    write_json(ws.protocol_path, doc)
    return ws.append("freeze", reason=reason, payload={"protocol_sha256": canonical_sha256(doc),
                                                       "configs": ws.referenced_hashes(doc)})


def amend(ws: Workspace, doc: dict, *, reason: str, affects: list[str], posthoc: bool | None = None) -> dict:
    issues = validate(doc)
    if issues:
        raise ProtocolError("; ".join(issues))
    state = ws.state()
    if state.frozen is None:
        raise ProtocolError("amend after freeze")
    if not affects:
        raise ProtocolError("an amendment names the sections it affects")
    posthoc = state.guarded_access_seen if posthoc is None else posthoc
    old = ws.load()
    number = len(state.amendments) + 1
    write_json(ws.config / "protocol-history" / f"protocol-{number - 1}.json", old)
    write_json(ws.protocol_path, doc)
    return ws.append("amend", reason=reason, payload={
        "number": number, "previous_sha256": canonical_sha256(old), "protocol_sha256": canonical_sha256(doc),
        "affects": sorted(affects), "posthoc": bool(posthoc), "configs": ws.referenced_hashes(doc)})


def seal_design(ws: Workspace, name: str, design: dict, *, reason: str) -> dict:
    if "/" in name or "\\" in name or not name:
        raise ProtocolError("design names are plain identifiers")
    path = ws.designs_dir / f"{name}.json"
    if path.exists():
        raise ProtocolError(f"design {name} exists; sealed designs are write-once")
    for key in ("study", "namespace", "hypotheses", "datasets", "analysis"):
        if key not in design:
            raise ProtocolError(f"design {name} lacks {key}")
    if design["namespace"] not in NAMESPACES:
        raise ProtocolError(f"design {name}: unknown namespace")
    write_json(path, design)
    try:
        return ws.append("seal_design", reason=reason, payload={
            "design": name, "sha256": canonical_sha256(design), "namespace": design["namespace"],
            "datasets": list(design["datasets"]), "hypotheses": list(design["hypotheses"])})
    except Exception:
        path.unlink()            # a refused seal leaves no design file behind
        raise


def sealed_design(ws: Workspace, name: str) -> dict:
    state = ws.state()
    if name not in state.designs:
        raise ProtocolError(f"design {name} is not sealed")
    doc = json.loads((ws.designs_dir / f"{name}.json").read_text(encoding="utf-8"))
    if canonical_sha256(doc) != state.designs[name]["sha256"]:
        raise ProtocolError(f"design {name} changed after sealing")
    return doc


def open_namespace(ws: Workspace, namespace: str, *, design: str, reason: str) -> dict:
    if namespace not in GUARDED:
        raise ProtocolError("only guarded namespaces are opened")
    return ws.append("open_namespace", reason=reason, payload={"namespace": namespace, "design": design})


def close_namespace(ws: Workspace, namespace: str, *, runs: dict[str, str], reason: str) -> dict:
    return ws.append("close_namespace", reason=reason, payload={"namespace": namespace, "runs": dict(runs)})


def record_access(ws: Workspace, dataset: str, *, namespace: str, stage: str, use: str, design: str | None = None,
                  reason: str) -> dict:
    if stage not in L.STAGES:
        raise ProtocolError(f"unknown stage {stage}")
    return ws.append("access", dataset=dataset, reason=reason,
                     payload={"namespace": namespace, "stage": stage, "use": use, "design": design})


def record_incident(ws: Workspace, kind: str, *, description: str, affects_validity: bool,
                    runs: list[str] | None = None) -> dict:
    return ws.append("incident", reason=f"incident: {kind}", payload={
        "kind": kind, "description": description, "affects_validity": bool(affects_validity), "runs": runs or []})


def record_attempt(ws: Workspace, *, study: str, run: str, outcome: str, namespace: str, detail: str = "") -> dict:
    return ws.append("attempt", reason=f"attempt {study}: {outcome}", payload={
        "study": study, "run": run, "outcome": outcome, "namespace": namespace, "detail": detail})


def verify(ws: Workspace) -> dict:
    """Ledger chain and rules, protocol hash, referenced-config hashes and sealed-design hashes."""
    issues: list[str] = []
    led = L.verify(ws.ledger_path)
    issues += led["issues"]
    state = ws.state() if led["valid"] else None
    doc = None
    if ws.protocol_path.exists():
        doc = ws.load()
        issues += validate(doc)
    if state is not None and state.frozen is not None:
        if doc is None:
            issues.append("frozen protocol file missing")
        elif canonical_sha256(doc) != state.frozen["protocol_sha256"]:
            issues.append("protocol.json differs from the frozen/amended hash")
        recorded = {}
        for e in L.read(ws.ledger_path):
            if e["event"] in ("freeze", "amend"):
                recorded = e["payload"].get("configs", {})
        for rel, digest in recorded.items():
            p = ws.config / rel
            if not p.exists() or file_sha256(p) != digest:
                issues.append(f"referenced config {rel} changed after freeze/amendment")
        for name, rec in state.designs.items():
            p = ws.designs_dir / f"{name}.json"
            if not p.exists() or canonical_sha256(json.loads(p.read_text(encoding="utf-8"))) != rec["sha256"]:
                issues.append(f"sealed design {name} missing or changed")
    return {"valid": not issues, "issues": issues, "protocol_sha256": canonical_sha256(doc) if doc else None,
            "state": state.to_dict() if state else None}


__all__ = ["SCHEMA", "STATUSES", "LIFECYCLE", "EVIDENCE_LEVELS", "ProtocolError", "Workspace", "validate", "freeze",
           "amend", "seal_design", "sealed_design", "open_namespace", "close_namespace", "record_access",
           "record_incident", "record_attempt", "verify", "canonical_sha256", "file_sha256", "write_json"]
