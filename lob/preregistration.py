"""v0.5 protocol freeze, append-only holdout consumption ledger and evidence binding.

The frozen protocol is an immutable JSON document identified by its canonical
SHA-256. Dataset freshness is never a mutable field: it is derived from a
hash-chained, append-only ledger. A consumed dataset cannot be declared fresh
again, amendments cannot follow first access of a dataset they affect, and a
registered external holdout cannot be accessed before an analysis design that
reads it has been sealed. Evidence binds to the protocol hash, a ledger anchor,
dataset identities, configuration, source and result hashes.

Hash chains detect accidental or silent mutation, truncation (through anchors
recorded in evidence) and reordering. They are not an external timestamp
authority and cannot prove that nobody looked at data outside this workflow.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .capabilities import EvidenceStatus
from .config import canonical_json

SCHEMA = "cleolob-v05-protocol-1"
LEDGER_SCHEMA = "cleolob-v05-ledger-1"
BINDING_SCHEMA = "cleolob-v05-binding-1"
STATUSES = tuple(status.value for status in EvidenceStatus)
ROLES = ("development", "selection", "internal_holdout", "external_holdout", "transfer_holdout",
         "mbo_development", "mbo_validation")
FRESH_ROLES = frozenset({"external_holdout", "transfer_holdout", "mbo_validation"})
FILL_CLASSES = ("OBSERVED_FILL", "GUARANTEED_FILL", "POSSIBLE_FILL", "GUARANTEED_NON_FILL",
                "INDETERMINATE", "UNSUPPORTED")
EVENTS = ("import_consumed", "declare_fresh", "freeze", "seal_design", "access", "consume", "amend")
_IDENTITY_FIELDS = ("id", "provider", "venue", "instrument", "date", "data_types", "capability_level", "role")
_REQUIRED = ("schema", "protocol_id", "frozen_at_utc", "base_commit", "statuses", "research_questions",
             "datasets", "holdout_rules", "endpoints", "hypotheses", "statistics", "success_gates",
             "completion_semantics", "invalid_episode_treatment", "historical_fill_uncertainty",
             "seeds", "compute_budget", "model_families", "normalization_sources", "referenced_configs")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9.-]{2,80}")


class ProtocolError(ValueError):
    """The protocol, ledger or binding is inconsistent; research must not proceed."""


def document_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def period_key(value: dict | str) -> str | None:
    """Venue/instrument/date alias key, so renamed identifiers cannot look fresh."""
    if isinstance(value, dict):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value.get("date", ""))):
            return None
        return f"{value['venue']}:{value['instrument']}:{value['date']}".lower()
    parts = str(value).split(":")
    if len(parts) == 3 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", parts[2]):
        return str(value).lower()
    return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_protocol(protocol: dict) -> dict:
    """Structural and semantic validation; returns the protocol unchanged."""
    if not isinstance(protocol, dict) or protocol.get("schema") != SCHEMA:
        raise ProtocolError("unsupported protocol schema")
    missing = [key for key in _REQUIRED if key not in protocol]
    if missing:
        raise ProtocolError(f"protocol missing required sections: {missing}")
    if tuple(protocol["statuses"]) != STATUSES:
        raise ProtocolError("protocol must use exactly the six evidence statuses")
    if not _DIGEST.fullmatch(str(protocol["base_commit"])) and not re.fullmatch("[0-9a-f]{40}", str(protocol["base_commit"])):
        raise ProtocolError("base_commit must be a full git SHA")
    ids = set()
    roles = []
    for dataset in protocol["datasets"]:
        for key in _IDENTITY_FIELDS + ("freshness_at_freeze",):
            if key not in dataset:
                raise ProtocolError(f"dataset declaration missing {key}")
        if not _SAFE_ID.fullmatch(dataset["id"]) or dataset["id"] in ids:
            raise ProtocolError(f"invalid or duplicate dataset id: {dataset['id']}")
        ids.add(dataset["id"])
        if dataset["role"] not in ROLES:
            raise ProtocolError(f"unknown dataset role: {dataset['role']}")
        if dataset["freshness_at_freeze"] not in {"fresh", "consumed"}:
            raise ProtocolError("freshness_at_freeze must be fresh or consumed")
        if dataset["role"] in FRESH_ROLES and dataset["freshness_at_freeze"] != "fresh":
            raise ProtocolError(f"{dataset['id']}: external/transfer/validation roles require fresh data")
        if dataset["freshness_at_freeze"] == "consumed" and not dataset.get("consumed_evidence"):
            raise ProtocolError(f"{dataset['id']}: consumed data must cite inspection evidence")
        if not isinstance(dataset["data_types"], list) or not dataset["data_types"]:
            raise ProtocolError("dataset data_types must be a nonempty list")
        roles.append(dataset["role"])
    for role in ("development", "selection", "internal_holdout", "external_holdout"):
        if role not in roles:
            raise ProtocolError(f"protocol must declare a {role} dataset")
    classes = tuple(protocol["historical_fill_uncertainty"].get("classes", ()))
    if classes != FILL_CLASSES:
        raise ProtocolError("historical fill classes must match the registered vocabulary")
    m7 = protocol["statistics"].get("M7", {})
    if m7.get("family_size") != 4 * 2 * 8 * 2 or m7.get("correction") != "bonferroni":
        raise ProtocolError("M7 comparison family/correction does not match its declared structure")
    for key in ("primary", "secondary"):
        if not protocol["endpoints"].get(key):
            raise ProtocolError(f"{key} endpoints are required")
    if not isinstance(protocol["referenced_configs"], list):
        raise ProtocolError("referenced_configs must list repository-relative paths")
    for name in protocol["referenced_configs"]:
        _portable(name)
    return protocol


def _portable(name: str) -> str:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name or name.startswith("/")
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ProtocolError(f"unsafe repository-relative path: {name!r}")
    return name


def load_protocol(path: str | Path) -> dict:
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8")))


def protocol_sha256(protocol: dict) -> str:
    return document_sha256(validate_protocol(protocol))


def dataset_declaration(protocol: dict, dataset_id: str) -> dict:
    for dataset in protocol["datasets"]:
        if dataset["id"] == dataset_id:
            return dataset
    raise ProtocolError(f"dataset {dataset_id!r} is not declared in the frozen protocol")


def dataset_identity_sha256(declaration: dict, source_hashes: dict[str, str] | None = None) -> str:
    """Identity = immutable declaration fields plus, once accessed, exact source bytes."""
    identity = {key: declaration[key] for key in _IDENTITY_FIELDS}
    identity["source_sha256"] = dict(sorted((source_hashes or {}).items()))
    return document_sha256(identity)


def text_sha256(path: str | Path) -> str:
    """SHA-256 of a text file with CRLF normalized, portable across git checkouts."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def referenced_config_hashes(protocol: dict, root: str | Path) -> dict[str, str]:
    root = Path(root)
    return {name: text_sha256(root / _portable(name)) for name in protocol["referenced_configs"]}


# ----------------------------------------------------------------------------- ledger


def _entry_hash(entry: dict) -> str:
    return document_sha256({key: value for key, value in entry.items() if key != "sha256"})


def read_ledger(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ProtocolError(f"ledger line {number} is blank; the ledger is append-only JSONL")
        entries.append(json.loads(line))
    return entries


class LedgerState:
    """Replay of the ledger rules; the only source of dataset freshness."""

    def __init__(self) -> None:
        self.head = "0" * 64
        self.count = 0
        self.consumed: dict[str, int] = {}
        self.consumed_keys: set[str] = set()
        self.fresh: dict[str, int] = {}
        self.accesses: dict[str, list[dict]] = {}
        self.sealed_reads: dict[str, int] = {}
        self.frozen: dict | None = None
        self.designs: dict[str, dict] = {}

    def freshness(self, dataset_id: str) -> str:
        if dataset_id in self.consumed:
            return "consumed"
        if dataset_id in self.fresh:
            return "fresh"
        return "undeclared"

    def apply(self, entry: dict, protocol: dict | None = None) -> None:
        if entry.get("schema") != LEDGER_SCHEMA or entry.get("index") != self.count:
            raise ProtocolError(f"ledger entry {self.count}: schema/index mismatch (truncated or reordered?)")
        if entry.get("prev_sha256") != self.head:
            raise ProtocolError(f"ledger entry {self.count}: broken hash chain")
        if entry.get("sha256") != _entry_hash(entry):
            raise ProtocolError(f"ledger entry {self.count}: content hash mismatch")
        event, payload = entry.get("event"), entry.get("payload", {})
        if event not in EVENTS or not isinstance(payload, dict):
            raise ProtocolError(f"ledger entry {self.count}: unknown event {event!r}")
        dataset = payload.get("dataset_id")
        key = payload.get("period_key") or period_key(dataset or "")
        if event == "import_consumed":
            self.consumed.setdefault(dataset, self.count)
            if key:
                self.consumed_keys.add(key)
        elif event == "declare_fresh":
            if dataset in self.consumed or (key and key in self.consumed_keys):
                raise ProtocolError(f"{dataset} is consumed; freshness cannot be restored")
            if dataset in self.fresh:
                raise ProtocolError(f"{dataset} already declared fresh")
            self.fresh[dataset] = self.count
        elif event == "freeze":
            if self.frozen is not None:
                raise ProtocolError("protocol already frozen; use an amendment")
            if not _DIGEST.fullmatch(str(payload.get("protocol_sha256"))):
                raise ProtocolError("freeze requires a protocol hash")
            self.frozen = payload
        elif event == "seal_design":
            if self.frozen is None:
                raise ProtocolError("analysis designs may only be sealed after protocol freeze")
            name = payload.get("analysis")
            if not name or name in self.designs or not _DIGEST.fullmatch(str(payload.get("design_sha256"))):
                raise ProtocolError("design seal requires a unique analysis name and design hash")
            for read in payload.get("reads", []):
                # Sealing a design that reads consumed data is allowed; it never restores freshness.
                self.sealed_reads.setdefault(read, self.count)
            self.designs[name] = payload
        elif event in {"access", "consume"}:
            if not dataset:
                raise ProtocolError(f"{event} requires dataset_id")
            if protocol is not None:
                declaration = dataset_declaration(protocol, dataset)
                if (declaration["role"] in {"external_holdout", "transfer_holdout"}
                        and dataset not in self.sealed_reads):
                    raise ProtocolError(f"{dataset}: no sealed analysis design precedes holdout access")
            if event == "access":
                for name, digest in payload.get("source_sha256", {}).items():
                    if not _DIGEST.fullmatch(str(digest)):
                        raise ProtocolError("access source hashes must be SHA-256")
                    for prior in self.accesses.get(dataset, []):
                        if name in prior.get("source_sha256", {}) and prior["source_sha256"][name] != digest:
                            raise ProtocolError(f"{dataset}/{name}: source bytes changed since first access")
                self.accesses.setdefault(dataset, []).append(payload)
            self.consumed.setdefault(dataset, self.count)
            if protocol is not None:
                key = period_key(dataset_declaration(protocol, dataset))
            if key:
                self.consumed_keys.add(key)
        elif event == "amend":
            if self.frozen is None:
                raise ProtocolError("amendments require a frozen protocol")
            if not _DIGEST.fullmatch(str(payload.get("amendment_sha256"))) or not payload.get("reason"):
                raise ProtocolError("amendment requires hash and reason")
            late = [d for d in payload.get("affects", []) if d in self.consumed]
            if late:
                raise ProtocolError(f"amendment affects already consumed data: {late}")
        self.head = entry["sha256"]
        self.count += 1


def replay_ledger(entries: Iterable[dict], protocol: dict | None = None) -> LedgerState:
    state = LedgerState()
    for entry in entries:
        state.apply(entry, protocol)
    if protocol is not None and state.frozen is not None:
        if state.frozen["protocol_sha256"] != protocol_sha256(protocol):
            raise ProtocolError("protocol changed after freeze (hash differs from ledger freeze entry)")
    return state


def append_event(path: str | Path, event: str, payload: dict, *, protocol: dict | None = None,
                 at: str | None = None) -> dict:
    """Validate the complete history plus the new entry, then append atomically."""
    path = Path(path)
    entries = read_ledger(path)
    state = replay_ledger(entries, protocol)
    entry = {"schema": LEDGER_SCHEMA, "index": state.count, "event": event,
             "payload": json.loads(canonical_json(payload)), "at": at or _utc_now(),
             "prev_sha256": state.head}
    entry["sha256"] = _entry_hash(entry)
    state.apply(entry, protocol)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(entry) + "\n")
    return entry


def ledger_anchor(path: str | Path) -> dict:
    entries = read_ledger(path)
    replay_ledger(entries)
    return {"index": len(entries) - 1, "sha256": entries[-1]["sha256"] if entries else "0" * 64}


def verify_ledger(path: str | Path, *, protocol: dict | None = None,
                  anchors: Iterable[dict] = ()) -> dict:
    issues = []
    state = None
    try:
        entries = read_ledger(path)
        state = replay_ledger(entries, protocol)
        for anchor in anchors:
            index = anchor["index"]
            if index >= len(entries) or entries[index]["sha256"] != anchor["sha256"]:
                issues.append(f"ledger anchor {index} missing or changed (truncation or rewrite)")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(str(exc))
    return {"valid": not issues, "issues": issues,
            "entries": state.count if state else None,
            "consumed": sorted(state.consumed) if state else None,
            "fresh": sorted(d for d in state.fresh if d not in state.consumed) if state else None}


def initialize_ledger(path: str | Path, protocol: dict, root: str | Path, *,
                      consumed_registry: dict) -> list[dict]:
    """Import prior inspection history, declare fresh holdouts, then freeze the protocol."""
    path = Path(path)
    if path.exists():
        raise ProtocolError("ledger already exists; it is append-only")
    written = []
    for period in consumed_registry["consumed_periods"]:
        written.append(append_event(path, "import_consumed", {
            "dataset_id": period["independent_period_id"], "evidence": period.get("evidence"),
            "source_sha256": period.get("source_sha256")}))
    for dataset in protocol["datasets"]:
        if dataset["freshness_at_freeze"] == "consumed":
            written.append(append_event(path, "import_consumed", {
                "dataset_id": dataset["id"], "evidence": dataset["consumed_evidence"],
                "period_key": period_key(dataset)}))
        else:
            written.append(append_event(path, "declare_fresh", {
                "dataset_id": dataset["id"], "role": dataset["role"], "period_key": period_key(dataset),
                "identity_sha256": dataset_identity_sha256(dataset)}))
    written.append(append_event(path, "freeze", {
        "protocol_sha256": protocol_sha256(protocol),
        "referenced_configs": referenced_config_hashes(protocol, root)}, protocol=protocol))
    return written


# ----------------------------------------------------------------------------- binding


def bind_evidence(*, protocol: dict, ledger_path: str | Path, analysis: str,
                  dataset_ids: Iterable[str], config: Any, result: Any,
                  provenance: dict | None = None, extra: dict | None = None) -> dict:
    """Bind a result to the frozen registration and exact dataset identities."""
    entries = read_ledger(ledger_path)
    state = replay_ledger(entries, protocol)
    if state.frozen is None:
        raise ProtocolError("evidence cannot bind before protocol freeze")
    datasets = {}
    for dataset_id in sorted(set(dataset_ids)):
        declaration = dataset_declaration(protocol, dataset_id)
        sources = {}
        for access in state.accesses.get(dataset_id, []):
            sources.update(access.get("source_sha256", {}))
        datasets[dataset_id] = {"identity_sha256": dataset_identity_sha256(declaration, sources),
                                "freshness": state.freshness(dataset_id), "role": declaration["role"],
                                "source_sha256": sources}
    if provenance is None:
        from .artifacts import portable_provenance
        provenance = portable_provenance()
    return {"schema": BINDING_SCHEMA, "analysis": analysis,
            "protocol_sha256": protocol_sha256(protocol),
            "ledger_anchor": {"index": state.count - 1, "sha256": state.head},
            "datasets": datasets, "config_sha256": document_sha256(config),
            "result_sha256": document_sha256(result),
            "source_sha256": provenance.get("source_sha256"), "git_commit": provenance.get("git_commit"),
            "git_dirty": provenance.get("git_dirty"), "extra": extra or {}}


def verify_binding(binding: dict, *, protocol: dict, ledger_path: str | Path, config: Any,
                   result: Any, source_sha256: str | None = None,
                   expected_commit: str | None = None) -> dict:
    issues = []
    try:
        if binding.get("schema") != BINDING_SCHEMA:
            raise ProtocolError("unsupported binding schema")
        if binding["protocol_sha256"] != protocol_sha256(protocol):
            issues.append("protocol changed since binding")
        checked = verify_ledger(ledger_path, protocol=protocol, anchors=[binding["ledger_anchor"]])
        issues.extend(checked["issues"])
        entries = read_ledger(ledger_path)
        for dataset_id, bound in binding["datasets"].items():
            declaration = dataset_declaration(protocol, dataset_id)
            # Only accesses up to the anchor are relevant to the bound result.
            anchored = {}
            for entry in entries[: binding["ledger_anchor"]["index"] + 1]:
                if entry["event"] == "access" and entry["payload"].get("dataset_id") == dataset_id:
                    anchored.update(entry["payload"].get("source_sha256", {}))
            if dataset_identity_sha256(declaration, anchored) != bound["identity_sha256"]:
                issues.append(f"dataset identity changed: {dataset_id}")
            if bound.get("source_sha256") != anchored:
                issues.append(f"dataset source hashes differ from ledger: {dataset_id}")
        if binding["config_sha256"] != document_sha256(config):
            issues.append("config changed since binding")
        if binding["result_sha256"] != document_sha256(result):
            issues.append("result changed since binding")
        if source_sha256 is not None and binding.get("source_sha256") != source_sha256:
            issues.append("implementation source differs from bound source")
        if expected_commit is not None and binding.get("git_commit") != expected_commit:
            issues.append("bound git commit differs from expected commit")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(str(exc))
    return {"valid": not issues, "issues": issues,
            "meaning": "registration/identity/byte consistency; not independent scientific validation"}


def verify_protocol_files(root: str | Path = ".", protocol_path: str = "configs/v05/protocol.json",
                          ledger_path: str = "configs/v05/consumption-ledger.jsonl") -> dict:
    """Check the committed protocol against its ledger freeze entry and referenced configs."""
    root = Path(root)
    issues = []
    state = None
    try:
        protocol = load_protocol(root / protocol_path)
        state = replay_ledger(read_ledger(root / ledger_path), protocol)
        if state.frozen is None:
            issues.append("protocol is not frozen in the ledger")
        else:
            current = referenced_config_hashes(protocol, root)
            for name, digest in state.frozen["referenced_configs"].items():
                if current.get(name) != digest:
                    issues.append(f"referenced config changed after freeze: {name}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(str(exc))
    return {"valid": not issues, "issues": issues,
            "protocol_sha256": state.frozen["protocol_sha256"] if state and state.frozen else None,
            "ledger_entries": state.count if state else None,
            "consumed": sorted(state.consumed) if state else None,
            "fresh": sorted(d for d in state.fresh if d not in state.consumed) if state else None}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "status"))
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    result = verify_protocol_files(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
