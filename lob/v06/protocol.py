"""v0.6 protocol freeze, append-only consumption ledger and evidence binding.

The v0.6 protocol (``configs/v06/protocol.json``) is identified by its canonical
SHA-256 and frozen by a ledger entry. Dataset freshness is never a mutable
field: it is replayed from the hash-chained ledger ``configs/v06/consumption-ledger.jsonl``.

Rules enforced by replay (any violation makes the ledger invalid):

* v0.5 history is imported, never rewritten: the protocol pins the v0.5 ledger
  anchor and every dataset consumed in v0.5 (or v0.4) is imported as consumed.
  A consumed dataset or calendar period can never be declared fresh again.
* Every access names a declared dataset, an analysis and a *use*; the use must
  be permitted for the dataset's role (development data cannot be "evaluated" as
  a holdout, selection data is only used to select, retrospective data only for
  labelled retrospective analyses, holdouts only to download and evaluate).
* A holdout may be accessed only by an analysis whose design was sealed before
  that holdout's first access. After first access a holdout is consumed: a new
  design that reads it must be declared ``posthoc`` and its accesses must use
  ``evaluate_posthoc``; it can never again yield registered evidence.
* Source bytes may not change after first access. Failed or invalid attempts
  are recorded as ``attempt`` events and are never removed.
* Amendments carry the new protocol hash and may not affect consumed datasets.

The hash chain detects mutation, truncation (through anchors stored in
evidence) and reordering. It is not an external timestamp authority and cannot
prove that nobody looked at data outside this workflow.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable

from ..config import canonical_json
from ..preregistration import (ProtocolError, document_sha256, file_sha256, period_key, read_ledger as _read_v05_ledger,
                               replay_ledger as _replay_v05_ledger, load_protocol as _load_v05_protocol, text_sha256)

SCHEMA = "cleolob-v06-protocol-1"
LEDGER_SCHEMA = "cleolob-v06-ledger-1"
BINDING_SCHEMA = "cleolob-v06-binding-1"
PROTOCOL_PATH = "configs/v06/protocol.json"
LEDGER_PATH = "configs/v06/consumption-ledger.jsonl"
DATASETS_PATH = "configs/v06/datasets.json"
V05_PROTOCOL_PATH = "configs/v05/protocol.json"
V05_LEDGER_PATH = "configs/v05/consumption-ledger.jsonl"

STATUSES = ("ESTABLISHED", "NOT_ESTABLISHED", "FAILED", "INVALID", "NOT_AVAILABLE", "INCONCLUSIVE",
            "EXPLORATORY", "ASSUMPTION_DEPENDENT")
EQUIVALENCE_STATUSES = ("EQUIVALENT_WITHIN_MARGIN", "NOT_ESTABLISHED", "FAILED_MARGIN", "NOT_EVALUABLE")
ROLE_USES = {
    "development": frozenset({"develop"}),
    "selection": frozenset({"select"}),
    "retrospective": frozenset({"retrospective"}),
    "fresh_external": frozenset({"download", "evaluate", "evaluate_posthoc"}),
    "cross_instrument_external": frozenset({"download", "evaluate", "evaluate_posthoc"}),
    "transfer_holdout": frozenset({"download", "evaluate", "evaluate_posthoc"}),
}
HOLDOUT_ROLES = frozenset({"fresh_external", "cross_instrument_external", "transfer_holdout"})
EVENTS = ("import_consumed", "declare_fresh", "freeze", "seal_design", "access", "attempt", "amend")
_IDENTITY_FIELDS = ("id", "provider", "venue", "instrument", "date", "data_types", "capability_level", "role")
_REQUIRED = ("schema", "protocol_id", "frozen_at_utc", "base_commit", "statuses", "equivalence_statuses",
             "research_question", "v05_history", "datasets", "holdout_rules", "realism", "calibration",
             "identifiability", "ensemble", "misspecification", "execution", "transfer", "domain_gap",
             "regimes", "hypotheses", "statistics", "seeds", "compute_budget", "invalidation",
             "referenced_configs", "exclusions")
_HYPOTHESIS_FIELDS = ("question", "estimator", "datasets", "sample_unit", "uncertainty", "family", "threshold",
                      "failure_semantics", "status_rule")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9.-]{2,80}")
_ATTEMPT_OUTCOMES = frozenset(STATUSES) | {"COMPLETED", "ABORTED"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _portable(name: str) -> str:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name or name.startswith("/")
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ProtocolError(f"unsafe repository-relative path: {name!r}")
    return name


# ----------------------------------------------------------------------------- protocol


def validate_protocol(protocol: dict) -> dict:
    """Structural and semantic validation; returns the protocol unchanged."""
    if not isinstance(protocol, dict) or protocol.get("schema") != SCHEMA:
        raise ProtocolError("unsupported v0.6 protocol schema")
    missing = [key for key in _REQUIRED if key not in protocol]
    if missing:
        raise ProtocolError(f"protocol missing required sections: {missing}")
    if tuple(protocol["statuses"]) != STATUSES:
        raise ProtocolError("protocol must use exactly the eight v0.6 evidence statuses")
    if tuple(protocol["equivalence_statuses"]) != EQUIVALENCE_STATUSES:
        raise ProtocolError("protocol must use exactly the four equivalence statuses")
    if not re.fullmatch("[0-9a-f]{40}", str(protocol["base_commit"])):
        raise ProtocolError("base_commit must be a full git SHA")
    history = protocol["v05_history"]
    for key in ("protocol_sha256", "ledger_entries", "ledger_head_sha256"):
        if key not in history:
            raise ProtocolError(f"v05_history missing {key}")
    if not _DIGEST.fullmatch(str(history["protocol_sha256"])) or not _DIGEST.fullmatch(str(history["ledger_head_sha256"])):
        raise ProtocolError("v05_history hashes must be SHA-256")
    ids, roles = set(), []
    for dataset in protocol["datasets"]:
        for key in _IDENTITY_FIELDS + ("freshness_at_freeze", "prior_use"):
            if key not in dataset:
                raise ProtocolError(f"dataset declaration missing {key}")
        if not _SAFE_ID.fullmatch(dataset["id"]) or dataset["id"] in ids:
            raise ProtocolError(f"invalid or duplicate dataset id: {dataset['id']}")
        ids.add(dataset["id"])
        if dataset["role"] not in ROLE_USES:
            raise ProtocolError(f"unknown dataset role: {dataset['role']}")
        if dataset["freshness_at_freeze"] not in {"fresh", "consumed"}:
            raise ProtocolError("freshness_at_freeze must be fresh or consumed")
        if (dataset["role"] in HOLDOUT_ROLES) != (dataset["freshness_at_freeze"] == "fresh"):
            raise ProtocolError(f"{dataset['id']}: holdout roles require fresh data and only holdouts are fresh")
        if not isinstance(dataset["data_types"], list) or not dataset["data_types"]:
            raise ProtocolError("dataset data_types must be a nonempty list")
        if period_key(dataset) is None:
            raise ProtocolError(f"{dataset['id']}: date must be YYYY-MM-DD")
        roles.append(dataset["role"])
    for role in ("development", "selection"):
        if roles.count(role) != 1:
            raise ProtocolError(f"protocol must declare exactly one {role} dataset")
    for name, hypothesis in protocol["hypotheses"].items():
        if not re.fullmatch(r"H\d{1,2}", name):
            raise ProtocolError(f"hypothesis ids must look like H1..H99: {name}")
        absent = [key for key in _HYPOTHESIS_FIELDS if key not in hypothesis]
        if absent:
            raise ProtocolError(f"{name} missing {absent}")
        for dataset_id in hypothesis["datasets"]:
            if dataset_id not in ids:
                raise ProtocolError(f"{name} references undeclared dataset {dataset_id}")
        if hypothesis["family"] not in protocol["statistics"]["families"]:
            raise ProtocolError(f"{name} references unknown multiplicity family {hypothesis['family']}")
    for name, family in protocol["statistics"]["families"].items():
        for key in ("size", "correction", "alpha", "adjusted_alpha", "kind"):
            if key not in family:
                raise ProtocolError(f"family {name} missing {key}")
        if family["kind"] not in {"confirmatory", "exploratory", "descriptive"}:
            raise ProtocolError(f"family {name}: kind must be confirmatory, exploratory or descriptive")
        if family["kind"] == "confirmatory" and family["correction"] not in {"bonferroni", "holm"}:
            raise ProtocolError(f"family {name}: confirmatory families require Bonferroni or Holm")
        size = family["size"]
        if isinstance(size, int):
            if size < 1:
                raise ProtocolError(f"family {name}: size must be positive")
            if family["correction"] == "bonferroni" and abs(family["adjusted_alpha"] - family["alpha"] / size) > 1e-12:
                raise ProtocolError(f"family {name}: adjusted alpha must equal alpha / size")
        elif not (isinstance(size, str) and size.startswith("sealed:")):
            raise ProtocolError(f"family {name}: size must be an integer or 'sealed:<rule>'")
    if not isinstance(protocol["referenced_configs"], list):
        raise ProtocolError("referenced_configs must list repository-relative paths")
    for name in protocol["referenced_configs"]:
        _portable(name)
    return protocol


def load_protocol(path: str | Path) -> dict:
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8")))


def protocol_sha256(protocol: dict) -> str:
    return document_sha256(validate_protocol(protocol))


def dataset_declaration(protocol: dict, dataset_id: str) -> dict:
    for dataset in protocol["datasets"]:
        if dataset["id"] == dataset_id:
            return dataset
    raise ProtocolError(f"dataset {dataset_id!r} is not declared in the v0.6 protocol")


def datasets_by_role(protocol: dict, role: str) -> list[str]:
    return [d["id"] for d in protocol["datasets"] if d["role"] == role]


def dataset_identity_sha256(declaration: dict, source_hashes: dict[str, str] | None = None) -> str:
    identity = {key: declaration[key] for key in _IDENTITY_FIELDS}
    identity["source_sha256"] = dict(sorted((source_hashes or {}).items()))
    return document_sha256(identity)


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
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"ledger line {number} is malformed JSON: {exc}") from exc
    return entries


class LedgerState:
    """Replay of the v0.6 rules; the only source of dataset freshness."""

    def __init__(self) -> None:
        self.head = "0" * 64
        self.count = 0
        self.consumed: dict[str, int] = {}
        self.consumed_keys: set[str] = set()
        self.fresh: dict[str, int] = {}
        self.first_access: dict[str, int] = {}
        self.accesses: dict[str, list[dict]] = {}
        self.frozen: dict | None = None
        self.protocol_sha256: str | None = None
        self.designs: dict[str, dict] = {}
        self.design_index: dict[str, int] = {}
        self.attempts: list[dict] = []
        self.amendments: list[dict] = []

    def freshness(self, dataset_id: str) -> str:
        if dataset_id in self.consumed:
            return "consumed"
        if dataset_id in self.fresh:
            return "fresh"
        return "undeclared"

    def _fail(self, message: str) -> None:
        raise ProtocolError(f"ledger entry {self.count}: {message}")

    def apply(self, entry: dict, protocol: dict | None = None) -> None:
        if not isinstance(entry, dict) or entry.get("schema") != LEDGER_SCHEMA or entry.get("index") != self.count:
            self._fail("schema/index mismatch (truncated, reordered or foreign entry?)")
        if entry.get("prev_sha256") != self.head:
            self._fail("broken hash chain")
        if entry.get("sha256") != _entry_hash(entry):
            self._fail("content hash mismatch")
        if not isinstance(entry.get("at"), str):
            self._fail("missing timestamp")
        event, payload = entry.get("event"), entry.get("payload")
        if event not in EVENTS or not isinstance(payload, dict):
            self._fail(f"unknown event {event!r} or malformed payload")
        getattr(self, f"_on_{event}")(payload, protocol)
        self.head = entry["sha256"]
        self.count += 1

    def _on_import_consumed(self, payload: dict, protocol: dict | None) -> None:
        if self.frozen is not None:
            self._fail("prior consumption must be imported before the freeze")
        dataset = payload.get("dataset_id")
        if not dataset or not payload.get("evidence"):
            self._fail("import_consumed requires dataset_id and evidence")
        self.consumed.setdefault(dataset, self.count)
        key = payload.get("period_key") or period_key(dataset)
        if key:
            self.consumed_keys.add(str(key).lower())

    def _on_declare_fresh(self, payload: dict, protocol: dict | None) -> None:
        if self.frozen is not None:
            self._fail("fresh holdouts must be declared before the freeze")
        dataset, key = payload.get("dataset_id"), payload.get("period_key")
        if not dataset or not key or not _DIGEST.fullmatch(str(payload.get("identity_sha256"))):
            self._fail("declare_fresh requires dataset_id, period_key and identity hash")
        if payload.get("role") not in HOLDOUT_ROLES:
            self._fail(f"{dataset}: only holdout roles can be declared fresh")
        if dataset in self.consumed or str(key).lower() in self.consumed_keys:
            self._fail(f"{dataset} is consumed; freshness cannot be restored")
        if dataset in self.fresh:
            self._fail(f"{dataset} already declared fresh")
        if protocol is not None:
            declaration = dataset_declaration(protocol, dataset)
            if declaration["role"] != payload["role"] or period_key(declaration) != str(key).lower():
                self._fail(f"{dataset}: fresh declaration differs from the protocol")
        self.fresh[dataset] = self.count

    def _on_freeze(self, payload: dict, protocol: dict | None) -> None:
        if self.frozen is not None:
            self._fail("protocol already frozen; use an amendment")
        if not _DIGEST.fullmatch(str(payload.get("protocol_sha256"))):
            self._fail("freeze requires a protocol hash")
        self.frozen = payload
        self.protocol_sha256 = payload["protocol_sha256"]

    def _on_seal_design(self, payload: dict, protocol: dict | None) -> None:
        if self.frozen is None:
            self._fail("analysis designs may only be sealed after the protocol freeze")
        name = payload.get("analysis")
        if not name or name in self.designs or not _DIGEST.fullmatch(str(payload.get("design_sha256"))):
            self._fail("design seal requires a unique analysis name and design hash")
        reads = payload.get("reads", [])
        if not isinstance(reads, list):
            self._fail("design reads must be a list")
        posthoc = bool(payload.get("posthoc", False))
        for read in reads:
            if protocol is not None:
                dataset_declaration(protocol, read)
            if read in self.first_access and not posthoc:
                self._fail(f"design {name!r} reads consumed holdout {read}; it must be declared posthoc")
        self.designs[name] = payload
        self.design_index[name] = self.count

    def _on_access(self, payload: dict, protocol: dict | None) -> None:
        dataset, use, analysis = payload.get("dataset_id"), payload.get("use"), payload.get("analysis")
        if not dataset or not use or not analysis:
            self._fail("access requires dataset_id, use and analysis")
        if self.frozen is None:
            self._fail("no data access before the protocol freeze")
        role = None
        if protocol is not None:
            role = dataset_declaration(protocol, dataset)["role"]
            if use not in ROLE_USES[role]:
                self._fail(f"{dataset}: use {use!r} is not permitted for role {role}")
        if role in HOLDOUT_ROLES or dataset in self.fresh:
            design = self.designs.get(analysis)
            if design is None or dataset not in design.get("reads", []):
                self._fail(f"{dataset}: holdout access by {analysis!r} without a sealed design that reads it")
            if bool(design.get("posthoc", False)) != (use == "evaluate_posthoc") and use != "download":
                self._fail(f"{dataset}: posthoc designs must use evaluate_posthoc and only they may")
        for name, digest in payload.get("source_sha256", {}).items():
            if not _DIGEST.fullmatch(str(digest)):
                self._fail("access source hashes must be SHA-256")
            for prior in self.accesses.get(dataset, []):
                if name in prior.get("source_sha256", {}) and prior["source_sha256"][name] != digest:
                    self._fail(f"{dataset}/{name}: source bytes changed since first access")
        self.accesses.setdefault(dataset, []).append(payload)
        self.first_access.setdefault(dataset, self.count)
        self.consumed.setdefault(dataset, self.count)
        if protocol is not None:
            self.consumed_keys.add(period_key(dataset_declaration(protocol, dataset)))

    def _on_attempt(self, payload: dict, protocol: dict | None) -> None:
        if not payload.get("analysis") or payload.get("outcome") not in _ATTEMPT_OUTCOMES:
            self._fail("attempt requires analysis and a registered outcome status")
        if not payload.get("directory") or not payload.get("note"):
            self._fail("attempt requires the retained directory and a note")
        self.attempts.append(payload)

    def _on_amend(self, payload: dict, protocol: dict | None) -> None:
        if self.frozen is None:
            self._fail("amendments require a frozen protocol")
        if not _DIGEST.fullmatch(str(payload.get("protocol_sha256"))) or not payload.get("reason"):
            self._fail("amendment requires the new protocol hash and a reason")
        late = [d for d in payload.get("affects", []) if d in self.first_access]
        if late:
            self._fail(f"amendment affects already accessed holdouts: {late}")
        self.amendments.append(payload)
        self.protocol_sha256 = payload["protocol_sha256"]


def replay_ledger(entries: Iterable[dict], protocol: dict | None = None) -> LedgerState:
    state = LedgerState()
    for entry in entries:
        state.apply(entry, protocol)
    if protocol is not None and state.frozen is not None and state.protocol_sha256 != protocol_sha256(protocol):
        raise ProtocolError("protocol changed after freeze (hash differs from the ledger's current protocol hash)")
    return state


def append_event(path: str | Path, event: str, payload: dict, *, protocol: dict | None = None,
                 at: str | None = None) -> dict:
    """Validate the complete history plus the new entry, then append."""
    path = Path(path)
    state = replay_ledger(read_ledger(path), protocol)
    entry = {"schema": LEDGER_SCHEMA, "index": state.count, "event": event,
             "payload": json.loads(canonical_json(payload)), "at": at or _utc_now(), "prev_sha256": state.head}
    entry["sha256"] = _entry_hash(entry)
    state.apply(entry, protocol)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(entry) + "\n")
    return entry


def amend_protocol(root: str | Path, document: dict, *, number: int, reason: str, affects: list[str]) -> dict:
    """Append the amendment (validated against the current protocol) and then write the new protocol."""
    root = Path(root)
    current = load_protocol(root / PROTOCOL_PATH)
    validate_protocol(document)
    entry = append_event(root / LEDGER_PATH, "amend", {"protocol_sha256": protocol_sha256(document), "reason": reason,
                                                       "affects": affects, "number": number}, protocol=current)
    (root / PROTOCOL_PATH).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                      newline="\n")
    return entry


def ledger_anchor(path: str | Path) -> dict:
    entries = read_ledger(path)
    replay_ledger(entries)
    return {"index": len(entries) - 1, "sha256": entries[-1]["sha256"] if entries else "0" * 64}


def v05_history(root: str | Path) -> dict:
    """Current v0.5 protocol hash, ledger head and consumed periods (read-only)."""
    root = Path(root)
    protocol = _load_v05_protocol(root / V05_PROTOCOL_PATH)
    entries = _read_v05_ledger(root / V05_LEDGER_PATH)
    state = _replay_v05_ledger(entries, protocol)
    from ..preregistration import protocol_sha256 as v05_protocol_sha256
    return {"protocol_sha256": v05_protocol_sha256(protocol), "ledger_entries": len(entries),
            "ledger_head_sha256": state.head, "consumed": sorted(state.consumed),
            "consumed_period_keys": sorted(state.consumed_keys)}


def initialize_ledger(root: str | Path, *, protocol_path: str = PROTOCOL_PATH, ledger_path: str = LEDGER_PATH) -> list[dict]:
    """Import v0.5 consumption, declare the fresh v0.6 holdouts, then freeze the protocol."""
    root = Path(root)
    path = root / ledger_path
    if path.exists():
        raise ProtocolError("v0.6 ledger already exists; it is append-only")
    protocol = load_protocol(root / protocol_path)
    history = v05_history(root)
    pinned = protocol["v05_history"]
    if (history["protocol_sha256"], history["ledger_head_sha256"], history["ledger_entries"]) != (
            pinned["protocol_sha256"], pinned["ledger_head_sha256"], pinned["ledger_entries"]):
        raise ProtocolError("v0.5 protocol or ledger differs from the pinned v0.5 history")
    evidence = f"{V05_LEDGER_PATH}@{history['ledger_entries'] - 1}:{history['ledger_head_sha256']}"
    written = []
    declared = {d["id"]: d for d in protocol["datasets"]}
    for dataset_id in history["consumed"]:
        key = period_key(declared[dataset_id]) if dataset_id in declared else period_key(dataset_id)
        written.append(append_event(path, "import_consumed", {"dataset_id": dataset_id, "evidence": evidence,
                                                              "period_key": key}))
    for key in history["consumed_period_keys"]:
        written.append(append_event(path, "import_consumed", {"dataset_id": f"period:{key}", "evidence": evidence,
                                                              "period_key": key}))
    for dataset in protocol["datasets"]:
        if dataset["freshness_at_freeze"] == "fresh":
            written.append(append_event(path, "declare_fresh", {
                "dataset_id": dataset["id"], "role": dataset["role"], "period_key": period_key(dataset),
                "identity_sha256": dataset_identity_sha256(dataset)}, protocol=protocol))
        elif dataset["id"] not in history["consumed"]:
            raise ProtocolError(f"{dataset['id']}: declared consumed but absent from the v0.5 ledger")
    written.append(append_event(path, "freeze", {"protocol_sha256": protocol_sha256(protocol),
                                                 "referenced_configs": referenced_config_hashes(protocol, root)},
                                protocol=protocol))
    return written


def verify_protocol_files(root: str | Path = ".", protocol_path: str = PROTOCOL_PATH,
                          ledger_path: str = LEDGER_PATH) -> dict:
    """Protocol hash, ledger chain and rules, referenced configs and pinned v0.5 history."""
    root = Path(root)
    issues: list[str] = []
    state = None
    protocol = None
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
        history = v05_history(root)
        pinned = protocol["v05_history"]
        if history["protocol_sha256"] != pinned["protocol_sha256"]:
            issues.append("v0.5 protocol changed (historical record must stay immutable)")
        v05_entries = _read_v05_ledger(root / V05_LEDGER_PATH)
        index = pinned["ledger_entries"] - 1
        if index >= len(v05_entries) or v05_entries[index]["sha256"] != pinned["ledger_head_sha256"]:
            issues.append("pinned v0.5 ledger anchor missing or rewritten")
        if len(v05_entries) != pinned["ledger_entries"]:
            issues.append("v0.5 ledger gained or lost entries after the v0.6 freeze")
        for dataset in protocol["datasets"]:
            if dataset["freshness_at_freeze"] == "consumed" and state.freshness(dataset["id"]) != "consumed":
                issues.append(f"{dataset['id']}: declared consumed but not consumed in the ledger")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(str(exc))
    return status_report(protocol, state, issues)


def status_report(protocol: dict | None, state: LedgerState | None, issues: list[str]) -> dict:
    holdouts = {}
    if protocol is not None and state is not None:
        for dataset in protocol["datasets"]:
            if dataset["role"] in HOLDOUT_ROLES:
                first = state.first_access.get(dataset["id"])
                holdouts[dataset["id"]] = {"role": dataset["role"], "freshness": state.freshness(dataset["id"]),
                                           "first_access_index": first,
                                           "sealed_readers": sorted(n for n, d in state.designs.items()
                                                                    if dataset["id"] in d.get("reads", []))}
    return {"valid": not issues, "issues": issues,
            "protocol_sha256": state.protocol_sha256 if state else None,
            "ledger_entries": state.count if state else None,
            "ledger_head_sha256": state.head if state else None,
            "fresh": sorted(d for d in state.fresh if d not in state.consumed) if state else None,
            "holdouts": holdouts,
            "sealed_designs": sorted(state.designs) if state else None,
            "attempts": [{k: a.get(k) for k in ("analysis", "outcome", "directory")} for a in state.attempts]
            if state else None,
            "amendments": len(state.amendments) if state else None}


# ----------------------------------------------------------------------------- binding


def bind_evidence(*, protocol: dict, ledger_path: str | Path, analysis: str, dataset_ids: Iterable[str],
                  config: Any, result: Any, provenance: dict, extra: dict | None = None) -> dict:
    """Bind a result to the frozen v0.6 registration and exact dataset identities."""
    state = replay_ledger(read_ledger(ledger_path), protocol)
    if state.frozen is None:
        raise ProtocolError("evidence cannot bind before the protocol freeze")
    datasets = {}
    for dataset_id in sorted(set(dataset_ids)):
        declaration = dataset_declaration(protocol, dataset_id)
        sources: dict[str, str] = {}
        for access in state.accesses.get(dataset_id, []):
            sources.update(access.get("source_sha256", {}))
        datasets[dataset_id] = {"identity_sha256": dataset_identity_sha256(declaration, sources),
                                "freshness": state.freshness(dataset_id), "role": declaration["role"],
                                "source_sha256": sources}
    return {"schema": BINDING_SCHEMA, "analysis": analysis, "protocol_sha256": protocol_sha256(protocol),
            "ledger_anchor": {"index": state.count - 1, "sha256": state.head}, "datasets": datasets,
            "config_sha256": document_sha256(config), "result_sha256": document_sha256(result),
            "source_sha256": provenance.get("source_sha256"), "git_commit": provenance.get("git_commit"),
            "git_dirty": provenance.get("git_dirty"),
            "package_version": provenance.get("runtime", {}).get("packages", {}).get("cleolob"),
            "extra": extra or {}}


def verify_binding(binding: dict, *, protocol: dict, ledger_path: str | Path, config: Any, result: Any,
                   source_sha256: str | None = None) -> dict:
    issues: list[str] = []
    try:
        if binding.get("schema") != BINDING_SCHEMA:
            raise ProtocolError("unsupported binding schema")
        entries = read_ledger(ledger_path)
        replay_ledger(entries, protocol)
        anchor = binding["ledger_anchor"]
        if anchor["index"] >= len(entries) or entries[anchor["index"]]["sha256"] != anchor["sha256"]:
            issues.append("ledger anchor missing or changed (truncation or rewrite)")
        else:
            # The bound protocol must be the ledger's protocol at the anchor; later amendments are
            # recorded in the ledger and reported, not treated as tampering.
            at_anchor = replay_ledger(entries[: anchor["index"] + 1])
            if binding["protocol_sha256"] != at_anchor.protocol_sha256:
                issues.append("bound protocol hash is not the ledger's protocol at the anchor")
            if binding["protocol_sha256"] != protocol_sha256(protocol):
                amended = [e["payload"].get("number") for e in entries[anchor["index"] + 1:] if e["event"] == "amend"]
                if not amended:
                    issues.append("protocol changed since binding without a ledger amendment")
        for dataset_id, bound in binding["datasets"].items():
            declaration = dataset_declaration(protocol, dataset_id)
            anchored: dict[str, str] = {}
            for entry in entries[: anchor["index"] + 1]:
                if entry["event"] == "access" and entry["payload"].get("dataset_id") == dataset_id:
                    anchored.update(entry["payload"].get("source_sha256", {}))
            if dataset_identity_sha256(declaration, anchored) != bound["identity_sha256"]:
                issues.append(f"dataset identity changed: {dataset_id}")
            if bound.get("role") != declaration["role"]:
                issues.append(f"dataset role changed: {dataset_id}")
        if binding["config_sha256"] != document_sha256(config):
            issues.append("config changed since binding")
        if binding["result_sha256"] != document_sha256(result):
            issues.append("result changed since binding")
        if source_sha256 is not None and binding.get("source_sha256") != source_sha256:
            issues.append("implementation source differs from bound source")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(str(exc))
    return {"valid": not issues, "issues": issues,
            "meaning": "byte integrity plus registration/identity binding; not independent scientific replication"}


__all__ = ["ProtocolError", "STATUSES", "EQUIVALENCE_STATUSES", "ROLE_USES", "HOLDOUT_ROLES", "validate_protocol",
           "load_protocol", "protocol_sha256", "dataset_declaration", "datasets_by_role", "read_ledger",
           "replay_ledger", "append_event", "ledger_anchor", "initialize_ledger", "verify_protocol_files",
           "bind_evidence", "verify_binding", "document_sha256", "file_sha256", "text_sha256", "period_key"]
