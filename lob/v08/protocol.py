"""v0.8 frozen protocol, append-only consumption ledger, seed namespaces and evidence binding (M1).

The v0.8 ledger (``configs/v08/consumption-ledger.jsonl``) is a hash chain of
JSON entries. Each entry records index, UTC time, event, dataset, role, design,
reason, git commit and dirty flag, the payload, the previous entry hash and its
own hash. Events:

* ``import_consumed`` — prior (v0.5-v0.7) consumption, imported before the freeze;
* ``declare_fresh`` — a never-accessed holdout period, before the freeze;
* ``freeze`` — protocol hash plus hashes of every referenced config;
* ``seal_design`` — a study design (hash, datasets it reads, seed namespaces it opens);
* ``open_namespace`` — first use of a confirmatory seed namespace (``confirm``) or of
  the sealed synthetic holdout (``holdout``, which can be opened exactly once);
* ``access`` — staged dataset access (downloaded, opened, parsed, evaluated);
* ``inspect`` — first look at an outcome;
* ``attempt`` — any FAILED / ABORTED / INVALID / NOT_AVAILABLE attempt, retained;
* ``amend`` — a protocol amendment (new hash; may not affect accessed holdouts).

``replay_ledger`` enforces the rules; ``verify_protocol_files`` additionally checks
that referenced configs are unchanged since the freeze and that the pinned v0.7
history (protocol hash, ledger length and head) is intact. The chain detects
mutation, truncation (with anchors) and reordering; it is not an external
timestamp authority.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

from ..config import canonical_json
from ..preregistration import ProtocolError, document_sha256, period_key, text_sha256
from .taxonomy import STATUSES

SCHEMA = "cleolob-v08-protocol-1"
LEDGER_SCHEMA = "cleolob-v08-ledger-1"
BINDING_SCHEMA = "cleolob-v08-binding-1"
CONFIG_DIR = "configs/v08"
PROTOCOL_PATH = f"{CONFIG_DIR}/protocol.json"
LEDGER_PATH = f"{CONFIG_DIR}/consumption-ledger.jsonl"
V07_PROTOCOL_PATH = "configs/v07/protocol.json"
V07_LEDGER_PATH = "configs/v07/consumption-ledger.jsonl"

ROLE_USES = {
    "development": frozenset({"develop"}),
    "calibration": frozenset({"calibrate"}),
    "selection": frozenset({"select"}),
    "retrospective": frozenset({"retrospective"}),
    "fresh_temporal": frozenset({"download", "evaluate", "evaluate_posthoc"}),
    "fresh_cross_instrument": frozenset({"download", "evaluate", "evaluate_posthoc"}),
    "fresh_cross_venue": frozenset({"download", "evaluate", "evaluate_posthoc"}),
    "final_holdout": frozenset({"download", "evaluate", "evaluate_posthoc"}),
}
HOLDOUT_ROLES = frozenset({"fresh_temporal", "fresh_cross_instrument", "fresh_cross_venue", "final_holdout"})
STAGES = ("downloaded", "opened", "parsed", "evaluated")
EVENTS = ("import_consumed", "declare_fresh", "freeze", "seal_design", "open_namespace", "access", "inspect",
          "attempt", "amend")
NAMESPACES = ("dev", "confirm", "holdout")
NAMESPACE_IDS = {"dev": 0, "confirm": 1, "holdout": 2}
ATTEMPT_OUTCOMES = frozenset(STATUSES) | {"COMPLETED", "ABORTED"}
REQUIRED_CONFIGS = ("hypotheses.json", "statistical-families.json", "dataset-registry.json", "compute-budget.json",
                    "claim-registry.json")
REQUIRED_SECTIONS = ("schema", "protocol_id", "frozen_at_utc", "base_commit", "research_question", "statuses",
                     "history", "datasets", "periods", "holdout_rules", "acquisition_rule", "seed_namespaces",
                     "experiment_families", "statistics", "resource_budgets", "failure_policies", "abstention_rules",
                     "completion_constraints", "evidence_schemas", "generator_gate", "mandate", "policies", "worlds",
                     "amendment_policy", "referenced_configs", "exclusions")
_IDENTITY = ("id", "provider", "venue", "instrument", "date", "data_types", "capability_level", "role")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9.-]{2,80}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_state(root: Path) -> dict:
    """HEAD and dirty flag at append time; the ledger file itself is excluded from the dirty check."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL, timeout=5).strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "--", ".", f":(exclude){LEDGER_PATH}"],
                                         cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status.strip())}


def portable(name: str) -> str:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name or name.startswith("/")
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ProtocolError(f"unsafe repository-relative path: {name!r}")
    return name


# ============================================================================= protocol document


def validate_protocol(protocol: dict, root: Path | None = None) -> dict:
    if not isinstance(protocol, dict) or protocol.get("schema") != SCHEMA:
        raise ProtocolError("unsupported v0.8 protocol schema")
    missing = [k for k in REQUIRED_SECTIONS if k not in protocol]
    if missing:
        raise ProtocolError(f"protocol missing required sections: {missing}")
    if tuple(protocol["statuses"]) != STATUSES:
        raise ProtocolError("protocol must use exactly the nine v0.8 statuses")
    if not re.fullmatch("[0-9a-f]{40}", str(protocol["base_commit"])):
        raise ProtocolError("base_commit must be a full git SHA")
    for key in ("v07_protocol_sha256", "v07_ledger_entries", "v07_ledger_head_sha256"):
        if key not in protocol["history"]:
            raise ProtocolError(f"history missing {key}")
    ids = set()
    for d in protocol["datasets"]:
        for key in _IDENTITY + ("freshness_at_freeze", "allowed_claims"):
            if key not in d:
                raise ProtocolError(f"dataset declaration missing {key}")
        if not _SAFE_ID.fullmatch(d["id"]) or d["id"] in ids:
            raise ProtocolError(f"invalid or duplicate dataset id: {d['id']}")
        ids.add(d["id"])
        if d["role"] not in ROLE_USES:
            raise ProtocolError(f"unknown dataset role: {d['role']}")
        if (d["role"] in HOLDOUT_ROLES) != (d["freshness_at_freeze"] == "fresh"):
            raise ProtocolError(f"{d['id']}: holdout roles require fresh data and only holdouts are fresh")
        if period_key(d) is None:
            raise ProtocolError(f"{d['id']}: date must be YYYY-MM-DD")
    for role in ("development", "calibration", "selection"):
        if sum(d["role"] == role for d in protocol["datasets"]) != 1:
            raise ProtocolError(f"protocol must declare exactly one {role} dataset")
    if tuple(protocol["seed_namespaces"]["names"]) != NAMESPACES:
        raise ProtocolError("seed namespaces must be exactly dev, confirm, holdout")
    names = [portable(n) for n in protocol["referenced_configs"]]
    for required in REQUIRED_CONFIGS:
        if f"{CONFIG_DIR}/{required}" not in names:
            raise ProtocolError(f"referenced_configs must include {CONFIG_DIR}/{required}")
    if root is not None:
        validate_configs(protocol, root)
    return protocol


def validate_configs(protocol: dict, root: Path) -> None:
    hypotheses = json.loads((root / CONFIG_DIR / "hypotheses.json").read_text(encoding="utf-8"))
    families = json.loads((root / CONFIG_DIR / "statistical-families.json").read_text(encoding="utf-8"))["families"]
    datasets = {d["id"] for d in protocol["datasets"]}
    needed = ("question", "domain", "estimator", "endpoint", "sample_unit", "family", "threshold", "status_rule",
              "evidence_level", "datasets", "failure_semantics")
    for hid, h in hypotheses["hypotheses"].items():
        if not re.fullmatch(r"H\d{1,2}", hid):
            raise ProtocolError(f"hypothesis ids must look like H1..H99: {hid}")
        absent = [k for k in needed if k not in h]
        if absent:
            raise ProtocolError(f"{hid} missing {absent}")
        if h["family"] not in families:
            raise ProtocolError(f"{hid} references unregistered family {h['family']}")
        unknown = set(h["datasets"]) - datasets
        if unknown:
            raise ProtocolError(f"{hid} references undeclared datasets {sorted(unknown)}")
    for k in range(1, 22):
        if f"H{k}" not in hypotheses["hypotheses"]:
            raise ProtocolError(f"core hypothesis H{k} is not registered")
    for name, fam in families.items():
        if fam["correction"] not in ("holm", "bonferroni", "benjamini_hochberg", "none"):
            raise ProtocolError(f"family {name}: unknown correction {fam['correction']}")
        if not 0 < fam["alpha"] < 0.5:
            raise ProtocolError(f"family {name}: alpha out of range")


def load_protocol(path: str | Path, root: Path | None = None) -> dict:
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8")), root)


def protocol_sha256(protocol: dict) -> str:
    return document_sha256(protocol)


def dataset_declaration(protocol: dict, dataset_id: str) -> dict:
    for d in protocol["datasets"]:
        if d["id"] == dataset_id:
            return d
    raise ProtocolError(f"dataset {dataset_id!r} is not declared in the v0.8 protocol")


def datasets_by_role(protocol: dict, role: str) -> list[str]:
    return [d["id"] for d in protocol["datasets"] if d["role"] == role]


def dataset_identity_sha256(declaration: dict, source_hashes: dict[str, str] | None = None) -> str:
    identity = {k: declaration[k] for k in _IDENTITY}
    identity["source_sha256"] = dict(sorted((source_hashes or {}).items()))
    return document_sha256(identity)


def referenced_config_hashes(protocol: dict, root: Path) -> dict[str, str]:
    return {n: text_sha256(root / portable(n)) for n in protocol["referenced_configs"]}


# ============================================================================= ledger


def entry_hash(entry: dict) -> str:
    return document_sha256({k: v for k, v in entry.items() if k != "sha256"})


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
    def __init__(self) -> None:
        self.head = "0" * 64
        self.count = 0
        self.consumed: dict[str, int] = {}
        self.consumed_keys: set[str] = set()
        self.fresh: dict[str, int] = {}
        self.first_access: dict[str, int] = {}
        self.stages: dict[str, list[str]] = {}
        self.inspected: dict[str, int] = {}
        self.accesses: dict[str, list[dict]] = {}
        self.frozen: dict | None = None
        self.protocol_sha256: str | None = None
        self.designs: dict[str, dict] = {}
        self.namespaces: dict[str, list[dict]] = {}
        self.attempts: list[dict] = []
        self.amendments: list[dict] = []

    def freshness(self, dataset_id: str) -> str:
        if dataset_id in self.consumed:
            return "consumed"
        return "fresh" if dataset_id in self.fresh else "undeclared"

    def _fail(self, message: str) -> None:
        raise ProtocolError(f"ledger entry {self.count}: {message}")

    def apply(self, entry: dict, protocol: dict | None = None) -> None:
        if not isinstance(entry, dict) or entry.get("schema") != LEDGER_SCHEMA or entry.get("index") != self.count:
            self._fail("schema/index mismatch (truncated, reordered or foreign entry?)")
        if entry.get("prev_sha256") != self.head:
            self._fail("broken hash chain")
        if entry.get("sha256") != entry_hash(entry):
            self._fail("content hash mismatch")
        for key in ("at", "event", "reason", "git_commit"):
            if key not in entry:
                self._fail(f"missing required field {key}")
        event, payload = entry["event"], entry.get("payload")
        if event not in EVENTS or not isinstance(payload, dict):
            self._fail(f"unknown event {event!r} or malformed payload")
        if not entry["reason"]:
            self._fail("every entry must state a reason")
        getattr(self, f"_on_{event}")(entry, payload, protocol)
        self.head = entry["sha256"]
        self.count += 1

    def _on_import_consumed(self, entry, payload, protocol) -> None:
        if self.frozen is not None:
            self._fail("prior consumption must be imported before the freeze")
        dataset = entry.get("dataset")
        if not dataset or not payload.get("evidence"):
            self._fail("import_consumed requires dataset and evidence")
        self.consumed.setdefault(dataset, self.count)
        key = payload.get("period_key") or period_key(dataset)
        if key:
            self.consumed_keys.add(str(key).lower())

    def _on_declare_fresh(self, entry, payload, protocol) -> None:
        if self.frozen is not None:
            self._fail("fresh holdouts must be declared before the freeze")
        dataset, key, role = entry.get("dataset"), payload.get("period_key"), entry.get("role")
        if not dataset or not key or not _DIGEST.fullmatch(str(payload.get("identity_sha256"))):
            self._fail("declare_fresh requires dataset, period_key and identity hash")
        if role not in HOLDOUT_ROLES:
            self._fail(f"{dataset}: only holdout roles can be declared fresh")
        if dataset in self.consumed or str(key).lower() in self.consumed_keys:
            self._fail(f"{dataset} is consumed; freshness cannot be restored")
        if dataset in self.fresh:
            self._fail(f"{dataset} already declared fresh")
        if protocol is not None:
            d = dataset_declaration(protocol, dataset)
            if d["role"] != role or period_key(d) != str(key).lower():
                self._fail(f"{dataset}: fresh declaration differs from the protocol")
        self.fresh[dataset] = self.count

    def _on_freeze(self, entry, payload, protocol) -> None:
        if self.frozen is not None:
            self._fail("protocol already frozen; use an amendment")
        if not _DIGEST.fullmatch(str(payload.get("protocol_sha256"))):
            self._fail("freeze requires a protocol hash")
        self.frozen, self.protocol_sha256 = payload, payload["protocol_sha256"]

    def _on_seal_design(self, entry, payload, protocol) -> None:
        if self.frozen is None:
            self._fail("designs may only be sealed after the protocol freeze")
        name = entry.get("design")
        if not name or name in self.designs or not _DIGEST.fullmatch(str(payload.get("design_sha256"))):
            self._fail("design seal requires a unique design name and design hash")
        reads, namespaces = payload.get("reads", []), payload.get("namespaces", [])
        posthoc = payload.get("posthoc", False)
        if not isinstance(reads, list) or not isinstance(posthoc, bool) or not isinstance(namespaces, list):
            self._fail("design reads/namespaces must be lists and posthoc a boolean")
        if set(namespaces) - set(NAMESPACES):
            self._fail(f"unknown seed namespaces {namespaces}")
        for read in reads:
            if protocol is not None:
                dataset_declaration(protocol, read)
            if read in self.first_access and not posthoc:
                self._fail(f"design {name!r} reads consumed holdout {read}; it must be declared posthoc")
        if "holdout" in namespaces and "holdout" in self.namespaces and not posthoc:
            self._fail("the synthetic holdout namespace was already opened; a new reader must be posthoc")
        self.designs[name] = {**payload, "posthoc": posthoc}

    def _on_open_namespace(self, entry, payload, protocol) -> None:
        if self.frozen is None:
            self._fail("seed namespaces open only after the freeze")
        ns, design = payload.get("namespace"), entry.get("design")
        if ns not in ("confirm", "holdout"):
            self._fail("only the confirm and holdout namespaces are opened through the ledger")
        sealed = self.designs.get(design)
        if sealed is None or ns not in sealed.get("namespaces", []):
            self._fail(f"namespace {ns} opened by {design!r} without a sealed design listing it")
        if ns == "holdout" and self.namespaces.get("holdout") and not sealed["posthoc"]:
            self._fail("the synthetic holdout namespace can be opened only once (later readers are posthoc)")
        self.namespaces.setdefault(ns, []).append({"design": design, "index": self.count})

    def _holdout_check(self, dataset: str, design_name: str, use: str, protocol) -> None:
        role = dataset_declaration(protocol, dataset)["role"] if protocol is not None else None
        if role in HOLDOUT_ROLES or dataset in self.fresh:
            design = self.designs.get(design_name)
            if design is None or dataset not in design.get("reads", []):
                self._fail(f"{dataset}: holdout access by {design_name!r} without a sealed design that reads it")
            if use != "download" and design["posthoc"] != (use == "evaluate_posthoc"):
                self._fail(f"{dataset}: posthoc designs must use evaluate_posthoc and only they may")

    def _on_access(self, entry, payload, protocol) -> None:
        dataset, use, design, stage = entry.get("dataset"), payload.get("use"), entry.get("design"), payload.get("stage")
        if not dataset or not use or not design:
            self._fail("access requires dataset, use and design")
        if stage not in STAGES:
            self._fail(f"access stage must be one of {STAGES}")
        if self.frozen is None:
            self._fail("no data access before the protocol freeze")
        if protocol is not None:
            role = dataset_declaration(protocol, dataset)["role"]
            if use not in ROLE_USES[role]:
                self._fail(f"{dataset}: use {use!r} is not permitted for role {role}")
            if entry.get("role") != role:
                self._fail(f"{dataset}: entry role {entry.get('role')!r} differs from the protocol role {role}")
        self._holdout_check(dataset, design, use, protocol)
        for name, digest in payload.get("source_sha256", {}).items():
            if not _DIGEST.fullmatch(str(digest)):
                self._fail("access source hashes must be SHA-256")
            for prior in self.accesses.get(dataset, []):
                if name in prior.get("source_sha256", {}) and prior["source_sha256"][name] != digest:
                    self._fail(f"{dataset}/{name}: source bytes changed since first access")
        self.accesses.setdefault(dataset, []).append(payload)
        self.stages.setdefault(dataset, []).append(stage)
        self.first_access.setdefault(dataset, self.count)
        self.consumed.setdefault(dataset, self.count)

    def _on_inspect(self, entry, payload, protocol) -> None:
        dataset = entry.get("dataset")
        if dataset not in self.first_access and not str(dataset).startswith("namespace:"):
            self._fail(f"{dataset}: outcome inspection recorded before any access")
        if not payload.get("run"):
            self._fail("inspect requires the run directory whose outcome was inspected")
        self.inspected.setdefault(dataset, self.count)

    def _on_attempt(self, entry, payload, protocol) -> None:
        if not entry.get("design") or payload.get("outcome") not in ATTEMPT_OUTCOMES:
            self._fail("attempt requires a design and a registered outcome status")
        if "directory" not in payload or not payload.get("note"):
            self._fail("attempt requires the retained directory (or null with a reason) and a note")
        if payload["directory"] is None and not payload.get("no_directory_reason"):
            self._fail("an attempt without a directory must say why none exists")
        self.attempts.append({**payload, "design": entry.get("design"), "dataset": entry.get("dataset"),
                              "index": self.count})

    def _on_amend(self, entry, payload, protocol) -> None:
        if self.frozen is None:
            self._fail("amendments require a frozen protocol")
        if not _DIGEST.fullmatch(str(payload.get("protocol_sha256"))):
            self._fail("amendment requires the new protocol hash")
        late = [d for d in payload.get("affects", []) if d in self.first_access]
        if late:
            self._fail(f"amendment affects already accessed holdouts: {late}")
        if "holdout" in payload.get("affects_namespaces", []) and self.namespaces.get("holdout"):
            self._fail("amendment affects the already opened synthetic holdout")
        self.amendments.append({**payload, "index": self.count})
        self.protocol_sha256 = payload["protocol_sha256"]


def replay_ledger(entries: Iterable[dict], protocol: dict | None = None) -> LedgerState:
    state = LedgerState()
    for entry in entries:
        state.apply(entry, protocol)
    if protocol is not None and state.frozen is not None and state.protocol_sha256 != protocol_sha256(protocol):
        raise ProtocolError("protocol changed after freeze (hash differs from the ledger's current protocol hash)")
    return state


class _LedgerLock:
    """Exclusive lock file next to the ledger so concurrent processes never interleave appends."""

    def __init__(self, path: Path, timeout: float = 120.0) -> None:
        self.lock = path.with_name(path.name + ".lock")
        self.timeout = timeout

    def __enter__(self):
        import os
        import time
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except (FileExistsError, PermissionError):
                if time.monotonic() > deadline:
                    raise ProtocolError(f"ledger lock {self.lock.name} held for more than {self.timeout} s") from None
                time.sleep(0.05)

    def __exit__(self, *exc) -> None:
        import os
        import time
        os.close(self.fd)
        for _ in range(100):
            try:
                self.lock.unlink(missing_ok=True)
                return
            except PermissionError:
                time.sleep(0.01)


def append_event(path: str | Path, event: str, *, reason: str, payload: dict | None = None, dataset: str | None = None,
                 role: str | None = None, design: str | None = None, protocol: dict | None = None,
                 root: Path | None = None, at: str | None = None) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LedgerLock(path):
        state = replay_ledger(read_ledger(path), protocol)
        entry = {"schema": LEDGER_SCHEMA, "index": state.count, "at": at or _utc_now(), "event": event,
                 "dataset": dataset, "role": role, "design": design, "reason": reason,
                 **git_state(root or path.resolve().parents[2]),
                 "payload": json.loads(canonical_json(payload or {})), "prev_sha256": state.head}
        entry["sha256"] = entry_hash(entry)
        state.apply(entry, protocol)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(entry) + "\n")
    return entry


def amend_protocol(root: Path, document: dict, *, number: int, reason: str, affects: list[str],
                   configs: list[str] | None = None) -> dict:
    """Append the amendment (validated against the current protocol) and only then write the new protocol."""
    current = load_protocol(root / PROTOCOL_PATH)
    validate_protocol(document, root)
    entry = append_event(root / LEDGER_PATH, "amend", reason=reason, protocol=current, root=root,
                         payload={"protocol_sha256": protocol_sha256(document), "affects": affects, "number": number,
                                  "configs": configs or [], "config_sha256": {
                                      n: text_sha256(root / portable(n)) for n in (configs or [])}})
    (root / PROTOCOL_PATH).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                      newline="\n")
    return entry


# ============================================================================= history and initialization


def history_state(root: Path) -> dict:
    """Read-only state of the v0.7 records that v0.8 pins and imports (v0.7 pins v0.5/v0.6 in turn)."""
    from ..v07.protocol import core as v07
    p7 = v07.load_protocol(root / V07_PROTOCOL_PATH)
    e7 = v07.read_ledger(root / V07_LEDGER_PATH)
    s7 = v07.replay_ledger(e7, p7)
    return {"v07_protocol_sha256": v07.protocol_sha256(p7), "v07_ledger_entries": len(e7),
            "v07_ledger_head_sha256": s7.head, "v07_frozen_protocol_sha256": s7.protocol_sha256,
            "consumed": sorted(d for d in s7.consumed if not d.startswith("period:")),
            "consumed_period_keys": sorted(s7.consumed_keys)}


def initialize_ledger(root: Path) -> list[dict]:
    """Import v0.5-v0.7 consumption, declare v0.8 fresh holdouts, then freeze."""
    path = root / LEDGER_PATH
    if path.exists():
        raise ProtocolError("v0.8 ledger already exists; it is append-only")
    protocol = load_protocol(root / PROTOCOL_PATH, root)
    history = history_state(root)
    for key in ("v07_protocol_sha256", "v07_ledger_entries", "v07_ledger_head_sha256"):
        if history[key] != protocol["history"][key]:
            raise ProtocolError(f"pinned history differs: {key}")
    evidence = f"{V07_LEDGER_PATH}@{history['v07_ledger_entries'] - 1}:{history['v07_ledger_head_sha256']}"
    written = []
    for dataset in history["consumed"]:
        written.append(append_event(path, "import_consumed", dataset=dataset, reason="consumed in v0.5-v0.7",
                                    payload={"evidence": evidence, "period_key": period_key(dataset)}, root=root))
    for key in history["consumed_period_keys"]:
        written.append(append_event(path, "import_consumed", dataset=f"period:{key}", reason="consumed calendar period",
                                    payload={"evidence": evidence, "period_key": key}, root=root))
    imported = set(history["consumed"])
    for d in protocol["datasets"]:
        if d["freshness_at_freeze"] == "fresh":
            written.append(append_event(path, "declare_fresh", dataset=d["id"], role=d["role"],
                                        reason="never accessed by any CleoLOB version (chronology audit)",
                                        payload={"period_key": period_key(d),
                                                 "identity_sha256": dataset_identity_sha256(d)},
                                        protocol=protocol, root=root))
        elif d["id"] not in imported:
            raise ProtocolError(f"{d['id']}: declared consumed but absent from the v0.7 ledger")
    written.append(append_event(path, "freeze", reason="v0.8 preregistration freeze", protocol=protocol, root=root,
                                payload={"protocol_sha256": protocol_sha256(protocol),
                                         "referenced_configs": referenced_config_hashes(protocol, root)}))
    return written


def verify_protocol_files(root: Path = Path(".")) -> dict:
    root = Path(root)
    issues: list[str] = []
    protocol = state = None
    try:
        protocol = load_protocol(root / PROTOCOL_PATH, root)
        state = replay_ledger(read_ledger(root / LEDGER_PATH), protocol)
        if state.frozen is None:
            issues.append("protocol is not frozen in the ledger")
        else:
            current = referenced_config_hashes(protocol, root)
            amended: dict[str, str] = {}
            for a in state.amendments:
                amended.update(a.get("config_sha256", {}))
            for name, digest in state.frozen["referenced_configs"].items():
                expected = amended.get(name, digest)
                if current.get(name) != expected:
                    issues.append(f"referenced config changed after freeze without amendment: {name}")
        history = history_state(root)
        for key in ("v07_protocol_sha256", "v07_ledger_entries", "v07_ledger_head_sha256"):
            if history[key] != protocol["history"][key]:
                issues.append(f"pinned history changed: {key} (v0.7 records must stay immutable)")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(str(exc))
    return status_report(protocol, state, issues)


def status_report(protocol: dict | None, state: LedgerState | None, issues: list[str]) -> dict:
    holdouts = {}
    if protocol is not None and state is not None:
        for d in protocol["datasets"]:
            if d["role"] in HOLDOUT_ROLES:
                holdouts[d["id"]] = {"role": d["role"], "freshness": state.freshness(d["id"]),
                                     "first_access_index": state.first_access.get(d["id"]),
                                     "stages": state.stages.get(d["id"], []),
                                     "sealed_readers": sorted(n for n, x in state.designs.items()
                                                              if d["id"] in x.get("reads", []))}
    return {"valid": not issues, "issues": issues,
            "protocol_sha256": state.protocol_sha256 if state else None,
            "ledger_entries": state.count if state else None, "ledger_head_sha256": state.head if state else None,
            "fresh_unread": sorted(d for d in state.fresh if d not in state.consumed) if state else None,
            "holdouts": holdouts, "sealed_designs": sorted(state.designs) if state else None,
            "namespaces_opened": {k: [x["design"] for x in v] for k, v in state.namespaces.items()} if state else None,
            "attempts": [{k: a.get(k) for k in ("design", "outcome", "directory", "dataset", "index")}
                         for a in state.attempts] if state else None,
            "amendments": len(state.amendments) if state else None}


# ============================================================================= designs and seeds


def seal_design(root: Path, name: str, design: dict, *, reads: list[str] | None = None,
                namespaces: list[str] | None = None, posthoc: bool = False, reason: str = "") -> dict:
    """Seal a study design before it runs; the design document is stored under configs/v08/designs."""
    protocol = load_protocol(root / PROTOCOL_PATH)
    target = root / CONFIG_DIR / "designs" / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json(design)
    if target.exists() and target.read_text(encoding="utf-8").strip() != body:
        raise ProtocolError(f"design file {target.name} exists with different content")
    target.write_text(body + "\n", encoding="utf-8", newline="\n")
    return append_event(root / LEDGER_PATH, "seal_design", design=name, reason=reason or f"seal {name}",
                        protocol=protocol, root=root,
                        payload={"design_sha256": document_sha256(design), "reads": reads or [],
                                 "namespaces": namespaces or [], "posthoc": posthoc,
                                 "file": f"{CONFIG_DIR}/designs/{name}.json"})


def sealed_design(root: Path, name: str) -> dict:
    """The sealed design document, verified against its ledger hash."""
    protocol = load_protocol(root / PROTOCOL_PATH)
    state = replay_ledger(read_ledger(root / LEDGER_PATH), protocol)
    if name not in state.designs:
        raise ProtocolError(f"design {name!r} is not sealed")
    document = json.loads((root / CONFIG_DIR / "designs" / f"{name}.json").read_text(encoding="utf-8"))
    if document_sha256(document) != state.designs[name]["design_sha256"]:
        raise ProtocolError(f"design {name!r} changed after sealing")
    return document


def open_namespace(root: Path, namespace: str, *, design: str, reason: str) -> dict:
    """Record the first (or posthoc) use of the confirm/holdout seed namespace by a sealed design."""
    protocol = load_protocol(root / PROTOCOL_PATH)
    state = replay_ledger(read_ledger(root / LEDGER_PATH), protocol)
    if namespace == "confirm" and any(x["design"] == design for x in state.namespaces.get("confirm", [])):
        return {"already_open": True}
    return append_event(root / LEDGER_PATH, "open_namespace", design=design, reason=reason, protocol=protocol,
                        root=root, payload={"namespace": namespace})


def namespace_guard(namespace: str, *, root: Path | None, design: str | None) -> None:
    """Seeds from ``confirm``/``holdout`` require a frozen protocol and a design that opened the namespace."""
    if namespace not in NAMESPACES:
        raise ProtocolError(f"unknown seed namespace {namespace!r}")
    if namespace == "dev":
        return
    if root is None or design is None:
        raise ProtocolError(f"seed namespace {namespace!r} requires a sealed design and the repository root")
    protocol = load_protocol(root / PROTOCOL_PATH)
    state = replay_ledger(read_ledger(root / LEDGER_PATH), protocol)
    if not any(x["design"] == design for x in state.namespaces.get(namespace, [])):
        raise ProtocolError(f"design {design!r} has not opened seed namespace {namespace!r}")


def seeds(namespace: str, study: str, count: int, *, root: Path | None = None, design: str | None = None) -> list[int]:
    """Seed-addressable integer seeds: ``SeedSequence([namespace id, hash(study), index])``; never schedule-dependent."""
    import numpy as np
    namespace_guard(namespace, root=root, design=design)
    tag = int(document_sha256(study)[:12], 16)
    out = []
    for i in range(count):
        ss = np.random.SeedSequence([NAMESPACE_IDS[namespace], tag, i])
        out.append(int(ss.generate_state(1, dtype=np.uint32)[0]))
    return out


# ============================================================================= evidence binding


def bind_evidence(*, protocol: dict, ledger_path: Path, analysis: str, dataset_ids: Iterable[str], config: Any,
                  result: Any, provenance: dict, extra: dict | None = None) -> dict:
    state = replay_ledger(read_ledger(ledger_path), protocol)
    datasets = {}
    for dataset_id in sorted(set(dataset_ids)):
        d = dataset_declaration(protocol, dataset_id)
        sources: dict[str, str] = {}
        for access in state.accesses.get(dataset_id, []):
            sources.update(access.get("source_sha256", {}))
        datasets[dataset_id] = {"identity_sha256": dataset_identity_sha256(d, sources), "role": d["role"],
                                "freshness": state.freshness(dataset_id), "source_sha256": sources}
    return {"schema": BINDING_SCHEMA, "analysis": analysis, "protocol_sha256": protocol_sha256(protocol),
            "frozen": state.frozen is not None,
            "ledger_anchor": {"index": state.count - 1, "sha256": state.head}, "datasets": datasets,
            "config_sha256": document_sha256(config), "result_sha256": document_sha256(result),
            "source_sha256": provenance.get("source_sha256"), "git_commit": provenance.get("git_commit"),
            "git_dirty": provenance.get("git_dirty"), "extra": extra or {}}


def verify_binding(binding: dict, *, protocol: dict, ledger_path: Path, config: Any, result: Any) -> list[str]:
    issues = []
    if binding.get("schema") != BINDING_SCHEMA:
        return ["unsupported binding schema"]
    entries = read_ledger(ledger_path)
    replay_ledger(entries, protocol)
    anchor = binding["ledger_anchor"]
    if anchor["index"] >= 0:
        if anchor["index"] >= len(entries) or entries[anchor["index"]]["sha256"] != anchor["sha256"]:
            issues.append("ledger anchor missing or changed (truncation or rewrite)")
        else:
            at_anchor = replay_ledger(entries[: anchor["index"] + 1])
            if at_anchor.protocol_sha256 is not None and binding["protocol_sha256"] != at_anchor.protocol_sha256:
                issues.append("bound protocol hash is not the ledger's protocol at the anchor")
    for dataset_id, bound in binding["datasets"].items():
        d = dataset_declaration(protocol, dataset_id)
        anchored: dict[str, str] = {}
        for e in entries[: anchor["index"] + 1]:
            if e["event"] == "access" and e.get("dataset") == dataset_id:
                anchored.update(e["payload"].get("source_sha256", {}))
        if dataset_identity_sha256(d, anchored) != bound["identity_sha256"]:
            issues.append(f"dataset identity changed: {dataset_id}")
    if binding["config_sha256"] != document_sha256(config):
        issues.append("config changed since binding")
    if binding["result_sha256"] != document_sha256(result):
        issues.append("result changed since binding")
    return issues
