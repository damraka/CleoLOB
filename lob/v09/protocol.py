"""v0.9 frozen protocol, consumption ledger v2, seed namespaces v2 and evidence binding (M1, M3, M4, M6).

The ledger ``configs/v09/consumption-ledger.jsonl`` is an append-only hash chain. Every entry carries index, UTC
time, event, dataset, role, design, reason, git commit and dirty flag, payload, previous hash and own hash.

Events and the firewall they enforce (M3):

* ``import_consumed`` / ``declare_fresh`` - prior consumption and never-accessed data, before the freeze;
* ``freeze`` - protocol hash plus hashes of every referenced config (candidate hypotheses included);
* ``seal_design`` - a study design (hash, datasets it reads, seed namespaces it opens, posthoc flag);
* ``open_namespace`` - first use of a guarded seed namespace (``confirm``, ``holdout``, ``final``);
* ``access`` - staged dataset access (downloaded, opened, parsed, evaluated) with role and use;
* ``promote`` - the frozen confirmatory registry (exactly once; later changes only by amendment). No
  confirmatory dataset or ``confirm`` seed may be touched before it;
* ``close_confirmatory`` - every confirmatory run sealed; the final vault opens only after it;
* ``inspect`` / ``attempt`` / ``amend`` / ``incident`` - outcome inspection, retained failed attempts, registered
  amendments and the scientific-incident record (M50 of the brief; never erased).

Exploratory and development data are readable after the freeze by any design; confirmatory data only by a sealed
design after promotion; the final vault only by the one sealed final design after ``close_confirmatory``. The
chain detects mutation, truncation (with anchors) and reordering; it is not an external timestamp authority.
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

SCHEMA = "cleolob-v09-protocol-1"
LEDGER_SCHEMA = "cleolob-v09-ledger-2"
BINDING_SCHEMA = "cleolob-v09-binding-1"
CONFIG_DIR = "configs/v09"
PROTOCOL_PATH = f"{CONFIG_DIR}/protocol.json"
LEDGER_PATH = f"{CONFIG_DIR}/consumption-ledger.jsonl"
INCIDENTS_PATH = f"{CONFIG_DIR}/incidents.jsonl"
CANDIDATES_PATH = f"{CONFIG_DIR}/hypotheses-candidates.json"
CONFIRMATORY_PATH = f"{CONFIG_DIR}/hypotheses-confirmatory.json"
V08_PROTOCOL_PATH = "configs/v08/protocol.json"
V08_LEDGER_PATH = "configs/v08/consumption-ledger.jsonl"

ENGINEERING = ("PLANNED", "IMPLEMENTED", "TESTED", "VERIFIED", "COMPLETE", "BLOCKED")
DATA_STATUSES = ("AVAILABLE", "NOT_AVAILABLE", "CAPABILITY_MISSING", "QUALITY_REJECTED")
LIFECYCLE = ("CANDIDATE", "FEASIBILITY_PASS", "FEASIBILITY_FAIL", "PROMOTED", "NOT_PROMOTED", "SEALED")
TERMINAL = ("ESTABLISHED", "FAILED", "NOT_ESTABLISHED", "NOT_AVAILABLE", "NOT_EVALUABLE", "INCONCLUSIVE")
PROMOTION_OUTCOMES = ("PROMOTED", "NOT_PROMOTED_INSUFFICIENT_POWER", "NOT_PROMOTED_DATA_CAPABILITY",
                      "NOT_PROMOTED_MEASUREMENT", "NOT_PROMOTED_COMPUTE", "NOT_PROMOTED_EFFECT_TOO_SMALL",
                      "NOT_PROMOTED_OTHER_REGISTERED_REASON")
REALISM = ("ACCEPTABLE", "PARTIALLY_ACCEPTABLE", "REJECTED", "NO_ACCEPTABLE_MODEL")
HISTORICAL_ID = ("IDENTIFIED", "PARTIALLY_IDENTIFIED", "INDETERMINATE", "NOT_IDENTIFIED")
CAUSAL_ID = ("IDENTIFIED", "ASSUMPTION_DEPENDENT", "NOT_IDENTIFIED")
EVIDENCE_LEVELS = tuple(f"E{k}" for k in range(11))

ROLE_USES = {
    "development": frozenset({"develop"}),
    "explore": frozenset({"explore"}),
    "confirm": frozenset({"confirm", "confirm_posthoc"}),
    "confirm_late": frozenset({"confirm", "confirm_posthoc"}),
    "final": frozenset({"final", "final_posthoc"}),
}
CONFIRM_ROLES = frozenset({"confirm", "confirm_late"})
FINAL_ROLES = frozenset({"final"})
GUARDED_ROLES = CONFIRM_ROLES | FINAL_ROLES
STAGES = ("downloaded", "opened", "parsed", "evaluated")
EVENTS = ("import_consumed", "declare_fresh", "freeze", "seal_design", "open_namespace", "access", "promote",
          "close_confirmatory", "inspect", "attempt", "amend", "incident")
NAMESPACES = ("dev", "explore", "pilot", "confirm", "holdout", "final")
NAMESPACE_IDS = {name: i for i, name in enumerate(NAMESPACES)}
GUARDED_NAMESPACES = ("confirm", "holdout", "final")
ATTEMPT_OUTCOMES = frozenset(TERMINAL) | {"INVALID", "ABORTED", "COMPLETED", "RESOURCE_EXHAUSTED"}
INCIDENT_KINDS = ("accidental_holdout_access", "wrong_result_pointer", "wrong_analysis_input", "contamination",
                  "posthoc_change", "data_provider_failure", "engineering_bug", "resource_abort",
                  "invalid_checkpoint", "code_provenance_mismatch", "other")
REQUIRED_CONFIGS = ("hypotheses-candidates.json", "statistical-families.json", "dataset-registry.json",
                    "capability-matrix.json", "model-capabilities.json", "compute-budget.json",
                    "claim-registry.json")
REQUIRED_SECTIONS = ("schema", "protocol_id", "frozen_at_utc", "base_commit", "research_question", "statuses",
                     "history", "datasets", "pools", "holdout_rules", "acquisition_rules", "seed_namespaces",
                     "promotion_rules", "power_rules", "statistics", "practical_thresholds",
                     "completion_constraints", "risk_measures", "mandates", "policies", "conclusions",
                     "realism_gate", "counterfactual_assumptions", "causal_assumptions", "resource_ceilings",
                     "failure_semantics", "evidence_schemas", "evidence_levels", "fidelities",
                     "amendment_policy", "referenced_configs", "exclusions")
_IDENTITY = ("id", "provider", "venue", "instrument", "date", "data_types", "role")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9.-]{2,96}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_state(root: Path) -> dict:
    """HEAD and dirty flag at append time; ledger files themselves are excluded from the dirty check."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL, timeout=5).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", f":(exclude){LEDGER_PATH}", f":(exclude){INCIDENTS_PATH}"],
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
        raise ProtocolError("unsupported v0.9 protocol schema")
    missing = [k for k in REQUIRED_SECTIONS if k not in protocol]
    if missing:
        raise ProtocolError(f"protocol missing required sections: {missing}")
    statuses = protocol["statuses"]
    for key, expected in (("engineering", ENGINEERING), ("data", DATA_STATUSES), ("lifecycle", LIFECYCLE),
                          ("terminal", TERMINAL), ("promotion", PROMOTION_OUTCOMES), ("realism", REALISM),
                          ("historical_identification", HISTORICAL_ID), ("causal_identification", CAUSAL_ID)):
        if tuple(statuses.get(key, ())) != expected:
            raise ProtocolError(f"status vocabulary {key!r} must be exactly {expected}")
    if tuple(protocol["evidence_levels"]) != EVIDENCE_LEVELS:
        raise ProtocolError("evidence levels must be exactly E0..E10")
    if not re.fullmatch("[0-9a-f]{40}", str(protocol["base_commit"])):
        raise ProtocolError("base_commit must be a full git SHA")
    for key in ("v08_protocol_sha256", "v08_ledger_entries", "v08_ledger_head_sha256"):
        if key not in protocol["history"]:
            raise ProtocolError(f"history missing {key}")
    ids: set[str] = set()
    for d in protocol["datasets"]:
        for key in _IDENTITY + ("freshness_at_freeze", "pool", "slot", "sources"):
            if key not in d:
                raise ProtocolError(f"dataset declaration missing {key}: {d.get('id')}")
        if not _SAFE_ID.fullmatch(d["id"]) or d["id"] in ids:
            raise ProtocolError(f"invalid or duplicate dataset id: {d['id']}")
        ids.add(d["id"])
        if d["role"] not in ROLE_USES:
            raise ProtocolError(f"unknown dataset role: {d['role']}")
        if d["freshness_at_freeze"] != "fresh":
            raise ProtocolError(f"{d['id']}: every v0.9 dataset must be fresh at the freeze")
        if period_key(d) is None:
            raise ProtocolError(f"{d['id']}: date must be YYYY-MM-DD")
        if d["pool"] not in protocol["pools"]:
            raise ProtocolError(f"{d['id']}: unknown pool {d['pool']}")
        if protocol["pools"][d["pool"]]["role"] != d["role"]:
            raise ProtocolError(f"{d['id']}: role differs from its pool role")
        for name, src in d["sources"].items():
            if not str(src.get("url", "")).startswith("https://"):
                raise ProtocolError(f"{d['id']}/{name}: sources must be https URLs")
    if sum(d["role"] == "development" for d in protocol["datasets"]) < 1:
        raise ProtocolError("protocol must declare a development dataset")
    if tuple(protocol["seed_namespaces"]["names"]) != NAMESPACES:
        raise ProtocolError(f"seed namespaces must be exactly {NAMESPACES}")
    names = [portable(n) for n in protocol["referenced_configs"]]
    for required in REQUIRED_CONFIGS:
        if f"{CONFIG_DIR}/{required}" not in names:
            raise ProtocolError(f"referenced_configs must include {CONFIG_DIR}/{required}")
    if root is not None:
        from . import promotion
        promotion.validate_candidates(json.loads((root / CANDIDATES_PATH).read_text(encoding="utf-8")), protocol,
                                      json.loads((root / CONFIG_DIR / "statistical-families.json").read_text(
                                          encoding="utf-8")))
    return protocol


def load_protocol(path: str | Path, root: Path | None = None) -> dict:
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8")), root)


def protocol_sha256(protocol: dict) -> str:
    return document_sha256(protocol)


_INDEX: dict[int, tuple[dict, dict[str, dict]]] = {}


def dataset_declaration(protocol: dict, dataset_id: str) -> dict:
    cached = _INDEX.get(id(protocol))
    if cached is None or cached[0] is not protocol or len(cached[1]) != len(protocol["datasets"]):
        cached = (protocol, {d["id"]: d for d in protocol["datasets"]})
        if len(_INDEX) > 64:
            _INDEX.clear()
        _INDEX[id(protocol)] = cached
    try:
        return cached[1][dataset_id]
    except KeyError:
        raise ProtocolError(f"dataset {dataset_id!r} is not declared in the v0.9 protocol") from None


def datasets_by(protocol: dict, *, role: str | None = None, pool: str | None = None) -> list[dict]:
    return [d for d in protocol["datasets"] if (role is None or d["role"] == role)
            and (pool is None or d["pool"] == pool)]


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
        self.accesses: dict[str, list[dict]] = {}
        self.inspected: dict[str, int] = {}
        self.frozen: dict | None = None
        self.protocol_sha256: str | None = None
        self.designs: dict[str, dict] = {}
        self.namespaces: dict[str, list[dict]] = {}
        self.promotion: dict | None = None
        self.confirmatory_sha256: str | None = None
        self.closed: dict | None = None
        self.attempts: list[dict] = []
        self.amendments: list[dict] = []
        self.incidents: list[dict] = []
        self.final_reader: str | None = None

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

    # -- before the freeze
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
            self._fail("fresh data must be declared before the freeze")
        dataset, key, role = entry.get("dataset"), payload.get("period_key"), entry.get("role")
        if not dataset or not key or not _DIGEST.fullmatch(str(payload.get("identity_sha256"))):
            self._fail("declare_fresh requires dataset, period_key and identity hash")
        if role not in ROLE_USES:
            self._fail(f"{dataset}: unknown role {role}")
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

    # -- designs and namespaces
    def _role(self, protocol, dataset: str) -> str | None:
        return dataset_declaration(protocol, dataset)["role"] if protocol is not None else None

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
        roles = {self._role(protocol, r) for r in reads} if protocol is not None else set()
        if ("confirm" in namespaces or roles & CONFIRM_ROLES) and self.promotion is None:
            self._fail(f"design {name!r} touches confirmatory data or seeds before the promotion freeze")
        if ("final" in namespaces or roles & FINAL_ROLES) and self.closed is None:
            self._fail(f"design {name!r} touches the final vault before the confirmatory phase closed")
        for read in reads:
            if read in self.first_access and self._role(protocol, read) in GUARDED_ROLES and not posthoc:
                self._fail(f"design {name!r} reads consumed guarded data {read}; it must be declared posthoc")
        for ns in ("holdout", "final"):
            if ns in namespaces and self.namespaces.get(ns) and not posthoc:
                self._fail(f"the {ns} namespace was already opened; a new reader must be posthoc")
        self.designs[name] = {**payload, "posthoc": posthoc}

    def _on_open_namespace(self, entry, payload, protocol) -> None:
        if self.frozen is None:
            self._fail("seed namespaces open only after the freeze")
        ns, design = payload.get("namespace"), entry.get("design")
        if ns not in GUARDED_NAMESPACES:
            self._fail(f"only {GUARDED_NAMESPACES} are opened through the ledger")
        sealed = self.designs.get(design)
        if sealed is None or ns not in sealed.get("namespaces", []):
            self._fail(f"namespace {ns} opened by {design!r} without a sealed design listing it")
        if ns == "confirm" and self.promotion is None:
            self._fail("the confirm namespace opens only after the promotion freeze")
        if ns == "final" and self.closed is None:
            self._fail("the final namespace opens only after the confirmatory phase closed")
        if ns in ("holdout", "final") and self.namespaces.get(ns) and not sealed["posthoc"]:
            self._fail(f"the {ns} namespace can be opened only once (later readers are posthoc)")
        self.namespaces.setdefault(ns, []).append({"design": design, "index": self.count})

    # -- data access
    def _on_access(self, entry, payload, protocol) -> None:
        dataset, use, design, stage = entry.get("dataset"), payload.get("use"), entry.get("design"), payload.get("stage")
        if not dataset or not use or not design:
            self._fail("access requires dataset, use and design")
        if stage not in STAGES:
            self._fail(f"access stage must be one of {STAGES}")
        if self.frozen is None:
            self._fail("no data access before the protocol freeze")
        role = self._role(protocol, dataset)
        if protocol is not None:
            if use not in ROLE_USES[role]:
                self._fail(f"{dataset}: use {use!r} is not permitted for role {role}")
            if entry.get("role") != role:
                self._fail(f"{dataset}: entry role {entry.get('role')!r} differs from the protocol role {role}")
        else:
            role = entry.get("role")
        if role in GUARDED_ROLES:
            sealed = self.designs.get(design)
            if sealed is None or dataset not in sealed.get("reads", []):
                self._fail(f"{dataset}: guarded access by {design!r} without a sealed design that reads it")
            if use.endswith("_posthoc") != sealed["posthoc"]:
                self._fail(f"{dataset}: posthoc designs must use a *_posthoc use and only they may")
            if role in CONFIRM_ROLES and self.promotion is None:
                self._fail(f"{dataset}: confirmatory data accessed before the promotion freeze")
            if role in FINAL_ROLES:
                if self.closed is None:
                    self._fail(f"{dataset}: final vault accessed before the confirmatory phase closed")
                if not sealed["posthoc"]:
                    if self.final_reader not in (None, design):
                        self._fail(f"{dataset}: the final vault is read by exactly one registered design")
                    self.final_reader = design
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

    # -- promotion and closing
    def _on_promote(self, entry, payload, protocol) -> None:
        if self.frozen is None:
            self._fail("promotion requires a frozen protocol")
        if self.promotion is not None:
            self._fail("the confirmatory registry is promoted exactly once; later changes need an amendment")
        for key in ("confirmatory_sha256", "candidates_sha256", "promotion_run_result_sha256"):
            if not _DIGEST.fullmatch(str(payload.get(key))):
                self._fail(f"promote requires {key}")
        if not isinstance(payload.get("promoted"), list) or not isinstance(payload.get("decisions"), dict):
            self._fail("promote requires the promoted list and per-candidate decisions")
        guarded = [d for d in self.first_access if protocol is not None and self._role(protocol, d) in GUARDED_ROLES]
        if guarded:
            self._fail(f"guarded data accessed before promotion: {guarded[:3]}")
        self.promotion, self.confirmatory_sha256 = payload, payload["confirmatory_sha256"]

    def _on_close_confirmatory(self, entry, payload, protocol) -> None:
        if self.promotion is None:
            self._fail("the confirmatory phase closes only after promotion")
        if self.closed is not None:
            self._fail("the confirmatory phase is closed exactly once")
        if not isinstance(payload.get("runs"), dict):
            self._fail("close_confirmatory requires the sealed confirmatory runs (design -> run)")
        self.closed = payload

    # -- records
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
        late = [d for d in payload.get("affects", []) if d in self.first_access
                and (protocol is None or self._role(protocol, d) in GUARDED_ROLES)]
        if late:
            self._fail(f"amendment affects already accessed guarded data: {late}")
        if payload.get("affects_confirmatory"):
            if self.promotion is None:
                self._fail("an amendment to the confirmatory registry requires a promotion to amend")
            touched = [d for d in self.first_access if protocol is not None and self._role(protocol, d)
                       in CONFIRM_ROLES]
            if touched or self.namespaces.get("confirm"):
                self._fail("the confirmatory registry cannot be amended after confirmatory access")
            if not _DIGEST.fullmatch(str(payload.get("confirmatory_sha256"))):
                self._fail("a confirmatory amendment requires the new confirmatory registry hash")
            self.confirmatory_sha256 = payload["confirmatory_sha256"]
        for ns in ("holdout", "final"):
            if ns in payload.get("affects_namespaces", []) and self.namespaces.get(ns):
                self._fail(f"amendment affects the already opened {ns} namespace")
        self.amendments.append({**payload, "index": self.count})
        self.protocol_sha256 = payload["protocol_sha256"]

    def _on_incident(self, entry, payload, protocol) -> None:
        if payload.get("kind") not in INCIDENT_KINDS:
            self._fail(f"incident kind must be one of {INCIDENT_KINDS}")
        if not isinstance(payload.get("affects_validity"), bool) or not payload.get("description"):
            self._fail("incident requires affects_validity (bool) and a description")
        self.incidents.append({**payload, "index": self.count})


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


def _root_of(path: Path) -> Path:
    return path.resolve().parents[2]


def append_event(path: str | Path, event: str, *, reason: str, payload: dict | None = None, dataset: str | None = None,
                 role: str | None = None, design: str | None = None, protocol: dict | None = None,
                 root: Path | None = None, at: str | None = None) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LedgerLock(path):
        state = replay_ledger(read_ledger(path), protocol)
        entry = {"schema": LEDGER_SCHEMA, "index": state.count, "at": at or _utc_now(), "event": event,
                 "dataset": dataset, "role": role, "design": design, "reason": reason,
                 **git_state(root or _root_of(path)),
                 "payload": json.loads(canonical_json(payload or {})), "prev_sha256": state.head}
        entry["sha256"] = entry_hash(entry)
        state.apply(entry, protocol)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(entry) + "\n")
    return entry


def state_of(root: Path) -> tuple[dict, LedgerState]:
    protocol = load_protocol(root / PROTOCOL_PATH)
    return protocol, replay_ledger(read_ledger(root / LEDGER_PATH), protocol)


def record_incident(root: Path, kind: str, *, description: str, affects_validity: bool, runs: list[str] | None = None,
                    classification: str = "", reason: str = "") -> dict:
    """Append a scientific incident (never erased) to the ledger chain."""
    protocol = load_protocol(root / PROTOCOL_PATH)
    return append_event(root / LEDGER_PATH, "incident", reason=reason or f"incident: {kind}", protocol=protocol,
                        root=root, payload={"kind": kind, "description": description,
                                            "affects_validity": affects_validity, "runs": runs or [],
                                            "classification": classification})


def amend_protocol(root: Path, document: dict, *, number: int, reason: str, affects: list[str],
                   configs: list[str] | None = None, confirmatory: dict | None = None) -> dict:
    """Append the amendment (validated against the current protocol) and only then write the new files."""
    current = load_protocol(root / PROTOCOL_PATH)
    validate_protocol(document)
    payload = {"protocol_sha256": protocol_sha256(document), "affects": affects, "number": number,
               "configs": configs or [],
               "config_sha256": {n: text_sha256(root / portable(n)) for n in (configs or [])}}
    if confirmatory is not None:
        payload.update({"affects_confirmatory": True, "confirmatory_sha256": document_sha256(confirmatory)})
    entry = append_event(root / LEDGER_PATH, "amend", reason=reason, protocol=current, root=root, payload=payload)
    (root / PROTOCOL_PATH).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                      newline="\n")
    if confirmatory is not None:
        (root / CONFIRMATORY_PATH).write_text(json.dumps(confirmatory, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8", newline="\n")
    return entry


# ============================================================================= history and initialization


def history_state(root: Path) -> dict:
    """Read-only state of the v0.8 records that v0.9 pins (v0.8 pins v0.7, which pins v0.5/v0.6)."""
    from ..v08 import protocol as v08
    p8 = v08.load_protocol(root / V08_PROTOCOL_PATH)
    e8 = v08.read_ledger(root / V08_LEDGER_PATH)
    s8 = v08.replay_ledger(e8, p8)
    return {"v08_protocol_sha256": v08.protocol_sha256(p8), "v08_ledger_entries": len(e8),
            "v08_ledger_head_sha256": s8.head, "v08_frozen_protocol_sha256": s8.protocol_sha256,
            "consumed": sorted(d for d in s8.consumed if not d.startswith("period:")),
            "consumed_period_keys": sorted(s8.consumed_keys)}


def initialize_ledger(root: Path) -> list[dict]:
    """Import v0.5-v0.8 consumption, declare every v0.9 dataset fresh, then freeze."""
    path = root / LEDGER_PATH
    if path.exists():
        raise ProtocolError("v0.9 ledger already exists; it is append-only")
    protocol = load_protocol(root / PROTOCOL_PATH, root)
    history = history_state(root)
    for key in ("v08_protocol_sha256", "v08_ledger_entries", "v08_ledger_head_sha256"):
        if history[key] != protocol["history"][key]:
            raise ProtocolError(f"pinned history differs: {key}")
    evidence = f"{V08_LEDGER_PATH}@{history['v08_ledger_entries'] - 1}:{history['v08_ledger_head_sha256']}"
    specs: list[dict] = []
    for dataset in history["consumed"]:
        specs.append({"event": "import_consumed", "dataset": dataset, "reason": "consumed in v0.5-v0.8",
                      "payload": {"evidence": evidence, "period_key": period_key(dataset)}})
    for key in history["consumed_period_keys"]:
        specs.append({"event": "import_consumed", "dataset": f"period:{key}", "reason": "consumed calendar period",
                      "payload": {"evidence": evidence, "period_key": key}})
    for d in protocol["datasets"]:
        specs.append({"event": "declare_fresh", "dataset": d["id"], "role": d["role"],
                      "reason": "never accessed by any CleoLOB version (chronology audit)",
                      "payload": {"period_key": period_key(d), "identity_sha256": dataset_identity_sha256(d)}})
    specs.append({"event": "freeze", "reason": "v0.9 preregistration freeze",
                  "payload": {"protocol_sha256": protocol_sha256(protocol),
                              "referenced_configs": referenced_config_hashes(protocol, root)}})
    return _append_batch(path, specs, protocol=protocol, root=root)


def _append_batch(path: Path, specs: list[dict], *, protocol: dict, root: Path) -> list[dict]:
    """Validate a batch of new entries in one pass and write them together (all or nothing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LedgerLock(path):
        state = replay_ledger(read_ledger(path), protocol)
        git = git_state(root)
        written = []
        for spec in specs:
            entry = {"schema": LEDGER_SCHEMA, "index": state.count, "at": _utc_now(), "event": spec["event"],
                     "dataset": spec.get("dataset"), "role": spec.get("role"), "design": spec.get("design"),
                     "reason": spec["reason"], **git, "payload": json.loads(canonical_json(spec.get("payload") or {})),
                     "prev_sha256": state.head}
            entry["sha256"] = entry_hash(entry)
            state.apply(entry, protocol)
            written.append(entry)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(canonical_json(e) + "\n" for e in written))
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
                if current.get(name) != amended.get(name, digest):
                    issues.append(f"referenced config changed after freeze without amendment: {name}")
        if state.confirmatory_sha256 is not None:
            confirmatory = json.loads((root / CONFIRMATORY_PATH).read_text(encoding="utf-8"))
            if document_sha256(confirmatory) != state.confirmatory_sha256:
                issues.append("confirmatory registry changed after the promotion freeze without amendment")
        elif (root / CONFIRMATORY_PATH).exists():
            issues.append("a confirmatory registry exists but was never promoted in the ledger")
        history = history_state(root)
        for key in ("v08_protocol_sha256", "v08_ledger_entries", "v08_ledger_head_sha256"):
            if history[key] != protocol["history"][key]:
                issues.append(f"pinned history changed: {key} (v0.8 records must stay immutable)")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(str(exc))
    return status_report(protocol, state, issues)


def status_report(protocol: dict | None, state: LedgerState | None, issues: list[str]) -> dict:
    guarded = {}
    if protocol is not None and state is not None:
        for d in protocol["datasets"]:
            if d["role"] in GUARDED_ROLES and d["id"] in state.consumed:
                guarded[d["id"]] = {"role": d["role"], "first_access_index": state.first_access.get(d["id"]),
                                    "stages": state.stages.get(d["id"], [])}
    by_role: dict[str, dict[str, int]] = {}
    if protocol is not None and state is not None:
        for d in protocol["datasets"]:
            row = by_role.setdefault(d["role"], {"declared": 0, "consumed": 0})
            row["declared"] += 1
            row["consumed"] += d["id"] in state.consumed
    return {"valid": not issues, "issues": issues,
            "protocol_sha256": state.protocol_sha256 if state else None,
            "ledger_entries": state.count if state else None, "ledger_head_sha256": state.head if state else None,
            "frozen": bool(state and state.frozen), "promoted": bool(state and state.promotion),
            "confirmatory_sha256": state.confirmatory_sha256 if state else None,
            "confirmatory_closed": bool(state and state.closed), "datasets_by_role": by_role,
            "guarded_accessed": guarded, "sealed_designs": sorted(state.designs) if state else None,
            "namespaces_opened": {k: [x["design"] for x in v] for k, v in state.namespaces.items()} if state else None,
            "attempts": [{k: a.get(k) for k in ("design", "outcome", "directory", "dataset", "index")}
                         for a in state.attempts] if state else None,
            "incidents": [{k: i.get(k) for k in ("kind", "affects_validity", "description", "index")}
                          for i in state.incidents] if state else None,
            "amendments": len(state.amendments) if state else None}


# ============================================================================= designs and seeds


def seal_design(root: Path, name: str, design: dict, *, reads: list[str] | None = None,
                namespaces: list[str] | None = None, posthoc: bool = False, reason: str = "") -> dict:
    """Seal a study design before it runs; the design document is stored under configs/v09/designs."""
    protocol = load_protocol(root / PROTOCOL_PATH)
    replay_ledger(read_ledger(root / LEDGER_PATH), protocol)
    target = root / CONFIG_DIR / "designs" / f"{name}.json"
    body = canonical_json(design)
    if target.exists() and target.read_text(encoding="utf-8").strip() != body:
        raise ProtocolError(f"design file {target.name} exists with different content")
    entry = append_event(root / LEDGER_PATH, "seal_design", design=name, reason=reason or f"seal {name}",
                         protocol=protocol, root=root,
                         payload={"design_sha256": document_sha256(design), "reads": reads or [],
                                  "namespaces": namespaces or [], "posthoc": posthoc,
                                  "file": f"{CONFIG_DIR}/designs/{name}.json"})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body + "\n", encoding="utf-8", newline="\n")
    return entry


def sealed_design(root: Path, name: str) -> dict:
    """The sealed design document, verified against its ledger hash."""
    protocol, state = state_of(root)
    if name not in state.designs:
        raise ProtocolError(f"design {name!r} is not sealed")
    document = json.loads((root / CONFIG_DIR / "designs" / f"{name}.json").read_text(encoding="utf-8"))
    if document_sha256(document) != state.designs[name]["design_sha256"]:
        raise ProtocolError(f"design {name!r} changed after sealing")
    return document


def open_namespace(root: Path, namespace: str, *, design: str, reason: str) -> dict:
    """Record the first (or posthoc) use of a guarded seed namespace by a sealed design."""
    protocol, state = state_of(root)
    if any(x["design"] == design for x in state.namespaces.get(namespace, [])):
        return {"already_open": True}
    return append_event(root / LEDGER_PATH, "open_namespace", design=design, reason=reason, protocol=protocol,
                        root=root, payload={"namespace": namespace})


def namespace_guard(namespace: str, *, root: Path | None, design: str | None) -> None:
    """Guarded seeds require a frozen protocol and a design that opened the namespace."""
    if namespace not in NAMESPACES:
        raise ProtocolError(f"unknown seed namespace {namespace!r}")
    if namespace not in GUARDED_NAMESPACES:
        return
    if root is None or design is None:
        raise ProtocolError(f"seed namespace {namespace!r} requires a sealed design and the repository root")
    _, state = state_of(root)
    if not any(x["design"] == design for x in state.namespaces.get(namespace, [])):
        raise ProtocolError(f"design {design!r} has not opened seed namespace {namespace!r}")


def seeds(namespace: str, study: str, count: int, *, root: Path | None = None, design: str | None = None) -> list[int]:
    """Seed-addressable seeds: ``SeedSequence([v09 tag, namespace id, hash(study), index])``; never schedule-dependent.

    The leading 9 separates v0.9 streams from the v0.8 streams, which used ``[namespace id, hash, index]``.
    """
    import numpy as np
    namespace_guard(namespace, root=root, design=design)
    tag = int(document_sha256(study)[:12], 16)
    return [int(np.random.SeedSequence([9, NAMESPACE_IDS[namespace], tag, i]).generate_state(1, dtype=np.uint32)[0])
            for i in range(count)]


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
            "frozen": state.frozen is not None, "confirmatory_sha256": state.confirmatory_sha256,
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
            if binding.get("confirmatory_sha256") != at_anchor.confirmatory_sha256:
                issues.append("bound confirmatory registry is not the ledger's registry at the anchor")
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
