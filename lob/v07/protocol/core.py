"""v0.7 protocol freeze, stage-aware append-only consumption ledger and evidence binding.

Every ledger entry records: sequential index, UTC timestamp, action (event),
dataset, role, reason, the sealed reader/design, the git commit (and dirty flag)
at the time of the entry, the previous entry hash and its own hash.

Rules enforced by replay:

* v0.5 and v0.6 consumption is imported before the freeze; the protocol pins the
  v0.6 protocol hash and ledger head. A consumed dataset or calendar period can
  never be declared fresh again.
* Uses are role-specific (see ``ROLE_USES``). Holdout access requires a design
  sealed before that holdout's first access and listing it in ``reads``.
* Access is staged: ``downloaded`` -> ``opened`` -> ``parsed`` -> ``evaluated``;
  ``inspect`` records when an outcome was first looked at. Any stage consumes.
* After first access, designs that read a holdout must be ``posthoc`` (default
  false, read as a strict bool) and use ``evaluate_posthoc``.
* Source bytes cannot change; attempts (failed, invalid, not available) are
  retained; amendments carry the new protocol hash and may not affect accessed
  holdouts.

The chain detects mutation, truncation (through anchors) and reordering; it is
not an external timestamp authority.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

from ...config import canonical_json
from ...preregistration import (ProtocolError, document_sha256, file_sha256, period_key, read_ledger as read_v05_ledger,
                                replay_ledger as replay_v05_ledger, load_protocol as load_v05_protocol, text_sha256)
from .taxonomy import EQUIVALENCE_STATUSES, STATUSES

SCHEMA = "cleolob-v07-protocol-1"
LEDGER_SCHEMA = "cleolob-v07-ledger-1"
BINDING_SCHEMA = "cleolob-v07-binding-1"
CONFIG_DIR = "configs/v07"
PROTOCOL_PATH = f"{CONFIG_DIR}/protocol.json"
LEDGER_PATH = f"{CONFIG_DIR}/consumption-ledger.jsonl"
V06_PROTOCOL_PATH = "configs/v06/protocol.json"
V06_LEDGER_PATH = "configs/v06/consumption-ledger.jsonl"
V05_PROTOCOL_PATH = "configs/v05/protocol.json"
V05_LEDGER_PATH = "configs/v05/consumption-ledger.jsonl"

ROLE_USES = {
    "development": frozenset({"develop"}),
    "selection": frozenset({"select"}),
    "validation": frozenset({"validate"}),
    "retrospective": frozenset({"retrospective"}),
    "mbo_retrospective": frozenset({"retrospective"}),
    "fresh_temporal": frozenset({"download", "evaluate", "evaluate_posthoc"}),
    "fresh_cross_instrument": frozenset({"download", "evaluate", "evaluate_posthoc"}),
    "fresh_cross_venue": frozenset({"download", "evaluate", "evaluate_posthoc"}),
    "final_transfer": frozenset({"download", "evaluate", "evaluate_posthoc"}),
}
HOLDOUT_ROLES = frozenset({"fresh_temporal", "fresh_cross_instrument", "fresh_cross_venue", "final_transfer"})
STAGES = ("downloaded", "opened", "parsed", "evaluated")
EVENTS = ("import_consumed", "declare_fresh", "freeze", "seal_design", "access", "inspect", "attempt", "amend")
_ATTEMPT_OUTCOMES = frozenset(STATUSES) | {"COMPLETED", "ABORTED"}
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9.-]{2,80}")
_IDENTITY = ("id", "provider", "venue", "instrument", "date", "data_types", "capability_level", "role")
REQUIRED_CONFIGS = ("dataset-registry.json", "hypotheses.json", "statistical-families.json", "compute-budget.json",
                    "result-taxonomy.json", "claim-registry.json", "scenario-taxonomy.json")
_REQUIRED = ("schema", "protocol_id", "frozen_at_utc", "base_commit", "research_question", "statuses",
             "equivalence_statuses", "history", "datasets", "holdout_rules", "model_selection", "generator_families",
             "calibration", "plausibility", "execution", "transfer", "statistics", "seeds", "early_stopping",
             "amendment_policy", "reporting", "referenced_configs", "exclusions")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_state(root: Path) -> dict:
    """HEAD commit and dirty flag at the time of an entry (None outside a git checkout).

    The ledger file itself is excluded from the dirty check: appending to it is the entry being recorded. Entries
    1-40 of the repository ledger predate this exclusion and are dirty only because the new ledger was untracked.
    """
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                         stderr=subprocess.DEVNULL, timeout=5).strip()
        status = subprocess.check_output(["git", "status", "--porcelain", "--", ".", f":(exclude){LEDGER_PATH}"],
                                         cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=10)
        dirty = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def _portable(name: str) -> str:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name or name.startswith("/")
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ProtocolError(f"unsafe repository-relative path: {name!r}")
    return name


# ----------------------------------------------------------------------------- protocol


def validate_protocol(protocol: dict, root: Path | None = None) -> dict:
    if not isinstance(protocol, dict) or protocol.get("schema") != SCHEMA:
        raise ProtocolError("unsupported v0.7 protocol schema")
    missing = [k for k in _REQUIRED if k not in protocol]
    if missing:
        raise ProtocolError(f"protocol missing required sections: {missing}")
    if tuple(protocol["statuses"]) != STATUSES:
        raise ProtocolError("protocol must use exactly the ten v0.7 statuses")
    if tuple(protocol["equivalence_statuses"]) != EQUIVALENCE_STATUSES:
        raise ProtocolError("protocol must use exactly the four equivalence statuses")
    if not re.fullmatch("[0-9a-f]{40}", str(protocol["base_commit"])):
        raise ProtocolError("base_commit must be a full git SHA")
    history = protocol["history"]
    for key in ("v06_protocol_sha256", "v06_ledger_entries", "v06_ledger_head_sha256", "v05_protocol_sha256"):
        if key not in history:
            raise ProtocolError(f"history missing {key}")
    ids, roles = set(), []
    for d in protocol["datasets"]:
        for key in _IDENTITY + ("freshness_at_freeze", "prior_use"):
            if key not in d:
                raise ProtocolError(f"dataset declaration missing {key}")
        if not _SAFE_ID.fullmatch(d["id"]) or d["id"] in ids:
            raise ProtocolError(f"invalid or duplicate dataset id: {d['id']}")
        ids.add(d["id"])
        if d["role"] not in ROLE_USES:
            raise ProtocolError(f"unknown dataset role: {d['role']}")
        if (d["role"] in HOLDOUT_ROLES) != (d["freshness_at_freeze"] == "fresh"):
            raise ProtocolError(f"{d['id']}: holdout roles require fresh data and only holdouts are fresh")
        if d["role"] != "mbo_retrospective" and period_key(d) is None:
            raise ProtocolError(f"{d['id']}: date must be YYYY-MM-DD")
        roles.append(d["role"])
    for role in ("development", "selection"):
        if roles.count(role) != 1:
            raise ProtocolError(f"protocol must declare exactly one {role} dataset")
    names = [_portable(n) for n in protocol["referenced_configs"]]
    for required in REQUIRED_CONFIGS:
        if f"{CONFIG_DIR}/{required}" not in names:
            raise ProtocolError(f"referenced_configs must include {CONFIG_DIR}/{required}")
    if root is not None:
        validate_configs(protocol, root)
    return protocol


def validate_configs(protocol: dict, root: Path) -> None:
    """Cross-checks between the protocol, hypotheses and statistical families."""
    from .taxonomy import MultiplicityRegistry
    hypotheses = json.loads((root / CONFIG_DIR / "hypotheses.json").read_text(encoding="utf-8"))
    families = json.loads((root / CONFIG_DIR / "statistical-families.json").read_text(encoding="utf-8"))
    registry = MultiplicityRegistry.from_config(families)
    datasets = {d["id"] for d in protocol["datasets"]}
    needed = ("question", "estimator", "datasets", "sample_unit", "uncertainty", "family", "threshold", "margin",
              "failure_semantics", "status_rule", "kind")
    for hid, h in hypotheses["hypotheses"].items():
        if not re.fullmatch(r"H\d{1,2}", hid):
            raise ProtocolError(f"hypothesis ids must look like H1..H99: {hid}")
        absent = [k for k in needed if k not in h]
        if absent:
            raise ProtocolError(f"{hid} missing {absent}")
        if h["family"] not in registry.families:
            raise ProtocolError(f"{hid} references unregistered family {h['family']}")
        for ds in h["datasets"]:
            if ds not in datasets:
                raise ProtocolError(f"{hid} references undeclared dataset {ds}")
    for name, fam in families["families"].items():
        if fam["kind"] == "confirmatory" and fam["correction"] == "bonferroni":
            expected = fam["alpha"] / fam["size"]
            if abs(fam["adjusted_alpha"] - expected) > 1e-12:
                raise ProtocolError(f"family {name}: adjusted alpha must equal alpha / size")
    for required in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12", "H13"):
        if required not in hypotheses["hypotheses"]:
            raise ProtocolError(f"core hypothesis {required} is not registered")


def load_protocol(path: str | Path, root: Path | None = None) -> dict:
    return validate_protocol(json.loads(Path(path).read_text(encoding="utf-8")), root)


def protocol_sha256(protocol: dict) -> str:
    return document_sha256(protocol)


def dataset_declaration(protocol: dict, dataset_id: str) -> dict:
    for d in protocol["datasets"]:
        if d["id"] == dataset_id:
            return d
    raise ProtocolError(f"dataset {dataset_id!r} is not declared in the v0.7 protocol")


def datasets_by_role(protocol: dict, role: str) -> list[str]:
    return [d["id"] for d in protocol["datasets"] if d["role"] == role]


def dataset_identity_sha256(declaration: dict, source_hashes: dict[str, str] | None = None) -> str:
    identity = {k: declaration[k] for k in _IDENTITY}
    identity["source_sha256"] = dict(sorted((source_hashes or {}).items()))
    return document_sha256(identity)


def referenced_config_hashes(protocol: dict, root: Path) -> dict[str, str]:
    return {n: text_sha256(root / _portable(n)) for n in protocol["referenced_configs"]}


# ----------------------------------------------------------------------------- ledger


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
        reads = payload.get("reads", [])
        posthoc = payload.get("posthoc", False)
        if not isinstance(reads, list) or not isinstance(posthoc, bool):
            self._fail("design reads must be a list and posthoc a boolean")
        for read in reads:
            if protocol is not None:
                dataset_declaration(protocol, read)
            if read in self.first_access and not posthoc:
                self._fail(f"design {name!r} reads consumed holdout {read}; it must be declared posthoc")
        self.designs[name] = {**payload, "posthoc": posthoc}

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
        if protocol is not None:
            key = period_key(dataset_declaration(protocol, dataset))
            if key:
                self.consumed_keys.add(key)

    def _on_inspect(self, entry, payload, protocol) -> None:
        dataset = entry.get("dataset")
        if dataset not in self.first_access:
            self._fail(f"{dataset}: outcome inspection recorded before any access")
        if not payload.get("run"):
            self._fail("inspect requires the run directory whose outcome was inspected")
        self.inspected.setdefault(dataset, self.count)

    def _on_attempt(self, entry, payload, protocol) -> None:
        if not entry.get("design") or payload.get("outcome") not in _ATTEMPT_OUTCOMES:
            self._fail("attempt requires a design and a registered outcome status")
        if "directory" not in payload or not payload.get("note"):
            self._fail("attempt requires the retained directory (or null with a reason) and a note")
        if payload["directory"] is None and not payload.get("no_directory_reason"):
            self._fail("an attempt without a directory must say why none exists")
        self.attempts.append({**payload, "design": entry.get("design"), "index": self.count})

    def _on_amend(self, entry, payload, protocol) -> None:
        if self.frozen is None:
            self._fail("amendments require a frozen protocol")
        if not _DIGEST.fullmatch(str(payload.get("protocol_sha256"))):
            self._fail("amendment requires the new protocol hash")
        late = [d for d in payload.get("affects", []) if d in self.first_access]
        if late:
            self._fail(f"amendment affects already accessed holdouts: {late}")
        self.amendments.append({**payload, "index": self.count})
        self.protocol_sha256 = payload["protocol_sha256"]


def replay_ledger(entries: Iterable[dict], protocol: dict | None = None) -> LedgerState:
    state = LedgerState()
    for entry in entries:
        state.apply(entry, protocol)
    if protocol is not None and state.frozen is not None and state.protocol_sha256 != protocol_sha256(protocol):
        raise ProtocolError("protocol changed after freeze (hash differs from the ledger's current protocol hash)")
    return state


def append_event(path: str | Path, event: str, *, reason: str, payload: dict | None = None, dataset: str | None = None,
                 role: str | None = None, design: str | None = None, protocol: dict | None = None,
                 root: Path | None = None, at: str | None = None) -> dict:
    """Validate history plus the new entry (with git state), then append."""
    path = Path(path)
    state = replay_ledger(read_ledger(path), protocol)
    entry = {"schema": LEDGER_SCHEMA, "index": state.count, "at": at or _utc_now(), "event": event,
             "dataset": dataset, "role": role, "design": design, "reason": reason,
             **git_state(root or path.resolve().parents[2]),
             "payload": json.loads(canonical_json(payload or {})), "prev_sha256": state.head}
    entry["sha256"] = entry_hash(entry)
    state.apply(entry, protocol)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(entry) + "\n")
    return entry


def amend_protocol(root: Path, document: dict, *, number: int, reason: str, affects: list[str]) -> dict:
    """Append the amendment (validated against the current protocol) and only then write the new protocol."""
    current = load_protocol(root / PROTOCOL_PATH)
    validate_protocol(document, root)
    entry = append_event(root / LEDGER_PATH, "amend", reason=reason, protocol=current, root=root,
                         payload={"protocol_sha256": protocol_sha256(document), "affects": affects, "number": number})
    (root / PROTOCOL_PATH).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                      newline="\n")
    return entry


def history_state(root: Path) -> dict:
    """Read-only state of the v0.5 and v0.6 records that v0.7 pins and imports."""
    from ...v06 import protocol as v06
    p6 = v06.load_protocol(root / V06_PROTOCOL_PATH)
    e6 = v06.read_ledger(root / V06_LEDGER_PATH)
    s6 = v06.replay_ledger(e6, p6)
    p5 = load_v05_protocol(root / V05_PROTOCOL_PATH)
    s5 = replay_v05_ledger(read_v05_ledger(root / V05_LEDGER_PATH), p5)
    from ...preregistration import protocol_sha256 as v05_sha
    consumed = sorted(set(s6.consumed) | set(s5.consumed))
    keys = sorted(s6.consumed_keys | s5.consumed_keys)
    return {"v06_protocol_sha256": v06.protocol_sha256(p6), "v06_ledger_entries": len(e6),
            "v06_ledger_head_sha256": s6.head, "v05_protocol_sha256": v05_sha(p5), "consumed": consumed,
            "consumed_period_keys": keys}


def initialize_ledger(root: Path) -> list[dict]:
    """Import v0.5/v0.6 consumption, declare v0.7 fresh holdouts, then freeze."""
    path = root / LEDGER_PATH
    if path.exists():
        raise ProtocolError("v0.7 ledger already exists; it is append-only")
    protocol = load_protocol(root / PROTOCOL_PATH, root)
    history = history_state(root)
    pinned = protocol["history"]
    for key in ("v06_protocol_sha256", "v06_ledger_entries", "v06_ledger_head_sha256", "v05_protocol_sha256"):
        if history[key] != pinned[key]:
            raise ProtocolError(f"pinned history differs: {key}")
    evidence = f"{V06_LEDGER_PATH}@{history['v06_ledger_entries'] - 1}:{history['v06_ledger_head_sha256']}"
    written = []
    for dataset in (d for d in history["consumed"] if not d.startswith("period:")):
        written.append(append_event(path, "import_consumed", dataset=dataset, reason="consumed in v0.5/v0.6",
                                    payload={"evidence": evidence, "period_key": period_key(dataset)}, root=root))
    for key in history["consumed_period_keys"]:
        written.append(append_event(path, "import_consumed", dataset=f"period:{key}", reason="consumed calendar period",
                                    payload={"evidence": evidence, "period_key": key}, root=root))
    for d in protocol["datasets"]:
        if d["freshness_at_freeze"] == "fresh":
            written.append(append_event(path, "declare_fresh", dataset=d["id"], role=d["role"],
                                        reason="never accessed by any CleoLOB version (chronology audit)",
                                        payload={"period_key": period_key(d), "identity_sha256":
                                                 dataset_identity_sha256(d)}, protocol=protocol, root=root))
        elif d["id"] not in history["consumed"] and d["role"] != "mbo_retrospective":
            raise ProtocolError(f"{d['id']}: declared consumed but absent from the v0.5/v0.6 ledgers")
    written.append(append_event(path, "freeze", reason="v0.7 M0 preregistration freeze", protocol=protocol, root=root,
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
            amended = {n for a in state.amendments for n in a.get("configs", [])}
            for name, digest in state.frozen["referenced_configs"].items():
                if current.get(name) != digest and name not in amended:
                    issues.append(f"referenced config changed after freeze without amendment: {name}")
        history = history_state(root)
        for key in ("v06_protocol_sha256", "v06_ledger_entries", "v06_ledger_head_sha256", "v05_protocol_sha256"):
            if history[key] != protocol["history"][key]:
                issues.append(f"pinned history changed: {key} (v0.5/v0.6 records must stay immutable)")
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
                                     "outcome_inspected_index": state.inspected.get(d["id"]),
                                     "sealed_readers": sorted(n for n, x in state.designs.items()
                                                              if d["id"] in x.get("reads", []))}
    return {"valid": not issues, "issues": issues,
            "protocol_sha256": state.protocol_sha256 if state else None,
            "ledger_entries": state.count if state else None, "ledger_head_sha256": state.head if state else None,
            "fresh": sorted(d for d in state.fresh if d not in state.consumed) if state else None,
            "holdouts": holdouts, "sealed_designs": sorted(state.designs) if state else None,
            "attempts": [{k: a.get(k) for k in ("design", "outcome", "directory", "index")} for a in state.attempts]
            if state else None, "amendments": len(state.amendments) if state else None}


# ----------------------------------------------------------------------------- binding


def bind_evidence(*, protocol: dict, ledger_path: Path, analysis: str, dataset_ids: Iterable[str], config: Any,
                  result: Any, provenance: dict, extra: dict | None = None) -> dict:
    state = replay_ledger(read_ledger(ledger_path), protocol)
    if state.frozen is None:
        raise ProtocolError("evidence cannot bind before the protocol freeze")
    datasets = {}
    for dataset_id in sorted(set(dataset_ids)):
        d = dataset_declaration(protocol, dataset_id)
        sources: dict[str, str] = {}
        for access in state.accesses.get(dataset_id, []):
            sources.update(access.get("source_sha256", {}))
        datasets[dataset_id] = {"identity_sha256": dataset_identity_sha256(d, sources), "role": d["role"],
                                "freshness": state.freshness(dataset_id), "source_sha256": sources}
    return {"schema": BINDING_SCHEMA, "analysis": analysis, "protocol_sha256": protocol_sha256(protocol),
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
    if anchor["index"] >= len(entries) or entries[anchor["index"]]["sha256"] != anchor["sha256"]:
        issues.append("ledger anchor missing or changed (truncation or rewrite)")
    else:
        at_anchor = replay_ledger(entries[: anchor["index"] + 1])
        if binding["protocol_sha256"] != at_anchor.protocol_sha256:
            issues.append("bound protocol hash is not the ledger's protocol at the anchor")
        if binding["protocol_sha256"] != protocol_sha256(protocol) and not any(
                e["event"] == "amend" for e in entries[anchor["index"] + 1:]):
            issues.append("protocol changed since binding without a ledger amendment")
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


__all__ = ["ProtocolError", "ROLE_USES", "HOLDOUT_ROLES", "STAGES", "validate_protocol", "load_protocol",
           "protocol_sha256", "dataset_declaration", "datasets_by_role", "read_ledger", "replay_ledger",
           "append_event", "amend_protocol", "initialize_ledger", "verify_protocol_files", "bind_evidence",
           "verify_binding", "document_sha256", "file_sha256", "text_sha256", "period_key", "git_state"]
