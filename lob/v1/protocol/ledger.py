"""Append-only, hash-chained consumption ledger (M4) with incidents and amendments (M9).

Each line is a JSON entry whose ``sha256`` covers every other field, and whose ``prev_sha256`` is the previous
entry's hash; a broken chain, a reordered or edited line or a replayed rule violation is reported by ``verify``.

Events (``EVENTS``):

* ``import_consumed`` - a dataset or calendar period consumed by an earlier version (never fresh again);
* ``declare_source`` / ``declare_fresh`` - a registered source or dataset slot, with its role and pool;
* ``acquire`` - raw bytes obtained and hashed (download or prospective capture). Acquisition is not access:
  nothing was parsed, so the data stay fresh;
* ``access`` - a dataset was opened, parsed or evaluated for a named design in a namespace (``use``);
* ``freeze`` / ``amend`` - the protocol hash at freeze and at every numbered amendment; an amendment after any
  guarded access must carry ``posthoc: true``;
* ``seal_design`` / ``open_namespace`` / ``close_namespace`` - the design life cycle;
* ``promote`` - the single promotion decision; ``attempt`` - every study attempt with its outcome, including
  failures; ``incident`` - engineering bugs, contamination, post-hoc changes.

Guarded events (accesses in confirm, replicate or vault namespaces; seals; openings) refuse a dirty git tree unless
the workspace is explicitly a test workspace: v0.9 recorded 240 of 700 entries from a dirty tree (audit section 1).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Iterator

SCHEMA = "cleolob-v1-ledger-1"
EVENTS = ("import_consumed", "declare_source", "declare_fresh", "acquire", "access", "freeze", "amend",
          "seal_design", "open_namespace", "close_namespace", "promote", "attempt", "incident")
STAGES = ("opened", "parsed", "evaluated")
INCIDENT_KINDS = ("engineering_bug", "contamination", "posthoc_change", "accidental_access", "wrong_input",
                  "resource", "other")
ATTEMPT_OUTCOMES = ("COMPLETED", "INVALID", "ABORTED", "RESOURCE_EXHAUSTED", "NOT_AVAILABLE")
GENESIS = "0" * 64


class LedgerError(RuntimeError):
    """A ledger rule would be broken; nothing was written."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def entry_hash(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "sha256"}
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def git_state(root: Path) -> dict:
    """Commit and cleanliness; anything that is not a readable git work tree counts as dirty."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=20)
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                                capture_output=True, text=True, timeout=20)
        if head.returncode != 0 or status.returncode != 0:
            return {"commit": None, "dirty": True}
        return {"commit": head.stdout.strip() or None, "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": True}


@dataclass
class LedgerState:
    entries: int = 0
    head: str = GENESIS
    consumed: dict[str, dict] = field(default_factory=dict)        # dataset/period id -> import record
    sources: dict[str, dict] = field(default_factory=dict)
    fresh: dict[str, dict] = field(default_factory=dict)           # dataset id -> declaration (role, pool)
    acquired: dict[str, dict] = field(default_factory=dict)        # dataset id -> {source: sha256}
    accesses: dict[str, list[dict]] = field(default_factory=dict)  # dataset id -> access records
    frozen: dict | None = None
    amendments: list[dict] = field(default_factory=list)
    designs: dict[str, dict] = field(default_factory=dict)         # sealed design name -> record
    open_namespaces: dict[str, str] = field(default_factory=dict)  # namespace -> design that opened it
    closed_namespaces: dict[str, dict] = field(default_factory=dict)
    promotion: dict | None = None
    attempts: list[dict] = field(default_factory=list)
    incidents: list[dict] = field(default_factory=list)
    guarded_access_seen: bool = False

    def role_of(self, dataset: str) -> str | None:
        d = self.fresh.get(dataset)
        return d.get("role") if d else None

    def accessed(self, dataset: str) -> bool:
        return bool(self.accesses.get(dataset))

    def to_dict(self) -> dict:
        return {"entries": self.entries, "head": self.head, "consumed": len(self.consumed),
                "sources": len(self.sources), "fresh": len(self.fresh), "acquired": len(self.acquired),
                "accessed": len(self.accesses), "frozen": self.frozen, "amendments": len(self.amendments),
                "designs": sorted(self.designs), "open_namespaces": dict(self.open_namespaces),
                "closed_namespaces": sorted(self.closed_namespaces), "promoted": self.promotion is not None,
                "attempts": len(self.attempts), "incidents": len(self.incidents)}


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError as exc:
                raise LedgerError(f"ledger line {n} is not JSON: {exc}") from None
    return out


def apply(state: LedgerState, e: dict, namespaces) -> None:
    """Apply one entry to ``state``; raises LedgerError on a rule violation (the replay is the rule book)."""
    ev, p, ds = e["event"], e.get("payload") or {}, e.get("dataset")
    if ev == "import_consumed":
        state.consumed[ds] = p
    elif ev == "declare_source":
        state.sources[p["source"]] = p
    elif ev == "declare_fresh":
        if ds in state.consumed:
            raise LedgerError(f"{ds} was consumed by an earlier version and can never be declared fresh")
        if ds in state.fresh:
            raise LedgerError(f"{ds} declared twice")
        state.fresh[ds] = p
    elif ev == "acquire":
        if ds not in state.fresh and p.get("role") != "development":
            raise LedgerError(f"acquire of undeclared dataset {ds}")
        prev = state.acquired.get(ds, {})
        for name, digest in p.get("sha256", {}).items():
            if name in prev and prev[name] != digest:
                raise LedgerError(f"{ds}: source {name} changed hash after first acquisition")
        state.acquired[ds] = {**prev, **p.get("sha256", {})}
    elif ev == "access":
        ns = p["namespace"]
        role = state.role_of(ds) or ("consumed" if ds in state.consumed else None)
        if role is None and p.get("use") != "development":
            raise LedgerError(f"access of undeclared dataset {ds}")
        namespaces.check_access(state, ds, role, ns, p.get("design"), p.get("use"))
        if ns in namespaces.GUARDED:
            state.guarded_access_seen = True
        state.accesses.setdefault(ds, []).append({"index": e["index"], **p})
    elif ev == "freeze":
        if state.frozen is not None:
            raise LedgerError("the protocol is frozen once; changes are amendments")
        state.frozen = {"protocol_sha256": p["protocol_sha256"], "index": e["index"]}
    elif ev == "amend":
        if state.frozen is None:
            raise LedgerError("amendments follow the freeze")
        if p["number"] != len(state.amendments) + 1:
            raise LedgerError("amendments are numbered consecutively")
        if state.guarded_access_seen and not p.get("posthoc"):
            raise LedgerError("an amendment after any guarded access must be declared posthoc")
        state.amendments.append({"index": e["index"], **p})
        state.frozen = {**state.frozen, "protocol_sha256": p["protocol_sha256"]}
    elif ev == "seal_design":
        if state.frozen is None:
            raise LedgerError("designs are sealed after the protocol freeze")
        name = p["design"]
        if name in state.designs:
            raise LedgerError(f"design {name} sealed twice")
        state.designs[name] = {"index": e["index"], **p}
    elif ev == "open_namespace":
        namespaces.check_open(state, p["namespace"], p["design"])
        state.open_namespaces[p["namespace"]] = p["design"]
    elif ev == "close_namespace":
        ns = p["namespace"]
        if ns not in state.open_namespaces:
            raise LedgerError(f"namespace {ns} is not open")
        state.closed_namespaces[ns] = {"index": e["index"], **p}
        del state.open_namespaces[ns]
    elif ev == "promote":
        if state.promotion is not None:
            raise LedgerError("promotion is decided once")
        if state.frozen is None:
            raise LedgerError("promotion follows the freeze")
        state.promotion = {"index": e["index"], **p}
    elif ev == "attempt":
        if p.get("outcome") not in ATTEMPT_OUTCOMES:
            raise LedgerError(f"unknown attempt outcome {p.get('outcome')}")
        state.attempts.append({"index": e["index"], **p})
    elif ev == "incident":
        if p.get("kind") not in INCIDENT_KINDS:
            raise LedgerError(f"unknown incident kind {p.get('kind')}")
        state.incidents.append({"index": e["index"], **p})
    else:
        raise LedgerError(f"unknown event {ev}")


def replay(entries: Iterable[dict]) -> LedgerState:
    from . import namespaces
    state = LedgerState()
    for i, e in enumerate(entries):
        if e.get("schema") != SCHEMA:
            raise LedgerError(f"entry {i}: schema {e.get('schema')}")
        if e.get("index") != i:
            raise LedgerError(f"entry {i}: index {e.get('index')}")
        if e.get("prev_sha256") != state.head:
            raise LedgerError(f"entry {i}: broken hash chain")
        if entry_hash(e) != e.get("sha256"):
            raise LedgerError(f"entry {i}: hash mismatch (edited entry)")
        if e.get("event") not in EVENTS:
            raise LedgerError(f"entry {i}: unknown event")
        apply(state, e, namespaces)
        state.head = e["sha256"]
        state.entries = i + 1
    return state


def verify(path: Path) -> dict:
    try:
        state = replay(read(path))
        return {"valid": True, "issues": [], "state": state.to_dict()}
    except LedgerError as exc:
        return {"valid": False, "issues": [str(exc)], "state": None}


@contextmanager
def _lock(path: Path, timeout: float = 30.0) -> Iterator[None]:
    lock = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise LedgerError(f"ledger lock {lock} held; remove it only if no writer is running") from None
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


GUARDED_EVENTS = frozenset({"seal_design", "open_namespace", "close_namespace", "promote", "freeze", "amend"})


def append(path: Path, event: str, *, reason: str, payload: dict | None = None, dataset: str | None = None,
           root: Path | None = None, allow_dirty: bool = False) -> dict:
    """Validate and append one entry atomically (the whole chain is replayed with the new entry first)."""
    from . import namespaces
    if event not in EVENTS:
        raise LedgerError(f"unknown event {event}")
    if not reason or not reason.strip():
        raise LedgerError("every ledger entry needs a reason")
    payload = dict(payload or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock(path):
        entries = read(path)
        state = replay(entries)
        git = git_state(root or path.parent)
        guarded = event in GUARDED_EVENTS or (event == "access" and payload.get("namespace") in namespaces.GUARDED)
        if guarded and git["dirty"] and not allow_dirty:
            raise LedgerError("guarded ledger events require a clean git tree (commit first)")
        entry = {"schema": SCHEMA, "index": len(entries), "at": utc_now(), "event": event, "dataset": dataset,
                 "reason": reason, "payload": payload, "git_commit": git["commit"], "git_dirty": git["dirty"],
                 "prev_sha256": state.head}
        entry["sha256"] = entry_hash(entry)
        apply(state, entry, namespaces)                      # raises before anything is written
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(canonical(entry) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return entry


__all__ = ["SCHEMA", "EVENTS", "STAGES", "INCIDENT_KINDS", "ATTEMPT_OUTCOMES", "LedgerError", "LedgerState",
           "read", "replay", "verify", "append", "entry_hash", "git_state", "canonical", "utc_now"]
