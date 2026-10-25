"""Event-sourced journal, snapshot and recovery (M37).

Every command the engine receives is appended to a ``Journal`` (kind, time, arguments) with a running SHA-256 chain.
``snapshot`` stores the journal position, chain head and the engine's canonical state; ``recover`` rebuilds an engine
from a snapshot's journal prefix (replaying the commands, because the engine is deterministic) and continues with the
tail. Recovery is verified: the rebuilt state must equal the snapshot state at the snapshot position, and the final
state must equal the live engine's state.
"""
from __future__ import annotations

import hashlib
import json

from .matching import MatchingEngine, Order
from .rules import Rules


class Journal:
    def __init__(self) -> None:
        self.entries: list[dict] = []
        self.head = "0" * 64

    def record(self, kind: str, ts: int, **kw) -> None:
        e = {"i": len(self.entries), "kind": kind, "ts": ts, **kw}
        body = json.dumps(e, sort_keys=True, separators=(",", ":"))
        self.head = hashlib.sha256((self.head + body).encode()).hexdigest()
        self.entries.append(e)

    def to_jsonl(self) -> str:
        return "".join(json.dumps(e, sort_keys=True) + "\n" for e in self.entries)


def attach(engine: MatchingEngine) -> Journal:
    j = Journal()
    engine.journal = j
    return j


def snapshot(engine: MatchingEngine) -> dict:
    j = engine.journal
    if j is None:
        raise ValueError("engine has no journal")
    return {"position": len(j.entries), "head": j.head, "state": engine.state(), "rules": engine.rules.to_dict()}


def _apply(engine: MatchingEngine, e: dict) -> None:
    if e["kind"] == "submit":
        engine.submit(Order(**{k: v for k, v in e["order"].items() if k not in ("remaining", "seq", "status")}),
                      e["ts"])
    elif e["kind"] == "cancel":
        engine.cancel(e["oid"], e["ts"], owner=e.get("owner"))
    elif e["kind"] == "replace":
        engine.replace(e["oid"], e["ts"], price=e.get("price"), qty=e.get("qty"), owner=e.get("owner"))
    else:
        raise ValueError(f"unknown journal entry {e['kind']}")


def replay(entries: list[dict], rules: Rules) -> MatchingEngine:
    engine = MatchingEngine(rules)
    attach(engine)
    for e in entries:
        _apply(engine, e)
    return engine


def recover(snap: dict, journal: Journal) -> MatchingEngine:
    """Engine rebuilt from the journal prefix of ``snap`` (verified) and then the journal tail."""
    rules = Rules(**snap["rules"])
    prefix = journal.entries[: snap["position"]]
    engine = replay(prefix, rules)
    if engine.state() != snap["state"] or engine.journal.head != snap["head"]:
        raise ValueError("journal prefix does not reproduce the snapshot")
    for e in journal.entries[snap["position"]:]:
        _apply(engine, e)
    return engine


__all__ = ["Journal", "attach", "snapshot", "replay", "recover"]
