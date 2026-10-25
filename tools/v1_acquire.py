"""Declare registered v1 datasets and acquire pool slots (hashing only; nothing is parsed, data stay fresh).

    python tools/v1_acquire.py declare
    python tools/v1_acquire.py acquire --pool explore --needed 16 [--stream bybit-linear:BTCUSDT ...]

``acquire`` fills each stream of a pool with its registered primaries, moving to the registered backups in order
when a provider does not serve a primary. A date is used for every stream or for none: if any stream of a date is
NOT_AVAILABLE, the registered dataset-slot rule still applies per stream (``fill_slots``) and the per-date coverage is
reported, so a later design can require all-stream dates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lob.v1.data import acquire as A, store as S  # noqa: E402
from lob.v1.protocol.document import Workspace  # noqa: E402


def registry(root: Path) -> dict:
    return json.loads((root / "configs/v1/dataset-registry.json").read_text())


def declare(root: Path) -> int:
    ws = Workspace(root)
    state = ws.state()
    n = 0
    for d in registry(root)["datasets"]:
        if d["id"] in state.fresh:
            continue
        ws.append("declare_fresh", dataset=d["id"], reason=f"registered {d['pool']} slot {d['slot']} ({d['slot_kind']})",
                  payload={"role": d["role"], "pool": d["pool"], "slot": d["slot"], "slot_kind": d["slot_kind"],
                           "stream": d["stream"], "date": d["date"]})
        n += 1
    return n


def acquire(root: Path, pool: str, needed: int, streams: list[str] | None) -> dict:
    ws = Workspace(root)
    raw = S.RawStore(root / "data/v1/raw")
    reg = registry(root)
    out = {}
    by_stream: dict[str, list] = {}
    for d in reg["datasets"]:
        if d["pool"] == pool and (not streams or d["stream"] in streams):
            by_stream.setdefault(d["stream"], []).append(
                A.DatasetSpec(d["id"], d["connector"], d["segment"], d["symbol"], d["date"], d["role"], d["pool"],
                              d["slot"], d["slot_kind"]))
    state = ws.state()
    for stream, specs in sorted(by_stream.items()):
        have = [s for s in specs if s.id in state.acquired]
        todo = [s for s in specs if s.id not in state.acquired]
        need = max(needed - len(have), 0)
        got = A.fill_slots(todo, need, raw, ws=ws) if need else []
        out[stream] = {"already": len(have), "acquired": [a.spec.date for a in got if a.status == "AVAILABLE"],
                       "not_available": [a.spec.date for a in got if a.status != "AVAILABLE"]}
        print(stream, json.dumps(out[stream]), flush=True)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("declare", "acquire"))
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--pool")
    p.add_argument("--needed", type=int)
    p.add_argument("--stream", action="append")
    a = p.parse_args(argv)
    if a.command == "declare":
        print(json.dumps({"declared": declare(a.root)}))
    else:
        print(json.dumps(acquire(a.root, a.pool, a.needed, a.stream), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
