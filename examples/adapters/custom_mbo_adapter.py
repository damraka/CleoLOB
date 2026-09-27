"""Template: a source-native order-level (MBO) adapter with a declared contract.

Copy this file for a new venue. The steps are:

1. Declare every interpretation rule in a ``SourceContract``. Its hash travels with
   results, so a rule cannot change silently between development and validation.
2. Convert native rows to ``LifecycleEvent`` in exact integer ticks and lots. Never
   invent identities and never repair gaps; emit ``GAP`` instead.
3. Supply independently published reference books, if the venue has them.
4. Run ``lob.mbo_sources.validate_source``. It replays twice (determinism), counts
   lifecycle anomalies without repairing them, and compares against references.

The CSV format here is illustrative: ``ts_ns,type,order_id,side,price,qty`` with
types ADD, CANCEL, EXECUTE and GAP; optional ``ts_ns,BOOK,bids,asks`` rows hold a
reference top of book as ``price:qty|price:qty``.

Running this file validates an embedded *synthetic* fixture. That exercises the
mechanics only and is never evidence about any venue.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

from lob.mbo_sources import SourceContract, exact_units, validate_source
from lob.order_lifecycle import LifecycleEvent, LifecycleInputError, LifecycleSemantics


def example_contract(instrument: str, tick_size: str, lot_size: str) -> SourceContract:
    return SourceContract(
        source="Example CSV order-level export (template)", venue="example-venue", instrument=instrument,
        tick_size=tick_size, lot_size=lot_size,
        identity="Venue order IDs, used unchanged; unknown or reused IDs are anomalies.",
        sequence="File row order; no venue sequence numbers, so losses are undetectable except as GAP rows.",
        timestamps="Exchange nanoseconds; no receive clock.",
        snapshot="The file starts empty (RESET); orders resting before the file are not represented.",
        priority="Not published; FIFO is not claimed.",
        execution="EXECUTE decrements the identified resting (maker) order.",
        hidden_liquidity="Not reported; hidden size is unknown.",
        fifo_established=False, reference_alignment="BOOK row after the preceding event",
        adapter_version="example-csv-1")


class ExampleCsvAdapter:
    def __init__(self, path: str | Path, *, instrument: str = "EXAMPLE", tick_size: str = "0.01",
                 lot_size: str = "1"):
        self.path = Path(path)
        self.contract = example_contract(instrument, tick_size, lot_size)
        self.stats: dict = {}

    def build(self) -> tuple[list[LifecycleEvent], list[dict]]:
        tick, lot = self.contract.tick_size, self.contract.lot_size
        events: list[LifecycleEvent] = [LifecycleEvent(sequence=0, kind="RESET")]
        references: list[dict] = []
        self.stats = {"rows": 0}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = [x.strip() for x in line.split(",")]
            ts, kind = int(fields[0]), fields[1]
            self.stats["rows"] += 1
            if kind == "BOOK":
                def levels(text: str) -> list[list[int]]:
                    return [[exact_units(p, tick, "price"), exact_units(q, lot, "qty")]
                            for p, q in (x.split(":") for x in text.split("|") if x)]
                references.append({"after_sequence": len(events) - 1, "exchange_ts_ns": ts,
                                   "bids": levels(fields[2]), "asks": levels(fields[3])})
                continue
            order_id, side = fields[2], fields[3]
            price = exact_units(fields[4], tick, "price") if fields[4] else None
            qty = exact_units(fields[5], lot, "qty") if fields[5] else None
            common = {"exchange_ts_ns": ts}
            if kind == "ADD":
                events.append(LifecycleEvent(len(events), "ADD", order_id=order_id, side=side, price=price,
                                             quantity=qty, **common))
            elif kind == "CANCEL":
                events.append(LifecycleEvent(len(events), "CANCEL", order_id=order_id, side=side,
                                             quantity=qty, **common))
            elif kind == "EXECUTE":
                events.append(LifecycleEvent(len(events), "EXECUTE", order_id=order_id, side=side, price=price,
                                             quantity=qty, **common))
            elif kind == "GAP":
                events.append(LifecycleEvent(len(events), "GAP", **common))
            else:
                raise LifecycleInputError(f"undeclared row type {kind!r}")  # refuse, never guess
        return events, references


SYNTHETIC_FIXTURE = """\
1,ADD,b1,BUY,99.99,10
2,ADD,b2,BUY,99.99,5
3,ADD,a1,SELL,100.01,8
4,BOOK,99.99:15,100.01:8
5,EXECUTE,b1,BUY,99.99,4
6,CANCEL,b2,BUY,,
7,BOOK,99.99:6,100.01:8
"""


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.csv"
        path.write_text(SYNTHETIC_FIXTURE, encoding="utf-8")
        result = validate_source(ExampleCsvAdapter(path), LifecycleSemantics(census_complete=False))
    summary = {"evidence": "SYNTHETIC fixture: mechanics only, never venue evidence",
               "contract_sha256": result["contract"]["contract_sha256"],
               "deterministic_replay": result["deterministic_replay"],
               "reference_comparison": {k: result["reference_comparison"].get(k)
                                        for k in ("compared", "exact", "exact_fraction")},
               "executions": result["executions"]}
    print(json.dumps(summary, indent=2))
    return 0 if result["deterministic_replay"] else 1


if __name__ == "__main__":
    sys.exit(main())
