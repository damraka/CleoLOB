"""Bounded, fail-closed MBO validation with explicit empirical evidence status."""
from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifacts import portable_provenance, seal_artifacts
from .capabilities import Capability, EvidenceStatus
from .datasets import CanonicalMBOAdapter, MappedMBOCSVAdapter, mbo_adapter, source_sha256
from .experiments.registry import write_json
from .replay.io import _json
from .replay.schema import _integer


@dataclass(frozen=True)
class AggregateReference:
    """An independently supplied state after one canonical event sequence.

    ``depth=None`` means the full displayed book; positive depth means top-N.
    The source must align native observation boundaries to canonical sequence.
    """

    sequence: int
    timestamp_ns: int
    bids: tuple[tuple[int, int], ...]
    asks: tuple[tuple[int, int], ...]
    depth: int | None = None

    def __post_init__(self) -> None:
        for name in ("sequence", "timestamp_ns"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=0))
        if self.depth is not None:
            object.__setattr__(self, "depth", _integer(self.depth, "depth"))
        for side in ("bids", "asks"):
            pairs = tuple((_integer(p, "price_ticks"), _integer(q, "quantity")) for p, q in getattr(self, side))
            prices = [p for p, _ in pairs]
            if len(set(prices)) != len(prices) or prices != sorted(prices, reverse=side == "bids"):
                raise ValueError("reference prices must be distinct and best-first")
            if self.depth is not None and len(pairs) > self.depth:
                raise ValueError("reference has more levels than declared depth")
            object.__setattr__(self, side, pairs)

    def to_dict(self) -> dict:
        return {"sequence": self.sequence, "timestamp_ns": self.timestamp_ns, "depth": self.depth,
                "bids": [list(pair) for pair in self.bids], "asks": [list(pair) for pair in self.asks]}


def load_references(path: str | Path, *, max_references: int = 100_000) -> list[AggregateReference]:
    path = Path(path)
    if path.stat().st_size > 128 * 1024**2:
        raise ValueError("aggregate reference file exceeds 128 MiB")
    result = []
    with path.open("rb") as handle:
        while True:
            line = handle.readline(2 * 1024**2 + 1)
            if not line:
                break
            if len(line) > 2 * 1024**2 or len(result) >= max_references:
                raise ValueError("aggregate reference resource limit exceeded")
            row = _json(line.decode("utf-8"))
            if not isinstance(row, dict) or set(row) != {"sequence", "timestamp_ns", "bids", "asks", "depth"}:
                raise ValueError("reference requires exactly sequence, timestamp_ns, bids, asks, depth")
            result.append(AggregateReference(**row))
    if not result:
        raise ValueError("aggregate reference source is empty")
    return result


def _error_code(exc: Exception) -> str:
    current = exc
    while current is not None:
        if hasattr(current, "code"):
            return str(current.code)
        current = current.__cause__
    return "INVALID_SOURCE"


def _aggregate(oracle: Mapping[str, tuple[str, int, int]]) -> dict[str, list[list[int]]]:
    result = {}
    for side, label in (("BUY", "bids"), ("SELL", "asks")):
        levels: Counter = Counter()
        for s, price, quantity in oracle.values():
            if s == side:
                levels[price] += quantity
        result[label] = [[p, levels[p]] for p in sorted(levels, reverse=side == "BUY")]
    return result


def validate_mbo(adapter: CanonicalMBOAdapter | MappedMBOCSVAdapter,
                 references: Iterable[AggregateReference] = ()) -> dict[str, Any]:
    """Validate the complete source or retain an invalid prefix plus first error.

    Independent aggregation/queue bookkeeping checks implementation consistency.
    Only supplied aggregate references check source agreement. Neither proves
    exchange truth or hypothetical passive fills. Counts after a first invalid
    event are intentionally unknown; the validator never skips a broken suffix.
    """
    pending = {}
    digest = hashlib.sha256()
    for reference in references:
        if not isinstance(reference, AggregateReference):
            raise ValueError("references must be AggregateReference objects")
        if len(pending) >= 100_000:
            raise ValueError("aggregate references exceed 100000 records")
        if pending and reference.sequence <= next(reversed(pending)):
            raise ValueError("reference sequences must be strictly increasing and distinct")
        pending[reference.sequence] = reference
        digest.update(json.dumps(reference.to_dict(), sort_keys=True, separators=(",", ":")).encode() + b"\n")
    reference_count = len(pending)
    counts = {name: 0 for name in (
        "events", "orders_observed", "snapshot_order_records", "adds", "modifications", "cancels", "deletes",
        "executions", "partial_executions", "terminal_executions", "executed_quantity", "trade_prints",
        "priority_resets", "left_censored_order_records", "census_censored_trajectories",
        "known_terminal_lifetimes", "terminal_lifetime_sum_ns", "internal_aggregate_mismatches",
        "queue_checks", "queue_inconsistencies", "reference_comparisons", "aggregate_book_mismatches",
        "price_level_mismatches", "volume_mismatches", "absolute_volume_difference_lots",
        "reference_timestamp_mismatches", "sequence_gaps", "duplicate_sequences", "duplicate_order_ids",
        "timestamp_reversals", "invalid_transitions",
    )}
    oracle: dict[str, tuple[str, int, int]] = {}
    births: dict[str, int | None] = {}
    seen = set()
    errors = []
    event_types: Counter = Counter()
    lifetime_min, lifetime_max = None, None
    last_time = None
    queue_available = adapter.capability_contract.supports(Capability.QUANTITY_AHEAD)
    try:
        for event in adapter.events():
            counts["events"] += 1
            event_types[event.event_type] += 1
            last_time = event.timestamp_ns
            kind, oid = event.event_type, event.order_id
            previous = oracle.get(oid)
            if kind in {"SNAPSHOT", "RESET"}:
                counts["census_censored_trajectories"] += len(oracle)
                oracle = {o.order_id: (o.side.value, o.price_ticks, o.quantity) for o in event.orders}
                births = {o.order_id: None for o in event.orders}
                seen.update(oracle)
                counts["snapshot_order_records"] += len(oracle)
                counts["left_censored_order_records"] += len(oracle)
            elif kind == "ADD":
                counts["adds"] += 1
                oracle[oid] = (event.side.value, event.price_ticks, event.quantity)
                births[oid] = event.timestamp_ns
                seen.add(oid)
            elif kind == "MODIFY":
                counts["modifications"] += 1
                side, price, quantity = previous
                new_price = price if event.price_ticks is None else event.price_ticks
                new_quantity = quantity if event.quantity is None else event.quantity
                if new_price != price or new_quantity > quantity:
                    counts["priority_resets"] += 1
                    del oracle[oid]
                oracle[oid] = side, new_price, new_quantity
            elif kind in {"CANCEL", "EXECUTE", "DELETE"}:
                side, price, quantity = previous
                remaining = 0 if kind == "DELETE" else quantity - event.quantity
                counts[{"CANCEL": "cancels", "EXECUTE": "executions", "DELETE": "deletes"}[kind]] += 1
                if kind == "EXECUTE":
                    counts["executed_quantity"] += event.quantity
                    counts["partial_executions" if remaining else "terminal_executions"] += 1
                if remaining:
                    oracle[oid] = side, price, remaining
                else:
                    del oracle[oid]
                    birth = births.pop(oid)
                    if birth is not None:
                        lifetime = event.timestamp_ns - birth
                        counts["known_terminal_lifetimes"] += 1
                        counts["terminal_lifetime_sum_ns"] += lifetime
                        lifetime_min = lifetime if lifetime_min is None else min(lifetime, lifetime_min)
                        lifetime_max = lifetime if lifetime_max is None else max(lifetime, lifetime_max)
            elif kind == "TRADE":
                counts["trade_prints"] += 1
            counts["internal_aggregate_mismatches"] += int(_aggregate(oracle) != adapter.book.aggregate_l2())
            if queue_available:
                queues: dict[tuple[str, int], list[str]] = {}
                for key, (side, price, _) in oracle.items():
                    queues.setdefault((side, price), []).append(key)
                # Check all surviving levels, including after census/terminal
                # events whose own ID is absent from the reconstructed book.
                for (side, price), queue in queues.items():
                    counts["queue_checks"] += 1
                    counts["queue_inconsistencies"] += int(tuple(queue) != adapter.book.queue(side, price))
            if queue_available and oid in oracle:
                side, price, _ = oracle[oid]
                queue = [key for key, (s, p, _) in oracle.items() if (s, p) == (side, price)]
                position = queue.index(oid)
                observed = adapter.book.queue_metrics(oid)
                counts["queue_checks"] += 1
                counts["queue_inconsistencies"] += int(
                    tuple(queue) != adapter.book.queue(side, price)
                    or observed["position"] != position
                    or observed["quantity_ahead"] != sum(oracle[key][2] for key in queue[:position]))
            reference = pending.pop(event.sequence, None)
            if reference is not None:
                actual = adapter.book.aggregate_l2(reference.depth)
                expected = {side: [list(pair) for pair in getattr(reference, side)] for side in ("bids", "asks")}
                counts["reference_comparisons"] += 1
                counts["aggregate_book_mismatches"] += int(actual != expected)
                counts["reference_timestamp_mismatches"] += int(reference.timestamp_ns != event.timestamp_ns)
                for side in ("bids", "asks"):
                    left, right = dict(actual[side]), dict(expected[side])
                    counts["price_level_mismatches"] += len(left.keys() ^ right.keys())
                    counts["volume_mismatches"] += sum(left.get(p, 0) != right.get(p, 0) for p in left.keys() | right.keys())
                    counts["absolute_volume_difference_lots"] += sum(
                        abs(left.get(p, 0) - right.get(p, 0)) for p in left.keys() | right.keys())
    except (ValueError, TypeError, OSError, EOFError, zlib.error) as exc:
        code = _error_code(exc)
        errors.append({"code": code, "message": str(exc), "valid_prefix_events": counts["events"]})
        names = {"SEQUENCE_GAP": "sequence_gaps", "DUPLICATE_SEQUENCE": "duplicate_sequences",
                 "DUPLICATE_ORDER_ID": "duplicate_order_ids", "TIMESTAMP_REVERSAL": "timestamp_reversals"}
        counts[names.get(code, "invalid_transitions")] += 1
    counts["orders_observed"] = len(seen)
    counts["unmatched_references"] = len(pending)
    counts["right_censored_at_eof"] = len(oracle) if adapter.stats["complete"] else None
    deterministic = None
    if not errors and adapter.stats["complete"]:
        try:
            twin = adapter.reopen()
            for _ in twin.events():
                pass
            deterministic = (adapter.stats == twin.stats and adapter.book.snapshot() == twin.book.snapshot())
            if not deterministic:
                errors.append({"code": "NONDETERMINISTIC_REPLAY", "message": "repeat digest/end state differ"})
        except (ValueError, TypeError, OSError, EOFError, zlib.error) as exc:
            errors.append({"code": "REPEAT_REPLAY_FAILED", "message": str(exc)})
    failed = any(counts[key] for key in ("internal_aggregate_mismatches", "queue_inconsistencies",
                                        "aggregate_book_mismatches", "reference_timestamp_mismatches", "unmatched_references"))
    status = EvidenceStatus.INVALID if errors else EvidenceStatus.FAILED if failed else EvidenceStatus.ESTABLISHED
    agreement = (EvidenceStatus.NOT_AVAILABLE if not reference_count else
                 EvidenceStatus.ESTABLISHED if status == EvidenceStatus.ESTABLISHED else status)
    historical = (EvidenceStatus.NOT_AVAILABLE if adapter.metadata.source_kind == "synthetic" else
                  EvidenceStatus.NOT_ESTABLISHED if not reference_count else status)
    return {
        "schema": "cleolob-mbo-validation-v1", "status": status.value,
        "evidence_kind": adapter.metadata.source_kind,
        "claim": "declared-source reconstruction consistency, not exchange truth or counterfactual fills",
        "real_historical_mbo_validation": historical.value,
        "source_aggregate_agreement": agreement.value,
        "deterministic_replay": deterministic,
        "determinism_scope": "complete canonical event digest, counters and final reconstructed state",
        "adapter": {"name": adapter.adapter_name, "version": adapter.adapter_version},
        "metadata": adapter.metadata.to_dict(), "capabilities": adapter.capability_contract.to_dict(),
        "source": adapter.stats, "counts": counts, "event_types": dict(sorted(event_types.items())),
        "reference_canonical_sha256": digest.hexdigest() if reference_count else None,
        "reference_count": reference_count,
        "queue_consistency_status": (
            EvidenceStatus.NOT_AVAILABLE.value if not queue_available else
            EvidenceStatus.NOT_ESTABLISHED.value if not counts["queue_checks"] else status.value),
        "lifetime_ns": {"minimum": lifetime_min, "maximum": lifetime_max,
                        "mean": counts["terminal_lifetime_sum_ns"] / counts["known_terminal_lifetimes"]
                        if counts["known_terminal_lifetimes"] else None,
                        "scope": "observed ADD to terminal cancellation/execution; censored orders excluded"},
        "last_valid_timestamp_ns": last_time,
        "diagnostic_scope": "validated prefix and first rejected event; no skipped or repaired suffix",
        "errors": errors,
        "limitations": ["Snapshot FIFO and venue amendment rules are declared source obligations.",
                        "Independent oracle checks implementation consistency, not exchange correctness.",
                        "Recorded executions behind visible orders are retained; they are not reassigned to FIFO head.",
                        "Sequence continuity of normalized indices cannot establish original feed completeness.",
                        "No hidden liquidity, passive counterfactual fills or profitability established."],
    }


def run_mbo_validation(source: str | Path, manifest: str | Path | Mapping[str, Any], out: str | Path,
                       *, reference_path: str | Path | None = None, max_events: int = 100_000) -> dict:
    """Write a fresh verifiable result; never overwrite a registered result."""
    if not isinstance(manifest, Mapping):
        manifest_path = Path(manifest)
        if manifest_path.stat().st_size > 1024**2:
            raise ValueError("adapter manifest exceeds 1 MiB")
        manifest = _json(manifest_path.read_text(encoding="utf-8"))
    adapter = mbo_adapter(source, manifest, max_events=max_events)
    reference_hash = source_sha256(reference_path, max_bytes=128 * 1024**2) if reference_path is not None else None
    references = load_references(reference_path) if reference_path is not None else []
    result = validate_mbo(adapter, references)
    if reference_path is not None and source_sha256(reference_path, max_bytes=128 * 1024**2) != reference_hash:
        raise ValueError("reference source changed during validation")
    result["reference_source_sha256"] = reference_hash
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "adapter-manifest.json", dict(manifest))
    write_json(root / "result.json", result)
    write_json(root / "provenance.json", portable_provenance())
    lines = ["# MBO validation", "", f"Status: **{result['status']}**; source kind: **{result['evidence_kind']}**.",
             f"Real historical MBO validation: **{result['real_historical_mbo_validation']}**.",
             f"Source aggregate agreement: **{result['source_aggregate_agreement']}**.", "",
             result["claim"], "", "| Diagnostic | Count |", "| --- | ---: |"]
    lines += [f"| {key} | {value} |" for key, value in result["counts"].items()]
    lines += ["", result["diagnostic_scope"], ""]
    lines += [f"- {item['code']}: {item['message']}" for item in result["errors"]]
    lines += ["", *result["limitations"], ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    seal_artifacts(root)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=100_000)
    args = parser.parse_args(argv)
    result = run_mbo_validation(args.source, args.manifest, args.out,
                                reference_path=args.references, max_events=args.max_events)
    print(json.dumps({"status": result["status"], "real_historical_mbo_validation":
                      result["real_historical_mbo_validation"], "out": str(args.out)}, indent=2))
    return 0 if result["status"] == EvidenceStatus.ESTABLISHED.value else 1


if __name__ == "__main__":
    raise SystemExit(main())
