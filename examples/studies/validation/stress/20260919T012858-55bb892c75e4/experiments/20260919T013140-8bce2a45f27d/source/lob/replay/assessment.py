"""Streaming L2 reconstruction checks against separately published top-five books.

This is a feed-mechanics assessment, not an execution backtest or calibration fit.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import uuid
import zlib
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Iterator

import numpy as np

from ..experiments.registry import runtime_metadata, sha256_file, source_manifest, write_json
from .l2 import L2Replay, L2State, exact_decimal, timestamp_us

IDENTITY = ["exchange", "symbol", "timestamp", "local_timestamp"]
SNAPSHOT_COLUMNS = IDENTITY + [f"{side}[{i}].{field}" for i in range(5)
                               for side in ("asks", "bids") for field in ("price", "amount")]
TRADE_COLUMNS = IDENTITY + ["id", "side", "price", "amount"]


def _positive(value: str) -> Decimal:
    return exact_decimal(value, "reference price/amount")


def _timestamp(value: str) -> int:
    return timestamp_us(value, "reference timestamp")


def _rows(path: Path, columns: list[str], max_rows: int) -> Iterator[dict[str, str]]:
    """Bound compressed bytes, expanded bytes, line length and row count separately."""
    if path.stat().st_size > 256 * 1024**2:
        raise ValueError("reference file exceeds 256 MiB")
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    expanded = 0
    try:
        with opener(path, "rb") as handle:
            for number in range(max_rows + 2):
                raw = handle.readline(65_537)
                if not raw:
                    break
                expanded += len(raw)
                if len(raw) > 65_536 or expanded > 2 * 1024**3:
                    raise ValueError("reference line or expanded-byte limit exceeded")
                parsed = next(csv.reader([raw.decode("utf-8").rstrip("\r\n")], strict=True))
                if number == 0:
                    if parsed != columns:
                        raise ValueError("reference CSV header does not match the documented schema")
                    continue
                if number > max_rows:
                    raise ValueError("reference row limit exceeded; assessment is incomplete")
                if len(parsed) != len(columns):
                    raise ValueError(f"wrong column count at reference row {number}")
                yield dict(zip(columns, parsed))
            if expanded == 0:
                raise ValueError("empty reference file")
    except (EOFError, gzip.BadGzipFile, UnicodeError, csv.Error, zlib.error) as exc:
        raise ValueError(f"invalid or truncated reference CSV/gzip: {exc}") from exc


@dataclass(frozen=True)
class ReferenceBook:
    timestamp_us: int
    local_timestamp_us: int
    exchange: str
    symbol: str
    bids: tuple[tuple[Decimal, Decimal], ...]
    asks: tuple[tuple[Decimal, Decimal], ...]


def read_snapshots(path: str | Path, max_rows: int = 20_000_000) -> Iterator[ReferenceBook]:
    last = -1
    identity = None
    for row in _rows(Path(path), SNAPSHOT_COLUMNS, max_rows):
        timestamp, local = _timestamp(row["timestamp"]), _timestamp(row["local_timestamp"])
        current = row["exchange"], row["symbol"]
        if not all(current) or (identity is not None and current != identity):
            raise ValueError("reference changes exchange/symbol or has empty identity")
        if local <= last:
            raise ValueError("reference local timestamps must strictly increase")
        identity, last = current, local
        sides = []
        for side in ("bids", "asks"):
            levels = []
            missing = False
            for i in range(5):
                price, amount = row[f"{side}[{i}].price"], row[f"{side}[{i}].amount"]
                if not price and not amount:
                    missing = True
                elif not price or not amount or missing:
                    raise ValueError("reference missing levels must be paired trailing blanks")
                else:
                    levels.append((_positive(price), _positive(amount)))
            prices = [p for p, _ in levels]
            if len(set(prices)) != len(prices) or prices != sorted(prices, reverse=side == "bids"):
                raise ValueError("reference price levels are duplicate or unsorted")
            sides.append(tuple(levels))
        if sides[0] and sides[1] and sides[0][0][0] >= sides[1][0][0]:
            raise ValueError("published reference contains a crossed or locked book")
        yield ReferenceBook(timestamp, local, *current, *sides)


def _metrics(state: L2State) -> dict[str, float] | None:
    if not state.bids or not state.asks or state.bids[0][0] >= state.asks[0][0]:
        return None
    with localcontext() as context:
        context.prec = 80
        bid, ask = state.bids[0][0], state.asks[0][0]
        mid = (bid + ask) / 2
        spread_bps = float((ask - bid) / mid * 10_000)
    bid_depth = sum(float(q) for _, q in state.bids)
    ask_depth = sum(float(q) for _, q in state.asks)
    return {"mid_price": float(mid), "spread_bps": spread_bps,
            "bid_depth_5_native": bid_depth, "ask_depth_5_native": ask_depth,
            "imbalance_5": (bid_depth - ask_depth) / (bid_depth + ask_depth)}


class _Sampler:
    """Previous-tick sampling on integer UTC seconds; no backward interpolation."""
    def __init__(self) -> None:
        self.previous = None
        self.next_us = None
        self.samples: dict[str, list[float]] = {}
        self.invalid = 0
        self.stale = 0
        self.max_age_us = 0

    def add(self, state: L2State) -> None:
        if self.next_us is None:
            self.next_us = ((state.local_timestamp_us + 999_999) // 1_000_000) * 1_000_000
        while self.previous is not None and self.next_us < state.local_timestamp_us:
            self._sample()
        self.previous = state

    def _sample(self) -> None:
        if sum((len(self.samples.get("mid_price", [])), self.invalid)) >= 172_800:
            raise ValueError("assessment sample limit exceeded (maximum two days per file)")
        values = _metrics(self.previous)
        age = self.next_us - self.previous.local_timestamp_us
        self.max_age_us = max(age, self.max_age_us)
        self.stale += age > 5_000_000
        if values is None:
            self.invalid += 1
        else:
            for key, value in values.items():
                self.samples.setdefault(key, []).append(value)
        self.next_us += 1_000_000

    def result(self) -> dict:
        if self.previous is not None and self.next_us == self.previous.local_timestamp_us:
            self._sample()
        summary = {}
        for key, data in self.samples.items():
            values = np.asarray(data, dtype=float)
            summary[key] = {"count": len(data), "mean": float(values.mean()), "min": float(values.min()),
                            "median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
                            "max": float(values.max())}
        return {"sampling": "1-second UTC grid, previous completed local-timestamp group; no extrapolation past EOF",
                "invalid_book_samples": self.invalid, "samples_older_than_5_seconds": self.stale,
                "maximum_sample_age_seconds": self.max_age_us / 1_000_000, "metrics": summary}


def _trades(path: Path, identity: tuple[str, str], max_rows: int) -> dict:
    counts: Counter = Counter()
    hours: Counter = Counter()
    ids = set()
    sizes = []
    total = Decimal(0)
    first = last = None
    exchange_last = None
    regressions = duplicate_ids = missing_ids = 0
    for row in _rows(path, TRADE_COLUMNS, max_rows):
        if (row["exchange"], row["symbol"]) != identity:
            raise ValueError("trades and book exchange/symbol differ")
        timestamp, local = _timestamp(row["timestamp"]), _timestamp(row["local_timestamp"])
        if last is not None and local < last:
            raise ValueError("trade capture timestamps regress")
        regressions += exchange_last is not None and timestamp < exchange_last
        exchange_last = timestamp
        _positive(row["price"])
        amount = _positive(row["amount"])
        if row["side"] not in {"buy", "sell", "unknown"}:
            raise ValueError("unknown trade side enum")
        if row["id"]:
            duplicate_ids += row["id"] in ids
            ids.add(row["id"])
        else:
            missing_ids += 1
        counts[row["side"]] += 1
        hours[str((local // 3_600_000_000) % 24)] += 1
        first = local if first is None else first
        last = local
        with localcontext() as context:
            context.prec = 80  # bounded 31 integer + 30 fractional digits + at most 8 count digits
            total += amount
        sizes.append(float(amount))
    if first is None:
        raise ValueError("trade file contains no records")
    return {"count": sum(counts.values()), "sides": dict(counts), "hourly_counts_utc": dict(hours),
            "first_local_timestamp_us": first, "last_local_timestamp_us": last,
            "duplicate_trade_ids": duplicate_ids, "missing_trade_ids": missing_ids,
            "exchange_timestamp_regressions": regressions, "total_amount_native": str(total),
            "mean_amount_native": float(np.mean(sizes)), "p95_amount_native": float(np.quantile(sizes, .95))}


def assess_l2(updates: str | Path, snapshots: str | Path, trades: str | Path | None = None,
              *, max_rows: int = 20_000_000) -> dict:
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or not 1 <= max_rows <= 20_000_000:
        raise ValueError("max_rows must be an integer in [1, 20000000]")
    implementation = source_manifest()
    files = {"updates": Path(updates).resolve(), "snapshots": Path(snapshots).resolve()}
    if trades is not None:
        files["trades"] = Path(trades).resolve()
    source = {k: {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)} for k, p in files.items()}
    replay = L2Replay(files["updates"], depth=5, max_rows=max_rows)
    references = iter(read_snapshots(files["snapshots"], max_rows))
    reference = None
    compared = matched = unmatched_reference = missing_reference = 0
    timestamp_mismatches = 0
    reference_rows = 0
    differences = []
    previous_top = None
    first_time = last_time = None
    identity = None
    sampler = _Sampler()

    def next_reference():
        value = next(references, None)
        if value is not None and (value.exchange, value.symbol) != identity:
            raise ValueError("snapshot and updates exchange/symbol differ")
        return value

    def issue(kind: str, local: int) -> None:
        if len(differences) < 20:
            differences.append({"kind": kind, "local_timestamp_us": local})

    for state in replay:
        first_state = identity is None
        identity = state.exchange, state.symbol
        if first_state:
            reference = next_reference()
        first_time = state.local_timestamp_us if first_time is None else first_time
        last_time = state.local_timestamp_us
        sampler.add(state)
        top = state.bids, state.asks
        while reference is not None and reference.local_timestamp_us < state.local_timestamp_us:
            unmatched_reference += 1
            reference_rows += 1
            issue("reference_without_update_group", reference.local_timestamp_us)
            reference = next_reference()
        if reference is not None and reference.local_timestamp_us == state.local_timestamp_us:
            if (reference.exchange, reference.symbol) != identity:
                raise ValueError("snapshot and updates exchange/symbol differ")
            compared += 1
            reference_rows += 1
            if reference.timestamp_us != state.timestamp_us:
                timestamp_mismatches += 1
                issue("different_exchange_timestamp", state.local_timestamp_us)
            if top == (reference.bids, reference.asks):
                matched += 1
            else:
                issue("different_depth_or_price", state.local_timestamp_us)
            reference = next_reference()
        elif top != previous_top:
            missing_reference += 1
            issue("changed_top_without_reference", state.local_timestamp_us)
        previous_top = top
    while reference is not None:
        unmatched_reference += 1
        reference_rows += 1
        issue("reference_after_updates", reference.local_timestamp_us)
        reference = next_reference()
    if first_time is None or not reference_rows:
        raise ValueError("assessment requires nonempty reconstructed and reference streams")
    trade_result = _trades(files["trades"], identity, max_rows) if trades is not None else None
    if trade_result is not None:
        start_day = first_time // 86_400_000_000
        if any(trade_result[k] // 86_400_000_000 != start_day for k in
               ("first_local_timestamp_us", "last_local_timestamp_us")):
            raise ValueError("trade dates do not match the book UTC date")
    if any(sha256_file(files[k]) != value["sha256"] for k, value in source.items()):
        raise ValueError("source file changed during assessment")
    comparison_ok = (compared > 0 and compared == matched and not unmatched_reference
                     and not missing_reference and not timestamp_mismatches)
    sampling = sampler.result()
    warnings = [
        "L2 amounts are native feed units; they are not automatically ETH units or stock shares.",
        "L2 does not expose order IDs, FIFO queue positions, hidden liquidity or hypothetical strategy fills.",
        "Both inputs come from one provider/feed; matching them is a reconstruction check, not independent market truth.",
        "The normalized CSV lacks original exchange sequence IDs; capture gaps cannot be proven absent.",
        "Provider snapshots may remove crossed levels; this replay never silently cleans crossed books.",
        "No model was fitted and no strategy profitability, calibration or out-of-sample claim is made.",
    ]
    if sampling["samples_older_than_5_seconds"]:
        warnings.append("Some grid samples carry a book older than five seconds; inspect feed gaps.")
    replay_summary = replay.summary()
    anomalies = (sampling["invalid_book_samples"] or replay_summary["crossed_groups"]
                 or replay_summary["one_sided_groups"] or replay_summary["empty_groups"]
                 or (trade_result and trade_result["duplicate_trade_ids"]))
    if source_manifest() != implementation:
        raise ValueError("implementation changed during assessment")
    return {"schema_version": 1, "kind": "historical_l2_mechanics_assessment",
            "status": "PASS" if comparison_ok and not anomalies else "WARNING",
            "exchange": identity[0], "symbol": identity[1], "inputs": source,
            "first_local_timestamp_us": first_time, "last_local_timestamp_us": last_time,
            "observed_hours": (last_time - first_time) / 3_600_000_000,
            "implementation": implementation, "replay": replay_summary, "reference_comparison": {
                "status": "MATCH" if comparison_ok else "MISMATCH", "reference_rows": reference_rows,
                "compared": compared, "exact_matches": matched, "different_books": compared - matched,
                "exchange_timestamp_mismatches": timestamp_mismatches,
                "reference_without_update_group": unmatched_reference,
                "changed_top_without_reference": missing_reference, "first_issues": differences},
            "regular_grid": sampling, "trades": trade_result, "warnings": warnings}


def save_assessment(result: dict, out: str | Path) -> Path:
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12])
    run.mkdir()
    write_json(run / "assessment.json", result)
    manifest = result["implementation"]
    write_json(run / "provenance.json", {"created_at": datetime.now(timezone.utc).isoformat(),
               "source_manifest": manifest, "runtime": runtime_metadata(),
               "source_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()})
    comparison = result["reference_comparison"]
    report = ["# Historical L2 mechanics assessment", "", f"Status: **{result['status']}**",
              f"Instrument: {result['exchange']} / {result['symbol']}",
              f"Observed span: {result['observed_hours']:.4f} hours", "",
              f"Exact top-five matches: {comparison['exact_matches']:,} / {comparison['reference_rows']:,} reference rows.",
              f"Different books: {comparison['different_books']:,}; missing reference groups: {comparison['changed_top_without_reference']:,}; unmatched reference rows: {comparison['reference_without_update_group']:,}.", "",
              "All distributions use a one-second UTC grid and the latest completed captured book.", "",
              "| Metric | Mean | Median | 95th percentile |", "|---|---:|---:|---:|"]
    for key, values in result["regular_grid"]["metrics"].items():
        report.append(f"| {key} | {values['mean']:.6g} | {values['median']:.6g} | {values['p95']:.6g} |")
    if result["trades"]:
        report.extend(["", f"Trade records checked: {result['trades']['count']:,}."])
    report.extend(["", "## Scope and limitations", ""] + [f"- {w}" for w in result["warnings"]])
    report.extend(["", "Full counters, exact input hashes, timestamps and mismatch locations are in `assessment.json`.",
                   "Source implementation hashes and runtime are in `provenance.json`.", ""])
    (run / "report.md").write_text("\n".join(report), encoding="utf-8")
    write_json(run / "manifest.json", {"algorithm": "sha256", "files": {
        p.name: sha256_file(p) for p in sorted(run.iterdir()) if p.is_file()}})
    return run
