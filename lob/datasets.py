"""Auditable adapters with exact canonical units and immutable source metadata.

These adapters never infer order IDs or venue semantics from aggregate data.
Metadata is a declared input contract, not independent verification of a vendor.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol, TypeVar, runtime_checkable

from .capabilities import Capability, CapabilityContract, DataLevel, L2_CONTRACT
from .mbo import MBOBook, MBOEvent, MBOReplay, _positive
from .replay.io import _json
from .replay.l2 import L2Replay, L2State, exact_decimal


def source_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4096:
        raise ValueError(f"{name} must be nonempty text of at most 4096 characters")
    return value


def _pairs(value: Any, name: str) -> tuple[tuple[str, str], ...]:
    values = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    result = tuple((_text(k, name), _text(v, name)) for k, v in values)
    if len(dict(result)) != len(result):
        raise ValueError(f"{name} contains duplicate keys")
    return tuple(sorted(result))


@dataclass(frozen=True)
class DatasetMetadata:
    source_id: str
    venue: str
    instrument: str
    data_level: DataLevel
    source_kind: str
    source_sha256: str
    timestamp_semantics: str
    sequence_semantics: str
    tick_size: str | None
    lot_size: str | None
    field_mapping: tuple[tuple[str, str], ...]
    provenance: tuple[tuple[str, str], ...]
    licensing: str
    snapshot_semantics: str
    execution_semantics: str
    priority_semantics: str
    timezone: str = "UTC"
    missing_data_behavior: str = "reject"

    def __post_init__(self) -> None:
        for name in ("source_id", "venue", "instrument", "timestamp_semantics", "sequence_semantics",
                     "licensing", "snapshot_semantics", "execution_semantics", "priority_semantics"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "data_level", DataLevel(self.data_level))
        if self.source_kind not in {"historical", "synthetic"}:
            raise ValueError("source_kind must explicitly be historical or synthetic")
        if self.source_kind == "historical" and "synthetic" in (self.venue + self.instrument).lower():
            raise ValueError("a synthetic fixture cannot be declared historical")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        if self.timezone != "UTC":
            raise ValueError("canonical timestamps must be normalized to UTC before ingestion")
        if self.missing_data_behavior != "reject":
            raise ValueError("missing_data_behavior must be reject; gaps cannot be silently repaired")
        for name in ("tick_size", "lot_size"):
            if getattr(self, name) is not None:
                exact_decimal(getattr(self, name), name)
        if self.data_level == DataLevel.MBO and (self.tick_size is None or self.lot_size is None):
            raise ValueError("integer-tick/lot MBO requires known tick_size and lot_size")
        object.__setattr__(self, "field_mapping", _pairs(self.field_mapping, "field_mapping"))
        object.__setattr__(self, "provenance", _pairs(self.provenance, "provenance"))
        if not self.field_mapping or not self.provenance:
            raise ValueError("field_mapping and provenance must be documented")

    def to_dict(self) -> dict:
        return {name: (value.value if isinstance(value, DataLevel) else
                       dict(value) if name in {"field_mapping", "provenance"} else value)
                for name, value in self.__dict__.items()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DatasetMetadata:
        if not isinstance(value, Mapping):
            raise ValueError("dataset metadata must be an object")
        try:
            return cls(**dict(value))
        except TypeError as exc:
            raise ValueError(f"incompatible dataset metadata: {exc}") from exc


Event = TypeVar("Event", covariant=True)


@runtime_checkable
class DatasetAdapter(Protocol[Event]):
    adapter_name: str
    adapter_version: str
    metadata: DatasetMetadata
    capability_contract: CapabilityContract

    def events(self) -> Iterator[Event]: ...

    @property
    def stats(self) -> dict: ...


class _Adapter:
    adapter_version = "1"

    def __init__(self, path: str | Path, metadata: DatasetMetadata, contract: CapabilityContract):
        if metadata.data_level != contract.level:
            raise ValueError("dataset data_level and capability contract disagree")
        self.path = Path(path)
        self._metadata = metadata
        self._capability_contract = contract
        self._started = False
        self._stats = {"complete": False, "events": 0, "source_sha256": None, "canonical_sha256": None}

    @property
    def metadata(self) -> DatasetMetadata:
        return self._metadata

    @property
    def capability_contract(self) -> CapabilityContract:
        return self._capability_contract

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def _begin(self, max_file_bytes: int) -> None:
        if self._started:
            raise ValueError("dataset adapters are one-pass; reopen for deterministic replay")
        self._started = True
        if not self.path.is_file() or self.path.stat().st_size > max_file_bytes:
            raise ValueError("source must be a regular file within max_file_bytes")
        if source_sha256(self.path) != self.metadata.source_sha256:
            raise ValueError("source SHA-256 does not match dataset metadata")

    def _identity(self, venue: str, instrument: str) -> None:
        if (venue, instrument) != (self.metadata.venue, self.metadata.instrument):
            raise ValueError("event identity does not match dataset metadata")

    def _finish(self, canonical_digest: str) -> None:
        if source_sha256(self.path) != self.metadata.source_sha256:
            raise ValueError("source changed during adapter iteration")
        self._stats.update(complete=True, source_sha256=self.metadata.source_sha256,
                           canonical_sha256=canonical_digest)


class TardisL2Adapter(_Adapter):
    """Existing Decimal/capture-group replay behind the dataset protocol.

    Normalized events are completed ``L2State`` observations. Snapshot resets,
    source-native amounts and crossing diagnostics retain v0.3 semantics.
    """

    adapter_name = "tardis-incremental-l2"

    def __init__(self, path: str | Path, metadata: DatasetMetadata, **replay_options):
        if metadata.data_level != DataLevel.L2:
            raise ValueError("TardisL2Adapter requires aggregate L2 metadata")
        super().__init__(path, metadata, L2_CONTRACT)
        self.replay = L2Replay(path, **replay_options)

    def events(self) -> Iterator[L2State]:
        self._begin(self.replay.max_file_bytes)
        digest = hashlib.sha256()
        for state in self.replay:
            self._identity(state.exchange, state.symbol)
            row = {"timestamp_us": state.timestamp_us, "local_timestamp_us": state.local_timestamp_us,
                   "exchange": state.exchange, "symbol": state.symbol, "is_snapshot": state.is_snapshot,
                   "bids": [[str(p), str(q)] for p, q in state.bids],
                   "asks": [[str(p), str(q)] for p, q in state.asks],
                   "bid_levels": state.bid_levels, "ask_levels": state.ask_levels,
                   "rows_in_group": state.rows_in_group}
            digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            self._stats["events"] += 1
            yield state
        self._stats["replay"] = self.replay.stats
        self._finish(digest.hexdigest())


def _mbo_contract(metadata: DatasetMetadata, contract: CapabilityContract) -> None:
    if metadata.data_level != DataLevel.MBO or contract.level != DataLevel.MBO:
        raise ValueError("MBO adapter requires order-level data and capability contract")
    contract.require(Capability.ORDER_IDENTITY, Capability.AGGREGATE_DEPTH)
    if metadata.sequence_semantics not in {"normalized_contiguous_per_instrument", "source_contiguous_per_instrument"}:
        raise ValueError("MBO canonical sequences must be contiguous per instrument; document native mapping separately")
    if metadata.snapshot_semantics not in {"complete_census_source_fifo", "complete_census_ordering_unknown"}:
        raise ValueError("MBO reconstruction requires a complete initial census; ordering may remain unknown")
    if metadata.priority_semantics != "same_price_reduction_retains_increase_or_reprice_resets":
        raise ValueError("unsupported priority semantics: translate native amendments explicitly")
    if metadata.execution_semantics != "maker_execute_decrements_once_trade_is_unlinked":
        raise ValueError("unsupported execution semantics: never map non-decrementing fills to EXECUTE")
    if contract.supports(Capability.EXACT_FIFO_POSITION):
        if metadata.snapshot_semantics != "complete_census_source_fifo":
            raise ValueError("exact FIFO requires a complete source-ordered snapshot census")
    if contract.supports(Capability.EXCHANGE_SEQUENCE_CONTINUITY):
        if metadata.sequence_semantics != "source_contiguous_per_instrument":
            raise ValueError("normalized sequence indices cannot prove exchange sequence continuity")


class CanonicalMBOAdapter(_Adapter):
    """Strict JSONL/NDJSON ingestion; no identity or semantic inference."""

    adapter_name = "canonical-mbo-jsonl"

    def __init__(self, path: str | Path, metadata: DatasetMetadata,
                 contract: CapabilityContract, **limits):
        _mbo_contract(metadata, contract)
        super().__init__(path, metadata, contract)
        self.limits = dict(limits)
        self.replay = MBOReplay(path, capability_contract=contract, **limits)
        self.book = self.replay.book

    def reopen(self) -> CanonicalMBOAdapter:
        return type(self)(self.path, self.metadata, self.capability_contract, **self.limits)

    def events(self) -> Iterator[MBOEvent]:
        self._begin(self.replay._limits["max_file_bytes"])
        for event in self.replay:
            self._identity(event.venue, event.symbol)
            self._stats["events"] += 1
            yield event
        self._finish(self.replay.stats["canonical_sha256"])


class MappedMBOCSVAdapter(_Adapter):
    """Map declared source columns/actions to canonical integer-tick events.

    The mapping is canonical-field -> source-column. Capture clocks are UTC
    integer nanoseconds, prices integer ticks and sizes integer lots; conversion
    into these units belongs in a documented source-specific normalization step.
    Missing IDs, unsupported fields and raw actions are rejected, never guessed.
    """

    adapter_name = "mapped-mbo-csv"

    def __init__(self, path: str | Path, metadata: DatasetMetadata, contract: CapabilityContract,
                 *, action_mapping: Mapping[str, str], side_mapping: Mapping[str, str],
                 max_file_bytes: int = 128 * 1024**2, max_expanded_bytes: int = 256 * 1024**2,
                 max_line_bytes: int = 2 * 1024**2, max_events: int = 1_000_000, **book_limits):
        _mbo_contract(metadata, contract)
        super().__init__(path, metadata, contract)
        self._action_mapping = _pairs(action_mapping, "action_mapping")
        self._side_mapping = _pairs(side_mapping, "side_mapping")
        for _, value in self._action_mapping:
            if value not in {"ADD", "MODIFY", "CANCEL", "DELETE", "EXECUTE", "TRADE",
                             "SNAPSHOT", "RESET", "HALT", "RESUME"}:
                raise ValueError(f"unsupported mapped canonical action {value}")
        if any(value not in {"BUY", "SELL"} for _, value in self._side_mapping):
            raise ValueError("side mapping values must be BUY or SELL")
        required = {"timestamp_ns", "sequence", "event_type", "symbol", "venue"}
        fields = required | {"exchange_timestamp_ns", "order_id", "side", "price_ticks", "quantity", "orders"}
        mapping = dict(metadata.field_mapping)
        if not required <= mapping.keys() or not mapping.keys() <= fields:
            raise ValueError("CSV mapping must include required canonical fields and no unknown fields")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("source columns cannot be mapped to multiple canonical fields")
        self.limits = {name: _positive(value, name) for name, value in {
            "max_file_bytes": max_file_bytes, "max_expanded_bytes": max_expanded_bytes,
            "max_line_bytes": max_line_bytes, "max_events": max_events}.items()}
        self.book_limits = dict(book_limits)
        self.book = MBOBook(capability_contract=contract, max_events=max_events, **book_limits)

    def reopen(self) -> MappedMBOCSVAdapter:
        return type(self)(self.path, self.metadata, self.capability_contract,
                          action_mapping=dict(self._action_mapping), side_mapping=dict(self._side_mapping),
                          **self.limits, **self.book_limits)

    def events(self) -> Iterator[MBOEvent]:
        self._begin(self.limits["max_file_bytes"])
        opener = gzip.open if self.path.name.endswith(".gz") else open
        digest = hashlib.sha256()
        expanded = 0
        mapping = dict(self.metadata.field_mapping)
        actions, sides = dict(self._action_mapping), dict(self._side_mapping)
        with opener(self.path, "rb") as handle:
            header = None
            line_number = 0
            while True:
                line = handle.readline(self.limits["max_line_bytes"] + 1)
                if not line:
                    break
                line_number += 1
                expanded += len(line)
                if len(line) > self.limits["max_line_bytes"] or expanded > self.limits["max_expanded_bytes"]:
                    raise ValueError("CSV source exceeds max_line_bytes or max_expanded_bytes")
                try:
                    values = next(csv.reader([line.decode("utf-8")], strict=True))
                    if header is None:
                        if len(set(values)) != len(values) or set(values) != set(mapping.values()):
                            raise ValueError("CSV header must match mapped source columns exactly")
                        header = values
                        continue
                    if self._stats["events"] >= self.limits["max_events"]:
                        raise ValueError("CSV source exceeds max_events")
                    if len(values) != len(header):
                        raise ValueError("CSV row has incorrect column count")
                    raw = dict(zip(header, values))
                    row = {field: raw[column] for field, column in mapping.items() if raw[column] != ""}
                    action = row.get("event_type")
                    if action not in actions:
                        raise ValueError(f"unmapped source action {action!r}")
                    row["event_type"] = actions[action]
                    if "side" in row:
                        if row["side"] not in sides:
                            raise ValueError(f"unmapped source side {row['side']!r}")
                        row["side"] = sides[row["side"]]
                    if "orders" in row:
                        row["orders"] = _json(row["orders"])
                    event = MBOEvent.from_mapping(row)
                    self._identity(event.venue, event.symbol)
                    self.book.apply(event)
                except (ValueError, TypeError, UnicodeError, csv.Error) as exc:
                    raise ValueError(f"Invalid mapped MBO CSV line {line_number}: {exc}") from exc
                digest.update(json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")).encode() + b"\n")
                self._stats["events"] += 1
                yield event
        if not self._stats["events"]:
            raise ValueError("MBO source has no events")
        self._stats["expanded_bytes"] = expanded
        self._finish(digest.hexdigest())


def mbo_adapter(path: str | Path, manifest: Mapping[str, Any], **limits) -> CanonicalMBOAdapter | MappedMBOCSVAdapter:
    """Load an explicit adapter manifest, including the capability declaration."""
    if not isinstance(manifest, Mapping) or set(manifest) - {"adapter", "metadata", "capabilities",
                                                           "action_mapping", "side_mapping"}:
        raise ValueError("invalid MBO adapter manifest fields")
    if not {"adapter", "metadata", "capabilities"} <= manifest.keys():
        raise ValueError("MBO manifest requires adapter, metadata and capabilities")
    metadata = DatasetMetadata.from_mapping(manifest["metadata"])
    contract = CapabilityContract(metadata.data_level, frozenset(manifest["capabilities"]), metadata.source_id)
    if manifest["adapter"] == CanonicalMBOAdapter.adapter_name:
        if "action_mapping" in manifest or "side_mapping" in manifest:
            raise ValueError("JSONL canonical adapter does not accept CSV mappings")
        return CanonicalMBOAdapter(path, metadata, contract, **limits)
    if manifest["adapter"] == MappedMBOCSVAdapter.adapter_name:
        if not {"action_mapping", "side_mapping"} <= manifest.keys():
            raise ValueError("mapped CSV requires explicit action_mapping and side_mapping")
        return MappedMBOCSVAdapter(path, metadata, contract, action_mapping=manifest["action_mapping"],
                                   side_mapping=manifest["side_mapping"], **limits)
    raise ValueError("unsupported MBO adapter")
