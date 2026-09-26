"""Immutable declarations of observations, not promises of market truth.

An order-level file is not automatically FIFO data. Adapters must narrow the
contract to the identities, ordering and execution semantics their source has.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class EvidenceStatus(str, Enum):
    ESTABLISHED = "ESTABLISHED"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"
    FAILED = "FAILED"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    INVALID = "INVALID"
    INCONCLUSIVE = "INCONCLUSIVE"


class DataLevel(str, Enum):
    TRADES = "trades_only"
    L1 = "mbp_l1"
    L2 = "aggregate_l2"
    MBO = "order_level_mbo"
    SIMULATOR = "simulator_internal"


class Capability(str, Enum):
    TRADE_PRINTS = "trade_prints"
    TOP_OF_BOOK = "top_of_book"
    SPREAD = "spread"
    AGGREGATE_DEPTH = "aggregate_depth"
    BOOK_IMBALANCE = "book_imbalance"
    AGGREGATE_VOLUME_CHANGES = "aggregate_volume_changes"
    ORDER_IDENTITY = "order_identity"
    EXACT_FIFO_POSITION = "exact_fifo_position"
    QUANTITY_AHEAD = "quantity_ahead"
    OBSERVED_ORDER_FILL = "observed_order_fill"
    SNAPSHOT_ORDERING = "snapshot_ordering"
    EXCHANGE_SEQUENCE_CONTINUITY = "exchange_sequence_continuity"
    HIDDEN_ORDER_QUANTITY = "hidden_order_quantity"
    COUNTERFACTUAL_PASSIVE_FILL = "counterfactual_passive_fill"
    SIMULATED_FILLS = "simulated_fills"


class CapabilityError(ValueError):
    """A requested observation is absent, or a declaration is inconsistent."""

    def __init__(self, capability: str, message: str):
        super().__init__(message)
        self.code = "UNSUPPORTED_CAPABILITY"
        self.capability = capability


_QUOTES = frozenset({Capability.TOP_OF_BOOK, Capability.SPREAD, Capability.BOOK_IMBALANCE})
_DEPTH = _QUOTES | {Capability.AGGREGATE_DEPTH, Capability.AGGREGATE_VOLUME_CHANGES}
_IDENTITY = frozenset({Capability.ORDER_IDENTITY, Capability.EXACT_FIFO_POSITION,
                       Capability.QUANTITY_AHEAD, Capability.OBSERVED_ORDER_FILL,
                       Capability.SNAPSHOT_ORDERING})
_SEQUENCE = frozenset({Capability.EXCHANGE_SEQUENCE_CONTINUITY})
_ALLOWED = {
    DataLevel.TRADES: frozenset({Capability.TRADE_PRINTS}) | _SEQUENCE,
    DataLevel.L1: _QUOTES | _SEQUENCE | {Capability.TRADE_PRINTS},
    DataLevel.L2: _DEPTH | _SEQUENCE | {Capability.TRADE_PRINTS},
    DataLevel.MBO: _DEPTH | _IDENTITY | _SEQUENCE | {Capability.TRADE_PRINTS},
    # Counterfactual market fills and real hidden liquidity are not established
    # even by a simulator. Simulated outcomes use their own capability name.
    DataLevel.SIMULATOR: _DEPTH | _IDENTITY | {Capability.SIMULATED_FILLS},
}
_DEPENDENCIES = {
    Capability.EXACT_FIFO_POSITION: {Capability.ORDER_IDENTITY, Capability.SNAPSHOT_ORDERING},
    Capability.QUANTITY_AHEAD: {Capability.EXACT_FIFO_POSITION, Capability.AGGREGATE_DEPTH},
    Capability.OBSERVED_ORDER_FILL: {Capability.ORDER_IDENTITY},
    Capability.SNAPSHOT_ORDERING: {Capability.ORDER_IDENTITY},
}


@dataclass(frozen=True)
class CapabilityContract:
    level: DataLevel
    available: frozenset[Capability]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", DataLevel(self.level))
        object.__setattr__(self, "available", frozenset(Capability(c) for c in self.available))
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("capability source must be a nonempty description")
        forbidden = self.available - _ALLOWED[self.level]
        if forbidden:
            raise CapabilityError(sorted(c.value for c in forbidden)[0],
                                  f"{self.level.value} cannot establish {sorted(c.value for c in forbidden)}")
        for capability, dependencies in _DEPENDENCIES.items():
            if capability in self.available and not dependencies <= self.available:
                raise CapabilityError(capability.value, f"{capability.value} requires "
                                      f"{sorted(c.value for c in dependencies - self.available)}")

    def supports(self, capability: Capability | str) -> bool:
        try:
            return Capability(capability) in self.available
        except ValueError as exc:
            raise CapabilityError(str(capability), f"Unknown capability {capability!r}") from exc

    def require(self, *capabilities: Capability | str) -> None:
        for capability in capabilities:
            if not self.supports(capability):
                name = Capability(capability).value
                raise CapabilityError(name, f"{name} is unavailable from {self.level.value} ({self.source})")

    def restrict(self, capabilities: Iterable[Capability | str]) -> CapabilityContract:
        """Narrow a contract; this method can never add evidence."""
        available = frozenset(Capability(c) for c in capabilities)
        if not available <= self.available:
            raise CapabilityError("declaration", "restrict cannot grant new capabilities")
        return CapabilityContract(self.level, available, self.source)

    def to_dict(self) -> dict:
        return {"data_level": self.level.value, "source": self.source,
                "available": sorted(capability.value for capability in self.available)}


TRADES_CONTRACT = CapabilityContract(DataLevel.TRADES, {Capability.TRADE_PRINTS}, "trade prints only")
L1_CONTRACT = CapabilityContract(DataLevel.L1, _QUOTES, "displayed best bid and ask")
L2_CONTRACT = CapabilityContract(DataLevel.L2, _DEPTH, "aggregate depth; no order identity or FIFO")
CANONICAL_MBO_CONTRACT = CapabilityContract(
    DataLevel.MBO, _DEPTH | _IDENTITY | {Capability.TRADE_PRINTS},
    "canonical source identities; complete capture and snapshot FIFO are caller obligations",
)
SIMULATOR_CONTRACT = CapabilityContract(
    DataLevel.SIMULATOR, (_DEPTH | _IDENTITY | {Capability.SIMULATED_FILLS}) - {Capability.OBSERVED_ORDER_FILL},
    "simulator state and simulated outcomes; no historical fill evidence",
)

REAL_HISTORICAL_MBO_VALIDATION = EvidenceStatus.NOT_AVAILABLE
