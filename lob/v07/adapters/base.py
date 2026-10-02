"""Adapter kit: the contract every v0.7 venue adapter implements, plus a conformance suite.

An adapter turns one source file (or capture) into canonical ``Record`` objects
and declares its ``VenueSpec``. The conformance suite checks the obligations any
new adapter must meet before its output is used by a study: schema validity,
nondecreasing local capture time, no fabricated identities, deterministic output
(same digest twice) and an honest capability declaration.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import islice
from typing import Iterator

from ..data.schema import Record, SchemaError, VenueSpec, digest


class Adapter(ABC):
    """Base class. Subclasses set ``spec`` and implement ``records``; ``stats`` describes the last pass."""

    spec: VenueSpec

    def __init__(self) -> None:
        self.stats: dict = {}

    @abstractmethod
    def records(self) -> Iterator[Record]:
        """Yield canonical records in source capture order."""

    def describe(self) -> dict:
        return {"adapter": type(self).__name__, "venue": self.spec.to_dict(),
                "capabilities": self.spec.contract.to_dict()}


def conformance(adapter: Adapter, *, limit: int = 200_000) -> dict:
    """Run the adapter-kit obligations on (a bounded prefix of) one source."""
    issues: list[str] = []
    level = adapter.spec.capability_level
    count, last_local, kinds = 0, -1, {}
    try:
        for record in islice(adapter.records(), limit):
            record.validate(level)
            if record.local_timestamp_us < last_local:
                issues.append(f"local capture time decreases at record {count}")
                break
            last_local = record.local_timestamp_us
            kinds[record.kind] = kinds.get(record.kind, 0) + 1
            count += 1
    except SchemaError as exc:
        issues.append(f"schema violation at record {count}: {exc}")
    if count == 0:
        issues.append("adapter produced no records")
    first = None
    if not issues:
        first = digest(islice(adapter.records(), min(limit, 20_000)))
        if first != digest(islice(adapter.records(), min(limit, 20_000))):
            issues.append("adapter output is not deterministic")
    if level == "aggregate_l2" and adapter.spec.contract.supports("order_identity"):
        issues.append("aggregate L2 adapter declares order identity")
    return {"valid": not issues, "issues": issues, "records_checked": count, "kinds": kinds,
            "prefix_digest": first, "adapter": adapter.describe()}
