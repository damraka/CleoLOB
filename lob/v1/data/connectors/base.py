"""Universal data-source connector interface (M10) and the connector registry.

A connector knows, for a (stream, date), which raw sources exist (URLs and formats), what they can identify
(``Capability``), under which terms they may be used (``Licence``), and how to turn the raw files into canonical
event chunks. Connectors never decide freshness or roles: that is the firewall's job (``lob.v1.data.firewall``).

Parsing is streaming: ``events`` yields chunks of at most ``chunk`` canonical events so a day of a deep book never
has to be materialized as Python objects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from ..schema import Capability, Instrument


@dataclass(frozen=True)
class Licence:
    provider: str
    terms_url: str
    checked: str                       # ISO date the terms were read
    use: str                           # what CleoLOB may do with the data
    redistribution: bool
    commercial: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Source:
    name: str                          # "l2", "trades", "depth", "capture", ...
    url: str
    format: str
    checksum_url: str | None = None
    optional: bool = False


class Connector(Protocol):
    name: str

    def instrument(self, segment: str, symbol: str, date: str) -> Instrument: ...

    def sources(self, segment: str, symbol: str, date: str) -> list[Source]: ...

    def capability(self, segment: str, symbol: str, date: str) -> Capability: ...

    def licence(self) -> Licence: ...

    def events(self, files: dict[str, Path], instrument: Instrument, *, chunk: int = 200_000) -> Iterator[np.ndarray]: ...


_REGISTRY: dict[str, Connector] = {}


def register(connector: Connector) -> Connector:
    if connector.name in _REGISTRY:
        raise ValueError(f"connector {connector.name} registered twice")
    _REGISTRY[connector.name] = connector
    return connector


def get(name: str) -> Connector:
    if name not in _REGISTRY:
        from . import binance, bitstamp, bybit  # noqa: F401 - registration side effect
    if name not in _REGISTRY:
        raise KeyError(f"no connector {name}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def known() -> list[str]:
    from . import binance, bitstamp, bybit  # noqa: F401
    return sorted(_REGISTRY)


__all__ = ["Licence", "Source", "Connector", "register", "get", "known"]
