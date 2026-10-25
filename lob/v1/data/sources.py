"""Source catalogue: licences, access policies and provenance metadata per connector (M16).

``catalogue()`` is the machine-readable statement of every source v1 may use: provider, terms URL and the date
they were read, permitted use, redistribution, capability, and the access policy (archive download, prospective
capture). It is exported into ``docs/v1-data-sources.md`` tables and into every dataset datasheet.
"""
from __future__ import annotations

from .connectors import base as CB

STREAMS = (
    ("bybit", "linear", "BTCUSDT"), ("bybit", "linear", "ETHUSDT"), ("bybit", "linear", "SOLUSDT"),
    ("bybit", "spot", "BTCUSDT"), ("bybit", "spot", "ETHUSDT"), ("bybit", "spot", "SOLUSDT"),
    ("binance", "um", "BTCUSDT"), ("binance", "um", "ETHUSDT"),
    ("bitstamp", "exchange", "btcusd"), ("bitstamp", "exchange", "ethusd"),
)
ACCESS = {"bybit": "public HTTPS archive download (daily files)",
          "binance": "public HTTPS archive download (daily files, published SHA-256 checksums)",
          "bitstamp": "prospective read-only public websocket capture plus public REST censuses"}


def catalogue(date: str = "2026-01-01") -> list[dict]:
    out = []
    for venue, segment, symbol in STREAMS:
        c = CB.get(venue)
        out.append({"stream": f"{venue}-{segment}:{symbol}", "connector": venue, "segment": segment, "symbol": symbol,
                    "licence": c.licence().to_dict(), "access": ACCESS[venue],
                    "capability": c.capability(segment, symbol, date).to_dict(),
                    "sources": [s.format for s in c.sources(segment, symbol, date)]})
    return out


__all__ = ["STREAMS", "ACCESS", "catalogue"]
