"""Latency, stale market data and cancel races (M35).

``Gateway`` puts a deterministic, seedable latency between agents and the matching engine:

* an agent command sent at local time ``t`` reaches the engine at ``t + entry_latency()``; commands are processed in
  arrival order (ties by send order), so a cancel can arrive after the order it targets was filled - the engine then
  rejects it with ``too_late`` and the race is counted;
* execution reports reach the agent at ``engine time + report_latency()``;
* ``view(t)`` returns the book snapshot the agent sees at local time ``t``: the latest published book whose
  publication time plus ``md_latency`` is <= ``t`` (stale market data).

Latency distributions are registered objects (constant, uniform, shifted lognormal) - simulation assumptions,
never measurements of a venue's real latency.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .matching import MatchingEngine, Order


@dataclass(frozen=True)
class LatencyModel:
    kind: str = "constant"          # constant | uniform | lognormal
    base_ns: int = 0
    spread_ns: int = 0              # uniform width or lognormal scale (ns)
    sigma: float = 0.5

    def draw(self, rng: np.random.Generator) -> int:
        if self.kind == "constant":
            return int(self.base_ns)
        if self.kind == "uniform":
            return int(self.base_ns + rng.integers(0, max(self.spread_ns, 0) + 1))
        if self.kind == "lognormal":
            return int(self.base_ns + self.spread_ns * math.exp(self.sigma * rng.standard_normal()))
        raise ValueError(f"unknown latency kind {self.kind}")


class Gateway:
    def __init__(self, engine: MatchingEngine, *, entry: LatencyModel = LatencyModel(),
                 report: LatencyModel = LatencyModel(), md: LatencyModel = LatencyModel(), seed: int = 0) -> None:
        self.engine, self.entry, self.report_lat, self.md = engine, entry, report, md
        self.rng = np.random.default_rng(seed)
        self._q: list = []
        self._n = 0
        self.now = 0
        self.inbox: dict[str, list] = {}
        self._published: list[tuple[int, dict]] = []
        self.races = 0

    def send(self, t: int, command: str, owner: str, **kw) -> int:
        arrive = t + self.entry.draw(self.rng)
        self._n += 1
        heapq.heappush(self._q, (arrive, self._n, command, owner, kw))
        return arrive

    def _publish(self) -> None:
        e = self.engine
        self._published.append((self.now, {"bid": e.best(1), "ask": e.best(-1), "bids": e.depth(1, 10),
                                           "asks": e.depth(-1, 10)}))

    def run_until(self, t: int) -> None:
        while self._q and self._q[0][0] <= t:
            arrive, _, command, owner, kw = heapq.heappop(self._q)
            self.now = arrive
            if command == "submit":
                reps = self.engine.submit(Order(owner=owner, **kw), arrive)
            elif command == "cancel":
                reps = self.engine.cancel(kw["oid"], arrive, owner=owner)
                self.races += sum(r.reason == "too_late" for r in reps)
            elif command == "replace":
                reps = self.engine.replace(kw.pop("oid"), arrive, owner=owner, **kw)
                self.races += sum(r.reason == "too_late" for r in reps)
            else:
                raise ValueError(f"unknown command {command}")
            for r in reps:
                self.inbox.setdefault(r.owner, []).append((arrive + self.report_lat.draw(self.rng), r))
            self._publish()
        self.now = max(self.now, t)

    def reports(self, owner: str, t: int) -> list:
        """Reports delivered to ``owner`` by local time ``t`` (removed from the inbox)."""
        box = self.inbox.get(owner, [])
        ready = [r for at, r in box if at <= t]
        self.inbox[owner] = [(at, r) for at, r in box if at > t]
        return ready

    def view(self, t: int) -> dict | None:
        """Book snapshot visible to agents at local time ``t`` (stale by the market-data latency)."""
        lag = self.md.draw(self.rng)
        seen = None
        for pub_t, snap in self._published:
            if pub_t + lag <= t:
                seen = snap
            else:
                break
        return seen


__all__ = ["LatencyModel", "Gateway"]
