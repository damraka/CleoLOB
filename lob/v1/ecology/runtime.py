"""Multi-agent ecology runtime on the order-level matching engine (M100) with endogenous feedback (M104).

Agents wake on independent seeded Poisson clocks, observe the book through the latency ``Gateway`` (stale market
data), and send commands that arrive after the entry latency; the deterministic matching engine v2 matches them.
Reports flow back to each agent (inventory, live orders) and every print enters the shared public tape, so agents
react to the strategy's own trades (self-impact and endogenous feedback, M104). An optional strategy callback acts
on a fixed decision grid through the same gateway.

``run`` returns 100 ms frames in the v0.8 ``EventFrame`` layout (touch, ten visible levels, event counts by mark,
traded volume, mids, trade samples) so the realism metrics and the gate apply unchanged, plus the tape and the
per-agent accounting. Everything is a simulation of assumed behaviour (F6); none of it is an observed market.
"""
from __future__ import annotations

import heapq
import itertools

import numpy as np

from ...v08.market.frame import EventFrame
from ..engines.latency import Gateway, LatencyModel
from ..engines.matching import BUY, MatchingEngine, Order
from ..engines.rules import Rules
from .agents import Agent, AgentContext

BIN_S = 0.1
NS = 1_000_000_000


class Ecology:
    def __init__(self, agents: list[Agent], *, seed: int = 0, rules: Rules | None = None, mid: int = 10_000,
                 entry_ns: int = 2_000_000, md_ns: int = 5_000_000, warm_levels: int = 10, warm_size: int = 10):
        self.agents = agents
        self.engine = MatchingEngine(rules or Rules(self_trade="cancel_taker"))
        self.gw = Gateway(self.engine, entry=LatencyModel("uniform", entry_ns, entry_ns),
                          report=LatencyModel("constant", entry_ns), md=LatencyModel("constant", md_ns), seed=seed)
        self.rng = np.random.default_rng(seed)
        self.oid = itertools.count(1)
        self.ctx = {a.name: AgentContext({}, 0, [], []) for a in agents}
        self.ctx["strategy"] = AgentContext({}, 0, [], [])
        self.tape: list[tuple] = []
        self.strategy_reports: list = []
        self.counts_bin = np.zeros(6, int)
        self.volume_bin = np.zeros(2)
        for k in range(warm_levels):                       # registered initial book (symmetric, warm_size lots)
            self._direct(Order(oid=next(self.oid), owner="init", side=BUY, qty=warm_size, price=mid - 1 - k), 0)
            self._direct(Order(oid=next(self.oid), owner="init", side=-BUY, qty=warm_size, price=mid + k), 0)
        self.gw._publish()                                  # the initial book is the first public snapshot

    def _direct(self, order: Order, ts: int) -> None:
        reps = self.engine.submit(order, ts)
        self._absorb(reps, ts)

    def _owner_ctx(self, owner: str):
        return self.ctx.get(owner.split("#", 1)[0])

    def _absorb(self, reps, ts: int) -> None:
        for r in reps:
            if r.owner == "strategy" and r.kind == "fill":
                self.strategy_reports.append(r)
            c = self._owner_ctx(r.owner)
            if r.kind == "fill" and r.liquidity == "taker":
                side = self.engine.orders[r.oid].side
                self.tape.append((ts / NS, r.price, r.qty, side))
                self.volume_bin[0 if side > 0 else 1] += r.qty
            if c is None:
                continue
            if r.kind == "fill":
                side = self.engine.orders[r.oid].side
                c.inventory += r.qty if side > 0 else -r.qty
                if r.remaining == 0:
                    c.live.pop(r.oid, None)
                elif r.oid in c.live:
                    s, p, _ = c.live[r.oid]
                    c.live[r.oid] = (s, p, r.remaining)
            elif r.kind == "ack":
                o = self.engine.orders[r.oid]
                if o.kind == "limit" and o.status == "live":
                    c.live[r.oid] = (o.side, o.price, o.remaining)
            elif r.kind in ("cancel", "expire", "reject"):
                c.live.pop(r.oid, None)

    def _send(self, owner: str, cmd, t_ns: int, population: bool = False) -> None:
        kind, arg = cmd
        if kind == "submit":
            oid = next(self.oid)
            o = arg
            self.counts_bin[(0 if o["side"] > 0 else 1) if o["kind"] == "market" else (2 if o["side"] > 0 else 3)] += 1
            self.gw.send(t_ns, "submit", f"{owner}#{oid}" if population else owner, oid=oid, **arg)
        elif kind == "cancel":
            live = self.ctx[owner].live.get(arg)
            if live is not None:
                self.counts_bin[4 if live[0] > 0 else 5] += 1
            real_owner = self.engine.orders[arg].owner if arg in self.engine.orders else owner
            self.gw.send(t_ns, "cancel", real_owner, oid=arg)

    def run(self, seconds: float, *, strategy=None, decision_s: float = 5.0) -> dict:
        n_bins = int(round(seconds / BIN_S))
        clocks = [(self.rng.exponential(1.0 / a.rate), i) for i, a in enumerate(self.agents)]
        heapq.heapify(clocks)
        rows = {k: [] for k in ("bid", "ask", "depth", "counts", "volume", "mid")}
        next_decision = 0.0
        for b in range(n_bins):
            t_end = (b + 1) * BIN_S
            while clocks and clocks[0][0] < t_end:
                t, i = heapq.heappop(clocks)
                a = self.agents[i]
                t_ns = int(t * NS)
                self._flush(t_ns)
                c = self.ctx[a.name]
                c.tape = self.tape[-200:]
                for cmd in a.act(t, self.gw.view(t_ns), c, self.rng):
                    self._send(a.name, cmd, t_ns, population=a.population)
                heapq.heappush(clocks, (t + self.rng.exponential(1.0 / a.rate), i))
            if strategy is not None and next_decision < t_end:
                t_ns = int(next_decision * NS)
                self._flush(t_ns)
                c = self.ctx["strategy"]
                for cmd in strategy(next_decision, self.gw.view(t_ns), c):
                    self._send("strategy", cmd, t_ns)
                next_decision += decision_s
            self._flush(int(t_end * NS))
            e = self.engine
            bid, ask = e.best(BUY), e.best(-BUY)
            rows["bid"].append(bid if bid is not None else -1)
            rows["ask"].append(ask if ask is not None else -1)
            d = np.zeros((2, 10))
            for k, (_, q) in enumerate(e.depth(BUY, 10)):
                d[0, k] = q
            for k, (_, q) in enumerate(e.depth(-BUY, 10)):
                d[1, k] = q
            rows["depth"].append(d)
            rows["counts"].append(self.counts_bin.copy())
            rows["volume"].append(self.volume_bin.copy())
            mid = 0.5 * (bid + ask) if bid is not None and ask is not None else np.nan
            rows["mid"].append(mid)
            for c in self.ctx.values():
                c.mids.append((t_end, mid))
                if len(c.mids) > 1200:
                    del c.mids[:600]
            self.counts_bin[:] = 0
            self.volume_bin[:] = 0
        return self._frame(rows)

    def _flush(self, t_ns: int) -> None:
        self.gw.run_until(t_ns)
        for owner in list(self.gw.inbox):
            self._absorb(self.gw.reports(owner, t_ns + 10 * NS), t_ns)

    def _frame(self, rows) -> dict:
        from ...v08.market.frame import features
        bid = np.asarray(rows["bid"], np.int64)
        ask = np.asarray(rows["ask"], np.int64)
        valid = (bid > 0) & (ask > 0) & (ask > bid)
        depth = np.asarray(rows["depth"])
        mid = np.asarray(rows["mid"], float)
        mid = np.where(np.isfinite(mid), mid, np.nan)
        idx = np.where(np.isfinite(mid), np.arange(len(mid)), 0)
        np.maximum.accumulate(idx, out=idx)
        mid = mid[idx]
        mid = np.where(np.isfinite(mid), mid, np.nanmean(mid) if np.isfinite(mid).any() else 0.0)
        mids_full = np.r_[mid, mid[-1]]
        state = np.stack([features(int(bid[i]), int(ask[i]), depth[i, 0], depth[i, 1],
                                   [mids_full[max(0, i - 10)], mids_full[i]]) for i in range(len(bid))])
        tape = np.asarray(self.tape, float).reshape(-1, 4)
        frame = EventFrame(np.asarray(rows["counts"], np.int64), state, bid, ask, depth.astype(np.int64),
                           np.asarray(rows["volume"], float), mids_full, valid,
                           samples={"trade_t": tape[:, 0], "trade_px": tape[:, 1], "trade_q": tape[:, 2],
                                    "trade_s": tape[:, 3], "market_size": tape[:, 2]},
                           meta={"source": "F6-ecology", "agents": [a.name for a in self.agents]})
        return {"frame": frame, "tape": self.tape, "inventory": {k: c.inventory for k, c in self.ctx.items()},
                "engine": {"fills": self.engine.stats.fills, "volume": self.engine.stats.volume,
                           "races": self.gw.races}}


__all__ = ["Ecology", "BIN_S"]
