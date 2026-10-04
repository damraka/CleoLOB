"""Multi-agent market ecology, strategic interaction and the market-maker module (workstreams 46-48).

Structural agents act on a shared ``ExchangeSimulator`` every ``dt`` seconds alongside (optional)
background flow: a liquidity provider, a random taker, a noise trader, an inventory-sensitive maker
and a meta-order trader. They are stylized mechanisms, never claims about real participant classes.

Accounting is exact: every trade moves inventory and cash between its two owners, so the sums over all
owners (including background ``ZI``) of inventory and cash are zero (conservation tests).

Strategic interaction: ``ReactiveMaker`` pulls and widens its quotes for ``cooldown`` seconds after it
observes aggressive volume on one side above a threshold; comparing an execution agent's cost against a
passive maker (same seeds) measures how much a conclusion depends on the reactive assumption.

Market-maker metrics: spread capture (fill price vs contemporaneous mid), adverse selection (mid move over
the next 5 s against the fill), inventory mark-to-market and an inventory penalty. Exploratory throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...engine import ExchangeSimulator, Side


@dataclass
class Account:
    inventory: int = 0
    cash: float = 0.0      # in ticks x lots


def settle(trades, accounts: dict[str, Account]) -> None:
    for t in trades:
        for owner, sign in ((t.taker_owner, 1 if t.taker_side is Side.BUY else -1),
                            (t.maker_owner, -1 if t.taker_side is Side.BUY else 1)):
            acc = accounts.setdefault(owner, Account())
            acc.inventory += sign * t.qty
            acc.cash -= sign * t.qty * t.price


class Agent:
    name = "agent"

    def __init__(self, name: str, seed: int) -> None:
        self.name, self.rng = name, np.random.default_rng(seed)
        self.orders: list[int] = []

    def act(self, sim: ExchangeSimulator, accounts: dict) -> None:   # pragma: no cover - interface
        raise NotImplementedError

    def cancel_all(self, sim) -> None:
        for oid in self.orders:
            o = sim.orders.get(oid)
            if o is not None and not o.is_terminal:
                sim.cancel(oid)
        self.orders = []


class LiquidityProvider(Agent):
    def __init__(self, name, seed, size=20, offset=1):
        super().__init__(name, seed)
        self.size, self.offset = size, offset

    def act(self, sim, accounts):
        self.cancel_all(sim)
        bb, ba = sim.book.best_bid(), sim.book.best_ask()
        mid = int(round(sim.book.mid()))
        bid = (bb if bb is not None else mid - 1) - (self.offset - 1)
        ask = (ba if ba is not None else mid + 1) + (self.offset - 1)
        if ask <= bid:
            ask = bid + 1
        self.orders += [sim.submit(Side.BUY, self.size, self.name, bid), sim.submit(Side.SELL, self.size, self.name, ask)]


class InventoryMaker(LiquidityProvider):
    """Quotes skewed against its inventory: long inventory -> lower quotes (sell more readily)."""

    def __init__(self, name, seed, size=10, skew_per_lot=0.05):
        super().__init__(name, seed, size, 1)
        self.skew = skew_per_lot

    def act(self, sim, accounts):
        self.cancel_all(sim)
        inventory = accounts.get(self.name, Account()).inventory
        mid = sim.book.mid()
        shift = -self.skew * inventory
        bid, ask = int(np.floor(mid - 1 + shift)), int(np.ceil(mid + 1 + shift))
        if ask <= bid:
            ask = bid + 1
        self.orders += [sim.submit(Side.BUY, self.size, self.name, max(1, bid)),
                        sim.submit(Side.SELL, self.size, self.name, max(2, ask))]


class Taker(Agent):
    def __init__(self, name, seed, rate=0.5, size=5):
        super().__init__(name, seed)
        self.rate, self.size = rate, size

    def act(self, sim, accounts):
        if self.rng.random() < self.rate:
            side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
            sim.submit(side, int(self.rng.integers(1, self.size + 1)), self.name)


class NoiseTrader(Agent):
    def act(self, sim, accounts):
        side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
        mid = int(round(sim.book.mid()))
        if self.rng.random() < 0.3:
            sim.submit(side, int(self.rng.integers(1, 4)), self.name)
        else:
            offset = int(self.rng.integers(1, 6))
            sim.submit(side, int(self.rng.integers(1, 6)), self.name, mid - offset if side is Side.BUY else mid + offset)


class MetaOrderTrader(Agent):
    def __init__(self, name, seed, quantity=60, slices=10, side=Side.BUY):
        super().__init__(name, seed)
        self.remaining, self.child, self.side = quantity, max(1, quantity // slices), side

    def act(self, sim, accounts):
        if self.remaining > 0:
            qty = min(self.child, self.remaining)
            sim.submit(self.side, qty, self.name)
            self.remaining -= qty


class ReactiveMaker(LiquidityProvider):
    """Withdraws and widens after seeing one-sided aggressive volume (strategic reaction)."""

    def __init__(self, name, seed, size=20, threshold=10, cooldown=5.0, widen=3):
        super().__init__(name, seed, size, 1)
        self.threshold, self.cooldown, self.widen = threshold, cooldown, widen
        self.until, self.cursor = -1.0, 0

    def act(self, sim, accounts):
        trades = sim.book.trades[self.cursor:]
        self.cursor = len(sim.book.trades)
        buy = sum(t.qty for t in trades if t.taker_side is Side.BUY and self.name not in (t.maker_owner,))
        sell = sum(t.qty for t in trades if t.taker_side is Side.SELL and self.name not in (t.maker_owner,))
        if max(buy, sell) >= self.threshold:
            self.until = sim.t + self.cooldown
        self.offset = self.widen if sim.t < self.until else 1
        super().act(sim, accounts)


@dataclass
class Ecology:
    world: object
    agents: list = field(default_factory=list)
    dt: float = 1.0

    def run(self, seed: int, seconds: float, *, warmup: float = 30.0) -> dict:
        sim = self.world.build(seed)
        sim.step(warmup)
        accounts: dict[str, Account] = {}
        cursor = len(sim.book.trades)
        mids = []
        while sim.t < warmup + seconds - 1e-9:
            for agent in self.agents:
                agent.act(sim, accounts)
            sim.step(self.dt)
            mids.append(sim.book.mid())
        settle(sim.book.trades[cursor:], accounts)
        sim.book.assert_invariants()
        return {"accounts": {k: (v.inventory, v.cash) for k, v in accounts.items()}, "mids": mids, "sim": sim,
                "trades": sim.book.trades[cursor:], "t0": warmup}


def maker_metrics(result: dict, owner: str, *, horizon: float = 5.0, penalty: float = 0.01) -> dict:
    """Spread capture, adverse selection, inventory mark and penalty of one maker (ticks; exploratory)."""
    sim, trades = result["sim"], result["trades"]
    mids = np.asarray(result["mids"])
    t0 = result["t0"]
    capture = adverse = 0.0
    fills = 0
    for t in trades:
        if t.maker_owner != owner:
            continue
        sign = -1 if t.taker_side is Side.BUY else 1           # maker sold (-1) or bought (+1)
        i = min(len(mids) - 1, max(0, int(t.time - t0)))
        j = min(len(mids) - 1, i + int(horizon))
        capture += sign * (mids[i] - t.price) * t.qty
        adverse += sign * (mids[j] - mids[i]) * t.qty
        fills += t.qty
    inventory, cash = result["accounts"].get(owner, (0, 0.0))
    mark = cash + inventory * sim.book.mid()
    return {"filled_lots": fills, "spread_capture_ticks": capture, "adverse_selection_ticks": adverse,
            "inventory": inventory, "mark_to_market_ticks": mark, "inventory_penalty": penalty * inventory ** 2,
            "net_after_penalty": mark - penalty * inventory ** 2}
