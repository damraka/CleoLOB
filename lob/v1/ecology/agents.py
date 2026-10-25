"""Order-level market participants for the multi-agent ecology (M101-M104).

Every agent is a **simulation assumption**, never an identified real trader: names describe the rule, not a
claim about who trades on any venue. Agents act through ``act(t, view, ctx, rng)`` and return commands
(``("submit", order_kwargs)`` / ``("cancel", oid)``); ``ctx`` gives the agent its own live orders, inventory and the
recent public tape (prints include the strategy's, so every reaction to it is endogenous self-impact, M104).

* ``MarketMaker`` (M101): quotes both sides around the (stale) mid with half-spread ``h`` ticks and inventory skew
  ``-gamma * inventory`` (Avellaneda-Stoikov style); it widens ``h`` by ``toxicity_widen`` ticks when the recent
  signed-flow imbalance exceeds a threshold (strategic liquidity response, M103) and refreshes stale quotes;
* ``NoiseTrader``: Poisson market and limit orders, random side, geometric sizes;
* ``Momentum`` / ``MeanReversion`` (M102): trade on the sign of the mid change over a lookback (with / against it);
* ``LiquidityProvider`` (M103): replenishes a depleted touch after a reaction delay and cancels its own resting
  orders when the recent flow runs against them (strategic cancellation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BUY, SELL = 1, -1


@dataclass
class AgentContext:
    live: dict            # oid -> (side, price, remaining)
    inventory: int
    tape: list            # recent prints (t, price, qty, aggressor side)
    mids: list            # recent visible mids (t, mid)


@dataclass
class Agent:
    name: str
    rate: float           # wake-ups per second (Poisson clock)
    params: dict = field(default_factory=dict)
    population: bool = False   # True: stands for many independent traders (one owner id per order, no self-trades)

    def act(self, t, view, ctx: AgentContext, rng) -> list:
        return []


def _flow_imbalance(tape, t, window=5.0) -> float:
    b = sum(q for (s, _, q, side) in tape if s >= t - window and side > 0)
    a = sum(q for (s, _, q, side) in tape if s >= t - window and side < 0)
    return (b - a) / max(a + b, 1)


class MarketMaker(Agent):
    def act(self, t, view, ctx, rng):
        if view is None or view["bid"] is None or view["ask"] is None:
            return []
        p = self.params
        mid = 0.5 * (view["bid"] + view["ask"])
        h = p.get("half_spread", 1)
        if abs(_flow_imbalance(ctx.tape, t)) > p.get("toxicity_threshold", 0.6):
            h += p.get("toxicity_widen", 2)
        skew = -p.get("gamma", 0.05) * ctx.inventory
        bid_px = int(np.floor(mid + skew - h + 0.5))
        ask_px = int(np.ceil(mid + skew + h - 0.5))
        if ask_px <= bid_px:
            ask_px = bid_px + 1
        cmds = [("cancel", oid) for oid, (side, price, _) in ctx.live.items()
                if (side == BUY and price != bid_px) or (side == SELL and price != ask_px)]
        have = {(side, price) for side, price, _ in ctx.live.values()}
        q = int(p.get("size", 5))
        if abs(ctx.inventory) < p.get("max_inventory", 200):
            if (BUY, bid_px) not in have:
                cmds.append(("submit", {"side": BUY, "qty": q, "kind": "limit", "price": bid_px}))
            if (SELL, ask_px) not in have:
                cmds.append(("submit", {"side": SELL, "qty": q, "kind": "limit", "price": ask_px}))
        return cmds


class NoiseTrader(Agent):
    def act(self, t, view, ctx, rng):
        if view is None or view["bid"] is None or view["ask"] is None:
            return []
        p = self.params
        side = BUY if rng.random() < 0.5 else SELL
        q = int(rng.geometric(1.0 / p.get("size_mean", 3)))
        if rng.random() < p.get("market_share", 0.35):
            return [("submit", {"side": side, "qty": q, "kind": "market"})]
        touch = view["bid"] if side == BUY else view["ask"]
        depth = int(rng.geometric(p.get("level_p", 0.5))) - 1
        return [("submit", {"side": side, "qty": q, "kind": "limit", "price": touch - side * depth})]


class Momentum(Agent):
    def act(self, t, view, ctx, rng):
        p = self.params
        old = [m for (s, m) in ctx.mids if s <= t - p.get("lookback", 10.0)]
        if not old or view is None or view["bid"] is None or view["ask"] is None:
            return []
        move = 0.5 * (view["bid"] + view["ask"]) - old[-1]
        sign = int(np.sign(move)) * (1 if self.params.get("follow", True) else -1)
        if abs(move) < p.get("threshold", 1.0) or sign == 0 or abs(ctx.inventory) > p.get("max_inventory", 100):
            return []
        return [("submit", {"side": sign, "qty": int(p.get("size", 2)), "kind": "market"})]


class MeanReversion(Momentum):
    def __init__(self, name, rate, params=None):
        super().__init__(name, rate, {**(params or {}), "follow": False})


class LiquidityProvider(Agent):
    def act(self, t, view, ctx, rng):
        if view is None or view["bid"] is None or view["ask"] is None:
            return []
        p = self.params
        cmds = []
        flow = _flow_imbalance(ctx.tape, t, p.get("window", 3.0))
        for oid, (side, price, _) in ctx.live.items():           # strategic cancellation against adverse flow
            if (side == BUY and flow < -p.get("adverse", 0.5)) or (side == SELL and flow > p.get("adverse", 0.5)):
                cmds.append(("cancel", oid))
        bids, asks = view["bids"], view["asks"]
        thin = p.get("thin", 3)
        q = int(p.get("size", 8))
        if not bids or bids[0][1] < thin:
            cmds.append(("submit", {"side": BUY, "qty": q, "kind": "limit", "price": view["bid"]}))
        if not asks or asks[0][1] < thin:
            cmds.append(("submit", {"side": SELL, "qty": q, "kind": "limit", "price": view["ask"]}))
        return cmds


def default_ecology(seed_scale: float = 1.0) -> list[Agent]:
    """The registered default population (rates per second)."""
    return [MarketMaker("mm1", 2.0 * seed_scale, {"half_spread": 1, "gamma": 0.05, "size": 6}),
            MarketMaker("mm2", 1.0 * seed_scale, {"half_spread": 2, "gamma": 0.02, "size": 10}),
            NoiseTrader("noise", 4.0 * seed_scale, {"size_mean": 3, "market_share": 0.3}, population=True),
            Momentum("mom", 0.5 * seed_scale, {"lookback": 10.0, "threshold": 2.0, "size": 3}),
            MeanReversion("mr", 0.5 * seed_scale, {"lookback": 30.0, "threshold": 3.0, "size": 3}),
            LiquidityProvider("lp", 1.0 * seed_scale, {"thin": 3, "size": 8})]


__all__ = ["Agent", "AgentContext", "MarketMaker", "NoiseTrader", "Momentum", "MeanReversion", "LiquidityProvider",
           "default_ecology"]
