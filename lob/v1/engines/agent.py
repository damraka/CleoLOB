"""Agent-side state: inventory, outstanding orders and self-impact feedback (M36).

``AgentState`` consumes execution reports and keeps the agent's open orders (with remaining quantities), inventory,
fill history and an implementation-shortfall ledger against an arrival price. Self-impact is endogenous in the
engine: the agent's own taker fills remove liquidity and move the touch, and its resting orders occupy queue
positions that background flow must clear first; ``impact_trace`` records the touch before and after each own
taker fill so studies can measure that feedback directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .matching import BUY, MatchingEngine, Order, Report


@dataclass
class AgentState:
    owner: str
    open: dict[int, int] = field(default_factory=dict)           # oid -> remaining
    side_of: dict[int, int] = field(default_factory=dict)
    inventory: int = 0
    fills: list[Report] = field(default_factory=list)
    impact_trace: list[dict] = field(default_factory=list)

    def track(self, order: Order) -> None:
        self.side_of[order.oid] = order.side

    def on_reports(self, reports: list[Report]) -> None:
        for r in reports:
            if r.owner != self.owner:
                continue
            if r.kind == "ack":
                self.open[r.oid] = r.qty
            elif r.kind == "fill":
                self.fills.append(r)
                side = self.side_of.get(r.oid, BUY)
                self.inventory += r.qty if side == BUY else -r.qty
                if r.remaining == 0:
                    self.open.pop(r.oid, None)
                else:
                    self.open[r.oid] = r.remaining
            elif r.kind in ("cancel", "expire", "reject"):
                self.open.pop(r.oid, None)
            elif r.kind == "replace":
                self.open[r.oid] = r.remaining

    @property
    def outstanding(self) -> int:
        return sum(self.open.values())

    def shortfall_ticks(self, arrival_mid: float) -> float:
        """Signed cost of own fills against the arrival mid, in tick-lots (positive = cost)."""
        cost = 0.0
        for r in self.fills:
            side = self.side_of.get(r.oid, BUY)
            cost += (r.price - arrival_mid) * r.qty * (1 if side == BUY else -1)
        return cost


def submit_tracked(engine: MatchingEngine, agent: AgentState, order: Order, ts: int) -> list[Report]:
    """Submit an own order, update the agent and record touch before/after when it takes liquidity."""
    before = (engine.best(1), engine.best(-1))
    agent.track(order)
    reps = engine.submit(order, ts)
    agent.on_reports(reps)
    taken = sum(r.qty for r in reps if r.kind == "fill" and r.oid == order.oid and r.liquidity == "taker")
    if taken:
        agent.impact_trace.append({"ts": ts, "oid": order.oid, "taken": taken, "touch_before": before,
                                   "touch_after": (engine.best(1), engine.best(-1))})
    return reps


__all__ = ["AgentState", "submit_tracked"]
