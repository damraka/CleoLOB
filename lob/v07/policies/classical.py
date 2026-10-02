"""Expanded classical execution suite (workstream 31).

All agents subclass the frozen v0.5 ``ExecutionAgent`` (same fill accounting,
risk checks, fees and ledger) and act once per decision step. Each is a fixed
rule: its parameters are declared constants, never re-fitted per world, so a
cost difference across worlds is a property of the world, not of re-tuning.

* ``liquidity_sensitive`` — TWAP schedule, but takes up to twice the slice when
  the opposite touch holds at least the slice, and only what the touch holds
  otherwise (catch-up by the shared completion constraint).
* ``urgency`` — front-loaded schedule: remaining fraction ``sinh(k(T-t))/sinh(kT)``
  with ``k T = 3`` (an Almgren-Chriss trajectory with high, fixed urgency).
* ``imbalance_aware`` — TWAP slices scaled by top-5 imbalance in the direction of
  the trade: x2 when the book leans against the parent (e.g. bid-heavy for a buy),
  x0 when it leans with it; deferred quantity is carried forward.
* ``spread_aware`` — crosses only when the spread is one tick; otherwise posts the
  slice passively at its own touch (cancelled and re-posted each step).
* ``mpc`` (exploratory) — receding horizon: chooses the child among {0, slice/2,
  slice, 2 slice, remaining} minimizing immediate walk cost plus a TWAP
  continuation estimate at the current half-spread plus a variance penalty.
"""
from __future__ import annotations

import math

from ...engine import Side
from ...execution import ExecutionAgent

PRIMARY_NEW = ("liquidity_sensitive", "urgency", "imbalance_aware", "spread_aware")
EXPLORATORY = ("mpc",)
N_SLICES = 20


class _Paced(ExecutionAgent):
    def __init__(self, total_qty: int, horizon: float, side: Side, start_time: float, decision_dt: float, **kwargs):
        super().__init__(total_qty, horizon, side, start_time, **kwargs)
        self.decision_dt = decision_dt
        self.steps = max(1, int(round(horizon / decision_dt)))

    def elapsed_steps(self, sim) -> int:
        return int(round((sim.t - self.start_time) / self.decision_dt))

    def schedule_remaining(self, sim) -> float:
        return self.total_qty * max(0.0, 1.0 - (self.elapsed_steps(sim) + 1) / self.steps)

    def deficit(self, sim) -> int:
        return max(0, int(math.ceil(self.remaining - self.risk.outstanding(sim) - self.schedule_remaining(sim))))

    def touch(self, sim) -> tuple[int | None, int]:
        """Opposite best price and its volume."""
        if self.side is Side.BUY:
            price = sim.book.best_ask()
            return price, sim.book.ask_vol.get(price, 0) if price is not None else 0
        price = sim.book.best_bid()
        return price, sim.book.bid_vol.get(price, 0) if price is not None else 0


class LiquiditySensitiveAgent(_Paced):
    label = "Liquidity-sensitive"

    def on_step(self, sim, owner) -> None:
        self.poll_fills(sim, owner)
        slice_ = max(1, int(math.ceil(self.total_qty / self.steps)))
        _, depth = self.touch(sim)
        want = min(2 * slice_, depth) if depth >= slice_ else min(depth, self.deficit(sim))
        self._send_market(sim, owner, max(want, 0))


class UrgencyAgent(_Paced):
    label = "Urgency"
    KAPPA_T = 3.0

    def on_step(self, sim, owner) -> None:
        self.poll_fills(sim, owner)
        tau = min(1.0, (self.elapsed_steps(sim) + 1) / self.steps)
        target_remaining = self.total_qty * math.sinh(self.KAPPA_T * (1 - tau)) / math.sinh(self.KAPPA_T)
        child = int(math.ceil(self.remaining - self.risk.outstanding(sim) - target_remaining))
        self._send_market(sim, owner, max(child, 0))


class ImbalanceAwareAgent(_Paced):
    label = "Imbalance-aware"
    THRESHOLD = 0.3

    def on_step(self, sim, owner) -> None:
        self.poll_fills(sim, owner)
        imbalance = sim.book.imbalance(5)
        lean = imbalance if self.side is Side.BUY else -imbalance   # > 0: book leans against the parent
        multiplier = 2.0 if lean > self.THRESHOLD else 0.0 if lean < -self.THRESHOLD else 1.0
        child = int(math.ceil(self.deficit(sim) * multiplier))
        self._send_market(sim, owner, min(child, self.remaining))


class SpreadAwareAgent(_Paced):
    label = "Spread-aware"

    def on_step(self, sim, owner) -> None:
        self.poll_fills(sim, owner)
        self.risk.cancel_all(sim, reason="repost")
        need = self.deficit(sim)
        if need <= 0:
            return
        spread = sim.book.spread()
        if spread is not None and spread <= 1:
            self._send_market(sim, owner, need)
            return
        own = sim.book.best_bid() if self.side is Side.BUY else sim.book.best_ask()
        if own is None:
            self._send_market(sim, owner, need)
            return
        qty = min(need, self.available(sim))
        if qty > 0 and self.risk.submit(sim, owner, qty, self.filled, self._ensure_ledger(sim), price_ticks=own):
            self.child_orders += 1


class MPCAgent(_Paced):
    label = "MPC (exploratory)"
    RISK = 1e-4

    def on_step(self, sim, owner) -> None:
        self.poll_fills(sim, owner)
        remaining = self.remaining - self.risk.outstanding(sim)
        if remaining <= 0:
            return
        steps_left = max(1, self.steps - self.elapsed_steps(sim))
        slice_ = max(1, int(math.ceil(remaining / steps_left)))
        spread = sim.book.spread() or 1
        best = None
        for q in sorted({0, max(1, slice_ // 2), slice_, 2 * slice_, remaining}):
            q = min(q, remaining)
            immediate = sim.book.walk_cost(self.side, q)[0] if q else 0.0
            mid = sim.book.mid()
            immediate_excess = (immediate - q * mid) * self.side.value if q else 0.0
            rest = remaining - q
            continuation = rest * spread / 2
            variance = self.RISK * rest * rest * (steps_left - 1)
            cost = immediate_excess + continuation + variance
            if best is None or cost < best[0] - 1e-12:
                best = (cost, q)
        if steps_left <= 1:
            best = (0.0, remaining)
        self._send_market(sim, owner, best[1])


AGENTS = {"liquidity_sensitive": LiquiditySensitiveAgent, "urgency": UrgencyAgent,
          "imbalance_aware": ImbalanceAwareAgent, "spread_aware": SpreadAwareAgent, "mpc": MPCAgent}
