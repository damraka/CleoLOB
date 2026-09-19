"""Bounded cancellation and fill reconciliation after strategy decisions stop.

Exchange time continues during settlement, so delayed strategy orders can still
fill. A cancellation rejected before its order arrives must be retried; merely
requesting cancellation does not release the parent reservation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Callable

from .engine import ExchangeSimulator
from .risk import ExecutionRisk

DEFAULT_SETTLEMENT_TIMEOUT = 5.0
DEFAULT_SETTLEMENT_POLL_DT = 0.01
MAX_SETTLEMENT_POLLS = 100_000


def validate_settlement(timeout: float, poll_dt: float) -> None:
    if (isinstance(timeout, bool) or isinstance(poll_dt, bool)
            or not isinstance(timeout, Real) or not isinstance(poll_dt, Real)
            or not math.isfinite(timeout) or not math.isfinite(poll_dt)
            or timeout < 0 or poll_dt <= 0):
        raise ValueError("settlement timeout must be finite/nonnegative and poll_dt finite/positive")
    if timeout / poll_dt > MAX_SETTLEMENT_POLLS:
        raise ValueError(f"settlement requires more than {MAX_SETTLEMENT_POLLS} polls")


@dataclass(frozen=True)
class SettlementResult:
    started_at: float
    ended_at: float
    complete: bool
    pending_order_ids: tuple[int, ...]
    polls: int

    @property
    def duration(self) -> float:
        return self.ended_at - self.started_at


def settle_orders(sim: ExchangeSimulator, risk: ExecutionRisk,
                  poll_fills: Callable[[], None], *,
                  timeout: float = DEFAULT_SETTLEMENT_TIMEOUT,
                  poll_dt: float = DEFAULT_SETTLEMENT_POLL_DT,
                  on_step: Callable[[], None] | None = None) -> SettlementResult:
    """Drain only cancellations and exchange events; never place a new child.

    Every exchange step is followed by fill reconciliation before checking order
    terminality. Polling uses a fixed, disclosed interval and clips exactly to the
    timeout. Zero-latency cancellation is processed even with a zero timeout.
    The exchange's own event cap remains in force.
    """
    validate_settlement(timeout, poll_dt)
    started_at = sim.t
    deadline = started_at + timeout
    if not math.isfinite(deadline):
        raise ValueError("settlement deadline overflow")

    def pending() -> tuple[int, ...]:
        return tuple(oid for oid in risk.order_ids if not sim.orders[oid].is_terminal)

    def advance(dt: float) -> None:
        # Even an exchange resource failure can occur after processing fills.
        # Reconcile that executed prefix before propagating the failure.
        try:
            sim.step(dt)
        finally:
            poll_fills()

    poll_fills()
    polls = 0
    while pending():
        risk.cancel_all(sim, reason="settlement_cancel")
        # Process same-time acknowledgements before concluding a zero-duration
        # window or advancing to the next polling boundary.
        advance(0.0)
        if not pending() or sim.t >= deadline:
            break
        step = min(poll_dt, deadline - sim.t)
        if sim.t + step <= sim.t:
            raise ValueError("settlement polling interval cannot advance the exchange clock")
        advance(step)
        polls += 1
        if on_step is not None:
            on_step()
        # A cancel rejected before order admission changes the state back to
        # IN_FLIGHT. The next iteration retries it without releasing its leaves.
    remaining = pending()
    return SettlementResult(started_at, sim.t, not remaining, remaining, polls)
