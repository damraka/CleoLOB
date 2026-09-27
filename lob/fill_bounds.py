"""Bounded fill evidence for hypothetical passive child orders.

A hypothetical order never receives a binary historical fill label by default.
Each order gets nested bounds under explicit assumption sets:

``conservative_lower <= fifo_lower <= fifo_upper <= optimistic_upper``

* conservative: only volume printed strictly *through* the order's price fills it.
  Requires price priority only; tolerates unknown hidden or priority-ahead size.
* observable FIFO (aggregate L2): the order joins behind the displayed level;
  non-trade level decreases are attributed behind the order (lower) or ahead of
  it (upper). Assumes price-time priority and no hidden liquidity at the level.
* identity FIFO (order-level data): the queue ahead is the exact set of source
  identities resting at the price; their executions and cancellations are
  observed. Under price-time priority this yields a point value.
* optimistic: the order is at the front of its level; every print at or through
  its price fills it.

All sets assume the hypothetical order does not change anyone else's behaviour.
Aggregate L2 never yields order identity, exact FIFO position or hidden size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .capabilities import Capability, CapabilityContract, DataLevel

CLASSES = ("OBSERVED_FILL", "GUARANTEED_FILL", "POSSIBLE_FILL", "GUARANTEED_NON_FILL",
           "INDETERMINATE", "UNSUPPORTED")
COMMON_ASSUMPTIONS = (
    "The hypothetical order does not change other participants' behaviour (no market impact).",
    "Trade prints are attributed to the aggressor side recorded by the source.",
)
L2_ASSUMPTIONS = ("Observable-FIFO bounds additionally assume price-time priority and no hidden "
                  "quantity at the order's price; neither is established by aggregate L2.",)
MBO_ASSUMPTIONS = ("Identity-FIFO values additionally assume price-time priority, which the source "
                   "feed does not itself establish.",)


@dataclass(frozen=True)
class ChildOrder:
    """A preregistered hypothetical passive order; quantities in source units."""

    order_id: str
    side: str  # BUY rests on the bid, SELL on the ask
    price: Any
    quantity: Any
    submit_ts: int
    expire_ts: int

    def __post_init__(self) -> None:
        if self.side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if not self.quantity > 0 or not self.expire_ts > self.submit_ts:
            raise ValueError("child quantity must be positive and lifetime nonempty")


@dataclass
class FillBound:
    order: ChildOrder
    capability_level: str
    classification: str
    conservative_lower: Any = 0
    fifo_lower: Any = None
    fifo_upper: Any = None
    optimistic_upper: Any = 0
    observable_evidence: dict = field(default_factory=dict)
    queue_evidence: dict = field(default_factory=dict)
    trade_evidence: dict = field(default_factory=dict)
    uncertainty_reason: str = ""
    assumptions: tuple = COMMON_ASSUMPTIONS
    capability_requirements: tuple = ()
    fifo_classification: str | None = None

    def to_dict(self) -> dict:
        def num(x):
            return None if x is None else (str(x) if isinstance(x, Decimal) else x)
        return {"order_id": self.order.order_id, "side": self.order.side, "price": num(self.order.price),
                "quantity": num(self.order.quantity), "submit_ts": self.order.submit_ts,
                "expire_ts": self.order.expire_ts, "capability_level": self.capability_level,
                "classification": self.classification, "conservative_lower": num(self.conservative_lower),
                "fifo_lower": num(self.fifo_lower), "fifo_upper": num(self.fifo_upper),
                "optimistic_upper": num(self.optimistic_upper), "fifo_classification": self.fifo_classification,
                "observable_evidence": _plain(self.observable_evidence), "queue_evidence": _plain(self.queue_evidence),
                "trade_evidence": _plain(self.trade_evidence), "uncertainty_reason": self.uncertainty_reason,
                "assumptions": list(self.assumptions), "capability_requirements": list(self.capability_requirements)}


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return str(value) if isinstance(value, Decimal) else value


def classify(lower, upper, quantity) -> str:
    if lower > upper:
        raise ValueError("fill bounds are inconsistent")
    if lower >= quantity:
        return "GUARANTEED_FILL"
    if upper <= 0:
        return "GUARANTEED_NON_FILL"
    return "POSSIBLE_FILL"


def _through(side: str, trade_price, order_price) -> bool:
    """A print strictly beyond the order's price on the order's side of the book."""
    return trade_price < order_price if side == "BUY" else trade_price > order_price


def _hits(side: str, aggressor: str) -> bool:
    """Only aggressors on the opposite side can fill a resting order."""
    return aggressor != side


class L2FillTracker:
    """Bounds for one child order from aggregate levels and aggressor-signed prints.

    Feed events in the declared order: prints observed up to a book update are
    applied before that update. ``displayed`` excludes the hypothetical order.
    """

    def __init__(self, order: ChildOrder, displayed_at_submit, contract: CapabilityContract):
        if contract.level != DataLevel.L2:
            raise ValueError("L2FillTracker requires an aggregate-L2 contract")
        contract.require(Capability.AGGREGATE_DEPTH, Capability.TRADE_PRINTS)
        for forbidden in (Capability.EXACT_FIFO_POSITION, Capability.ORDER_IDENTITY,
                          Capability.HIDDEN_ORDER_QUANTITY, Capability.COUNTERFACTUAL_PASSIVE_FILL):
            if contract.supports(forbidden):
                raise ValueError("aggregate L2 cannot declare identity, FIFO or hidden-liquidity evidence")
        self.order = order
        zero = order.quantity * 0
        self.zero = zero
        self.displayed = displayed_at_submit
        self.ahead_optimistic = displayed_at_submit   # FIFO upper path: cancels come from ahead
        self.ahead_pessimistic = displayed_at_submit  # FIFO lower path: cancels come from behind
        self.fill = {"conservative": zero, "fifo_lower": zero, "fifo_upper": zero, "optimistic": zero}
        self.at_level_since_update = zero
        self.evidence = {"displayed_at_submit": displayed_at_submit, "prints_at_price": 0, "volume_at_price": zero,
                         "prints_through": 0, "volume_through": zero, "level_decreases_unexplained": zero,
                         "level_increases": zero, "level_updates": 0}
        self.closed = False

    def _add(self, key: str, amount) -> None:
        room = self.order.quantity - self.fill[key]
        if amount > 0 and room > 0:
            self.fill[key] += min(room, amount)

    def on_print(self, price, amount, aggressor: str) -> None:
        if self.closed or not _hits(self.order.side, aggressor):
            return
        if _through(self.order.side, price, self.order.price):
            self.evidence["prints_through"] += 1
            self.evidence["volume_through"] += amount
            for key in self.fill:
                self._add(key, amount)
            self.ahead_optimistic = self.ahead_pessimistic = self.zero
        elif price == self.order.price:
            self.evidence["prints_at_price"] += 1
            self.evidence["volume_at_price"] += amount
            self.at_level_since_update += amount
            self._add("optimistic", amount)
            for key, attr in (("fifo_upper", "ahead_optimistic"), ("fifo_lower", "ahead_pessimistic")):
                ahead = getattr(self, attr)
                self._add(key, amount - ahead)
                setattr(self, attr, max(self.zero, ahead - amount))

    def on_level(self, displayed_now) -> None:
        """Displayed quantity at the order's price after a book update (0 if absent)."""
        if self.closed:
            return
        self.evidence["level_updates"] += 1
        decrease = self.displayed - displayed_now
        if decrease > 0:
            cancels = max(self.zero, decrease - self.at_level_since_update)
            self.evidence["level_decreases_unexplained"] += cancels
            self.ahead_optimistic = max(self.zero, self.ahead_optimistic - cancels)
            behind = max(self.zero, self.displayed - self.at_level_since_update - self.ahead_pessimistic)
            from_ahead = max(self.zero, cancels - behind)
            self.ahead_pessimistic = max(self.zero, self.ahead_pessimistic - from_ahead)
        elif decrease < 0:
            self.evidence["level_increases"] += -decrease
        # Displayed quantity bounds anything that can still be ahead.
        self.ahead_pessimistic = min(self.ahead_pessimistic, displayed_now)
        self.ahead_optimistic = min(self.ahead_optimistic, displayed_now)
        self.displayed = displayed_now
        self.at_level_since_update = self.zero

    def result(self, contract: CapabilityContract, *, indeterminate_reason: str | None = None) -> FillBound:
        self.closed = True
        q = self.order.quantity
        requirements = (Capability.AGGREGATE_DEPTH.value, Capability.TRADE_PRINTS.value)
        if indeterminate_reason:
            return FillBound(self.order, contract.level.value, "INDETERMINATE", self.zero, None, None, q,
                             self.evidence, {}, {}, indeterminate_reason, COMMON_ASSUMPTIONS + L2_ASSUMPTIONS,
                             requirements)
        f = self.fill
        assert f["conservative"] <= f["fifo_lower"] <= f["fifo_upper"] <= f["optimistic"]
        return FillBound(
            self.order, contract.level.value, classify(f["conservative"], f["optimistic"], q),
            f["conservative"], f["fifo_lower"], f["fifo_upper"], f["optimistic"],
            observable_evidence={k: v for k, v in self.evidence.items() if k.startswith(("displayed", "level"))},
            queue_evidence={"displayed_ahead_at_submit": self.evidence["displayed_at_submit"],
                            "fifo_ahead_remaining_upper_path": self.ahead_optimistic,
                            "fifo_ahead_remaining_lower_path": self.ahead_pessimistic,
                            "exact_queue_position": "UNSUPPORTED by aggregate L2"},
            trade_evidence={k: v for k, v in self.evidence.items() if "print" in k or "volume" in k},
            uncertainty_reason=("Exact FIFO position, hidden quantity and cancellation location are unobservable "
                                "in aggregate L2; bounds span these unknowns."),
            assumptions=COMMON_ASSUMPTIONS + L2_ASSUMPTIONS, capability_requirements=requirements,
            fifo_classification=classify(f["fifo_lower"], f["fifo_upper"], q))


class MBOFillTracker:
    """Bounds from exact source identities at the order's price.

    ``ahead`` maps resting identities at the price at submission to quantities.
    Executions of later identities at the price while identities ahead still rest
    are counted as observed departures from price-time priority.
    """

    def __init__(self, order: ChildOrder, ahead: dict[str, int], contract: CapabilityContract):
        if contract.level != DataLevel.MBO:
            raise ValueError("MBOFillTracker requires an order-level contract")
        contract.require(Capability.ORDER_IDENTITY, Capability.OBSERVED_ORDER_FILL)
        self.order = order
        self.ahead = dict(ahead)
        self.fifo_point = 0
        self.conservative = 0
        self.optimistic = 0
        self.evidence = {"orders_ahead_at_submit": len(ahead), "quantity_ahead_at_submit": sum(ahead.values()),
                         "ahead_executed": 0, "ahead_cancelled": 0, "later_executed_at_price": 0,
                         "later_executed_while_ahead_resting": 0, "executions_through": 0, "volume_through": 0,
                         "volume_at_price": 0}
        self.closed = False

    def _room(self, value: int) -> int:
        return self.order.quantity - value

    def on_execute(self, order_id: str, side: str, price: int, quantity: int) -> None:
        """Observed maker execution of a source identity."""
        if self.closed or side != self.order.side:
            return
        if _through(self.order.side, price, self.order.price):
            self.evidence["executions_through"] += 1
            self.evidence["volume_through"] += quantity
            self.conservative += min(self._room(self.conservative), quantity)
            self.fifo_point += min(self._room(self.fifo_point), quantity)
            self.optimistic += min(self._room(self.optimistic), quantity)
            self.ahead.clear()
        elif price == self.order.price:
            self.evidence["volume_at_price"] += quantity
            self.optimistic += min(self._room(self.optimistic), quantity)
            if order_id in self.ahead:
                self.evidence["ahead_executed"] += quantity
                self.ahead[order_id] -= quantity
                if self.ahead[order_id] <= 0:
                    del self.ahead[order_id]
            else:
                self.evidence["later_executed_at_price"] += quantity
                if self.ahead:
                    self.evidence["later_executed_while_ahead_resting"] += quantity
                # Under price-time priority this execution would have reached the
                # hypothetical order before any later identity.
                self.fifo_point += min(self._room(self.fifo_point), quantity)

    def on_reduce(self, order_id: str, quantity_removed: int) -> None:
        """Cancellation or reduction of an identity; removes it from the queue ahead if listed."""
        if self.closed or order_id not in self.ahead:
            return
        self.evidence["ahead_cancelled"] += min(quantity_removed, self.ahead[order_id])
        self.ahead[order_id] -= quantity_removed
        if self.ahead[order_id] <= 0:
            del self.ahead[order_id]

    def on_priority_loss(self, order_id: str) -> None:
        """A repriced or size-increased identity leaves the queue ahead."""
        self.ahead.pop(order_id, None)

    def result(self, contract: CapabilityContract, *, indeterminate_reason: str | None = None) -> FillBound:
        self.closed = True
        q = self.order.quantity
        requirements = (Capability.ORDER_IDENTITY.value, Capability.OBSERVED_ORDER_FILL.value)
        if indeterminate_reason:
            return FillBound(self.order, contract.level.value, "INDETERMINATE", 0, None, None, q, self.evidence,
                             {}, {}, indeterminate_reason, COMMON_ASSUMPTIONS + MBO_ASSUMPTIONS, requirements)
        assert self.conservative <= self.fifo_point <= self.optimistic
        fifo = contract.supports(Capability.EXACT_FIFO_POSITION)
        return FillBound(
            self.order, contract.level.value, classify(self.conservative, self.optimistic, q),
            self.conservative, self.fifo_point, self.fifo_point, self.optimistic,
            observable_evidence={"identities_ahead_at_submit": self.evidence["orders_ahead_at_submit"]},
            queue_evidence={"quantity_ahead_at_submit": self.evidence["quantity_ahead_at_submit"],
                            "ahead_executed": self.evidence["ahead_executed"],
                            "ahead_cancelled": self.evidence["ahead_cancelled"],
                            "later_executed_while_ahead_resting": self.evidence["later_executed_while_ahead_resting"],
                            "fifo_established_by_source": fifo},
            trade_evidence={k: self.evidence[k] for k in ("volume_at_price", "volume_through", "executions_through",
                                                            "later_executed_at_price")},
            uncertainty_reason=("Identity-FIFO value is exact only under price-time priority, which the source does "
                                "not establish; conservative/optimistic bounds do not assume it."),
            assumptions=COMMON_ASSUMPTIONS + MBO_ASSUMPTIONS, capability_requirements=requirements,
            fifo_classification=classify(self.fifo_point, self.fifo_point, q))


def unsupported(order: ChildOrder, contract: CapabilityContract, reason: str) -> FillBound:
    return FillBound(order, contract.level.value, "UNSUPPORTED", uncertainty_reason=reason,
                     conservative_lower=None, optimistic_upper=None)


def summarize(bounds: list[FillBound]) -> dict:
    """Shares by class plus normalized widths; never averages into a point estimate."""
    total = len(bounds)
    counts = {name: sum(b.classification == name for b in bounds) for name in CLASSES}
    evaluable = [b for b in bounds if b.classification not in {"INDETERMINATE", "UNSUPPORTED"}]
    def width(lo, hi, q):
        return float((hi - lo) / q)
    fifo = [b for b in evaluable if b.fifo_lower is not None]
    return {"orders": total, "classes": counts,
            "determinate_share_conservative": (counts["GUARANTEED_FILL"] + counts["GUARANTEED_NON_FILL"]) / total
            if total else None,
            "mean_conservative_width": sum(width(b.conservative_lower, b.optimistic_upper, b.order.quantity)
                                           for b in evaluable) / len(evaluable) if evaluable else None,
            "mean_fifo_width": sum(width(b.fifo_lower, b.fifo_upper, b.order.quantity) for b in fifo) / len(fifo)
            if fifo else None,
            "fifo_classes": {name: sum(b.fifo_classification == name for b in fifo) for name in CLASSES},
            "mean_conservative_fill_fraction": sum(float(b.conservative_lower / b.order.quantity)
                                                   for b in evaluable) / len(evaluable) if evaluable else None,
            "mean_optimistic_fill_fraction": sum(float(b.optimistic_upper / b.order.quantity)
                                                 for b in evaluable) / len(evaluable) if evaluable else None}
