"""Preregistered hypothetical passive child orders evaluated with bounded historical fill evidence.

Aggregate-L2 driver: one streaming pass over a Tardis incremental L2 file plus the
matching trades file. Each child order joins the best price observed strictly
before its submission time, so no future book state is used. Prints with capture
time in (submit, expire] are applied before the book update that follows them.

Order-level driver: one pass over normalized lifecycle events. Besides
hypothetical orders, genuine source orders that joined the back of the best
price are re-evaluated from their queue position at entry and the public flow at
their price. Their actually observed fills then test whether the conservative
bound holds (prints through the price must fill a resting order under price
priority) and whether the venue behaved as price-time FIFO. Observed outcomes
use OBSERVED_FILL.
"""
from __future__ import annotations

import csv
from decimal import Decimal
import gzip
from pathlib import Path
from typing import Any, Iterable, Iterator

from .capabilities import Capability, CapabilityContract, DataLevel
from .fill_bounds import ChildOrder, FillBound, L2FillTracker, MBOFillTracker, summarize, unsupported
from .order_lifecycle import LifecycleEvent, LifecycleSemantics, OrderLifecycleBook
from .replay.l2 import L2Replay, exact_decimal

L2_WITH_TRADES = CapabilityContract(
    DataLevel.L2, {Capability.TOP_OF_BOOK, Capability.SPREAD, Capability.BOOK_IMBALANCE,
                   Capability.AGGREGATE_DEPTH, Capability.AGGREGATE_VOLUME_CHANGES, Capability.TRADE_PRINTS},
    "aggregate L2 levels plus aggressor-signed trade prints; no identity, FIFO or hidden size")
MBO_IDENTITY = CapabilityContract(
    DataLevel.MBO, {Capability.TOP_OF_BOOK, Capability.SPREAD, Capability.BOOK_IMBALANCE,
                    Capability.AGGREGATE_DEPTH, Capability.AGGREGATE_VOLUME_CHANGES, Capability.TRADE_PRINTS,
                    Capability.ORDER_IDENTITY, Capability.OBSERVED_ORDER_FILL},
    "source identities and observed maker executions; FIFO not established by the source")
DESIGN_KEYS = ("start_offset_s", "interval_s", "end_margin_s", "sides", "sizes", "lifetimes_s",
               "price_rule", "max_gap_s", "max_staleness_s")


def validate_design(design: dict) -> dict:
    missing = [key for key in DESIGN_KEYS if key not in design]
    if missing:
        raise ValueError(f"child-order design missing {missing}")
    if design["price_rule"] != "join_best":
        raise ValueError("only the preregistered join_best price rule is implemented")
    if not set(design["sides"]) <= {"BUY", "SELL"} or not design["sizes"] or not design["lifetimes_s"]:
        raise ValueError("design requires sides, sizes and lifetimes")
    if design["interval_s"] <= 0 or design["max_gap_s"] <= 0 or design["max_staleness_s"] < 0:
        raise ValueError("design intervals must be positive")
    return design


def schedule(design: dict, first_us: int, last_us: int) -> list[tuple[int, str, Any, int]]:
    """Deterministic (submit_us, side, size_text, lifetime_us) grid inside the observed file."""
    validate_design(design)
    times = []
    t = first_us + int(design["start_offset_s"] * 1e6)
    stop = last_us - int(design["end_margin_s"] * 1e6)
    while t <= stop:
        for side in design["sides"]:
            for size in design["sizes"]:
                for life in design["lifetimes_s"]:
                    times.append((t, side, size, int(life * 1e6)))
        t += int(design["interval_s"] * 1e6)
    return times


def _trades(path: Path) -> Iterator[tuple[int, Decimal, Decimal, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    last = None
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            local = int(row["local_timestamp"])
            if last is not None and local < last:
                raise ValueError("trade capture timestamps regress; order cannot be established")
            last = local
            side = row["side"].lower()
            if side not in {"buy", "sell"}:
                continue  # unknown aggressor cannot sign a print
            yield (local, exact_decimal(row["price"], "trade price"), exact_decimal(row["amount"], "trade amount"),
                   "BUY" if side == "buy" else "SELL")


def run_l2(updates: str | Path, trades: str | Path, design: dict, *, first_us: int | None = None,
           last_us: int | None = None, dataset_id: str = "") -> dict:
    """Stream one L2 day; returns per-order bounds and a summary."""
    validate_design(design)
    if last_us is None:
        last_us = replay_last_hint(updates)
    replay = L2Replay(Path(updates), depth=1)
    trade_iter = _trades(Path(trades))
    pending_trade = next(trade_iter, None)
    plan: list[tuple[int, str, Any, int]] | None = None
    plan_index = 0
    active: list[tuple[L2FillTracker, str | None]] = []
    results: list[FillBound] = []
    previous = None
    max_gap = int(design["max_gap_s"] * 1e6)
    stale = int(design["max_staleness_s"] * 1e6)
    counter = 0
    for state in replay:
        now = state.local_timestamp_us
        if plan is None:
            plan = schedule(design, first_us or now, last_us)
        # 1) submissions strictly before this update use the previous completed state
        while plan_index < len(plan) and plan[plan_index][0] < now:
            submit, side, size_text, life = plan[plan_index]
            plan_index += 1
            counter += 1
            size = Decimal(str(size_text))
            if previous is None or previous.crossed or previous.one_sided or previous.empty \
                    or submit - previous.local_timestamp_us > stale:
                order = ChildOrder(f"{dataset_id}-{counter}", side, None, size, submit, submit + life)
                results.append(FillBound(order, L2_WITH_TRADES.level.value, "INDETERMINATE", Decimal(0), None, None,
                                         size, uncertainty_reason="invalid or stale book at submission"))
                continue
            price, displayed = (previous.bids[0] if side == "BUY" else previous.asks[0])
            order = ChildOrder(f"{dataset_id}-{counter}", side, price, size, submit, submit + life)
            active.append((L2FillTracker(order, displayed, L2_WITH_TRADES), None))
        # 2) prints up to this update, each applied inside every order's lifetime
        while pending_trade is not None and pending_trade[0] <= now:
            local, price, amount, aggressor = pending_trade
            for tracker, _ in active:
                if tracker.order.submit_ts < local <= tracker.order.expire_ts:
                    tracker.on_print(price, amount, aggressor)
            pending_trade = next(trade_iter, None)
        # 3) close expired orders, flag data gaps, then apply this level update
        still = []
        for tracker, reason in active:
            if previous is not None and now - previous.local_timestamp_us > max_gap and reason is None:
                reason = "capture gap longer than max_gap_s during order lifetime"
            if tracker.order.expire_ts < now:
                results.append(tracker.result(L2_WITH_TRADES, indeterminate_reason=reason))
                continue
            book = replay._bids if tracker.order.side == "BUY" else replay._asks
            tracker.on_level(book.get(tracker.order.price, Decimal(0)))
            still.append((tracker, reason))
        active = still
        previous = state
    for tracker, reason in active:
        results.append(tracker.result(L2_WITH_TRADES, indeterminate_reason=reason or
                                      "order lifetime extends beyond the end of the file (right-censored)"))
    if not replay.stats["complete"]:
        raise ValueError("L2 source incomplete; bounds withheld")
    return {"dataset_id": dataset_id, "capability": L2_WITH_TRADES.to_dict(), "design": design,
            "summary": summarize(results), "orders": [b.to_dict() for b in results],
            "source_stats": replay.stats}


def replay_last_hint(updates: str | Path) -> int:
    """Last capture timestamp, read in a separate pass so the schedule stays inside the file."""
    last = None
    opener = gzip.open if str(updates).endswith(".gz") else open
    with opener(updates, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        column = header.index("local_timestamp")
        for row in reader:
            last = int(row[column])
    if last is None:
        raise ValueError("empty L2 source")
    return last


def run_mbo(events: Iterable[LifecycleEvent], design: dict, *, semantics: LifecycleSemantics | None = None,
            dataset_id: str = "", observed_lifetime_s: float = 60.0, max_observed: int = 20000) -> dict:
    """Hypothetical orders plus observed-order coverage on genuine identity data."""
    validate_design(design)
    events = list(events)
    stamps = [e.exchange_ts_ns for e in events if e.exchange_ts_ns is not None]
    if not stamps:
        raise ValueError("order-level stream has no exchange timestamps")
    first_us, last_us = stamps[0] // 1000, max(stamps) // 1000
    plan = schedule(design, first_us, last_us)
    book = OrderLifecycleBook(semantics or LifecycleSemantics())
    plan_index = 0
    counter = 0
    hypothetical: list[tuple[MBOFillTracker, None]] = []
    observed: dict[str, dict] = {}
    finished: list[FillBound] = []
    observed_results = []
    life_ns = int(observed_lifetime_s * 1e9)

    def close(tracker, reason=None):
        return tracker.result(MBO_IDENTITY, indeterminate_reason=reason)

    for event in events:
        ts = event.exchange_ts_ns
        if ts is not None:
            while plan_index < len(plan) and plan[plan_index][0] * 1000 < ts:
                submit_us, side, size_text, life = plan[plan_index]
                plan_index += 1
                counter += 1
                size = int(size_text)
                bid, ask = book.best("BUY"), book.best("SELL")
                if not book.initialized or bid is None or ask is None or bid >= ask:
                    order = ChildOrder(f"{dataset_id}-{counter}", side, None, size, submit_us * 1000,
                                       submit_us * 1000 + life * 1000)
                    finished.append(FillBound(order, MBO_IDENTITY.level.value, "INDETERMINATE", 0, None, None, size,
                                              uncertainty_reason="invalid book at submission"))
                    continue
                price = bid if side == "BUY" else ask
                ahead = {oid: book.order(oid)["quantity"] for oid in book.tracked_level_order(side, price)}
                order = ChildOrder(f"{dataset_id}-{counter}", side, price, size, submit_us * 1000,
                                   submit_us * 1000 + life * 1000)
                hypothetical.append((MBOFillTracker(order, ahead, MBO_IDENTITY), None))
            # expire hypothetical orders strictly before this event
            keep = []
            for tracker, _ in hypothetical:
                if tracker.order.expire_ts < ts:
                    finished.append(close(tracker))
                else:
                    keep.append((tracker, None))
            hypothetical = keep
            for oid in [o for o, item in observed.items() if item["tracker"].order.expire_ts < ts]:
                item = observed.pop(oid)
                observed_results.append(_observed_outcome(item, close(item["tracker"])))
        before = book.order(event.order_id) if event.order_id else None
        joins_back = (event.kind == "ADD" and book.initialized and len(observed) + len(observed_results) < max_observed
                      and event.price == book.best(event.side) and event.order_id not in observed)
        if joins_back:
            ahead = {oid: book.order(oid)["quantity"] for oid in book.tracked_level_order(event.side, event.price)}
        book.apply(event)
        if joins_back and book.order(event.order_id) is not None:
            order = ChildOrder(event.order_id, event.side, event.price, event.quantity, ts, ts + life_ns)
            # The order's own executions are public flow at its price from an identity that is not
            # ahead of it, so they enter its bounds exactly as any later identity's would.
            observed[event.order_id] = {"tracker": MBOFillTracker(order, ahead, MBO_IDENTITY),
                                        "filled": 0, "terminal": None}
        trackers = [t for t, _ in hypothetical] + [item["tracker"] for item in observed.values()]
        if event.kind == "EXECUTE" and before is not None:
            for tracker in trackers:
                tracker.on_execute(event.order_id, before["side"], before["price"], event.quantity)
            if event.order_id in observed:
                observed[event.order_id]["filled"] += event.quantity
        elif event.kind == "CANCEL" and before is not None:
            removed = before["quantity"] if event.quantity is None else event.quantity
            for tracker in trackers:
                tracker.on_reduce(event.order_id, removed)
        elif event.kind == "MODIFY" and before is not None:
            after = book.order(event.order_id)
            for tracker in trackers:
                if after is None:
                    tracker.on_reduce(event.order_id, before["quantity"])
                elif after["price"] != before["price"] or after["quantity"] > before["quantity"]:
                    tracker.on_priority_loss(event.order_id)
                else:
                    tracker.on_reduce(event.order_id, before["quantity"] - after["quantity"])
        elif event.kind in {"GAP", "SNAPSHOT"} and (event.kind == "GAP" or event.snapshot_mode == "reset"):
            for tracker, _ in hypothetical:
                finished.append(close(tracker, "gap or census reset during lifetime"))
            hypothetical = []
            for item in observed.values():
                observed_results.append(_observed_outcome(item, close(item["tracker"], "gap during lifetime")))
            observed = {}
        # an observed order that left the book ends its own observation window
        if event.order_id in observed and book.order(event.order_id) is None:
            item = observed.pop(event.order_id)
            tracker = item["tracker"]
            if ts is None or ts <= tracker.order.submit_ts:
                # Joined and left within one source timestamp: no observation window exists.
                observed_results.append({"order_id": tracker.order.order_id, "indeterminate": True,
                                         "zero_length_window": True, "observed_fill": item["filled"]})
                continue
            tracker.order = ChildOrder(tracker.order.order_id, tracker.order.side, tracker.order.price,
                                       tracker.order.quantity, tracker.order.submit_ts, ts)
            observed_results.append(_observed_outcome(item, close(tracker)))
    for tracker, _ in hypothetical:
        finished.append(close(tracker, "order lifetime extends beyond the capture (right-censored)"))
    for item in observed.values():
        observed_results.append(_observed_outcome(item, close(item["tracker"], "right-censored at capture end")))
    return {"dataset_id": dataset_id, "capability": MBO_IDENTITY.to_dict(), "design": design,
            "summary": summarize(finished), "orders": [b.to_dict() for b in finished],
            "observed_order_validation": _coverage(observed_results)}


def _observed_outcome(item: dict, bound: FillBound) -> dict:
    return {"order_id": bound.order.order_id, "quantity": bound.order.quantity, "observed_fill": item["filled"],
            "classification": "OBSERVED_FILL" if item["filled"] else "OBSERVED_NON_FILL_IN_WINDOW",
            "conservative_lower": bound.conservative_lower, "fifo_point": bound.fifo_lower,
            "optimistic_upper": bound.optimistic_upper, "indeterminate": bound.classification == "INDETERMINATE",
            "later_executed_while_ahead_resting": bound.queue_evidence.get("later_executed_while_ahead_resting", 0)
            if bound.queue_evidence else None}


def _coverage(rows: list[dict]) -> dict:
    usable = [r for r in rows if not r["indeterminate"]]
    n = len(usable)
    zero_length = sum(bool(r.get("zero_length_window")) for r in rows)
    if not n:
        return {"orders": len(rows), "evaluable": 0, "zero_length_windows_excluded": zero_length}
    inside = sum(r["conservative_lower"] <= r["observed_fill"] <= r["optimistic_upper"] for r in usable)
    exact = sum(r["observed_fill"] == r["fifo_point"] for r in usable)
    above = sum(r["observed_fill"] > r["fifo_point"] for r in usable)
    below = sum(r["observed_fill"] < r["fifo_point"] for r in usable)
    violations = sum(1 for r in usable if r["later_executed_while_ahead_resting"])
    filled = sum(r["observed_fill"] > 0 for r in usable)
    return {"orders": len(rows), "evaluable": n, "zero_length_windows_excluded": zero_length, "observed_filled": filled,
            "within_conservative_optimistic_bounds": inside, "coverage_fraction": inside / n,
            "equal_to_identity_fifo_point": exact, "fifo_point_agreement": exact / n,
            "observed_above_fifo_point": above, "observed_below_fifo_point": below,
            "orders_with_later_identity_executed_while_ahead_resting": violations,
            "interpretation": ("Coverage below 1 falsifies the conservative/optimistic assumption set; "
                               "FIFO-point disagreement is evidence against price-time priority or against "
                               "complete observation. Observed outcomes are the source orders' own, not "
                               "hypothetical fills.")}


def unsupported_design(orders: Iterable[ChildOrder], contract: CapabilityContract, reason: str) -> list[dict]:
    return [unsupported(order, contract, reason).to_dict() for order in orders]
