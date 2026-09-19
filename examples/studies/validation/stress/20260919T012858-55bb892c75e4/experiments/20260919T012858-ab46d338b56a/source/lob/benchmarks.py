"""Measured local throughput; correctness checks bracket each timed workload."""
from __future__ import annotations

import platform
import statistics
import time
import tracemalloc
from typing import Any

from .engine import Order, OrderBook, OrderType, Side


def benchmark_matching(pairs: int = 2000, repeats: int = 3) -> dict[str, Any]:
    if not 1 <= pairs <= 100_000 or not 1 <= repeats <= 10:
        raise ValueError("pairs must be 1..100000 and repeats 1..10")
    rates = []
    for _ in range(repeats):
        book = OrderBook()
        started = time.perf_counter()
        for index in range(pairs):
            now = float(index)
            book.process(Order(2 * index + 1, Side.SELL, 10, OrderType.LIMIT, "maker", now, price=10001), now)
            book.process(Order(2 * index + 2, Side.BUY, 10, OrderType.MARKET, "taker", now), now)
        elapsed = time.perf_counter() - started
        if len(book.trades) != pairs or book.orders:
            raise AssertionError("matching benchmark failed fill/book reconciliation")
        book.assert_invariants()
        rates.append(2 * pairs / elapsed)
    # Memory instrumentation changes timing; measure it in a separate untimed pass.
    tracemalloc.start()
    book = OrderBook()
    for index in range(pairs):
        book.process(Order(index + 1, Side.BUY, 10, OrderType.LIMIT, "maker", 0.0, price=10000), 0.0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    book.assert_invariants()
    return {"workload": "alternating one resting limit and one fully matching market order",
            "orders_per_repeat": 2 * pairs, "fills_per_repeat": pairs, "repeats": repeats,
            "orders_per_second_median": statistics.median(rates), "orders_per_second_samples": rates,
            "resting_orders_memory_count": pairs, "python_tracemalloc_peak_bytes": peak,
            "python": platform.python_version(), "platform": platform.platform(),
            "note": "Local Python microbenchmark, not full market throughput; no latency/agent/replay workload included."}
