"""Warm, repeated local Python workload measurements, never colocation latency."""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import statistics
import tempfile
import time
import tracemalloc
from decimal import Decimal
from pathlib import Path
from typing import Callable

import numpy as np

from .artifacts import portable_provenance, seal_artifacts
from .engine import ExchangeSimulator, Order, OrderBook, OrderType, Side, SimConfig
from .experiments.registry import sha256_file, write_json
from .replay.l2 import L2Replay, _Row

SIZES = {"small": 10, "medium": 100, "deep": 1000}


def _book(levels: int) -> OrderBook:
    book = OrderBook()
    for index in range(levels):
        for offset, side, price in ((1, Side.BUY, 10000 - index), (2, Side.SELL, 10002 + index)):
            book.process(Order(2 * index + offset, side, 10, OrderType.LIMIT, "fixture", 0., price=price), 0.)
    book.assert_invariants()
    return book


def latency_summary(samples: list[int]) -> dict:
    if not samples or any(s <= 0 for s in samples):
        raise ValueError("latency samples must be positive nanoseconds")
    return {"samples": len(samples), "p50_ns": float(np.percentile(samples, 50)),
            "p95_ns": float(np.percentile(samples, 95)), "p99_ns": float(np.percentile(samples, 99)),
            "operations_per_second": len(samples) * 1e9 / sum(samples)}


def _measure(factory: Callable, operations: int, repeats: int, warmup: int) -> dict:
    """Measure call latency and wall-clock throughput in separate fresh runs.

    Per-call latency instrumentation is intentionally kept out of the throughput
    pass so timer calls and sample bookkeeping do not define the reported batch
    throughput.
    """
    samples = []
    timed_call_rates = []
    wall_clock_rates = []

    for _ in range(repeats):
        # Latency pass: instrument each operation individually.
        operation, check = factory()
        for _ in range(warmup):
            operation()

        batch = []
        for _ in range(operations):
            started = time.perf_counter_ns()
            operation()
            batch.append(max(1, time.perf_counter_ns() - started))

        check()
        samples.extend(batch)
        timed_call_rates.append(len(batch) * 1e9 / sum(batch))

        # Throughput pass: use a fresh equivalent state and one outer timer.
        # This avoids per-operation timing and list-append instrumentation.
        operation, check = factory()
        for _ in range(warmup):
            operation()

        started = time.perf_counter_ns()
        for _ in range(operations):
            operation()
        elapsed = max(1, time.perf_counter_ns() - started)

        check()
        wall_clock_rates.append(operations * 1e9 / elapsed)

    summary = latency_summary(samples)

    return {
        **summary,
        # Explicit alias: this rate is inferred from summed instrumented
        # per-call latencies and is not wall-clock batch throughput.
        "latency_derived_operations_per_second": summary["operations_per_second"],
        "timed_call_operations_per_second": timed_call_rates,
        "median_timed_call_operations_per_second": statistics.median(timed_call_rates),
        # Backwards-facing batch field now has its literal wall-clock meaning.
        "batch_operations_per_second": wall_clock_rates,
        "median_batch_operations_per_second": statistics.median(wall_clock_rates),
    }

def _synthetic_factory(levels: int, snapshot: bool = False):
    book = _book(levels)
    count = 0

    def mutate():
        nonlocal count
        count += 1
        # One measured operation is a same-price quantity amendment.
        book.modify(1, 9 if count % 2 else 10, now=float(count))

    def check():
        book.assert_invariants()
        assert len(book.orders) == 2 * levels

    return (lambda: book.snapshot(5)) if snapshot else mutate, check


def _l2_factory(levels: int):
    book = L2Replay(Path("unused-input"))
    for index in range(levels):
        for side, price in (("bid", 10000 - index), ("ask", 10002 + index)):
            book._update(_Row("SYNTHETIC", "SMOKE", 0, 0, True, side, Decimal(price), Decimal(10)))
    count = 0

    def mutate():
        nonlocal count
        count += 1
        book._update(_Row("SYNTHETIC", "SMOKE", count, count, False,
                          "bid", Decimal(10000), Decimal(9 if count % 2 else 10)))

    def check():
        assert len(book._bids) == len(book._asks) == levels
        assert book._bids[Decimal(10000)] in {Decimal(9), Decimal(10)}

    return mutate, check


def _mbo_factory(levels: int):
    from .mbo import MBOBook, MBOEvent
    from .replay.schema import SnapshotOrder
    book = MBOBook()
    orders = tuple(SnapshotOrder(str(2 * i + offset), side, price, 10)
                   for i in range(levels)
                   for offset, side, price in ((1, "BUY", 10000 - i), (2, "SELL", 10002 + i)))
    book.apply(MBOEvent(timestamp_ns=0, sequence=0, event_type="SNAPSHOT", symbol="SMOKE",
                        venue="SYNTHETIC", orders=orders))
    count = 0

    def mutate():
        nonlocal count
        count += 1
        book.apply(MBOEvent(timestamp_ns=count, sequence=count, event_type="MODIFY", symbol="SMOKE",
                            venue="SYNTHETIC", order_id="1", quantity=9 if count % 2 else 10))

    def check():
        state = book.aggregate_l2()
        assert len(state["bids"]) == len(state["asks"]) == levels

    return mutate, check


def _fixture(path: Path, levels: int, updates: int) -> None:
    with path.open("x", encoding="ascii", newline="\n") as handle:
        handle.write("exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount\n")
        for i in range(levels):
            for side, price in (("bid", 10000 - i), ("ask", 10002 + i)):
                handle.write(f"SYNTHETIC,SMOKE,0,0,true,{side},{price},10\n")
        for i in range(1, updates + 1):
            handle.write(f"SYNTHETIC,SMOKE,{i},{i},false,bid,10000,{9 if i % 2 else 10}\n")


def profile_workload(seconds: float = 30.) -> dict:
    profile = cProfile.Profile()
    profile.enable()
    sim = ExchangeSimulator(SimConfig(seed=27))
    for _ in range(round(seconds / 0.1)):
        sim.step(0.1)
    for _ in range(1000):
        sim.book.snapshot(5)
    profile.disable()
    stats = pstats.Stats(profile).strip_dirs()
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
        rows.append({"file": filename, "line": line, "function": name, "calls": calls,
                     "primitive_calls": primitive, "own_seconds": own, "cumulative_seconds": cumulative})
    return {"workload": "30 simulated seconds then 1000 top-five snapshots", "seed": 27,
            "top_cumulative": sorted(rows, key=lambda r: r["cumulative_seconds"], reverse=True)[:20],
            "decision": "Validation and background flow dominate; retain correctness checks. No compiled fast path."}


def benchmark_suite(*, operations: int = 200, repeats: int = 3, warmup: int = 20,
                    sizes: tuple[str, ...] = ("small", "medium", "deep")) -> dict:
    for name, value, upper in (("operations", operations, 10000), ("repeats", repeats, 20), ("warmup", warmup, 1000)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
            raise ValueError(f"{name} must be an integer in 1..{upper}")
    if not sizes or len(set(sizes)) != len(sizes) or set(sizes) - set(SIZES):
        raise ValueError("sizes must be unique small/medium/deep names")
    result = {"kind": "local_python_research_workloads", "production_latency": False,
              "config": {"operations": operations, "repeats": repeats, "warmup": warmup,
                         "levels_per_side": {s: SIZES[s] for s in sizes}},
              "provenance": portable_provenance(), "dataset": {"kind": "synthetic", "symbol": "SMOKE",
              "period": "integer event clock; no historical date", "seed_policy": "fixed simulation seed 27"},
              "clock": {"implementation": time.get_clock_info("perf_counter").implementation,
                        "resolution_seconds": time.get_clock_info("perf_counter").resolution},
              "notes": ["In-process timings include Python/runtime and OS scheduling effects; no CPU affinity imposed.",
                        "Per-call p50/p95/p99 use instrumented calls; operations_per_second is derived from those sampled call durations.",
                        "batch_operations_per_second uses a separate fresh-state outer wall-clock pass without per-call timing instrumentation.",
                        "L2 mutation excludes parsing; replay includes CSV parsing and top-five snapshots.",
                        "MBO timing excludes queue watches; memory uses a separate untimed tracemalloc pass.",
                        "Python allocations are not process RSS. Percentiles summarize calls, not worst-case guarantees."],
              "sizes": {}}
    for size in sizes:
        levels = SIZES[size]
        rows = {}
        for name, factory in (("book_modify", lambda: _synthetic_factory(levels)),
                              ("snapshot_top5", lambda: _synthetic_factory(levels, True)),
                              ("l2_update", lambda: _l2_factory(levels)),
                              ("mbo_update", lambda: _mbo_factory(levels))):
            rows[name] = _measure(factory, operations, repeats, warmup)
        with tempfile.TemporaryDirectory(prefix="cleolob-bench-") as directory:
            path = Path(directory) / "synthetic-l2.csv"
            _fixture(path, levels, operations)
            samples = []
            for index in range(repeats + 1):
                started = time.perf_counter_ns()
                replay = L2Replay(path)
                states = sum(1 for _ in replay)
                elapsed = time.perf_counter_ns() - started
                assert states == operations + 1 and replay.stats["complete"]
                if index:
                    samples.append(elapsed)
            rows["l2_replay"] = {**latency_summary(samples), "rows_per_batch": 2 * levels + operations,
                                 "rows_per_second": (2 * levels + operations) * len(samples) * 1e9 / sum(samples),
                                 "fixture_sha256": sha256_file(path)}
        memory = {}
        for name, factory in (("synthetic_book", lambda: _synthetic_factory(levels)),
                              ("l2_book", lambda: _l2_factory(levels)), ("mbo_book", lambda: _mbo_factory(levels))):
            tracemalloc.start()
            try:
                operation, check = factory()
                operation()
                check()
                current, peak = tracemalloc.get_traced_memory()
                memory[name] = {"current_bytes": current, "peak_bytes": peak}
                del operation, check
            finally:
                tracemalloc.stop()
        rows["python_memory"] = memory
        result["sizes"][size] = rows
    from .runner import run_episode
    params = {"seed": 27, "qty": 100, "horizon": 2., "dt": .2, "latency_ms": 10.,
              "resilience": .8, "market_rate": 6., "risk_aversion": 1e-6, "warmup_seconds": .5}

    def episode_factory():
        def episode():
            row = run_episode("twap", params)
            assert row["status"] != "INVALID" and row["effective_bps"] is not None
        return episode, lambda: None

    result["simulation_episode"] = {**_measure(episode_factory, 3, repeats, 1), "params": params,
                                     "agent": "twap", "unit": "whole execution episodes"}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--operations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    # Profiling precedes the timed study and is not mixed into latency samples.
    write_json(args.out / "profile.json", profile_workload())
    write_json(args.out / "result.json", benchmark_suite(
        operations=args.operations, repeats=args.repeats, warmup=args.warmup))
    seal_artifacts(args.out)
    print(json.dumps({"status": "COMPLETE", "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
