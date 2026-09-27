"""M11: registered local benchmarks for the v0.5 research paths (never exchange latency).

Every workload is built from a deterministic synthetic generator so the benchmark
is redistributable and repeatable. For each workload and size the study records
workload units, wall time (median of timed repetitions after one warmup),
throughput, per-call p50/p95/p99 where the operation is separable, and peak
traced Python allocation in a separate pass. Process RSS is not reliable across
platforms here and is reported as NOT_AVAILABLE.

Workloads: Bitstamp-format MBO parsing, lifecycle replay, MBO-to-L2 aggregation,
fill-bound updates, streaming aggregate-L2 historical execution, impact and
resilience extraction, calibration-v2 simulation summaries, extended simulator
stepping, historical replay episodes, policy evaluation episodes, and evidence
serialization.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time
import tracemalloc
from typing import Callable

import numpy as np

from .artifacts import portable_provenance

SIZES = {"small": 1, "medium": 4, "large": 16}


def _percentiles(samples: list[int]) -> dict:
    return {"calls": len(samples), "p50_us": float(np.percentile(samples, 50)) / 1e3,
            "p95_us": float(np.percentile(samples, 95)) / 1e3, "p99_us": float(np.percentile(samples, 99)) / 1e3}


def measure(name: str, size: str, units: int, unit: str, setup: Callable, run: Callable, *,
            repeats: int = 3, per_call: Callable | None = None) -> dict:
    """``setup()`` builds fresh state; ``run(state)`` performs the whole workload once."""
    run(setup())  # warmup
    walls = []
    for _ in range(repeats):
        state = setup()
        started = time.perf_counter()
        run(state)
        walls.append(time.perf_counter() - started)
    state = setup()
    tracemalloc.start()
    run(state)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    median = statistics.median(walls)
    row = {"workload": name, "size": size, "units": units, "unit": unit, "repeats": repeats,
           "wall_seconds": walls, "median_wall_seconds": median, "throughput_per_second": units / median,
           "peak_traced_python_bytes": peak, "process_rss": "NOT_AVAILABLE"}
    if per_call is not None:
        row["per_call"] = _percentiles(per_call(setup()))
    return row


# ----------------------------------------------------------------------------- generators

def _bitstamp_capture(path: Path, orders: int, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    census = {"microtimestamp": "1000000000", "bids": [[f"{100 - i * 0.01:.2f}", "1.00000000", f"b{i}"] for i in range(50)],
              "asks": [[f"{100.01 + i * 0.01:.2f}", "1.00000000", f"a{i}"] for i in range(50)]}
    rows = [{"kind": "rest_snapshot", "recv_ns": 1, "body": json.dumps(census)}]
    last = None
    for i in range(orders):
        ts = 1_000_001 + i
        side = int(rng.integers(0, 2))
        price = 99.5 - rng.integers(0, 40) * 0.01 if side == 0 else 100.5 + rng.integers(0, 40) * 0.01
        for event in ("order_created", "order_deleted"):
            data = {"id_str": f"o{i}", "order_type": side, "microtimestamp": str(ts * 1000), "price_str": f"{price:.2f}",
                    "amount_str": "0.50000000", "amount_traded": "0"}
            message = {"channel": "live_orders_btcusd", "event": event, "data": data, "event_id": f"e{i}{event[6]}",
                       "pre_event_id": last}
            last = message["event_id"]
            rows.append({"kind": "ws", "recv_ns": ts, "raw": json.dumps(message)})
        if i % 20 == 0:
            book = {"microtimestamp": str(ts * 1000 + 500), "bids": [r[:2] for r in census["bids"][:10]],
                    "asks": [r[:2] for r in census["asks"][:10]]}
            rows.append({"kind": "ws", "recv_ns": ts, "raw": json.dumps({"channel": "order_book_btcusd", "event": "data",
                                                                        "data": book})})
    rows.append({"kind": "capture_end", "recv_ns": orders + 2, "status": "COMPLETE"})
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def _tardis(updates: Path, trades: Path, seconds: int, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    rows = ["exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount"]
    for i in range(20):
        rows.append(f"x,S,0,0,true,bid,{100 - i * 0.05:.2f},{rng.integers(100, 1000)}")
        rows.append(f"x,S,0,0,true,ask,{100.05 + i * 0.05:.2f},{rng.integers(100, 1000)}")
    trade_rows = ["exchange,symbol,timestamp,local_timestamp,id,side,price,amount"]
    for k in range(seconds * 20):
        t = 1_000_000 + k * 50_000
        side = "bid" if rng.random() < 0.5 else "ask"
        level = int(rng.integers(0, 5))
        price = 100 - level * 0.05 if side == "bid" else 100.05 + level * 0.05
        rows.append(f"x,S,{t},{t},false,{side},{price:.2f},{rng.integers(50, 1000)}")
        if k % 10 == 0:
            aggressor = "sell" if side == "bid" else "buy"
            trade_rows.append(f"x,S,{t},{t},t{k},{aggressor},{100 if aggressor == 'sell' else 100.05:.2f},{rng.integers(1, 200)}")
    for path, lines in ((updates, rows), (trades, trade_rows)):
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")


def _tape(seconds: int, seed: int = 5):
    from .calibration_v2 import tape_from_simulator
    from .sim_v2 import SimulatorSpec
    from .engine import SimConfig
    from dataclasses import asdict
    return tape_from_simulator(SimulatorSpec(asdict(SimConfig(record_events=False)), {}), seed, seconds=seconds)


# ----------------------------------------------------------------------------- suite

def suite(scale: float = 1.0, repeats: int = 3) -> list[dict]:
    from .fill_bounds import ChildOrder, L2FillTracker
    from .historical_execution import L2_WITH_TRADES, run_l2
    from .impact import event_table
    from .mbo_sources import BitstampCaptureAdapter
    from .order_lifecycle import LifecycleSemantics, OrderLifecycleBook
    from .resilience import depletion_events, recovery
    from .runner import run_episode
    from .scenarios import scenario_params
    from .sim_v2 import ExtendedSimulator, FlowExtensions
    from .engine import SimConfig
    from .config import canonical_json
    results = []
    workdir = Path(tempfile.mkdtemp(prefix="cleolob-bench-"))
    for size, factor in SIZES.items():
        n = max(1, int(factor * scale))
        # 1-3 MBO parsing, lifecycle replay, aggregation
        capture = workdir / f"capture-{size}.jsonl.gz"
        _bitstamp_capture(capture, 2000 * n)
        results.append(measure("mbo_parsing_normalization", size, 4000 * n, "order messages",
                               lambda: BitstampCaptureAdapter(capture), lambda a: a.build(), repeats=repeats))
        events, _ = BitstampCaptureAdapter(capture).build()

        def replay_calls(_state, events=events):
            book = OrderLifecycleBook(LifecycleSemantics(crossing_add="transient_aggressor"))
            samples = []
            for event in events:
                started = time.perf_counter_ns()
                book.apply(event)
                samples.append(max(1, time.perf_counter_ns() - started))
            return samples
        results.append(measure("lifecycle_replay", size, len(events), "normalized events",
                               lambda: None, lambda _s, events=events: replay_calls(None, events), repeats=repeats,
                               per_call=replay_calls))

        def aggregated_book(events=events):
            book = OrderLifecycleBook(LifecycleSemantics(crossing_add="transient_aggressor"))
            for event in events[: len(events) // 2]:
                book.apply(event)
            return book

        def aggregate_calls(book):
            samples = []
            for _ in range(500):
                started = time.perf_counter_ns()
                book.aggregate(10)
                samples.append(max(1, time.perf_counter_ns() - started))
            return samples
        results.append(measure("mbo_to_l2_aggregation", size, 500, "top-10 aggregations", aggregated_book,
                               lambda book: [book.aggregate(10) for _ in range(500)], repeats=repeats,
                               per_call=aggregate_calls))
        # 4 fill bounds
        def trackers(n=n):
            return [L2FillTracker(ChildOrder(str(i), "BUY", 100.0, 10.0, 0, 1), 50.0, L2_WITH_TRADES)
                    for i in range(25 * n)]

        def bound_calls(ts):
            samples = []
            for k in range(2000):
                started = time.perf_counter_ns()
                for t in ts:
                    t.on_print(100.0, 3.0, "SELL") if k % 2 else t.on_level(40.0 + k % 7)
                samples.append(max(1, time.perf_counter_ns() - started))
            return samples
        results.append(measure("fill_bound_updates", size, 2000 * 25 * n, "tracker updates", trackers,
                               lambda ts: bound_calls(ts), repeats=repeats, per_call=bound_calls))
        # 5 streaming historical execution
        updates, trades = workdir / f"u-{size}.csv.gz", workdir / f"t-{size}.csv.gz"
        _tardis(updates, trades, 300 * n)
        design = {"start_offset_s": 1, "interval_s": 5, "end_margin_s": 5, "sides": ["BUY", "SELL"],
                  "sizes": ["100", "1000"], "lifetimes_s": [1, 5], "price_rule": "join_best", "max_gap_s": 10,
                  "max_staleness_s": 5}
        results.append(measure("historical_execution_l2", size, 300 * n * 20, "L2 rows", lambda: None,
                               lambda _s: run_l2(updates, trades, design, dataset_id="bench"), repeats=repeats))
        # 6 impact / resilience
        tape = _tape(300 * n)
        results.append(measure("impact_resilience", size, len(tape.trade_t), "aggressive events", lambda: tape,
                               lambda tp: (event_table(tp), recovery(tp, depletion_events(tp))), repeats=repeats))
        # 7 calibration-v2 simulation summary
        from .calibration_v2 import simulated_summary
        from .sim_v2 import SimulatorSpec
        from dataclasses import asdict
        spec = SimulatorSpec(asdict(SimConfig(record_events=False)), {})
        results.append(measure("calibration_v2_summary", size, 120 * n, "simulated seconds", lambda: spec,
                               lambda s, n=n: simulated_summary(s, [1], 120 * n), repeats=repeats))
        # 8 simulator stepping with all extensions
        ext = FlowExtensions(limit_size_quantiles=(1, 5, 30), market_size_quantiles=(1, 3, 40), inside_spread_prob=0.2,
                             cancel_depth_exponent=0.5, imbalance_beta=0.3, regime_multiplier=2, regime_switch_rate=0.02,
                             hawkes_alpha=0.5, hawkes_decay=2.0)
        results.append(measure("extended_simulation", size, 60 * n, "simulated seconds",
                               lambda: ExtendedSimulator(SimConfig(seed=3, record_events=False), ext),
                               lambda sim, n=n: [sim.step(1.0) for _ in range(60 * n)], repeats=repeats))
        # 9 policy evaluation episodes (identical runner as registered studies)
        def episodes(n=n):
            rows = []
            for seed in range(4 * n):
                params = scenario_params("calm", seed)
                params.update(completion={"enabled": True}, horizon=2.0, qty=300)
                rows.append(run_episode("twap", params))
            return rows
        results.append(measure("policy_evaluation_episodes", size, 4 * n, "episodes", lambda: None,
                               lambda _s: episodes(), repeats=repeats))
        # 10 evidence serialization
        records = [{"i": i, "cost": float(i) * 0.1, "flag": bool(i % 2), "nested": {"a": [i, i + 1]}} for i in range(5000 * n)]
        results.append(measure("evidence_serialization", size, len(records), "records", lambda: records,
                               lambda r: hashlib.sha256(canonical_json(r).encode()).hexdigest(), repeats=repeats))
    return results


def run(out: str | Path, *, scale: float = 1.0, repeats: int = 3) -> dict:
    from .v05_evidence import finalize, new_run
    out = new_run(out)
    provenance = portable_provenance()
    started = time.time()
    rows = suite(scale, repeats)
    result = {"workloads": rows, "wall_seconds_total": time.time() - started,
              "environment": {"cpu": provenance["runtime"].get("cpu"),
                              "logical_cpu_count": provenance["runtime"].get("logical_cpu_count"),
                              "platform": provenance["runtime"].get("platform"),
                              "python": provenance["runtime"].get("python")},
              "interpretation": ("Local single-process Python research workloads on synthetic generators; not "
                                 "production, colocation or HFT latency. Throughputs of different units are not comparable.")}
    finalize(out, analysis="m11-benchmarks", dataset_ids=[], config={"scale": scale, "repeats": repeats,
                                                                    "sizes": SIZES}, result=result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    result = run(args.out, scale=args.scale, repeats=args.repeats)
    for row in result["workloads"]:
        print(f"{row['workload']:<30} {row['size']:<7} {row['throughput_per_second']:>14.1f} {row['unit']}/s")


if __name__ == "__main__":
    main()
