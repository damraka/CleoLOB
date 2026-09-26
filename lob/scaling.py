"""Registered local scaling measurements; throughput is not exchange latency."""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
import tracemalloc
from pathlib import Path

from .config import ResearchConfig, canonical_json
from .evidence import document_hash, finalize_evidence, register_evidence
from .experiments.registry import write_json
from .generalization import evaluate_observable_model, fit_observable_model, synthetic_fixture
from .mbo import MBOBook, MBOEvent, MBOReplay
from .performance import _book, _fixture, latency_summary, profile_workload
from .replay.l2 import L2Replay
from .replay.schema import SnapshotOrder


def _measure_batch(operation, *, units: int, repeats: int, warmup: int) -> dict:
    for _ in range(warmup):
        operation()
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        operation()
        elapsed.append(max(1, time.perf_counter_ns() - started))
    # Separate memory pass: tracing never contributes to timed throughput.
    tracemalloc.start()
    try:
        operation()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    rates = [units * 1e9 / value for value in elapsed]
    return {"batch_duration": latency_summary(elapsed), "batch_nanoseconds": elapsed,
            "mean_wall_seconds": statistics.mean(elapsed) / 1e9,
            "median_wall_seconds": statistics.median(elapsed) / 1e9,
            "stdev_wall_seconds": statistics.stdev(elapsed) / 1e9 if repeats > 1 else None,
            "units_per_batch": units, "batch_units_per_second": rates,
            "median_units_per_second": statistics.median(rates),
            "mean_nanoseconds_per_unit": statistics.mean(elapsed) / units,
            "python_peak_bytes_separate_pass": peak,
            "process_rss": None, "process_rss_status": "NOT_AVAILABLE"}


def _mutation_batch(family: str, depth: int, events: int, snapshot_every: int):
    def run():
        if family == "engine":
            book = _book(depth)
            for i in range(1, events + 1):
                book.modify(1, 9 if i % 2 else 10, now=float(i))
                if i % snapshot_every == 0:
                    book.snapshot(5)
            book.assert_invariants()
            assert len(book.orders) == 2 * depth
        else:
            book = MBOBook(max_events=events + 2, max_orders=2 * depth + 1)
            orders = tuple(SnapshotOrder(str(2 * i + j), side, price, 10)
                           for i in range(depth)
                           for j, side, price in ((1, "BUY", 10000 - i), (2, "SELL", 10002 + i)))
            book.apply(MBOEvent(0, 0, "SNAPSHOT", "SCALING", "SYNTHETIC", orders=orders))
            for i in range(1, events + 1):
                book.apply(MBOEvent(i, i, "MODIFY", "SCALING", "SYNTHETIC", "1", quantity=9 if i % 2 else 10))
                if i % snapshot_every == 0:
                    book.aggregate_l2()
            assert len(book.aggregate_l2()["bids"]) == depth
    return run


def _mbo_fixture(path: Path, depth: int, events: int):
    orders = tuple(SnapshotOrder(str(2 * i + j), side, price, 10)
                   for i in range(depth)
                   for j, side, price in ((1, "BUY", 10000 - i), (2, "SELL", 10002 + i)))
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(MBOEvent(0, 0, "SNAPSHOT", "SCALING", "SYNTHETIC", orders=orders).to_dict()) + "\n")
        for i in range(1, events + 1):
            handle.write(canonical_json(MBOEvent(i, i, "MODIFY", "SCALING", "SYNTHETIC", "1",
                                                  quantity=9 if i % 2 else 10).to_dict()) + "\n")


def validate_config(config: dict) -> dict:
    result = json.loads(canonical_json(config))
    limits = {"depths": (1, 2000), "event_counts": (1, 10000), "snapshot_every": (1, 10000),
              "calibration_rows": (65, 10000), "experiment_sizes": (1, 32)}
    if set(result) != set(limits) | {"repeats", "warmup"}:
        raise ValueError("scaling config requires exactly the documented dimensions")
    for key, (low, high) in limits.items():
        values = result[key]
        if (not isinstance(values, list) or not 1 <= len(values) <= 4
                or any(isinstance(v, bool) or not isinstance(v, int) or not low <= v <= high for v in values)
                or sorted(set(values)) != values):
            raise ValueError(f"{key} must be increasing distinct integers in {low}..{high}")
    for key, low, high in (("repeats", 1, 20), ("warmup", 1, 5)):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"invalid {key}")
    return result


DEFAULT_CONFIG = {"depths": [10, 100, 1000], "event_counts": [100, 1000], "snapshot_every": [1, 10],
                  "calibration_rows": [128, 512, 2048], "experiment_sizes": [2, 8], "repeats": 3, "warmup": 1}


def scaling_suite(config: dict) -> dict:
    config = validate_config(config)
    measurements = []

    def record(family, dimensions, operation, units, unit):
        measurements.append({"family": family, "dimensions": dimensions, "unit": unit,
                             **_measure_batch(operation, units=units, repeats=config["repeats"],
                                              warmup=config["warmup"])})

    with tempfile.TemporaryDirectory(prefix="cleo-scaling-") as temp:
        scratch = Path(temp)
        for depth in config["depths"]:
            for events in config["event_counts"]:
                dims = {"levels_per_side": depth, "price_levels": depth * 2, "events": events}
                for stride in config["snapshot_every"]:
                    for family in ("engine", "mbo"):
                        record(family + "_mutation_and_snapshot", {**dims, "active_orders": depth * 2,
                               "snapshot_every": stride, "snapshot_depth": 5 if family == "engine" else "full"},
                               _mutation_batch(family, depth, events, stride), events, "mutation events")
                l2 = scratch / f"l2-{depth}-{events}.csv"
                _fixture(l2, depth, events)

                def replay_l2():
                    replay = L2Replay(l2)
                    assert sum(1 for _ in replay) == events + 1 and replay.stats["complete"]

                record("l2_csv_replay", {**dims, "snapshot_depth": 5, "file_bytes": l2.stat().st_size}, replay_l2,
                       2 * depth + events, "CSV rows including initial snapshot")
                mbo = scratch / f"mbo-{depth}-{events}.jsonl"
                _mbo_fixture(mbo, depth, events)

                def replay_mbo():
                    replay = MBOReplay(mbo, max_events=events + 2)
                    assert replay.run() == events + 1

                record("mbo_jsonl_replay", {**dims, "active_orders": depth * 2,
                       "snapshot_depth": "none", "file_bytes": mbo.stat().st_size}, replay_mbo,
                       events + 1, "canonical events including initial census")
        for count in config["calibration_rows"]:
            frame = synthetic_fixture({"rows": count, "seed": 27401, "start_us": 0}, 1_000_000)
            target = synthetic_fixture({"rows": count, "seed": 27404,
                                        "start_us": (count + 1) * 1_000_000}, 1_000_000)

            def calibration():
                model = fit_observable_model(frame, family="iid_joint", interval_us=1_000_000,
                                             seed=27402, max_pool_rows=128)
                result = evaluate_observable_model(model, target, seeds=[27403], simulation_rows=count)
                assert result["rows"] == count

            record("calibration_fit_and_diagnostic", {"rows": count}, calibration, count, "feature rows")
        for count in config["event_counts"]:
            payload = [{"event": i, "price": 10000, "quantity": 10, "source": "SYNTHETIC"} for i in range(count)]

            def serialization():
                encoded = canonical_json(payload)
                assert len(json.loads(encoded)) == count

            record("json_serialization_roundtrip", {"records": count}, serialization, count, "records")
        from .experiments.runner import run_experiment
        from .runner import run_episode
        for count in config["experiment_sizes"]:
            config_episode = ResearchConfig.model_validate({"name": "scaling_synthetic",
                "execution": {"quantity": 100, "horizon": 1., "decision_dt": .1, "warmup_seconds": .2},
                "evaluation": {"agents": ["heuristic"], "reference": "heuristic",
                               "seeds": list(range(27500, 27500 + count)), "bootstrap_samples": 100}})

            def policy_evaluation():
                rows = [run_episode("heuristic", config_episode.runner_params(seed))
                        for seed in config_episode.evaluation.seeds]
                assert all(row["status"] in {"VALID", "WARNING"} for row in rows)

            def orchestration():
                run = run_experiment(config_episode, scratch / "experiments")
                result = json.loads((run / "result.json").read_text(encoding="utf-8"))
                assert result["coverage"]["episodes_valid_or_warning"] == count

            record("heuristic_policy_evaluation", {"episodes": count}, policy_evaluation, count, "episodes")
            record("experiment_orchestration", {"episodes": count}, orchestration, count, "episodes")
    return {"measurements": measurements, "config": config, "production_latency": False,
            "limitations": ["Synthetic fixed workloads, local Python, no CPU affinity/frequency control.",
                "Batch duration includes state setup and correctness checks; per-unit cost is amortized, not event latency.",
                "Replay includes parsing; mutation/snapshot batches exclude parsing. Initial census is explicitly counted.",
                "p50/p95/p99 summarize whole-batch samples; three repetitions do not estimate reliable extreme tails.",
                "Memory is traced Python peak allocation from a separate pass; process RSS is unavailable.",
                "Policy family measures complete heuristic episodes, not neural-network inference latency.",
                "Calibration timing uses fixed separate chronological synthetic fixtures; this is not historical evidence."]}


def run_scaling(out: str | Path, config: dict | None = None) -> dict:
    config = validate_config(config or DEFAULT_CONFIG)
    register_evidence(out, kind="local_python_scaling", config=config,
        inputs=[{"identity": "cleolob-scaling-fixtures-v1", "adapter_version": "1", "role": "benchmark",
                 "kind": "synthetic", "sha256": document_hash(config)}],
        seeds={"fixture": [27401, 27404], "calibration_fit": [27402], "calibration_sample": [27403],
               "market": list(range(27500, 27500 + max(config["experiment_sizes"])))},
        hypotheses=["Describe measured runtime/allocation scaling without a performance acceptance threshold.",
                    "No production latency or historical performance inference is permitted."])
    write_json(Path(out) / "profile.json", profile_workload())
    metrics = scaling_suite(config)
    return finalize_evidence(out, status="ESTABLISHED", metrics=metrics)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    result = run_scaling(args.out, config)
    print(json.dumps({"status": result["status"], "workloads": len(result["metrics"]["measurements"])}))


if __name__ == "__main__":
    main()
