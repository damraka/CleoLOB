"""M16: registered local benchmarks of v0.6 research workloads.

Each workload reports wall-clock seconds over repeats and a throughput in its own
unit (blocks/s, comparisons/s, bootstrap draws/s, candidate evaluations/s,
episodes/s, windows/s, runs/s). Units differ across workloads and are never
compared with each other. These are single-process Python measurements on the
local machine; they are not HFT, latency or exchange benchmarks. Inputs are
synthetic or simulated, so no restricted data is needed.
"""
from __future__ import annotations

import platform
import statistics
import time
from pathlib import Path

import numpy as np

from ..experiments.registry import runtime_metadata
from .evidence import finalize, new_run


def _time(function, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        values.append(time.perf_counter() - started)
    return values


def _summary(name: str, unit: str, work: float, times: list[float]) -> dict:
    median = statistics.median(times)
    return {"workload": name, "unit": unit, "work_per_repeat": work, "repeats": len(times),
            "median_seconds": median, "min_seconds": min(times), "max_seconds": max(times),
            "throughput": work / median if median > 0 else None}


def run(out: str | Path, *, repeats: int = 5, scale: float = 1.0) -> dict:
    from .calibration import base_config, spec_from_unit
    from .domain_gap import FEATURES, Logistic
    from .observables import build_design, measure, pool, sketch, stack, weighted
    from .realism import compare, quick_objective
    from .tape import tape_from_simulator
    from .worlds import World, episode_params
    tables = {"market_size_quantiles": [1.0] * 10 + [2.0] * 11, "limit_size_quantiles": [1.0] * 10 + [4.0] * 11}
    spec = spec_from_unit(np.full(14, 0.35), base_config(3.79), tables)
    seconds = 600.0 * max(1, round(6 * scale))
    workloads = []
    simulate_times = _time(lambda: tape_from_simulator(spec, 1, seconds=seconds), max(1, repeats // 2))
    workloads.append(_summary("simulate_10_level_tape", "simulated seconds/s", seconds, simulate_times))
    tape = tape_from_simulator(spec, 1, seconds=seconds)
    blocks = tape.blocks(600.0)
    raws = [measure(b) for b in blocks]
    workloads.append(_summary("measure_600s_blocks", "blocks/s", len(blocks),
                              _time(lambda: [measure(b) for b in blocks], repeats)))
    design = build_design(raws, tick_bps=3.79)
    workloads.append(_summary("sketch_blocks", "blocks/s", len(raws),
                              _time(lambda: [sketch(r, design) for r in raws], repeats)))
    sketches = [sketch(r, design) for r in raws]
    h = pool(sketches)
    workloads.append(_summary("realism_compare", "comparisons/s", 20,
                              _time(lambda: [compare(h, h, design) for _ in range(20)], repeats)))
    matrix = stack(sketches)
    rng = np.random.default_rng(0)

    def bootstrap() -> None:
        for _ in range(50):
            weights = np.bincount(rng.integers(0, len(sketches), len(sketches)), minlength=len(sketches)).astype(float)
            quick_objective(h, weighted(matrix, weights), design)
    workloads.append(_summary("block_bootstrap_draws", "bootstrap draws/s", 50, _time(bootstrap, repeats)))
    from . import calibration as cal
    cal._init({"design": design, "scales": None, "targets": {"t": h}})
    task = ("bench", {"config": spec.config, "extensions": spec.extensions}, [1], 600.0, "t")
    workloads.append(_summary("calibration_candidate_1x600s", "candidate evaluations/s", 1,
                              _time(lambda: cal.evaluate(task), max(1, repeats // 2))))
    world = World("bench", "selected", spec.config, spec.extensions)
    mandate = {"side": "buy", "quantity": 14, "horizon_s": 120.0, "decision_dt_s": 6.0, "warmup_s": 60.0,
               "settlement_timeout_s": 5.0, "fees": {"maker_bps": 0.0, "taker_bps": 1.0}, "terminal_penalty_bps": 25.0,
               "completion_urgency_fraction": 0.8, "pov_participation": 0.3}
    from ..runner import run_episode
    workloads.append(_summary("execution_episode_twap", "episodes/s", 3,
                              _time(lambda: [run_episode("twap", episode_params(world, mandate, 670000 + i))
                                             for i in range(3)], max(1, repeats // 2))))
    x = rng.normal(size=(2000, len(FEATURES)))
    y = (rng.random(2000) < 0.5).astype(float)
    workloads.append(_summary("logistic_discriminator_fit", "windows/s", 2000,
                              _time(lambda: Logistic().fit(x, y), repeats)))
    out = new_run(out)
    result = {"workloads": workloads, "machine": {"platform": platform.platform(), "processor": platform.processor()},
              "runtime": {k: v for k, v in runtime_metadata().items() if k != "executable"},
              "scale": scale, "repeats": repeats,
              "interpretation": ("Local single-process research throughput; units differ by workload and are not "
                                 "comparable; not an HFT, latency or exchange benchmark.")}
    finalize(out, analysis="m16-benchmarks", dataset_ids=[], config={"repeats": repeats, "scale": scale},
             result=result, seeds={"synthetic": [0, 1]})
    return result
