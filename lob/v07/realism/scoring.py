"""Score any v0.7 world (v0.6 ``SimulatorSpec`` or a v0.7 generator spec) with the frozen v0.6 measurement.

Simulated tapes come from ``lob.v06.tape.tape_from_simulator`` (unchanged), are
cut into 600 s blocks and reduced to the development-frozen sketches; the
objective is the v0.6 sealed objective. Failures (exceptions, the 500 events/s
implausibility guard) are retained as ``objective=None`` with the reason.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import os
import time

import numpy as np

from ...sim_v2 import SimulatorSpec
from ..v06_compat import measure, pool, sketch, tape_from_simulator
from ...v06.realism import compare, family_vector
from ...v06.calibration import EVENTS_PER_SECOND_CAP

_CONTEXT: dict = {}


def _init(context: dict) -> None:
    _CONTEXT.clear()
    _CONTEXT.update(context)


def capped(spec, seconds: float):
    if isinstance(spec, SimulatorSpec):
        return SimulatorSpec({**spec.config, "max_events": int(EVENTS_PER_SECOND_CAP * (seconds + 60))},
                             spec.extensions)
    return spec.capped(seconds, EVENTS_PER_SECOND_CAP)


def simulate_sketches(spec, seeds: list[int], seconds: float, design: dict) -> list[dict]:
    out = []
    for seed in seeds:
        tape = tape_from_simulator(capped(spec, seconds), seed, seconds=seconds)
        out.append(pool([sketch(measure(block), design) for block in tape.blocks(600.0)]))
    return out


def _finite(value) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


def evaluate(task: tuple) -> dict:
    """(key, spec, seeds, seconds, target name) -> objective, family errors, per-seed sketches/objectives."""
    key, spec, seeds, seconds, target_name = task
    design, scales = _CONTEXT["design"], _CONTEXT["scales"]
    target = _CONTEXT["targets"][target_name] if target_name else None
    started = time.perf_counter()
    try:
        per_seed = simulate_sketches(spec, seeds, seconds, design)
        out = {"key": key, "sketches": per_seed if _CONTEXT.get("keep_sketches", True) else [], "error": None,
               "elapsed_s": time.perf_counter() - started}
        if target is not None:
            result = compare(target, pool(per_seed), design, scales)
            out.update(objective=_finite(result["objective"]), families=[_finite(v) for v in family_vector(result)],
                       seed_objectives=[_finite(compare(target, s, design, scales)["objective"]) for s in per_seed])
        return out
    except (ValueError, RuntimeError, AssertionError, OverflowError, ZeroDivisionError, FloatingPointError) as exc:
        return {"key": key, "objective": None, "families": None, "seed_objectives": [], "sketches": [],
                "error": f"{type(exc).__name__}: {exc}", "elapsed_s": time.perf_counter() - started}


def run_tasks(tasks: list[tuple], context: dict, workers: int | None = None) -> list[dict]:
    workers = workers or int(os.environ.get("CLEOLOB_WORKERS", max(1, min(14, (os.cpu_count() or 2) - 2))))
    if workers <= 1:
        _init(context)
        return [evaluate(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(context,)) as executor:
        return list(executor.map(evaluate, tasks, chunksize=1))


def objective_of(target: dict, sketches: list[dict], design: dict, scales: dict) -> float | None:
    return _finite(compare(target, pool(sketches), design, scales)["objective"]) if sketches else None


def seed_se(values: list[float | None]) -> float | None:
    v = np.asarray([x for x in values if x is not None], float)
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
