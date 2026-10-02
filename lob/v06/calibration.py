"""Calibration v3 (M6/M9): multi-start, multi-objective search with sealed chronology.

Model family: ``lob.sim_v2.ExtendedSimulator`` (unchanged) with every v0.5
extension available and the empirical size tables fixed from development. The
search runs in a 14-dimensional transformed unit box (``PARAMETERS``); the two
v0.5 size means are inert whenever size tables are active and are therefore not
searched (searching them would manufacture non-identifiability).

develop: 8 seeded starts x (96 global draws + 3 refinement rounds of 64) on the
development day; candidates of a round are drawn before any of them is
evaluated, so results do not depend on process scheduling. The 64 best are
re-scored on fresh development seeds; the 32 best re-scored advance.
select: the 32 candidates and the v0.5 control are scored on the selection day;
the near-optimal region, the selected model and the materially distinct set are
derived from selection data only and sealed in the ledger before any holdout.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
import os
import time
from typing import Callable

import numpy as np

from ..engine import SimConfig
from ..sim_v2 import SimulatorSpec
from .observables import FAMILIES, measure, pool, sketch
from .realism import compare, family_vector
from .tape import tape_from_simulator

MAX_EVENTS = 3_000_000
# Deterministic implausibility guard: >500 engine events per simulated second (5.6x the v0.5 selected
# model, ~70x the development net book-change rate) ends the path; the candidate is a retained failure.
EVENTS_PER_SECOND_CAP = 500
SIZE_GRID = tuple(np.linspace(0.0, 1.0, 21))


@dataclass(frozen=True)
class Parameter:
    name: str
    low: float
    high: float
    scale: str            # log | linear
    integer: bool = False
    target: str = "config"   # config | extension | hawkes

    def value(self, u: float) -> float:
        u = float(min(1.0, max(0.0, u)))
        if self.scale == "log":
            v = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            v = self.low + u * (self.high - self.low)
        return float(int(round(v))) if self.integer else float(v)

    def unit(self, value: float) -> float:
        if self.scale == "log":
            return (math.log(value) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (value - self.low) / (self.high - self.low)


PARAMETERS: tuple[Parameter, ...] = (
    Parameter("target_level_vol", 5, 600, "log", integer=True),
    Parameter("limit_rate", 0.1, 60.0, "log"),
    Parameter("market_rate", 0.005, 5.0, "log"),
    Parameter("cancel_rate", 0.002, 1.0, "log"),
    Parameter("offset_p", 0.05, 0.95, "linear"),
    Parameter("resilience", 0.001, 2.0, "log"),
    Parameter("inside_spread_prob", 0.0, 0.9, "linear", target="extension"),
    Parameter("cancel_depth_exponent", -3.0, 3.0, "linear", target="extension"),
    Parameter("imbalance_beta", -0.9, 0.9, "linear", target="extension"),
    Parameter("regime_multiplier", 1.0, 10.0, "log", target="extension"),
    Parameter("regime_switch_rate", 0.002, 0.2, "log", target="extension"),
    Parameter("regime_high_share", 0.05, 0.6, "linear", target="extension"),
    Parameter("hawkes_branching", 0.0, 0.95, "linear", target="hawkes"),
    Parameter("hawkes_decay", 0.1, 10.0, "log", target="hawkes"),
)
NAMES = tuple(p.name for p in PARAMETERS)


def base_config(tick_bps: float) -> dict:
    """The v0.5 base geometry (relative tick of the development instrument), larger event cap guard."""
    from dataclasses import asdict
    mid_ticks = max(21, round(10000 / tick_bps))
    return asdict(SimConfig(initial_mid_ticks=mid_ticks, tick_size=100 / mid_ticks, target_level_vol=200,
                            limit_rate=4, market_rate=0.5, cancel_rate=0.1, limit_qty_mean=35, market_qty_mean=40,
                            resilience=0.15, record_events=False, max_events=MAX_EVENTS))


def size_tables(dev_raws: list[dict]) -> dict:
    """v0.5 rule on v0.6 development samples: 21 quantiles of event and addition sizes, floored at one lot."""
    from .observables import concatenate
    tables = {}
    for key, source in (("market_size_quantiles", "trade_size"), ("limit_size_quantiles", "add_size")):
        values = concatenate(dev_raws, source)
        q = np.maximum.accumulate(np.quantile(values, SIZE_GRID)) if len(values) else np.ones(len(SIZE_GRID))
        tables[key] = [float(max(1.0, v)) for v in np.maximum.accumulate(np.maximum(q, 1.0))]
    return tables


def spec_from_unit(u: np.ndarray, base: dict, tables: dict) -> SimulatorSpec:
    config, extensions = dict(base), dict(tables)
    hawkes = {}
    for parameter, value in zip(PARAMETERS, u):
        v = parameter.value(value)
        if parameter.target == "config":
            config[parameter.name] = int(v) if parameter.integer else v
        elif parameter.target == "extension":
            extensions[parameter.name] = v
        else:
            hawkes[parameter.name] = v
    extensions["hawkes_decay"] = hawkes["hawkes_decay"]
    extensions["hawkes_alpha"] = hawkes["hawkes_branching"] * hawkes["hawkes_decay"]
    return SimulatorSpec(config, extensions)


def unit_from_spec(spec: SimulatorSpec) -> np.ndarray:
    values = []
    for parameter in PARAMETERS:
        if parameter.name == "hawkes_branching":
            decay = spec.extensions.get("hawkes_decay", 1.0)
            value = spec.extensions.get("hawkes_alpha", 0.0) / decay
        elif parameter.target == "config":
            value = spec.config[parameter.name]
        else:
            value = spec.extensions.get(parameter.name, 0.0)
        values.append(parameter.unit(value) if value > 0 or parameter.scale == "linear" else float("nan"))
    return np.asarray(values)


# ----------------------------------------------------------------------------- evaluation


_CONTEXT: dict = {}


def _init(context: dict) -> None:
    _CONTEXT.clear()
    _CONTEXT.update(context)


def simulate_sketches(spec: SimulatorSpec, seeds: list[int], seconds: float, design: dict) -> list[dict]:
    """One pooled sketch per seed (600 s blocks summed within the seed)."""
    if seconds < 600.0:
        raise ValueError("simulated seconds must cover at least one 600 s measurement block")
    result = []
    capped = SimulatorSpec({**spec.config, "max_events": int(EVENTS_PER_SECOND_CAP * (seconds + 60))}, spec.extensions)
    for seed in seeds:
        tape = tape_from_simulator(capped, seed, seconds=seconds)
        result.append(pool([sketch(measure(block), design) for block in tape.blocks(600.0)]))
    return result


def evaluate(task: tuple) -> dict:
    """Objective, family errors and per-seed objectives of one candidate (failures retained as +inf)."""
    key, spec_dict, seeds, seconds, target_name = task
    design, scales, target = _CONTEXT["design"], _CONTEXT["scales"], _CONTEXT["targets"][target_name]
    spec = SimulatorSpec(spec_dict["config"], spec_dict["extensions"])
    started = time.perf_counter()
    try:
        per_seed = simulate_sketches(spec, seeds, seconds, design)
        result = compare(target, pool(per_seed), design, scales)
        seed_objectives = [_finite(compare(target, s, design, scales)["objective"]) for s in per_seed]
        return {"key": key, "objective": _finite(result["objective"]),
                "families": [_finite(v) for v in family_vector(result)], "seed_objectives": seed_objectives,
                "error": None, "elapsed_s": time.perf_counter() - started}
    except (ValueError, RuntimeError, AssertionError, OverflowError, ZeroDivisionError) as exc:
        return {"key": key, "objective": None, "families": [None] * len(FAMILIES), "seed_objectives": [],
                "error": f"{type(exc).__name__}: {exc}", "elapsed_s": time.perf_counter() - started}


def _finite(value) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


def run_tasks(tasks: list[tuple], context: dict, workers: int | None = None) -> list[dict]:
    workers = workers or int(os.environ.get("CLEOLOB_WORKERS", max(1, min(14, (os.cpu_count() or 2) - 2))))
    if workers <= 1:
        _init(context)
        return [evaluate(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(context,)) as pool_:
        return list(pool_.map(evaluate, tasks, chunksize=1))


def _rank(record: dict) -> tuple:
    return (record["objective"] if record["objective"] is not None else math.inf, record["key"])


def search(*, starts: int, global_draws: int, rounds: tuple[float, ...], per_round: int, seed: int, base: dict,
           tables: dict, seeds: list[int], seconds: float, context: dict, target: str,
           runner: Callable = run_tasks, label: str = "c") -> list[dict]:
    """Multi-start search; candidate parameters of each phase are drawn before evaluation."""
    dim = len(PARAMETERS)
    rngs = [np.random.default_rng([seed, s]) for s in range(starts)]
    records: list[dict] = []

    def submit(units: list[tuple[int, int, np.ndarray]], stage: str) -> list[dict]:
        tasks, meta = [], {}
        for start, index, u in units:
            key = f"{label}{start:02d}-{stage}-{index:03d}"
            spec = spec_from_unit(u, base, tables)
            tasks.append((key, {"config": spec.config, "extensions": spec.extensions}, seeds, seconds, target))
            meta[key] = {"start": start, "stage": stage, "unit": u.tolist(), "config": spec.config,
                         "extensions": spec.extensions}
        return [{**meta[r["key"]], **r} for r in runner(tasks, context)]

    first = [(s, i, rngs[s].random(dim)) for s in range(starts) for i in range(global_draws)]
    records += submit(first, "global")
    for r, step in enumerate(rounds):
        units = []
        for s in range(starts):
            incumbent = min((x for x in records if x["start"] == s), key=_rank)
            centre = np.asarray(incumbent["unit"])
            for i in range(per_round):
                units.append((s, i, np.clip(centre + rngs[s].normal(0, step, dim), 0.0, 1.0)))
        records += submit(units, f"round{r + 1}")
    return records


# ----------------------------------------------------------------------------- selection rules


def near_optimal(scores: dict[str, float], best: str, se: float, *, relative: float = 0.10) -> list[str]:
    """Candidates with objective <= min + max(relative * min, 2 * SE)."""
    threshold = scores[best] + max(relative * scores[best], 2 * se)
    return sorted((k for k, v in scores.items() if v is not None and v <= threshold), key=lambda k: (scores[k], k))


def distinct_set(ordered: list[str], units: dict[str, np.ndarray], *, distance: float = 0.25,
                 limit: int | None = None) -> list[str]:
    """Greedy materially distinct subset (L-infinity >= ``distance`` in the unit box)."""
    kept: list[str] = []
    for key in ordered:
        if all(np.max(np.abs(units[key] - units[other])) >= distance for other in kept):
            kept.append(key)
        if limit is not None and len(kept) >= limit:
            break
    return kept


def pareto_front(vectors: dict[str, np.ndarray]) -> list[str]:
    """Non-dominated candidates (minimization; NaN components treated as +inf)."""
    keys = sorted(vectors)
    matrix = np.vstack([np.where(np.isfinite(vectors[k]), vectors[k], np.inf) for k in keys])
    front = []
    for i, key in enumerate(keys):
        dominated = np.any(np.all(matrix <= matrix[i], axis=1) & np.any(matrix < matrix[i], axis=1))
        if not dominated:
            front.append(key)
    return front
