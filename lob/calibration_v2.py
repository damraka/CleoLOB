"""Calibration v2 (M3): model-class comparison under the frozen chronology.

development (fit) -> selection (choose one model) -> selection seal (ledger) ->
sealed internal holdout (retrospective) -> fresh external holdout -> cross-instrument holdout.

Each family is fitted only on development observables by a bounded, seeded
two-stage search with an identical budget for every family (including the v0.4
ZI control). Selection uses only the selection period. The selected model and
the control are then evaluated, unchanged, on the holdouts; the paired
chronological block bootstrap compares their losses. Nothing is refitted after
the selection seal, and holdout results never feed back into any choice.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import numpy as np

from .engine import SimConfig
from .observables_v2 import FAMILIES, GATE, SAMPLE_DT, Tape, family_errors, group_prints, loss, measure, summarize
from .replay.l2 import L2Replay
from .sim_v2 import SimulatorSpec

EXTENSION_FAMILIES = ("empirical_sizes", "spread_conditioned_arrivals", "depth_conditioned_cancellation",
                      "imbalance_conditioned_market_flow", "markov_activity_regime", "self_exciting_market_orders")
BASE_BOUNDS = {"target_level_vol": (20, 600), "limit_rate": (0.5, 60.0), "market_rate": (0.01, 5.0),
               "cancel_rate": (0.005, 1.0), "offset_p": (0.1, 0.9), "limit_qty_mean": (5.0, 400.0),
               "market_qty_mean": (5.0, 800.0), "resilience": (0.01, 2.0)}
EXTENSION_BOUNDS = {
    "spread_conditioned_arrivals": {"inside_spread_prob": (0.02, 0.9)},
    "depth_conditioned_cancellation": {"cancel_depth_exponent": (-2.0, 2.0)},
    "imbalance_conditioned_market_flow": {"imbalance_beta": (-0.9, 0.9)},
    "markov_activity_regime": {"regime_multiplier": (1.5, 10.0), "regime_switch_rate": (0.002, 0.2),
                               "regime_high_share": (0.05, 0.6)},
    "self_exciting_market_orders": {"hawkes_alpha": (0.05, 5.0), "hawkes_decay": (0.1, 10.0)},
}
SIZE_GRID = tuple(np.linspace(0.0, 1.0, 21))
HAWKES_DISPERSION_THRESHOLD = 1.5


# ----------------------------------------------------------------------------- tapes

def tape_from_tardis(updates: str | Path, trades: str | Path, *, depth_scale: float,
                     max_staleness_s: float = 5.0) -> tuple[Tape, dict]:
    """Causal 100 ms samples of the top five levels plus grouped aggressive events."""
    import csv
    import gzip

    replay = L2Replay(Path(updates), depth=5)
    step_us = int(SAMPLE_DT * 1e6)
    stale_us = int(max_staleness_s * 1e6)
    rows_t, bp, bq, ap, aq = [], [], [], [], []
    previous = None
    next_us = None

    def emit(state, at_us):
        rows_t.append(at_us / 1e6)
        ok = state is not None and at_us - state.local_timestamp_us <= stale_us and not state.crossed
        for levels, prices, quantities in ((state.bids if ok else (), bp, bq), (state.asks if ok else (), ap, aq)):
            p = [float(x[0]) for x in levels[:5]] + [math.nan] * (5 - len(levels[:5]))
            q = [float(x[1]) / depth_scale for x in levels[:5]] + [0.0] * (5 - len(levels[:5]))
            prices.append(p)
            quantities.append(q)

    for state in replay:
        now = state.local_timestamp_us
        if next_us is None:
            next_us = (now // step_us + 1) * step_us
        while previous is not None and next_us < now:
            emit(previous, next_us)
            next_us += step_us
        previous = state
    if not replay.stats["complete"]:
        raise ValueError("L2 source incomplete")
    t0 = rows_t[0]
    times, prices, sizes, sides, keys = [], [], [], [], []
    opener = gzip.open if str(trades).endswith(".gz") else open
    with opener(trades, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            side = row["side"].lower()
            if side not in {"buy", "sell"}:
                continue
            local = int(row["local_timestamp"]) / 1e6
            if local < t0 or local >= rows_t[-1] + SAMPLE_DT:
                continue
            times.append(local)
            prices.append(float(row["price"]))
            sizes.append(float(row["amount"]) / depth_scale)
            sides.append(1 if side == "buy" else -1)
            keys.append(int(row["timestamp"]) * 4 + (1 if side == "buy" else 2))
    event_t, event_p, event_q, event_s = group_prints(np.asarray(times), np.asarray(prices), np.asarray(sizes),
                                                      np.asarray(sides), np.asarray(keys, dtype=np.int64))
    tape = Tape(np.asarray(rows_t) - t0, np.asarray(bp), np.asarray(bq), np.asarray(ap), np.asarray(aq),
                event_t - t0, event_p, event_q, event_s.astype(float))
    return tape, {"first_local_us": int(round(t0 * 1e6)), "samples": len(rows_t), "events": int(len(event_t)),
                  "replay": replay.stats}


def tape_from_simulator(spec: SimulatorSpec, seed: int, *, seconds: float, warmup: float = 60.0) -> Tape:
    sim = spec.build(seed)
    sim.step(warmup)
    tick = sim.cfg.tick_size
    first_trade = len(sim.book.trades)
    n = int(round(seconds / SAMPLE_DT))
    bp, bq, ap, aq = (np.full((n, 5), np.nan), np.zeros((n, 5)), np.full((n, 5), np.nan), np.zeros((n, 5)))
    start = sim.t
    for i in range(n):
        sim.step(SAMPLE_DT)
        bids, asks = sim.book.depth(5)
        for j, (p, q) in enumerate(bids):
            bp[i, j], bq[i, j] = p * tick, q
        for j, (p, q) in enumerate(asks):
            ap[i, j], aq[i, j] = p * tick, q
    trades = sim.book.trades[first_trade:]
    if trades:
        t = np.asarray([x.time for x in trades]) - start
        keep = t <= n * SAMPLE_DT
        keys = np.asarray([(x.taker_order_id or 0) for x in trades], dtype=np.int64)
        et, ep, eq, es = group_prints(t[keep], np.asarray([x.price * tick for x in trades])[keep],
                                      np.asarray([float(x.qty) for x in trades])[keep],
                                      np.asarray([1.0 if x.taker_side.value == 1 else -1.0 for x in trades])[keep],
                                      keys[keep])
    else:
        et = ep = eq = es = np.asarray([])
    # Sample i is the book at the end of interval i: shift times so sample 0 is at 0 s.
    return Tape(np.arange(n) * SAMPLE_DT, bp, bq, ap, aq, et - SAMPLE_DT, ep, eq, es)


def simulated_summary(spec: SimulatorSpec, seeds: list[int], seconds: float) -> dict:
    """Pool samples over seeds (each seed measured separately; no cross-seed returns)."""
    pooled: dict[str, list] = {}
    for seed in seeds:
        samples = measure(tape_from_simulator(spec, seed, seconds=seconds))
        for key, value in samples.items():
            pooled.setdefault(key, []).append(value)
    merged = {}
    for key, values in pooled.items():
        if key in {"trade_rate", "top_change_rate", "net_cancel_rate", "addition_rate", "acf_return",
                   "acf_abs_return", "acf_spread", "corr_spread_depth", "corr_imbalance_next_return",
                   "corr_flow_return"}:
            finite = [float(v[0]) for v in values if math.isfinite(float(v[0]))]
            merged[key] = np.asarray([np.mean(finite) if finite else math.nan])
        else:
            merged[key] = np.concatenate(values)
    return summarize(merged)


# ----------------------------------------------------------------------------- candidates

def base_config(tick_bps: float) -> dict:
    mid_ticks = max(21, round(10000 / tick_bps))
    return asdict(SimConfig(initial_mid_ticks=mid_ticks, tick_size=100 / mid_ticks, target_level_vol=200,
                            limit_rate=4, market_rate=0.5, cancel_rate=0.1, limit_qty_mean=35, market_qty_mean=40,
                            resilience=0.15, record_events=False, max_events=5_000_000))


def _draw(rng: np.random.Generator, low: float, high: float, log: bool) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high)))) if log else float(rng.uniform(low, high))


def random_base(rng: np.random.Generator, base: dict) -> dict:
    values = {}
    for name, (low, high) in BASE_BOUNDS.items():
        if name == "target_level_vol":
            values[name] = int(rng.integers(low, high + 1))
        else:
            values[name] = _draw(rng, low, high, name != "offset_p")
    return {**base, **values}


def perturb_base(rng: np.random.Generator, config: dict, scale: float = 0.35) -> dict:
    values = {}
    for name, (low, high) in BASE_BOUNDS.items():
        if name == "target_level_vol":
            values[name] = int(np.clip(round(config[name] * math.exp(rng.normal(0, scale))), low, high))
        elif name == "offset_p":
            values[name] = float(np.clip(config[name] + rng.normal(0, 0.1), low, high))
        else:
            values[name] = float(np.clip(config[name] * math.exp(rng.normal(0, scale)), low, high))
    return {**config, **values}


def random_extension(rng: np.random.Generator, family: str) -> dict:
    values = {}
    for name, (low, high) in EXTENSION_BOUNDS.get(family, {}).items():
        values[name] = _draw(rng, low, high, name in {"regime_multiplier", "regime_switch_rate", "hawkes_decay",
                                                         "hawkes_alpha", "inside_spread_prob"})
    if family == "self_exciting_market_orders" and values["hawkes_alpha"] >= values["hawkes_decay"]:
        values["hawkes_alpha"] = 0.9 * values["hawkes_decay"]
    return values


def empirical_size_tables(dev_samples: dict) -> dict:
    trade = np.quantile(dev_samples["trade_sizes"], SIZE_GRID) if len(dev_samples["trade_sizes"]) else ()
    add = np.quantile(dev_samples["addition_sizes"], SIZE_GRID) if len(dev_samples["addition_sizes"]) else ()
    clean = lambda q: tuple(float(max(1.0, v)) for v in np.maximum.accumulate(np.asarray(q)))  # noqa: E731
    return {"market_size_quantiles": clean(trade), "limit_size_quantiles": clean(add)}


# ----------------------------------------------------------------------------- fitting

def evaluate_spec(spec: SimulatorSpec, target: dict, seeds: list[int], seconds: float) -> dict:
    try:
        simulated = simulated_summary(spec, seeds, seconds)
        errors = family_errors(target, simulated)
        return {"loss": loss(errors), "errors": errors, "simulated": simulated, "error": None}
    except (ValueError, RuntimeError, AssertionError, OverflowError) as exc:
        return {"loss": math.inf, "errors": None, "simulated": None, "error": f"{type(exc).__name__}: {exc}"}


def _evaluate_task(task: tuple) -> dict:
    config, extensions, target, seeds, seconds = task
    outcome = evaluate_spec(SimulatorSpec(config, extensions), target, seeds, seconds)
    value = outcome["loss"]
    return {"config": config, "extensions": extensions,
            "loss": value if value is not None and math.isfinite(value) else None,
            "error": outcome["error"], "errors": outcome["errors"]}


def _rank(record: dict) -> tuple:
    """Failed or non-finite candidates rank last; ties are broken deterministically."""
    return (record["loss"] if record["loss"] is not None else math.inf, json_key(record))


def _map(tasks: list[tuple], executor) -> list[dict]:
    return list(executor.map(_evaluate_task, tasks)) if executor is not None else [_evaluate_task(t) for t in tasks]


def _refine_extension(rng: np.random.Generator, family: str, extensions: dict) -> dict:
    result = dict(extensions)
    for name, (low, high) in EXTENSION_BOUNDS.get(family, {}).items():
        logscale = name in {"regime_multiplier", "regime_switch_rate", "hawkes_decay", "hawkes_alpha",
                            "inside_spread_prob"}
        value = result[name] * math.exp(rng.normal(0, 0.3)) if logscale else result[name] + rng.normal(0, 0.2 * (high - low))
        result[name] = float(np.clip(value, low, high))
    if family == "self_exciting_market_orders" and result["hawkes_alpha"] >= result["hawkes_decay"]:
        result["hawkes_alpha"] = 0.9 * result["hawkes_decay"]
    return result


def fit_family(family: str, target: dict, *, base: dict, size_tables: dict, baseline_best: dict | None,
               seeds: list[int], seconds: float, candidates: int, search_seed: int, executor=None,
               start_extensions: dict | None = None, extension_families: tuple[str, ...] | None = None) -> dict:
    """Two batches with an identical budget for every family.

    Batch 1 (half the budget): global draws (control) or baseline-perturbed draws with
    random extension parameters. Batch 2: perturbations around the batch-1 optimum.
    Candidate parameters are drawn before any evaluation of their batch, so results
    do not depend on process scheduling.
    """
    rng = np.random.default_rng([search_seed, FAMILY_INDEX[family]])
    fixed = size_tables if family in {"empirical_sizes"} else {}
    members = extension_families or ((family,) if family in EXTENSION_BOUNDS else ())
    half = candidates // 2
    first = []
    if baseline_best is None:
        first = [(random_base(rng, base), dict(fixed)) for _ in range(half)]
    else:
        def draw_extensions():
            values = dict(fixed)
            for member in members:
                values.update(random_extension(rng, member))
            return values
        seed_ext = dict(start_extensions) if start_extensions is not None else draw_extensions()
        first = [(baseline_best["config"], seed_ext)]
        first += [(perturb_base(rng, baseline_best["config"]), draw_extensions()) for _ in range(half - 1)]
    records = [dict(r, family=family, stage="batch1") for r in _map([(c, e, target, seeds, seconds) for c, e in first],
                                                                    executor)]
    best = min(records, key=_rank)
    second = []
    for _ in range(candidates - half):
        extensions = dict(best["extensions"])
        for member in members:
            extensions = _refine_extension(rng, member, extensions)
        second.append((perturb_base(rng, best["config"], 0.2), extensions))
    records += [dict(r, family=family, stage="batch2") for r in _map([(c, e, target, seeds, seconds) for c, e in second],
                                                                     executor)]
    best = min(records, key=_rank)
    parameters = len(BASE_BOUNDS) + sum(len(EXTENSION_BOUNDS.get(m, {})) for m in members)
    return {"family": family, "best": best, "candidates": len(records), "records": records,
            "parameters": parameters, "members": list(members)}


def json_key(record: dict) -> str:
    import json
    return json.dumps([record["config"], record["extensions"]], sort_keys=True)


FAMILY_INDEX = {"v04_zi_baseline": 0, **{name: i + 1 for i, name in enumerate(EXTENSION_FAMILIES)}, "combined": 99}


def combine(bests: dict[str, dict], improving: list[str], size_tables: dict) -> dict:
    extensions = {}
    for family in improving:
        extensions.update(bests[family]["best"]["extensions"])
    if "empirical_sizes" in improving:
        extensions.update(size_tables)
    return extensions


# ----------------------------------------------------------------------------- inference

def block_bootstrap_loss_difference(blocks: list[dict], sim_a: dict, sim_b: dict, *, samples: int,
                                    alpha: float, seed: int) -> dict:
    """Paired resampling of historical 10-minute blocks; simulated summaries fixed."""
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(blocks)
    for _ in range(samples):
        pick = rng.integers(0, n, n)
        merged = {}
        for key in blocks[0]:
            values = [blocks[i][key] for i in pick]
            if key in {"trade_rate", "top_change_rate", "net_cancel_rate", "addition_rate", "acf_return",
                       "acf_abs_return", "acf_spread", "corr_spread_depth", "corr_imbalance_next_return",
                       "corr_flow_return"}:
                finite = [float(v[0]) for v in values if math.isfinite(float(v[0]))]
                merged[key] = np.asarray([np.mean(finite) if finite else math.nan])
            else:
                merged[key] = np.concatenate(values)
        target = summarize(merged)
        diffs.append(loss(family_errors(target, sim_a)) - loss(family_errors(target, sim_b)))
    diffs = np.asarray(diffs)
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"blocks": n, "samples": samples, "alpha": alpha, "ci_low": float(low), "ci_high": float(high),
            "mean_bootstrap": float(diffs.mean())}


def block_samples(tape: Tape, block_seconds: float = 600.0) -> list[dict]:
    blocks = []
    start = float(tape.t[0])
    stop = float(tape.t[-1])
    while start + block_seconds <= stop + SAMPLE_DT:
        blocks.append(measure(tape.block(start, start + block_seconds)))
        start += block_seconds
    return blocks


def gate_status(errors: dict, valid_fraction: float, samples: int) -> dict:
    families = {f: errors[f] for f in FAMILIES}
    evaluated = {f: e for f, e in families.items() if e["error"] is not None}
    passed = (all(e["passed"] for e in evaluated.values()) and valid_fraction >= 0.95 and samples >= 1000)
    return {"status": "ESTABLISHED" if passed else "FAILED", "gate": GATE, "valid_fraction": valid_fraction,
            "samples": samples, "failed_families": sorted(f for f, e in evaluated.items() if not e["passed"]),
            "not_evaluable_families": sorted(f for f, e in families.items() if e["error"] is None)}

