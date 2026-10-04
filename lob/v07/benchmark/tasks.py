"""Frozen benchmark tasks, the public replication subset and local performance benchmarks (workstreams 86, 93, 18M).

``TASKS`` are stable definitions (reconstruction, calibration, realism, identifiability, execution, transfer,
performance) that run on synthetic fixtures only, so anyone can run them; each returns metrics and a digest.

The public replication subset is the ``public`` group: deterministic results computed from code and synthetic
fixtures alone (queue-model bound ordering on random level histories, SMC-ABC recovery on an analytic
problem, decision-certification truth table, matching-engine quantity conservation, v0.6/v0.7 tape
equality on a synthetic file). ``cleo benchmark-v07 public`` reruns them and compares the digest with the
stored expectation (reproduction tier 3).

Performance timings are local research-workload measurements (wall time, throughput), never exchange,
network or HFT latency.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from ...config import canonical_json


def _digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _fixture(directory: Path, seconds: float = 600.0):
    from ...v07.adapters.fixtures import write_tardis
    return write_tardis(directory, seconds=seconds)


def task_reconstruction() -> dict:
    from ...v06.tape import tape_from_tardis
    from ..data.tape import build_tape
    from ..replay.deterministic import BookReplay
    from ..adapters.tardis import TardisAdapter
    with tempfile.TemporaryDirectory() as tmp:
        files = _fixture(Path(tmp))
        t0 = time.perf_counter()
        new, quality = build_tape(files["l2"], files["trades"], tick=0.05)
        old = tape_from_tardis(files["l2"], files["trades"], tick=0.05)
        chain = BookReplay().run(TardisAdapter("deribit", "ETH-PERPETUAL", l2=files["l2"], trades=None).records()).finish()
        return {"tapes_equal": bool(np.array_equal(new.bq, old.bq) and np.array_equal(new.trade_t, old.trade_t)),
                "replay_chain": chain, "samples": int(len(new.t)), "wall_s": time.perf_counter() - t0}


def task_queue() -> dict:
    from ..queue.models import MODELS, LevelEvent, all_models
    rng = np.random.default_rng(93)
    ordered = 0
    for _ in range(500):
        events, level = [], float(rng.integers(10, 200))
        join = level
        for t in range(int(rng.integers(1, 30))):
            kind = rng.choice(["size", "trade_at", "trade_through"], p=[0.5, 0.4, 0.1])
            q = float(rng.integers(0, 50))
            events.append(LevelEvent(float(t), str(kind), q))
        out = all_models(events, qty=float(rng.integers(1, 40)), level_at_join=join)
        filled = [out[m]["filled"] for m in MODELS]
        ordered += all(a <= b + 1e-9 for a, b in zip(filled, filled[1:]))
    return {"histories": 500, "bound_order_holds": ordered}


def task_calibration() -> dict:
    from ..posterior import smc_abc
    truth = np.full(smc_abc.DIM, 0.3)

    class Analytic:
        evaluations = 0

        def __call__(self, units, tag):
            rng = np.random.default_rng(tag)
            d = np.max(np.abs(units - truth), axis=1) + rng.normal(0, 0.005, len(units))
            return d, [{"objective": float(v)} for v in d]
    post = smc_abc.run(Analytic(), particles=200, generations=8, seed=1)
    return {"tolerance": round(post["tolerance"], 10), "coverage": smc_abc.coverage(post["particles"], truth)["covered_fraction"],
            "mean_abs_error": round(float(np.max(np.abs(post["particles"].mean(0) - truth))), 10)}


def task_identifiability() -> dict:
    from ..identifiability import analysis as ia
    a = np.array([[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    j, _ = ia.jacobian(lambda th: a @ th, np.full(4, 0.5))
    return {"effective_rank": ia.sloppiness(j)["effective_rank"]}


def task_certification() -> dict:
    from itertools import product
    from ..robustness.certification import Evidence, certify
    table = {}
    for reg, sig, mat, model in product([True, False], repeat=4):
        e = Evidence(registered=reg, estimate=-2.0 if mat else -0.5, ci=(-3.0, -1.0) if sig else (-3.0, 0.5), margin=1.0,
                     mde=0.5, opposite_worlds=[] if model else ["w"], worst_plausible=-0.5,
                     history={"conservative": -1, "optimistic": -1})
        if sig and not mat:
            e.ci = (-0.8, -0.2)
        table[f"{int(reg)}{int(sig)}{int(mat)}{int(model)}"] = certify(e)["status"]
    return {"truth_table": table}


def task_execution() -> dict:
    from ...engine import Order, OrderBook, OrderType, Side
    rng = np.random.default_rng(5)
    book = OrderBook(check_invariants=True)
    submitted = filled = 0
    for i in range(1, 400):
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        qty = int(rng.integers(1, 20))
        market = rng.random() < 0.3
        trades = book.process(Order(i, side, qty, OrderType.MARKET if market else OrderType.LIMIT, "B", 0.0,
                                    price=None if market else int(rng.integers(95, 106))), float(i))
        submitted += qty
        filled += sum(t.qty for t in trades)
    resting = sum(book.bid_vol.values()) + sum(book.ask_vol.values())
    cancelled = sum(o.remaining for o in book.order_history.values() if o.is_terminal)
    return {"conserved": resting + 2 * filled + cancelled == submitted, "filled": filled}


def task_transfer() -> dict:
    from ..transfer.matrix import variance_decomposition
    rng = np.random.default_rng(7)
    vectors = {f"{g}-{m}": np.array([s + rng.normal(0, 0.1), rng.normal(1, 0.1)]) for g, s in (("eth", 0), ("btc", 5))
               for m in range(5)}
    out = variance_decomposition(vectors, {k: k.split("-")[0] for k in vectors})
    return {"between_share": [round(c["between_share"], 6) for c in out["coefficients"]]}


def task_performance() -> dict:
    from ...engine import ExchangeSimulator, SimConfig
    t0 = time.perf_counter()
    sim = ExchangeSimulator(SimConfig(seed=1, record_events=False))
    sim.step(600.0)
    wall = time.perf_counter() - t0
    return {"simulated_seconds_per_wall_second": 600.0 / wall, "events": sim.event_count,
            "note": "local research-workload timing; not exchange or HFT latency"}


TASKS = {"reconstruction": task_reconstruction, "queue": task_queue, "calibration": task_calibration,
         "identifiability": task_identifiability, "certification": task_certification, "execution": task_execution,
         "transfer": task_transfer, "performance": task_performance}
PUBLIC = ("queue", "calibration", "identifiability", "certification", "execution", "transfer", "reconstruction")
VOLATILE = {"reconstruction": ("wall_s",)}


def run(names=None) -> dict:
    out = {}
    for name in names or TASKS:
        result = TASKS[name]()
        out[name] = result
    return out


def public_digest(results: dict) -> str:
    stable = {k: {f: v for f, v in results[k].items() if f not in VOLATILE.get(k, ())} for k in PUBLIC}
    return _digest(stable)


def definitions() -> dict:
    return {"schema": "cleolob-v07-benchmark-tasks-1", "tasks": {k: (v.__doc__ or k) for k, v in TASKS.items()},
            "public_subset": list(PUBLIC), "volatile_fields": {k: list(v) for k, v in VOLATILE.items()},
            "rule": "task definitions are frozen; timing fields are excluded from the public digest"}


def write_expected(path: Path) -> dict:
    results = run(PUBLIC)
    doc = {"digest": public_digest(results), "results": json.loads(canonical_json(
        {k: {f: v for f, v in r.items() if f not in VOLATILE.get(k, ())} for k, r in results.items()}))}
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    return doc
