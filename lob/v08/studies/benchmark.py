"""``benchmark-v08``: local research-workload measurements (brief section 22).

Measured on this machine (context recorded): the frozen v0.5 matching engine (limit/market pairs), the v0.8
simulation driver per generator family, ABC summary simulations, MPC decisions, robust-criterion evaluation,
counterfactual replays, artifact sealing and serial-versus-parallel runner throughput. These are research-workload
timings. They are not HFT or exchange latency benchmarks and support no production claim.
"""
from __future__ import annotations

import os
import platform
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from ...artifacts import portable_provenance, seal_artifacts
from ...engine import Order, OrderBook, OrderType, Side
from .. import counterfactual as cf
from ..execution.mpc import MPC
from ..execution.simulator import Episode, EpisodeContext, Mandate, Observation
from ..generators import REGISTRY
from ..inference.abc import simulate_summary
from ..market.sim import BookModel, MarketSim
from ..robust import select_all
from ..runner import Task, run_tasks
from ..worlds import World
from . import common as C

DEFAULTS = {"tiny": {"repeats": 1, "scale": 0.2}, "small": {"repeats": 2, "scale": 0.5},
            "medium": {"repeats": 3, "scale": 1.0}}
FAMILIES = ("poisson", "hawkes", "queue_reactive", "regime_switching", "state_dependent", "mixture", "ecology")


def _time(fn, repeats: int) -> dict:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        n = fn()
        values.append((time.perf_counter() - start, n))
    per = [t / max(n, 1) for t, n in values]
    return {"median_seconds_per_unit": float(np.median(per)), "units_per_second": float(1 / np.median(per)),
            "repeats": repeats, "units": int(values[0][1])}


def _engine_pairs(n: int) -> int:
    book = OrderBook(tick_size=0.01)
    oid = 0
    for i in range(n):
        oid += 1
        book.process(Order(oid, Side.SELL, 5, OrderType.LIMIT, "B", float(i), price=1000 + i % 5), float(i))
        oid += 1
        book.process(Order(oid, Side.BUY, 5, OrderType.MARKET, "T", float(i)), float(i))
    return n


def _sim(family: str, bins: int) -> int:
    g = REGISTRY[family]()
    MarketSim(g, g.book, 1).run(bins)
    return bins


def _reseal(directory: Path) -> int:
    """Seal ``directory`` again; checksums.json is write-once, so a repeat removes the previous manifest first."""
    (directory / "checksums.json").unlink(missing_ok=True)
    return len(seal_artifacts(directory)["files"])


def run(args) -> dict:
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    r, sc = d["repeats"], d["scale"]
    results = {"v05_matching_engine_pairs": _time(lambda: _engine_pairs(int(4000 * sc)), r)}
    results["v08_simulation_bins"] = {f: _time(lambda f=f: _sim(f, int(3000 * sc)), r) for f in FAMILIES}
    g = REGISTRY["poisson"]()
    results["abc_summary_simulations"] = _time(
        lambda: sum(1 for k in range(int(10 * sc) + 1) if simulate_summary(g.to_dict(), ["market"], [1.0], 600, k)), r)
    obs = Observation(t=0.0, elapsed=10.0, remaining_time=110.0, horizon=120.0, step=2, steps=24, observed_at=0.0,
                      bid=4000, ask=4001, bid_depth=[10] * 10, ask_depth=[10] * 10, volume_since_last=20.0,
                      recent_mids=list(4000.5 + np.cumsum(np.random.default_rng(0).normal(0, 0.3, 300))), target=40,
                      side=1, filled=5, resting=[], tick_bps=2.5, context={"in_flight": 0, "pending_limit": 0})
    policy = MPC()
    policy.reset(EpisodeContext(Mandate(), 0.0, 2.5, np.random.default_rng(0)))
    results["mpc_decisions"] = _time(lambda: sum(1 for _ in range(int(500 * sc) + 1) if policy.decide(obs) is not None), r)
    cost = np.random.default_rng(1).normal(3, 1, (16, 24))
    results["robust_selection_16x24"] = _time(lambda: sum(1 for _ in range(int(20 * sc) + 1)
                                                          if select_all([str(i) for i in range(16)], cost, np.ones(24))), r)
    world = World("bench", g, BookModel())
    m = Mandate(qty=40)
    sim = MarketSim(g, world.book, 3)
    sim.run(1700)
    frame = sim.frame()
    results["counterfactual_replays"] = _time(
        lambda: sum(1 for _ in range(int(40 * sc) + 1)
                    if cf.replay(frame, C.policy("twap"), m, cf.Assumption(), start_bin=300)), r)
    results["episodes"] = _time(lambda: sum(1 for s in range(int(10 * sc) + 1)
                                            if Episode(world, C.policy("twap"), m, s).run()), r)
    tmp = Path(tempfile.mkdtemp(prefix="cleo-bench-"))
    try:
        for i in range(200):
            (tmp / f"f{i:03d}.json").write_text("{}" * 500, encoding="utf-8")
        results["artifact_sealing_200_files"] = _time(lambda: _reseal(tmp), r)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tasks = [Task(f"e{i:03d}", "lob.v08.studies.common:episode_task",
                  {"world": world.to_dict(), "policy_name": "twap", "params": {}, "mandate": m.to_dict(), "seed": i})
             for i in range(int(32 * sc) + 2)]
    t0 = time.perf_counter()
    serial, _ = run_tasks(tasks, workers=1)
    t1 = time.perf_counter()
    parallel, _ = run_tasks(tasks, workers=min(4, os.cpu_count() or 1))
    t2 = time.perf_counter()
    results["runner"] = {"episodes": len(tasks), "serial_seconds": t1 - t0, "parallel_seconds": t2 - t1,
                         "workers": min(4, os.cpu_count() or 1), "identical_results": serial == parallel}
    prov = portable_provenance()
    context = {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
               "cpu": prov["runtime"].get("cpu"), "logical_cpus": os.cpu_count()}
    result = {"summary": results, "machine": context,
              "label": "local research-workload measurements; not HFT, exchange or production latency benchmarks"}
    return C.finish(args, out, analysis="benchmark", config={"design": d}, result=C.json_safe(result))
