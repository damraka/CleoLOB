"""M13 exploratory ecology study (workstreams 46-48): conservation over seeds, market-maker metrics, and a
passive-versus-reactive environment comparison of two parent-order schedules (fast: 5 slices; slow: 20)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...experiments.registry import PROJECT_ROOT
from ...sim_v2 import SimulatorSpec
from ...v06.identifiability import clean
from ..evidence.runs import finalize, new_run
from ..generators.study import v06_inputs
from . import ecology as ec

SEEDS = tuple(range(791000, 791032))


def _shortfall(result, owner="META"):
    trades = [t for t in result["trades"] if t.taker_owner == owner]
    if not trades:
        return None
    arrival = result["mids"][0]
    return float(sum((t.price - arrival) * t.qty for t in trades) / sum(t.qty for t in trades) / arrival * 1e4)


def run(out: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    selected = v06_inputs(root)["selection"]["selected"]
    world = SimulatorSpec(selected["config"], selected["extensions"])
    out = new_run(root / out)
    conservation, makers = [], []
    for s in SEEDS[:16]:
        r = ec.Ecology(world, [ec.LiquidityProvider("LP", s), ec.Taker("TK", s + 1), ec.NoiseTrader("NT", s + 2),
                               ec.InventoryMaker("MM", s + 3), ec.MetaOrderTrader("META", s + 4)]).run(s, 120.0)
        conservation.append({"inventory_sum": sum(v[0] for v in r["accounts"].values()),
                             "cash_sum": sum(v[1] for v in r["accounts"].values())})
        makers.append(ec.maker_metrics(r, "MM"))
    envs = {}
    for env, maker in (("passive", ec.LiquidityProvider), ("reactive", ec.ReactiveMaker)):
        rows = {"fast": [], "slow": []}
        for s in SEEDS:
            for schedule, slices in (("fast", 5), ("slow", 20)):
                r = ec.Ecology(world, [maker("MK", s), ec.MetaOrderTrader("META", s + 9, quantity=100, slices=slices)]).run(s, 25.0)
                rows[schedule].append(_shortfall(r))
        d = np.asarray([a - b for a, b in zip(rows["fast"], rows["slow"]) if a is not None and b is not None])
        idx = np.random.default_rng(1).integers(0, len(d), (2000, len(d)))
        lo, hi = np.quantile(d[idx].mean(1), [0.025, 0.975])
        envs[env] = {"fast_minus_slow_bps": float(d.mean()), "ci": [float(lo), float(hi)], "n": int(len(d)),
                     "direction": -1 if hi < 0 else 1 if lo > 0 else 0}
    dirs = {e: v["direction"] for e, v in envs.items()}
    status = ("MODEL_DEPENDENT" if dirs["passive"] * dirs["reactive"] == -1 or
              (dirs["passive"] != dirs["reactive"] and 0 in dirs.values()) else "EXPLORATORY")
    result = {"conservation": {"max_abs_inventory_sum": max(abs(c["inventory_sum"]) for c in conservation),
                               "max_abs_cash_sum": max(abs(c["cash_sum"]) for c in conservation)},
              "market_maker": {k: float(np.mean([m[k] for m in makers])) for k in makers[0]},
              "strategic_interaction": {"environments": envs, "status": status},
              "label": "EXPLORATORY; stylized agents are not real participant classes"}
    finalize(out, analysis="m13-ecology", dataset_ids=[], config={"seeds": list(SEEDS)}, result=clean(result),
             root=root, seeds={"seeds": list(SEEDS)})
    return result
