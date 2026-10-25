"""v1 hybrid-twin decisions and synthetic-to-hybrid agreement (H5 = v0.9 H8, H9 = v0.9 H29 under new conditions).

* Hybrid days: the unchanged v0.9 ``decisions.hybrid_day`` (F3 twin on a real backbone, impact and survival models
  fitted on the training units, registered policy set over the 24 windows with common random numbers), executed
  on ledgered v1 caches through ``v09_adapter``.
* Synthetic: per stream, every classical family and the strategic ecology are calibrated on the training units
  with the **corrected v1 book model** (the registered "new conditions" of H9) and run the registered policy set
  in the F4 simulator (``twins.classical.synthetic_episode``), 8 seeds per policy.
* Analysis: the unchanged v0.9 ``decisions.analyse`` produces the unit values; H5 = v0.9 ``h8`` (KM minus
  calibrated survival-passive cost, bps), H9 = v0.9 ``h29`` (pairwise sign agreement of the synthetic consensus
  with the hybrid directions).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

from . import common as CM
from .replication import _mapping, _pool

ANALYSIS = "decisions-v1"
SYN_SEEDS = 8


def synthetic_task_v1(stream: str, train_paths: list[str], seed: int, notional: float) -> dict:
    from ....v09.execution import policies as P
    from ....v09.generators.strategic import StrategicEcology
    from ....v09.studies.bounds import RANKING_POLICIES
    from ...data import daycache as DC
    from ...runtime import cap_threads
    from ...twins import classical as TC
    cap_threads()
    caches = [DC.load(Path(p)) for p in train_paths]
    fit = TC.fit(caches, book="v1")
    tick, lot = float(caches[0]["tick"]), float(caches[0]["lot"])
    del caches
    gens = dict(fit["generators"])
    strat = StrategicEcology(None, fit["book"]).fit(fit["frame"])
    strat.book = fit["book"]
    gens["strategic_ecology"] = strat
    qty = int(notional / (fit["book"].mid_ticks * tick) / lot)
    costs = {}
    for gname, g in gens.items():
        costs[gname] = {}
        for p in RANKING_POLICIES:
            vals = []
            for k in range(SYN_SEEDS):
                r = TC.synthetic_episode(g, P.make(p), qty=qty, seed=seed * 1000 + k, side=1 if k % 2 == 0 else -1)
                vals.append(r["is_net_bps"] if r["status"] == "OK" else None)
            costs[gname][p] = vals
    return {"stream": stream, "qty_lots": qty, "costs": costs, "book": fit["book"].calibration.get("improvement")}


def run(ctx: CM.Context, out: Path, *, evaluate: list[str], train: list[str] | None, seed: int,
        workers: int = 2) -> dict:
    from ....v09.studies import decisions as D9
    t0 = time.time()
    if train is None:                       # pilot: first half of the dates trains, second half evaluates
        dates = sorted({CM.date_of(d) for d in evaluate})
        half = set(dates[: len(dates) // 2])
        train = [d for d in evaluate if CM.date_of(d) in half]
        evaluate = [d for d in evaluate if CM.date_of(d) not in half]
    prot = json.loads((ctx.root / "configs/v1/protocol.json").read_text())
    notional = prot["carried_from_v09"]["mandates"]["classes"]["medium"]["notional_usd"]
    prep = ctx.prepare(train + evaluate, use=f"{ANALYSIS} (v0.9 hybrid twin, v1 synthetic book)", workers=workers)
    mp = _mapping(prep)
    streams = sorted({CM.stream_of(d) for d in evaluate})
    plan = {s: ([d for d in train if CM.stream_of(d) == s and d in mp],
                [d for d in evaluate if CM.stream_of(d) == s and d in mp]) for s in streams}
    jobs = {}
    for s, (tr, te) in plan.items():
        inst = s.split(":")[1]
        for d in tr + te:
            jobs[f"h|{d}"] = ("lob.v09.studies.decisions:hybrid_day", mp,
                              {"day": d, "train": tr, "root": str(ctx.root), "design": "v1", "seed": seed,
                               "notional": float(notional[inst])})
    with ProcessPoolExecutor(workers) as ex:
        hyb = _pool(ex, jobs)
        syn_f = {s: ex.submit(synthetic_task_v1, s, [mp[d] for d in tr], seed, float(notional[s.split(":")[1]]))
                 for s, (tr, _te) in plan.items()}
        syn = {s: f.result() for s, f in syn_f.items()}
    units = {}
    for s, (tr, te) in plan.items():
        a = D9.analyse([hyb[f"h|{d}"] for d in tr], [hyb[f"h|{d}"] for d in te], syn[s], None)
        units.update(a["units"])
    h5 = {d: u.get("h8") for d, u in units.items()}
    h9 = {d: u.get("h29") for d, u in units.items()}
    res = {"plan": plan, "units_v09": units, "units": {"H5": h5, "H9": h9},
           "clusters": {"H5": CM.cluster_values(h5), "H9": CM.cluster_values(h9)},
           "synthetic": {s: {"qty_lots": v["qty_lots"], "book": v["book"]} for s, v in syn.items()},
           "compute_hours": (time.time() - t0) / 3600,
           "label": "HYBRID (F3) and SYNTHETIC (F4/F6) outcomes; never historical fills"}
    info = CM.finalize(ctx, out, analysis=ANALYSIS, config={"plan": plan, "seed": seed, "synthetic_seeds": SYN_SEEDS},
                       result=res)
    return {**info, "result": res}


__all__ = ["run", "synthetic_task_v1", "ANALYSIS"]
