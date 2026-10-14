"""realism-benchmark-v09 (M51-M54): realism scorecard v3 of the full-book generators (F4, F6) per instrument.

Explore namespace: per instrument the generators are fitted on the first half of the explore days (chronological)
and evaluated against the other half. Confirm namespace (sealed design): fitted on all explore days, evaluated on the
design's confirm days (the authoritative gate input). Each generator simulates the registered number of windows
(3300 bins each, seeded) and is scored per capability against the pooled held-out real metrics, normalized by real
day-to-day variation; a cross-validated classifier AUC is reported. Neural generators (F5) model trades or paths only
and are scored in the TPP and path studies, not here.
"""
from __future__ import annotations

import time

import numpy as np

from .. import protocol as pr
from ..data.daycache import BIN_MS, HORIZON_S, WARMUP_S, WINDOW_STARTS_S, frame as cache_frame
from ..generators import classical as G
from ..generators.strategic import StrategicEcology
from ..realism import benchmark as RB
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "realism-benchmark-v09"
SIM_WINDOWS = 12


def _windows(cache):
    out = []
    for s in WINDOW_STARTS_S:
        a, b = int((s - WARMUP_S) * 1000 // BIN_MS), int((s + HORIZON_S) * 1000 // BIN_MS)
        out.append(cache_frame(cache, a, b))
    return out


def instrument_task(stream: str, train: list[str], test: list[str], root: str, design: str, seed: int) -> dict:
    caches = [common.load_cache(d, root, design, "realism benchmark") for d in train]
    frame = G.training_frame(caches)
    book = G.book_model(frame)
    gens = {n: G.fit_family(n, frame, book) for n in list(G.FAMILIES) + list(G.STATIC_ECOLOGY)}
    strat = StrategicEcology(None, book).fit(frame)
    strat.book = book
    gens["strategic_ecology"] = strat
    del caches
    real_day = {}
    real_frames = []
    for d in test:
        fr = _windows(common.load_cache(d, root, design, "realism benchmark"))
        real_day[d] = RB.metrics(fr)
        real_frames.extend(fr[::2])
    ref = RB.reference(list(real_day.values()))
    real = {cap: np.mean([m[cap] for m in real_day.values()], axis=0).tolist() for cap in RB.CAPABILITIES}
    out = {}
    for name, g in gens.items():
        sims = [g.simulate(3300, seed * 100 + k, warmup_bins=600) for k in range(SIM_WINDOWS)]
        m = RB.metrics(sims)
        auc = RB.classifier_auc(real_frames, sims, seed=seed)
        r = RB.ratios(m, real, ref)
        out[name] = {"metrics": m, "ratios": r, "auc": auc, "decisions": RB.decide(r, auc)}
    return {"stream": stream, "train": train, "test": test, "real": real, "reference": ref, "real_by_day": real_day,
            "generators": out, "global": RB.global_gate({n: v["decisions"] for n, v in out.items()})}


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    t0 = time.time()
    if not getattr(args, "design", None):            # pilot: registered explore slots; else the sealed design
        res = common.units(args, "explore", common.BYBIT_STREAMS, protocol["pools"]["explore"]["count"])
        plan = {}
        for s in common.BYBIT_STREAMS:
            days = sorted(res[s]["datasets"], key=lambda d: d[-10:])
            plan[s] = (days[: len(days) // 2], days[len(days) // 2:])
        design_name = "explore-realism"
    else:
        design = common.design_document(args)
        plan = {s: (list(v["train"]), list(v["test"])) for s, v in design["plan"].items()}
        design_name = args.design
    out = common.start(args)
    tasks = [Task(s, "lob.v09.studies.realism:instrument_task", {"stream": s, "train": tr, "test": te,
                                                                 "root": str(root), "design": design_name,
                                                                 "seed": args.seed}, namespace=args.namespace)
             for s, (tr, te) in plan.items()]
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args))
    result = {"namespace": args.namespace, "streams": {r["stream"]: r for r in rows}, "runner": info,
              "gate_rule": protocol["realism_gate"], "hours": (time.time() - t0) / 3600}
    datasets = [d for tr, te in plan.values() for d in tr + te]
    return common.finish(args, out, analysis=ANALYSIS, config={"plan": plan, "seed": args.seed,
                                                               "sim_windows": SIM_WINDOWS},
                         result=result, datasets=datasets)
