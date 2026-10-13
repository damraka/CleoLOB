"""meta-adaptation-v09 (H15): few-shot hierarchical adaptation vs full-day refit (non-inferiority).

For each target day k >= 2 of an instrument (chronological), the hierarchical empirical-Bayes prior is built from
the earlier days. Few-shot: prior + the first 2 registered windows of day k (8.3 % of the day's windows, within the
registered 10 %). Full refit: a flat Dirichlet(1) prior + the first 12 windows (50 %) of day k. Both predict the last
12 windows. Unit endpoint (oriented, positive favours few-shot): NLL(full refit) - NLL(few-shot); the registered
test is non-inferiority with margin 0.01 nats per bin. Split-half: odd vs even evaluation windows.
"""
from __future__ import annotations

import time

import numpy as np

from .. import protocol as pr
from ..inference import bayes as B
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "meta-adaptation-v09"
FEW, HALF = 2, 12


def evaluate(sequence: list[dict]) -> dict:
    rows = {}
    for k in range(1, len(sequence)):
        prior_days = [B.counts(np.asarray(s["b"]), np.asarray(s["y"])) for s in sequence[:k]]
        prior = B.empirical_bayes(prior_days)
        b, y, wid = (np.asarray(sequence[k][key]) for key in ("b", "y", "wid"))
        few = prior + B.counts(b[wid < FEW], y[wid < FEW])
        full = 1.0 + B.counts(b[wid < HALF], y[wid < HALF])
        ev = wid >= HALF
        a, f = B.nll(full, b[ev], y[ev]), B.nll(few, b[ev], y[ev])
        odd = wid[ev] % 2 == 1
        rows[sequence[k]["dataset"]] = {"nll_full_refit": float(a.mean()), "nll_few_shot": float(f.mean()),
                                        "effect": float(a.mean() - f.mean()),
                                        "odd": float(a[odd].mean() - f[odd].mean()),
                                        "even": float(a[~odd].mean() - f[~odd].mean()),
                                        "adaptation_fraction_few": FEW / 24, "adaptation_fraction_full": HALF / 24}
    return rows


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    t0 = time.time()
    if args.phase == "pilot":
        res = common.units(args, "explore", common.BYBIT_STREAMS, protocol["pools"]["explore"]["count"])
        sequences = {s: sorted(res[s]["datasets"], key=lambda d: d[-10:]) for s in common.BYBIT_STREAMS}
        design_name = "explore-meta"
    else:
        design = common.design_document(args)
        sequences, design_name = {s: list(v) for s, v in design["sequences"].items()}, args.design
    out = common.start(args)
    ids = [d for v in sequences.values() for d in v]
    tasks = [Task(d, "lob.v09.studies.inference:day_counts", {"d": d, "root": str(root), "design": design_name},
                  namespace=args.namespace) for d in ids]
    days, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args))
    by = {d["dataset"]: d for d in days}
    rows = {}
    for s, seq in sequences.items():
        rows.update(evaluate([by[d] for d in seq]))
    hours = (time.time() - t0) / 3600 / max(len(rows), 1)
    caps = common.caps_for(protocol, ("incremental_l2", "trades_with_side"))
    pilot = {"H15": common.pilot_record([r["effect"] for r in rows.values()], [r["odd"] for r in rows.values()],
                                        [r["even"] for r in rows.values()], caps=caps, hours=hours)}
    result = {"phase": args.phase, "sequences": sequences, "units": rows,
              "pilot": pilot if args.phase == "pilot" else None, "runner": info}
    return common.finish(args, out, analysis=ANALYSIS, config={"phase": args.phase, "sequences": sequences,
                                                               "few_windows": FEW, "half_windows": HALF},
                         result=result, datasets=ids)
