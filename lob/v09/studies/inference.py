"""inference-v09 (M56-M59; H14): sequential posterior updating vs frozen calibration on chronological days.

For each instrument the explore (pilot) or confirm days are ordered chronologically. The Dirichlet-categorical
bucket model is updated after every day (power-prior discount 0.9); the frozen model keeps the posterior after the
first day. Unit (instrument, day k >= 2): next-day log score difference NLL(frozen) - NLL(sequential) on day k's
registered windows (nats per bin, positive favours sequential updating). Split-half: odd vs even windows. Also
reports SBI v3 / SBC products when run with the ``sbc-v09`` command (separate study).
"""
from __future__ import annotations

import time

import numpy as np

from .. import protocol as pr
from ..inference import bayes as B
from ..representations import sequence as S
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "inference-v09"


def day_counts(d: str, root: str, design: str) -> dict:
    X, y, wid, _ = S.frame_features(common.load_cache(d, root, design, "sequential posterior"))
    b = B.buckets(X, B.current_class(X))
    return {"dataset": d, "b": b.tolist(), "y": y.tolist(), "wid": wid.tolist()}


def evaluate(sequence: list[dict]) -> dict:
    cs = [B.counts(np.asarray(s["b"]), np.asarray(s["y"])) for s in sequence]
    post = B.sequential(cs)
    frozen = post[0]
    rows = {}
    for k in range(1, len(sequence)):
        b, y, wid = (np.asarray(sequence[k][key]) for key in ("b", "y", "wid"))
        a = B.nll(frozen, b, y)
        s = B.nll(post[k - 1], b, y)
        odd = wid % 2 == 1
        rows[sequence[k]["dataset"]] = {"nll_frozen": float(a.mean()), "nll_sequential": float(s.mean()),
                                        "effect": float(a.mean() - s.mean()),
                                        "odd": float(a[odd].mean() - s[odd].mean()),
                                        "even": float(a[~odd].mean() - s[~odd].mean())}
    return rows


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    t0 = time.time()
    if args.phase == "pilot":
        res = common.units(args, "explore", common.BYBIT_STREAMS, protocol["pools"]["explore"]["count"])
        sequences = {s: sorted(res[s]["datasets"], key=lambda d: d[-10:]) for s in common.BYBIT_STREAMS}
        design_name = "explore-inference"
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
    pilot = {"H14": common.pilot_record([r["effect"] for r in rows.values()], [r["odd"] for r in rows.values()],
                                        [r["even"] for r in rows.values()], caps=caps, hours=hours)}
    result = {"phase": args.phase, "sequences": sequences, "units": rows,
              "pilot": pilot if args.phase == "pilot" else None, "discount": B.DISCOUNT, "runner": info}
    return common.finish(args, out, analysis=ANALYSIS, config={"phase": args.phase, "sequences": sequences,
                                                               "discount": B.DISCOUNT}, result=result, datasets=ids)
