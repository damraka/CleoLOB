"""continual-calibration-v09 (H16): forgetting under sequential training on chronologically ordered days.

For each instrument the GRU conditional-event model is trained day by day in registered chronological order with
three procedures: ``naive`` fine-tuning, ``replay`` (half of every batch from a buffer of earlier days) and ``ewc``
(elastic weight consolidation, lambda registered). Forgetting on an earlier day k = NLL_k after the last day minus NLL_k
right after day k (validation bins of day k). Units are (instrument, k) for every earlier day. Unit endpoint:
forgetting(naive) - forgetting(continual method); the pilot reports both continual methods and selects the one with
the larger mean effect for the confirmatory design (fixed there).
"""
from __future__ import annotations

import time

import numpy as np

from .. import protocol as pr
from ..representations import sequence as S
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "continual-calibration-v09"
STEPS_PER_DAY = 600
EWC_LAMBDA = 50.0


def sequence_task(days: list[str], root: str, design: str, seed: int) -> dict:
    parts = []
    for d in days:
        X, y, wid, _ = S.frame_features(common.load_cache(d, root, design, "continual sequences"))
        parts.append((X, y, wid))
    mu, sd = S.standardizer(parts[0][0])
    data = []
    for X, y, wid in parts:
        Xs = (X - mu) / sd
        idx = S.windows_of(Xs, wid, S.HISTORY)
        rng = np.random.default_rng(seed)
        val = np.sort(rng.choice(idx[wid[idx] % 2 == 1], min(5000, int((wid[idx] % 2 == 1).sum())), replace=False))
        tr = idx[wid[idx] % 2 == 0]
        data.append((Xs, y, tr, val))
    out = {}
    for method in ("naive", "replay", "ewc"):
        model = S.SeqClassifier("gru", seed=seed, steps=STEPS_PER_DAY)
        after_own, after_last = {}, {}
        anchor = None
        for k, (Xs, y, tr, val) in enumerate(data):
            replay = None
            if method == "replay" and k > 0:
                Xr = np.vstack([data[j][0] for j in range(k)])
                yr = np.concatenate([data[j][1] for j in range(k)])
                offs = np.cumsum([0] + [len(data[j][0]) for j in range(k)])
                ir = np.concatenate([data[j][2] + offs[j] for j in range(k)])
                replay = (Xr, yr, ir)
            ewc = (anchor[0], anchor[1], EWC_LAMBDA) if method == "ewc" and anchor is not None else None
            model.fit(Xs, y, tr, init=(k == 0), steps=STEPS_PER_DAY, replay=replay, ewc=ewc)
            after_own[k] = float(model.nll(Xs, y, val).mean())
            if method == "ewc":
                fisher, params = model.fisher(Xs, y, tr)
                if anchor is None:
                    anchor = (fisher, params)
                else:
                    anchor = ([0.5 * (a + b) for a, b in zip(anchor[0], fisher)], params)
        for k, (Xs, y, tr, val) in enumerate(data):
            after_last[k] = float(model.nll(Xs, y, val).mean())
        out[method] = {"after_own": after_own, "after_last": after_last,
                       "forgetting": {k: after_last[k] - after_own[k] for k in range(len(data) - 1)}}
    return {"days": days, "methods": out}


def units_of(rows: list[dict]) -> dict:
    """Per (instrument, earlier day) forgetting of every method, in chronological order."""
    units = {}
    for r in rows:
        f = r["methods"]
        # task results pass through canonical JSON, so the day-position keys arrive as strings
        for k in sorted(f["naive"]["forgetting"], key=int):
            units[r["days"][int(k)]] = {m: f[m]["forgetting"][k] for m in f}
    return units


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    t0 = time.time()
    if args.phase == "pilot":
        res = common.units(args, "explore", common.BYBIT_STREAMS, protocol["pools"]["explore"]["count"])
        sequences = {s: sorted(res[s]["datasets"], key=lambda d: d[-10:]) for s in common.BYBIT_STREAMS}
        design_name = "explore-continual"
    else:
        design = common.design_document(args)
        sequences, design_name = {s: list(v) for s, v in design["sequences"].items()}, args.design
    out = common.start(args)
    tasks = [Task(s, "lob.v09.studies.continual:sequence_task", {"days": d, "root": str(root), "design": design_name,
                                                                 "seed": args.seed}, namespace=args.namespace)
             for s, d in sequences.items()]
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args))
    hours = (time.time() - t0) / 3600 / max(sum(len(r["days"]) - 1 for r in rows), 1)
    units = units_of(rows)
    effects = {m: [u["naive"] - u[m] for u in units.values()] for m in ("replay", "ewc")}
    selected = max(effects, key=lambda m: float(np.mean(effects[m])) if effects[m] else -np.inf)
    keys = list(units)
    caps = common.caps_for(protocol, ("incremental_l2", "trades_with_side", "neural_backend"))
    odd = [effects[selected][i] for i in range(len(keys)) if i % 2 == 1]
    even = [effects[selected][i] for i in range(len(keys)) if i % 2 == 0]
    k = min(len(odd), len(even))
    pilot = {"H16": common.pilot_record(effects[selected], odd[:k], even[:k], caps=caps, hours=hours,
                                        extra={"selected_method": selected,
                                               "note": "split-half pairs alternate units (no within-unit split)"})}
    result = {"phase": args.phase, "sequences": sequences, "units": units, "effects": effects, "selected": selected,
              "pilot": pilot if args.phase == "pilot" else None, "runner": info}
    datasets = [d for v in sequences.values() for d in v]
    return common.finish(args, out, analysis=ANALYSIS, config={"phase": args.phase, "sequences": sequences,
                                                               "steps_per_day": STEPS_PER_DAY, "ewc": EWC_LAMBDA},
                         result=result, datasets=datasets)
