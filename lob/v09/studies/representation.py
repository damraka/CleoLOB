"""representation-transfer-v09 (M38-M43; H35): learned vs hand-engineered representations under instrument transfer.

A contrastive GRU encoder (InfoNCE, registered augmentations) is trained on the BTCUSDT training days; linear probes
for the 10 s-ahead mid direction (down / flat / up) are trained on BTCUSDT training days on (a) hand-engineered window
summaries and (b) the learned embedding, then evaluated on each ETHUSDT day (the transfer target). Unit endpoint:
log loss(hand) - log loss(learned), nats per sample (positive favours the learned representation). Split-half: odd vs
even windows of the ETH day. Pilot: train on BTC explore days, evaluate on ETH explore days. Confirm: the same
training (explore BTC days) evaluated on ETH confirm days.
"""
from __future__ import annotations

import time

import numpy as np

from .. import protocol as pr
from ..representations import sequence as S
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "representation-transfer-v09"
MAX_TRAIN = 60_000


def _day(d, root, design):
    X, y, wid, mid = S.frame_features(common.load_cache(d, root, design, "representations"))
    idx, lab = S.direction_labels(mid, wid)
    return X, wid, idx, lab


def evaluate_task(train: list[str], tests: list[str], root: str, design: str, seed: int) -> dict:
    parts = [_day(d, root, design) for d in train]
    mu, sd = S.standardizer(np.vstack([p[0] for p in parts]))
    Xs = [(p[0] - mu) / sd for p in parts]
    rng = np.random.default_rng(seed)
    allX = np.vstack(Xs)
    offs = np.cumsum([0] + [len(x) for x in Xs])
    all_idx = np.concatenate([p[2] + offs[k] for k, p in enumerate(parts)])
    all_lab = np.concatenate([p[3] for p in parts])
    pick = np.sort(rng.choice(len(all_idx), min(MAX_TRAIN, len(all_idx)), replace=False))
    enc = S.ContrastiveEncoder(seed=seed).fit(allX, all_idx[pick])
    Htr = S.hand_features(allX, all_idx[pick])
    Etr = enc.embed(allX, all_idx[pick])
    ytr = all_lab[pick]
    out = {}
    for d in tests:
        X, wid, idx, lab = _day(d, root, design)
        Xs_t = (X - mu) / sd
        a = S.probe_nll(Htr, ytr, S.hand_features(Xs_t, idx), lab)
        b = S.probe_nll(Etr, ytr, enc.embed(Xs_t, idx), lab)
        odd = wid[idx] % 2 == 1
        out[d] = {"logloss_hand": float(a.mean()), "logloss_learned": float(b.mean()),
                  "effect": float(a.mean() - b.mean()), "odd": float(a[odd].mean() - b[odd].mean()),
                  "even": float(a[~odd].mean() - b[~odd].mean()), "samples": int(len(idx))}
    return out


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    t0 = time.time()
    if args.phase == "pilot":
        res = common.units(args, "explore", common.BYBIT_STREAMS, protocol["pools"]["explore"]["count"])
        train, tests, design_name = res[common.BYBIT_STREAMS[0]]["datasets"], res[common.BYBIT_STREAMS[1]]["datasets"], \
            "explore-representation"
    else:
        design = common.design_document(args)
        train, tests, design_name = list(design["train_datasets"]), list(design["test_datasets"]), args.design
    out = common.start(args)
    chunks = [tests]                                   # one task: the encoder is trained once (seeded)
    tasks = [Task(f"chunk{i}", "lob.v09.studies.representation:evaluate_task",
                  {"train": train, "tests": c, "root": str(root), "design": design_name, "seed": args.seed},
                  namespace=args.namespace) for i, c in enumerate(chunks)]
    parts, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args))
    rows = {k: v for p in parts for k, v in p.items()}
    hours = (time.time() - t0) / 3600 / max(len(rows), 1)
    caps = common.caps_for(protocol, ("incremental_l2", "trades_with_side", "neural_backend"))
    pilot = {"H35": common.pilot_record([r["effect"] for r in rows.values()], [r["odd"] for r in rows.values()],
                                        [r["even"] for r in rows.values()], caps=caps, hours=hours)}
    result = {"phase": args.phase, "train": train, "units": rows, "pilot": pilot if args.phase == "pilot" else None,
              "runner": info, "note": "the encoder and probes are refit identically in each chunk (seeded)"}
    return common.finish(args, out, analysis=ANALYSIS, config={"phase": args.phase, "train": train, "tests": tests,
                                                               "seed": args.seed, "max_train": MAX_TRAIN},
                         result=result, datasets=train + tests)
