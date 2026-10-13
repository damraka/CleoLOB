"""sequence-generator-v09 (M46; H3): conditional event realism of sequence models vs the classical model.

Pilot: leave-one-explore-day-out. Inputs are standardized with training-day statistics. The classical model
(multinomial logistic regression) and the three sequence models (GRU, LSTM, causal Transformer) are fitted on a seeded
subsample of training sequences (registered cap); the sequence model with the lowest NLL on an inner validation split
(last quarter of training days) is the selected one. Unit endpoint: NLL(classical) - NLL(selected sequence model) on
the held-out day's registered windows (nats per bin, positive favours sequence models). Split-half: odd vs even
windows.
"""
from __future__ import annotations

import time

import numpy as np

from .. import protocol as pr
from ..representations import sequence as S
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "sequence-generator-v09"
MAX_TRAIN = 150_000


def _data(ids, root, design):
    out = []
    for d in ids:
        X, y, wid, _ = S.frame_features(common.load_cache(d, root, design, "bin sequences"))
        out.append((X, y, wid))
    return out


def _stack(parts, mu, sd, cap, seed):
    X = np.vstack([(p[0] - mu) / sd for p in parts])
    y = np.concatenate([p[1] for p in parts])
    wid = np.concatenate([p[2] + 100 * k for k, p in enumerate(parts)])
    idx = S.windows_of(X, wid, S.HISTORY)
    if len(idx) > cap:
        idx = np.sort(np.random.default_rng(seed).choice(idx, cap, replace=False))
    return X, y, wid, idx


def fold_task(held: str, train: list[str], root: str, design: str, seed: int) -> dict:
    parts = _data(train, root, design)
    mu, sd = S.standardizer(np.vstack([p[0] for p in parts]))
    n_inner = max(1, int(round(0.75 * len(parts))))
    Xi, yi, wi, ii = _stack(parts[:n_inner], mu, sd, MAX_TRAIN, seed)
    Xv, yv, wv, iv = _stack(parts[n_inner:], mu, sd, 40_000, seed + 1)
    val = {kind: float(S.SeqClassifier(kind, seed=seed).fit(Xi, yi, ii).nll(Xv, yv, iv).mean())
           for kind in ("gru", "lstm", "transformer")}
    best = min(val, key=val.get)
    X, y, w, idx = _stack(parts, mu, sd, MAX_TRAIN, seed)
    seq = S.SeqClassifier(best, seed=seed).fit(X, y, idx)
    Z = S.MultinomialLR.design(X, w)
    lr = S.MultinomialLR().fit(Z[idx], y[idx])
    hX, hy, hw = _data([held], root, design)[0]
    hX = (hX - mu) / sd
    hidx = S.windows_of(hX, hw, S.HISTORY)
    a = lr.nll(S.MultinomialLR.design(hX, hw)[hidx], hy[hidx])
    b = seq.nll(hX, hy, hidx)
    odd = hw[hidx] % 2 == 1
    return {"held": held, "selected": best, "val_nll": val, "nll_classical": float(a.mean()),
            "nll_sequence": float(b.mean()), "effect": float(a.mean() - b.mean()),
            "odd": float(a[odd].mean() - b[odd].mean()), "even": float(a[~odd].mean() - b[~odd].mean()),
            "bins": int(len(hidx))}


def run(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    t0 = time.time()
    if args.phase == "pilot":
        units = common.explore_units(args, protocol)
        folds, design_name = [(d, [x for x in units if x != d]) for d in units], "explore-sequence"
    else:
        design = common.design_document(args)
        units, design_name = list(design["test_datasets"]), args.design
        folds = [(d, list(design["train_datasets"])) for d in units]
    out = common.start(args)
    tasks = [Task(h, "lob.v09.studies.sequence:fold_task", {"held": h, "train": tr, "root": str(root),
                                                            "design": design_name, "seed": args.seed},
                  namespace=args.namespace) for h, tr in folds]
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root / (args.out + ".checkpoint"), design={"folds": folds}, inputs=units)
    hours = (time.time() - t0) / 3600 / max(len(rows), 1)
    caps = common.caps_for(protocol, ("incremental_l2", "trades_with_side", "neural_backend"))
    pilot = {"H3": common.pilot_record([r["effect"] for r in rows], [r["odd"] for r in rows],
                                       [r["even"] for r in rows], caps=caps, hours=hours)}
    result = {"phase": args.phase, "units": {r["held"]: r for r in rows},
              "pilot": pilot if args.phase == "pilot" else None, "runner": info}
    return common.finish(args, out, analysis=ANALYSIS, config={"phase": args.phase, "units": units,
                                                               "seed": args.seed, "max_train": MAX_TRAIN},
                         result=result, datasets=units)
