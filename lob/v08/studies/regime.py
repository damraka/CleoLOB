"""``regime-study-v08``: regime detection framework (M5).

Synthetic: regime-switching truths with recorded regime paths; every registered labeler is fitted on the first
half (development blocks) and scored on the second half against the true block regimes (adjusted Rand index) and
by the generic diagnostics (persistence, separation, next-block R^2, shuffled-label negative control, shift).
Real (RETROSPECTIVE, EXPLORATORY): labelers fitted on development-day blocks only and applied unchanged to the
calibration, selection and other consumed days.
"""
from __future__ import annotations

import numpy as np

from .. import realdata
from ..generators import RegimeSwitchingGenerator
from ..generators.base import MarketGenerator
from ..market.sim import MarketSim
from ..regimes import METHODS, Labeler, block_features, compare
from ..runner import Task, run_tasks
from . import common as C

DEFAULTS = {"tiny": {"seeds": 2, "blocks": 40, "real_days": 3, "real_windows": 2},
            "small": {"seeds": 6, "blocks": 120, "real_days": 6, "real_windows": 4},
            "medium": {"seeds": 16, "blocks": 240, "real_days": 10, "real_windows": 8}}
BLOCK = 100


def adjusted_rand(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return 0.0
    from math import comb
    ua, ub = np.unique(a), np.unique(b)
    table = np.asarray([[np.sum((a == x) & (b == y)) for y in ub] for x in ua])
    s = sum(comb(int(n), 2) for n in table.ravel())
    sa = sum(comb(int(n), 2) for n in table.sum(1))
    sb = sum(comb(int(n), 2) for n in table.sum(0))
    total = comb(len(a), 2)
    expected = sa * sb / total if total else 0.0
    maximum = 0.5 * (sa + sb)
    return float((s - expected) / (maximum - expected)) if maximum != expected else 0.0


def synthetic_task(base: dict, ratio: float, duration_s: float, blocks: int, seed: int) -> dict:
    g0 = MarketGenerator.from_dict(base)
    mu = np.asarray(g0._mu) if hasattr(g0, "_mu") else np.full(6, 1.0)
    p = 0.1 / duration_s
    g = RegimeSwitchingGenerator({"rates": [mu.tolist(), (mu * ratio).tolist()], "trans": [[1 - p, p], [p, 1 - p]],
                                  "init": [0.5, 0.5]}, g0.book)
    sim = MarketSim(g, g0.book, seed)
    path = []
    for _ in range(blocks * BLOCK):
        sim.run_bin()
        path.append(sim.mstate["regime"])
    frame = sim.frame()
    feats = block_features(frame, BLOCK)
    truth = np.asarray([int(round(np.mean(path[i * BLOCK:(i + 1) * BLOCK]))) for i in range(len(feats))])
    half = len(feats) // 2
    res = compare(feats[:half], feats[half:], seed=seed)
    for m in METHODS:
        lab = Labeler(m, seed=seed).fit(feats[:half]).label(feats[half:])
        res[m]["ari_vs_truth"] = adjusted_rand(lab, truth[half:])
        res[m].pop("labeler")
    return res


def run(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    prun = C.input_run(args, "posterior")
    base = C.read_result(prun)["fits"]["poisson"]
    truths = [(2.0, 60.0), (4.0, 60.0), (2.0, 300.0), (4.0, 300.0)]
    seeds = C.seeds(args, "regime", d["seeds"] * len(truths))
    tasks = [Task(f"{r:g}-{t:g}|{i:03d}", "lob.v08.studies.regime:synthetic_task",
                  {"base": base, "ratio": r, "duration_s": t, "blocks": d["blocks"], "seed": seeds[j * d["seeds"] + i]})
             for j, (r, t) in enumerate(truths) for i in range(d["seeds"])]
    rows, _ = run_tasks(tasks, workers=args.workers, checkpoint=out.parent / (out.name + ".ckpt") / "synthetic",
                        binding={"study": "regime"})
    keys = ("ari_vs_truth", "persistence_blocks", "separation_activity", "next_block_r2_activity",
            "negative_control_shuffled_r2_activity", "distribution_shift_tv")
    synthetic = {m: {k: float(np.mean([r[m][k] for r in rows])) for k in keys} for m in METHODS}
    from .adaptive import CHRONOLOGY, USE
    order = ["deribit-eth-perp-2020-10-01"] + [x for x in CHRONOLOGY if x != "deribit-eth-perp-2020-10-01"]
    days = order[: d["real_days"]]
    windows = realdata.registered_windows()[: d["real_windows"]]
    feats = {}
    for ds in days:
        frames, _ = realdata.load_frames(ds, use=USE.get(ds, "retrospective"), design="regime-real", root=root,
                                         purpose="regime labeling (labelers fitted on the development day only)",
                                         windows=windows)
        feats[ds] = np.vstack([block_features(f, 300) for f in frames])
    dev = feats[days[0]]
    real = {ds: compare(dev, feats[ds], seed=args.seed) for ds in days[1:]}
    for ds in days:
        realdata.mark_evaluated(ds, use=USE.get(ds, "retrospective"), design="regime-real",
                                run=out.relative_to(root).as_posix(), root=root)
    result = {"synthetic": synthetic, "real": {ds: {m: {k: v for k, v in r[m].items() if k != "labeler"} for m in METHODS}
                                               for ds, r in real.items()},
              "real_development_day": days[0], "labels": {"synthetic": "SYNTHETIC", "real": "RETROSPECTIVE; EXPLORATORY"}}
    return C.finish(args, out, analysis="regime_study", config={"design": d, "truths": truths},
                    result=C.json_safe(result), datasets=days, inputs=[prun.relative_to(root).as_posix()])
