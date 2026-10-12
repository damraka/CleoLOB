"""Decision cases shared by the selection, abstention, OOD and conformal studies.

A case is one world (a plausible world in a registered regime, or a stressed version of it). Its *true* policy
costs are the means over the case's seeds; its *observable context* is the summary vector of a strategy-free
pre-trade window simulated with the case's first seed (what a selector could see before acting).

Registered regimes (relative to each base world): ``calm`` (all rates x0.6), ``active`` (market and cancel rates
x1.8), ``thin`` (refill depth x0.4).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..generators.scaled import ScaledGenerator
from ..market.frame import summaries
from ..selection import Case
from ..stress import apply as stress_apply
from ..worlds import World
from . import common as C

REGIMES = {"calm": [0.6] * 6, "active": [1.8, 1.8, 1.0, 1.0, 1.8, 1.8], "thin": None}
REGIME_INDEX = {"calm": 0, "active": 1, "thin": 2}
POLICIES = ("twap", "ac", "adaptive_twap", "heuristic", "pov", "mpc")
PREFIX_BINS = 1200


def regime_world(base: World, regime: str) -> World:
    if regime == "thin":
        book = base.book.replace(refill_mean=max(1.0, base.book.refill_mean * 0.4))
        gd = base.generator.to_dict()
        gd["book"] = book.to_dict()
        from ..generators.base import MarketGenerator
        return replace(base, world_id=f"{base.world_id}@thin", generator=MarketGenerator.from_dict(gd), book=book,
                       regime_id="thin")
    g = ScaledGenerator.wrap(base.generator, REGIMES[regime], base.book)
    return replace(base, world_id=f"{base.world_id}@{regime}", generator=g, regime_id=regime)


def regime_worlds(bases: list[World]) -> list[World]:
    return [regime_world(b, r) for b in bases for r in REGIMES]


def stressed_worlds(bases: list[World], scenarios: list[str]) -> list[World]:
    out = []
    for b in bases:
        for regime in REGIMES:
            w = regime_world(b, regime)
            for s in scenarios:
                sw = stress_apply(w, s)
                out.append(replace(sw, regime_id=regime))
    return out


def prefix_task(world: dict, seed: int, bins: int = PREFIX_BINS) -> dict:
    """Strategy-free pre-trade window: summary vector (observable context)."""
    w = World.from_dict(world)
    frame = w.generator.simulate(bins, seed, book=w.book)
    return {"summaries": summaries(frame).tolist(), "frame": {k: v.tolist() for k, v in frame.to_arrays().items()
                                                              if k in ("counts", "state", "valid")}}


def prefixes(args, out, worlds: list[World], seed_of: dict[str, int], tag: str) -> dict[str, dict]:
    from ..runner import Task, run_tasks
    tasks = [Task(f"{tag}|{w.world_id}", "lob.v08.studies.cases:prefix_task", {"world": w.to_dict(),
                                                                               "seed": seed_of[w.world_id]})
             for w in worlds]
    rows, _ = run_tasks(tasks, workers=args.workers, checkpoint=None)
    ids = sorted(t.task_id for t in tasks)
    return {i.split("|", 1)[1]: r for i, r in zip(ids, rows)}


def disagreement(prefix: dict, fits: list) -> float:
    """Standard deviation across development fits of the mean one-step log score of the pre-trade window."""
    from ..market.frame import EventFrame
    n = len(prefix["frame"]["counts"])
    frame = EventFrame(np.asarray(prefix["frame"]["counts"], np.int64), np.asarray(prefix["frame"]["state"], float),
                       np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros((n, 2, 10)), np.zeros((n, 2)),
                       np.zeros(n + 1), np.asarray(prefix["frame"]["valid"], bool))
    scores = [float(np.nanmean(g.loglik(frame))) for g in fits]
    return float(np.std(scores))


def build_cases(rows: list[dict], worlds: list[World], policies=POLICIES) -> list[Case]:
    means = C.world_means(rows)
    cases = []
    for w in worlds:
        costs = {p: means.get((w.world_id, p)) for p in policies}
        if any(v is None for v in costs.values()):
            continue
        cases.append(Case(w.world_id, REGIME_INDEX.get(w.regime_id, -1), costs=costs))
    return cases


def first_seed(seeds: list[int], worlds: list[World]) -> dict[str, int]:
    return {w.world_id: seeds[0] for w in worlds}
