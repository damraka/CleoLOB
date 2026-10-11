"""Sealed synthetic multi-axis holdout for H21.

Only reached when H8 or H9 carries an ESTABLISHED conclusion. The registered holdout worlds are an unseen family
(stylized ecology with endogenous response and a regime transition) with parameters drawn from the registered box
using ``holdout``-namespace seeds; the namespace is opened exactly once, through a design sealed here.
Each carried conclusion is re-tested with the registered primary mandate: same sign, interval excluding zero at
alpha 0.05 and |effect| >= 0.25 bps.
"""
from __future__ import annotations

import numpy as np

from .. import protocol as pr
from ..generators.ecology import AGENTS, EcologyGenerator
from ..generators.endogenous import EndogenousGenerator
from ..generators.families import MixtureGenerator
from ..generators.scaled import ScaledGenerator
from ..statistics import cvar, hierarchical_paired
from ..worlds import World
from . import common as C
from .mpc import paired_matrix

DESIGN = "synthetic-holdout-h21"
WORLDS, SEEDS = 12, 32


def holdout_worlds(book, seeds: list[int]) -> list[World]:
    out = []
    for k in range(WORLDS):
        rng = np.random.default_rng(seeds[k])
        base = EcologyGenerator(book=book)
        agents = {a: {**p, "intensity": float(p["intensity"] * rng.uniform(0.5, 2.0))}
                  for a, p in base.params["agents"].items() if a in AGENTS}
        eco = EcologyGenerator({"agents": agents}, book).fit(EcologyGenerator(book=book).simulate(600, seeds[k]))
        endo = EndogenousGenerator.wrap(eco, depletion=float(rng.uniform(0.5, 2.0)), replenish=float(rng.uniform(0, 1)))
        burst = ScaledGenerator.wrap(endo, [2.0, 2.0, 1.0, 1.0, 1.5, 1.5], book)
        g = MixtureGenerator.of([endo, burst], [0.5, 0.5], switch_prob=float(rng.uniform(0.002, 0.02)), book=book)
        out.append(World(f"holdout-{k:02d}", g, book, label="HOLDOUT"))
    return out


def evaluate_holdout(args, root, carried: list[dict], out, row) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), protocol)
    if DESIGN not in state.designs:
        pr.seal_design(root, DESIGN, {"worlds": WORLDS, "seeds": SEEDS, "carried": carried,
                                      "box": protocol["worlds"]["synthetic_holdout"]["box"]},
                       namespaces=["holdout"], reason="seal the synthetic holdout reader (H21)")
    pr.open_namespace(root, "holdout", design=DESIGN, reason="H21 opens the sealed synthetic holdout once")
    seeds = pr.seeds("holdout", "h21.worlds", WORLDS, root=root, design=DESIGN)
    ep_seeds = pr.seeds("holdout", "h21.episodes", SEEDS, root=root, design=DESIGN)
    wrun = C.input_run(args, "worlds")
    book = C.load_world_set(wrun, "selection").worlds[0].book
    worlds = holdout_worlds(book, seeds)
    registered = C.registered_policies(root)
    m = C.mandate(root)
    ids = sorted(w.world_id for w in worlds)
    results, survived = {}, 0
    for c in carried:
        if c["kind"] == "member_vs_twap":
            names = [("twap", {}), (c["policy"], registered.get(c["policy"], {}))]
        else:
            names = [(c["nominal_choice"], {}), (c["cvar_choice"], {})]
        args.namespace, args.design = "holdout", DESIGN
        rows = C.evaluate(args, out, worlds, names, m, ep_seeds, tag=f"h21-{c['source']}-{c.get('member', 'h8')}")
        a, b = (paired_matrix(rows, n, ids, ep_seeds) for n, _ in names)
        if a is None or b is None:
            results[c["source"]] = {"status": "INVALID"}
            continue
        s = hierarchical_paired(a, b, lambda x: cvar(x, 0.8), alpha=0.05, n_boot=2000, seed=0)
        ok = s["ci_low"] > 0 and s["estimate"] >= 0.25
        survived += int(ok)
        results[f"{c['source']}:{c.get('member', 'robust')}"] = {"stats": s, "survived": ok}
    return row("H21", "ESTABLISHED" if survived else "NOT_ESTABLISHED", estimate=float(survived), threshold=1.0,
               n=len(carried), alpha=0.05, qualifiers=("SYNTHETIC_HOLDOUT",), members=results,
               reason="estimate = number of carried conclusions surviving the sealed synthetic holdout")
