"""``identifiability-v3``: identifiability diagnostics per family (M15).

For every family: the local Fisher approximation at the development fit (log-scale steps, common random numbers),
first-order global indices over the family's posterior prior box, the profile-distance proxy and posterior
diagnostics from the calibration posterior, the structural flags, and a verdict per parameter
(PRACTICALLY_IDENTIFIED / WEAKLY_IDENTIFIED / NOT_IDENTIFIED). Nothing here claims structural identification.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..inference import identifiability as idf
from ..inference.abc import Prior, _seed, simulate_summary
from . import common as C

DEFAULTS = {"tiny": {"replicates": 3, "global_samples": 8, "bins": 300},
            "small": {"replicates": 8, "global_samples": 40, "bins": 600},
            "medium": {"replicates": 16, "global_samples": 160, "bins": 900}}
SUMMARIES = list(range(20))


def _call(a):
    return simulate_summary(*a)


def run(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    prun = C.input_run(args, "posterior")
    post = C.read_result(prun)
    seeds = C.seeds(args, "identifiability", 10)
    pool = ProcessPoolExecutor(args.workers) if args.workers > 1 else None
    mapper = (lambda f, a: pool.map(f, a, chunksize=2)) if pool else map
    families = {}
    try:
        for fi, (name, gd) in enumerate(sorted(post["fits"].items())):
            p = post["posteriors"][name]
            prior = Prior(**{k: tuple(v) for k, v in p["prior"].items()})
            g = C.generator_from(gd)
            theta0 = np.asarray([g.get_flat()[k] for k in prior.names])
            theta0 = np.clip(theta0, prior.low, prior.high)
            local = idf.local(g, prior, theta0, n_bins=d["bins"], replicates=d["replicates"], seed=seeds[fi],
                              summary_index=SUMMARIES, map_fn=mapper)
            rng = np.random.default_rng(seeds[fi] + 1)
            thetas = prior.sample(d["global_samples"], rng)
            args_list = [(gd, list(prior.names), list(map(float, t)), d["bins"], _seed(seeds[fi], 700, k))
                         for k, t in enumerate(thetas)]
            summ = np.asarray(list(mapper(_call, args_list)), float)
            glob = idf.global_indices(thetas, summ, list(prior.names))
            prof = idf.profile(p)
            families[name] = {"local": local, "global": glob, "profile": prof,
                              "posterior": idf.posterior_diagnostics(p), "assessment": idf.assess(name, local, p, prof)}
    finally:
        if pool:
            pool.shutdown()
    result = {"families": families, "assessment": {k: v["assessment"] for k, v in families.items()},
              "labels": {"evidence": "SYNTHETIC simulations around RETROSPECTIVE fits",
                         "validity": "local Fisher results are local and rely on a Gaussian summary approximation"}}
    return C.finish(args, out, analysis="identifiability", config={"design": d}, result=C.json_safe(result),
                    inputs=[prun.relative_to(root).as_posix()])
