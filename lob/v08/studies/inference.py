"""``inference-v08``: development fits, calibration posteriors, plausible world sets, SBI recovery (H4),
family recovery and hierarchical pooling (M12, M13, brief section 14).

Phases:

* ``posterior`` — fit the six core families on the development day (ledgered access), fit the book model,
  and run ABC-SMC per family against calibration-day summaries;
* ``worlds`` — draw the registered selection and held-out plausible world sets from a posterior run;
* ``recovery`` — synthetic ground-truth recovery for ABC-SMC (primary, H4), ABC rejection and synthetic
  likelihood (secondary), plus model-family recovery;
* ``hierarchical`` — partial pooling of per-day estimates across instrument, venue and day (retrospective)
  and a synthetic check that pooling reduces estimation error.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import math
import time

import numpy as np

from .. import realdata
from ..generators import (HawkesGenerator, MixtureGenerator, PoissonGenerator, QueueReactiveGenerator,
                          RegimeSwitchingGenerator, StateDependentGenerator)
from ..inference import abc
from ..inference.adaptive import concat
from ..inference.hierarchical import pool_tree
from ..market.frame import summaries
from ..market.sim import BookModel
from ..model_averaging import stacking_weights
from ..statistics import wilson
from ..worlds import World, WorldSet, from_posterior
from . import common as C

DEV = "deribit-eth-perp-2020-10-01"
CAL = "deribit-eth-perp-2020-11-01"
FAMILIES = {"poisson": PoissonGenerator, "hawkes": HawkesGenerator, "queue_reactive": QueueReactiveGenerator,
            "regime_switching": RegimeSwitchingGenerator, "state_dependent": StateDependentGenerator}
DEFAULTS = {
    "tiny": {"particles": 24, "generations": 2, "bins": 300, "windows": 4, "truths": 2, "draws": 2},
    "small": {"particles": 60, "generations": 3, "bins": 600, "windows": 8, "truths": 4, "draws": 3},
    "medium": {"particles": 120, "generations": 5, "bins": 900, "windows": 24, "truths": 12, "draws": 4},
}
RECOVERY_PRIORS = {
    "poisson": ({"market": (0.2, 3.0), "add": (1.0, 12.0), "cancel": (1.0, 12.0)}, {"market", "add", "cancel"}),
    "hawkes": ({"branching": (0.05, 0.85), "beta": (0.5, 10.0)}, {"beta"}),
    "queue_reactive": ({"gamma.0": (-1.5, 1.5), "gamma.2": (-1.5, 1.5), "gamma.4": (-1.5, 1.5)}, set()),
}
EVENT_GUARD = 60.0


def _pool(workers: int):
    return ProcessPoolExecutor(max_workers=workers) if workers > 1 else None


def _map(pool):
    if pool is None:
        return map
    return lambda f, a: pool.map(f, a, chunksize=4)


def run(args) -> dict:
    return {"posterior": posterior, "worlds": worlds, "recovery": recovery, "hierarchical": hierarchical}[args.phase](args)


# ============================================================================= posterior


def _fit_all(dev, book) -> dict:
    fits = {}
    for name, cls in FAMILIES.items():
        start = time.perf_counter()
        fits[name] = cls(book=book).fit(dev)
        fits[name].fit_info["seconds"] = time.perf_counter() - start
    comp = np.column_stack([fits[n].loglik(dev) for n in FAMILIES])
    ok = np.all(np.isfinite(comp), 1)
    w = stacking_weights(comp[ok])
    fits["mixture"] = MixtureGenerator.of([fits[n] for n in FAMILIES], w.tolist(), book=book)
    fits["mixture"].fit_info = {"method": "stacking weights on development log scores", "weights": dict(zip(FAMILIES, w.tolist()))}
    return fits


def _prior_for(name: str, g) -> abc.Prior:
    space = dict(g.parameter_space())
    log = set()
    if name == "poisson":
        space = {k: (g.params[k] / 3, g.params[k] * 3) for k in ("market", "add", "cancel")}
        log = set(space)
    elif name == "hawkes":
        log = {"beta"}
    return abc.Prior.from_space(space, log=log)


def posterior(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    windows = realdata.registered_windows()[: d["windows"]]
    dev_frames, dev_meta = realdata.load_frames(DEV, use="develop", design="inference-posterior", root=root,
                                                purpose="fit core families", windows=windows)
    cal_frames, cal_meta = realdata.load_frames(CAL, use="calibrate", design="inference-posterior", root=root,
                                                purpose="ABC target summaries", windows=windows)
    dev = concat(dev_frames)
    mid = float(np.median(np.concatenate([f.mid for f in dev_frames])))
    book = BookModel.fit(dev, mid_ticks=int(round(mid)))
    fits = _fit_all(dev, book)
    observed = np.mean([summaries(f) for f in cal_frames], axis=0)
    posts, timings = {}, {}
    pool = _pool(args.workers)
    try:
        for name, g in fits.items():
            prior = _prior_for(name, g)
            start = time.perf_counter()
            problem = abc.Problem(g, prior, observed, n_bins=d["bins"], seed=args.seed + 17, map_fn=_map(pool),
                                  pilot=max(12, d["particles"] // 4), reference=np.asarray(
                                      [g.get_flat()[k] for k in prior.names]))
            posts[name] = abc.smc(problem, particles=d["particles"], generations=d["generations"], quantile=0.5)
            timings[name] = time.perf_counter() - start
    finally:
        if pool:
            pool.shutdown()
    run_rel = out.relative_to(root).as_posix()
    for ds in (DEV, CAL):
        realdata.mark_evaluated(ds, use="develop" if ds == DEV else "calibrate", design="inference-posterior",
                                run=run_rel, root=root)
    result = {"development": {"meta": dev_meta, "windows": len(dev_frames)},
              "calibration": {"meta": cal_meta, "windows": len(cal_frames), "observed_summaries": observed.tolist()},
              "book": book.to_dict(), "fits": {k: v.to_dict() for k, v in fits.items()},
              "dev_loglik_per_bin": {k: v.score(dev)["loglik_per_bin"] for k, v in fits.items()},
              "posteriors": posts, "seconds": timings,
              "labels": {"evidence": "RETROSPECTIVE (consumed development and calibration days)"}}
    return C.finish(args, out, analysis="inference_posterior", config={"phase": "posterior", **d}, result=C.json_safe(result),
                    datasets=[DEV, CAL])


# ============================================================================= worlds


def worlds(args) -> dict:
    root = C.root_of(args)
    post_run = C.input_run(args, "posterior")
    post = C.read_result(post_run)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    selection, heldout = [], []
    for name, gd in post["fits"].items():
        g = C.generator_from(gd)
        p = post["posteriors"][name]
        for seed, bucket, tag in ((801, selection, "sel"), (802, heldout, "hold")):
            ws = from_posterior(g, [dict(zip(p["names"], x)) for x in p["particles"]], p["weights"], n=d["draws"],
                                seed=seed, prefix=f"{name}-{tag}")
            bucket.extend(ws.worlds)
    nominal = World("nominal-poisson", C.generator_from(post["fits"]["poisson"]),
                    C.generator_from(post["fits"]["poisson"]).book, label="CALIBRATED")
    sel = WorldSet("selection", [nominal] + selection, provenance={"posterior_run": post_run.relative_to(root).as_posix(),
                                                                   "resampling_seed": 801})
    hold = WorldSet("heldout", heldout, provenance={"posterior_run": post_run.relative_to(root).as_posix(),
                                                    "resampling_seed": 802})
    from .. import worlds as W
    W.save(sel, out / "selection.json")
    W.save(hold, out / "heldout.json")
    result = {"selection": {"worlds": len(sel), "families": sel.families(), "digest": sel.digest()},
              "heldout": {"worlds": len(hold), "families": hold.families(), "digest": hold.digest()},
              "nominal": nominal.world_id, "draws_per_family": d["draws"]}
    return C.finish(args, out, analysis="inference_worlds", config={"phase": "worlds", **d}, result=result,
                    inputs=[post_run.relative_to(root).as_posix()])


# ============================================================================= recovery (H4)


def _truths(name: str, n: int, seed: int) -> list[dict]:
    space, log = RECOVERY_PRIORS[name]
    prior = abc.Prior.from_space(space, log=log)
    rng = np.random.default_rng(seed)
    out, attempts = [], 0
    base = {"poisson": PoissonGenerator, "hawkes": HawkesGenerator, "queue_reactive": QueueReactiveGenerator}[name]()
    while len(out) < n and attempts < 20 * n:
        attempts += 1
        theta = dict(zip(prior.names, map(float, prior.sample(1, rng)[0])))
        g = base.with_params(theta)
        frame = g.simulate(200, seed=int(rng.integers(1 << 30)), warmup_bins=50)
        if frame.counts.sum(1).mean() > EVENT_GUARD:
            continue   # registered event guard: implausible truths are rejected and redrawn (seeded, recorded)
        out.append(theta)
    return out, attempts


def recovery(args) -> dict:
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    seeds = C.seeds(args, "inference.recovery", 10_000)
    pool = _pool(args.workers)
    rows, mech_rows, attempts = [], [], {}
    k = 0
    try:
        for name in RECOVERY_PRIORS:
            space, log = RECOVERY_PRIORS[name]
            prior = abc.Prior.from_space(space, log=log)
            truths, attempts[name] = _truths(name, d["truths"], seeds[k])
            k += 1
            base = {"poisson": PoissonGenerator, "hawkes": HawkesGenerator, "queue_reactive": QueueReactiveGenerator}[name]()
            for i, theta in enumerate(truths):
                g = base.with_params(theta)
                data = g.simulate(d["bins"], seed=seeds[k])
                k += 1
                problem = abc.Problem(base, prior, summaries(data), n_bins=d["bins"], seed=seeds[k], map_fn=_map(pool),
                                      pilot=max(12, d["particles"] // 4))
                k += 1
                post = abc.smc(problem, particles=d["particles"], generations=d["generations"], quantile=0.5)
                rows.append(_recovery_row(name, i, theta, post, prior, "abc_smc"))
                if i < max(1, d["truths"] // 3):
                    rej = abc.rejection(problem, draws=d["particles"] * 4, accept=0.25)
                    mech_rows.append(_recovery_row(name, i, theta, rej, prior, "abc_rejection"))
                    if name == "poisson":
                        mle = PoissonGenerator().fit(data)
                        n = data.valid.sum()
                        mech = {"family": name, "truth_index": i, "method": "mle_analytic", "parameters": {}}
                        for p in ("market", "add", "cancel"):
                            se = math.sqrt(mle.params[p] / n)
                            lo, hi = mle.params[p] - 1.645 * se, mle.params[p] + 1.645 * se
                            mech["parameters"][p] = {"truth": theta[p], "lo": lo, "hi": hi, "covered": lo <= theta[p] <= hi}
                        mech_rows.append(mech)
    finally:
        if pool:
            pool.shutdown()
    summary = _recovery_summary(rows)
    family = family_recovery(seeds[k:k + 40], d)
    result = {"rows": rows, "secondary_mechanisms": mech_rows, "summary": summary, "truth_attempts": attempts,
              "family_recovery": family, "registered_rule": "H4: coverage >= 0.80 with Wilson lower >= 0.60 and median "
                                                             "relative error <= 0.15 for every parameter"}
    return C.finish(args, out, analysis="inference_recovery", config={"phase": "recovery", **d,
                                                                       "priors": {k: v[0] for k, v in RECOVERY_PRIORS.items()}},
                    result=C.json_safe(result), seed_info={"namespace": args.namespace, "study": "inference.recovery"})


def _recovery_row(name: str, i: int, theta: dict, post: dict, prior: abc.Prior, method: str) -> dict:
    row = {"family": name, "truth_index": i, "method": method, "parameters": {}}
    particles = np.asarray(post["particles"], float)
    w = np.asarray(post["weights"], float)
    for j, p in enumerate(post["names"]):
        lo, hi = abc.credible_interval(post, p, 0.9)
        med = post["summary"]["parameters"][p]["q50"]
        u = prior.to_unit(np.asarray([[theta[q] for q in prior.names]]))[0]
        um = prior.to_unit(np.asarray([[post["summary"]["parameters"][q]["q50"] for q in prior.names]]))[0]
        rank = float(np.sum(w[particles[:, j] < theta[p]]))
        row["parameters"][p] = {"truth": theta[p], "median": med, "lo90": lo, "hi90": hi,
                                "covered": bool(lo <= theta[p] <= hi), "relative_error": float(abs(um[j] - u[j])),
                                "sbc_rank": rank, "contraction": post["summary"]["parameters"][p]["contraction"]}
    row["simulations"] = post["simulations"]
    return row


def _recovery_summary(rows: list[dict]) -> dict:
    out = {}
    for row in rows:
        for p, v in row["parameters"].items():
            out.setdefault(row["family"], {}).setdefault(p, []).append(v)
    summary = {}
    for fam, params in out.items():
        summary[fam] = {}
        for p, vals in params.items():
            k = sum(v["covered"] for v in vals)
            lo, hi = wilson(k, len(vals))
            ranks = np.asarray([v["sbc_rank"] for v in vals])
            hist = np.histogram(ranks, bins=4, range=(0, 1))[0]
            summary[fam][p] = {"coverage": k / len(vals), "wilson": [lo, hi], "n": len(vals),
                               "median_relative_error": float(np.median([v["relative_error"] for v in vals])),
                               "mean_contraction": float(np.mean([v["contraction"] for v in vals])),
                               "sbc_rank_histogram": hist.tolist()}
    return summary


def family_recovery(seeds: list[int], d: dict) -> dict:
    """Fit every family on data from each truth family; select by held-out (second half) log score."""
    truths = {"poisson": PoissonGenerator(), "hawkes": HawkesGenerator(), "queue_reactive": QueueReactiveGenerator(),
              "regime_switching": RegimeSwitchingGenerator()}
    reps = max(2, min(len(seeds) // len(truths), d.get("truths", 4)))
    confusion = {t: {f: 0 for f in FAMILIES} for t in truths}
    k = 0
    for tname, tg in truths.items():
        for r in range(reps):
            frame = tg.simulate(2 * d["bins"], seed=seeds[k % len(seeds)] + r)
            k += 1
            train, test = frame.slice(0, d["bins"]), frame.slice(d["bins"], 2 * d["bins"])
            scores = {}
            for fname, cls in FAMILIES.items():
                try:
                    scores[fname] = float(np.nanmean(cls().fit(train).loglik(test)))
                except (ValueError, np.linalg.LinAlgError):
                    scores[fname] = -math.inf
            confusion[tname][max(scores, key=scores.get)] += 1
    correct = sum(confusion[t][t] for t in truths)
    total = sum(sum(v.values()) for v in confusion.values())
    return {"confusion": confusion, "accuracy": correct / total, "replicates_per_truth": reps,
            "selection": "held-out log score (second half of each sequence)"}


# ============================================================================= hierarchical (M13)


HIER_DATASETS = [("deribit-eth-perp-2020-04-01", "ETH", "deribit"), ("deribit-eth-perp-2020-07-01", "ETH", "deribit"),
                 ("deribit-eth-perp-2020-09-01", "ETH", "deribit"), ("deribit-eth-perp-2021-01-01", "ETH", "deribit"),
                 ("deribit-btc-perp-2020-07-01", "BTC", "deribit"), ("deribit-btc-perp-2020-09-01", "BTC", "deribit"),
                 ("deribit-btc-perp-2020-11-01", "BTC", "deribit"), ("bitmex-xbtusd-2020-11-01", "BTC", "bitmex")]


def hierarchical(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    windows = realdata.registered_windows()[: d["windows"]]
    leaves, used = [], []
    for ds, instrument, venue in HIER_DATASETS:
        try:
            frames, meta = realdata.load_frames(ds, use="retrospective", design="inference-hierarchical", root=root,
                                                purpose="per-day Poisson rate estimates for pooling", windows=windows)
        except realdata.DataNotAvailable as exc:
            leaves.append({"dataset": ds, "status": "NOT_AVAILABLE", "reason": str(exc)})
            continue
        used.append(ds)
        frame = concat(frames)
        fit = PoissonGenerator().fit(frame)
        n = int(frame.valid.sum())
        mean = {p: math.log(max(fit.params[p], 1e-9)) for p in ("market", "add", "cancel")}
        var = {p: 1.0 / max(fit.params[p] * n, 1e-9) for p in mean}   # delta-method variance of log rate
        leaves.append({"dataset": ds, "global": "all", "instrument": instrument, "venue": venue, "day": ds,
                       "mean": mean, "var": var, "contraction": {p: 1.0 for p in mean}, "bins": n})
    tree = pool_tree([x for x in leaves if "mean" in x], ("instrument", "venue", "day"))
    synthetic = _synthetic_pooling_check(C.seeds(args, "inference.hierarchical", 200))
    run_rel = out.relative_to(root).as_posix()
    for ds in used:
        realdata.mark_evaluated(ds, use="retrospective", design="inference-hierarchical", run=run_rel, root=root)
    result = {"tree": tree, "leaves": leaves, "synthetic_check": synthetic,
              "labels": {"evidence": "RETROSPECTIVE (consumed days); EXPLORATORY",
                         "identification": "per-day log rates are identified by bin counts; between-day variance "
                                           "is estimated by DerSimonian-Laird and can be zero"}}
    return C.finish(args, out, analysis="inference_hierarchical", config={"phase": "hierarchical", **d},
                    result=C.json_safe(result), datasets=used)


def _synthetic_pooling_check(seeds: list[int]) -> dict:
    """Known hierarchy of log market rates; compare no-pooling vs partial-pooling error."""
    rng = np.random.default_rng(seeds[0])
    errors = {"none": [], "partial": []}
    for rep in range(20):
        mu, tau = math.log(1.0), 0.3
        groups = []
        for g in range(6):
            truth = rng.normal(mu, tau)
            n = int(rng.integers(200, 2000))
            counts = rng.poisson(math.exp(truth), n)
            est = math.log(max(counts.mean(), 1e-6))
            groups.append({"global": "all", "day": f"d{g}", "mean": {"m": est}, "var": {"m": 1 / max(counts.sum(), 1)},
                           "truth": truth})
        tree = pool_tree(groups, ("day",))
        for g in groups:
            shrunk = tree["pooling"]["groups"][g["day"]]["shrunk_mean"]["m"]
            errors["none"].append((g["mean"]["m"] - g["truth"]) ** 2)
            errors["partial"].append((shrunk - g["truth"]) ** 2)
    return {"mse_no_pooling": float(np.mean(errors["none"])), "mse_partial_pooling": float(np.mean(errors["partial"])),
            "replicates": 20, "label": "SYNTHETIC"}
