"""``adaptive-calibration-v08``: frozen versus adaptive calibration (M14) - H1, H2, H3, H18.

* ``drift`` (H1, H18): the four registered drift scenarios; 12 windows of 600 bins; template ``state_dependent``;
  schemes frozen / rolling / expanding / forgetting / change-triggered. H1 = relative loss reduction of rolling
  versus frozen over windows >= 2; H18(b) the same over windows >= 6; H18(a) a half-life fit to the mean
  degradation curve of the frozen window-1 fit in the two gradual scenarios.
* ``regime`` (H2): four persistent regime-switching truths; template ``poisson``; global (frozen) versus
  regime-conditioned fits with the registered threshold labeler fitted on the initial windows.
* ``switch`` (H3): sequences whose generating family changes at windows 4 and 8; components fitted on windows 0-3;
  best single family chosen on windows 4-5; dynamic model averaging (forgetting 0.99) versus that family on
  windows 6-11.
* ``real`` (secondary, RETROSPECTIVE): the same comparisons on the consumed ETH chronology.
"""
from __future__ import annotations

import numpy as np

from .. import realdata
from ..drift import half_life
from ..generators import (HawkesGenerator, MixtureGenerator, PoissonGenerator, QueueReactiveGenerator,
                          RegimeSwitchingGenerator, StateDependentGenerator)
from ..generators.base import MarketGenerator
from ..inference.adaptive import compare, loss
from ..model_averaging import dynamic
from ..regimes import Labeler, block_features
from ..runner import Task, run_tasks
from . import common as C

DEFAULTS = {"tiny": {"seeds": 2, "windows": 6, "bins": 300, "orders": 2, "real_days": 3, "real_windows": 2},
            "small": {"seeds": 6, "windows": 12, "bins": 600, "orders": 6, "real_days": 10, "real_windows": 4},
            "medium": {"seeds": 24, "windows": 12, "bins": 600, "orders": 6, "real_days": 10, "real_windows": 4}}
SCENARIOS = ("step_rate", "branching_change", "gradual_rates", "gradual_mix")
ORDERS = [("poisson", "hawkes", "queue_reactive"), ("hawkes", "queue_reactive", "regime_switching"),
          ("queue_reactive", "regime_switching", "poisson"), ("regime_switching", "poisson", "hawkes"),
          ("poisson", "regime_switching", "queue_reactive"), ("hawkes", "poisson", "regime_switching")]
CHANGE = 6
FAM = {"poisson": PoissonGenerator, "hawkes": HawkesGenerator, "queue_reactive": QueueReactiveGenerator,
       "regime_switching": RegimeSwitchingGenerator, "state_dependent": StateDependentGenerator}


def _scaled_poisson(base: dict, factor_market: float, factor_add: float, factor_cancel: float) -> MarketGenerator:
    g = MarketGenerator.from_dict(base)
    p = dict(g.params)
    p.update(market=p["market"] * factor_market, add=p["add"] * factor_add, cancel=p["cancel"] * factor_cancel)
    return PoissonGenerator(p, g.book)


def scenario_windows(name: str, fits: dict, n: int, bins: int, seed: int) -> list:
    rng_seeds = [int(x) for x in np.random.SeedSequence([seed, 11]).generate_state(n)]
    out = []
    for k in range(n):
        if name == "step_rate":
            f = 2.0 if k >= CHANGE else 1.0
            g = _scaled_poisson(fits["poisson"], f, f, 1.0)
        elif name == "branching_change":
            g = MarketGenerator.from_dict(fits["hawkes"]).with_params({"branching": 0.7 if k >= CHANGE else 0.3})
        elif name == "gradual_rates":
            f = 1.0 + 0.08 * k
            g = _scaled_poisson(fits["poisson"], f, f, f)
        else:
            calm, active = _scaled_poisson(fits["poisson"], 0.6, 0.6, 0.6), _scaled_poisson(fits["poisson"], 1.8, 1.8, 1.8)
            w = 0.1 + 0.8 * k / max(n - 1, 1)
            g = MixtureGenerator.of([calm, active], [1 - w, w], switch_prob=0.01, book=calm.book)
        out.append(g.simulate(bins, rng_seeds[k], warmup_bins=100))
    return out


def drift_task(scenario: str, fits: dict, n: int, bins: int, seed: int) -> dict:
    windows = scenario_windows(scenario, fits, n, bins, seed)
    book = MarketGenerator.from_dict(fits["poisson"]).book
    template = StateDependentGenerator(book=book)
    res = compare(template, windows, initial=2, window=3, rho=0.3,
                  schemes=("frozen", "rolling", "expanding", "forgetting", "change_triggered"))
    frozen1 = template.fit(windows[1])
    baseline = loss(frozen1, windows[1])
    curve = [loss(frozen1, windows[1 + h]) for h in range(1, n - 1)]
    return {"losses": res["losses"], "change_refits": res["change_refits"], "baseline": baseline, "curve": curve}


def regime_task(duration_s: float, ratio: float, fits: dict, n: int, bins: int, seed: int) -> dict:
    base = MarketGenerator.from_dict(fits["poisson"])
    mu = np.asarray(base._mu)
    p = 0.1 / duration_s
    g = RegimeSwitchingGenerator({"rates": [mu.tolist(), (mu * ratio).tolist()], "trans": [[1 - p, p], [p, 1 - p]],
                                  "init": [0.5, 0.5]}, base.book)
    frame = g.simulate(n * bins, seed, warmup_bins=200)
    windows = frame.windows(bins)
    block = max(20, bins // 6)        # six labelled blocks per window (registered labeler needs >= 6 blocks)
    labeler = Labeler("threshold").fit(np.vstack([block_features(w, block=block) for w in windows[:2]]))
    res = compare(PoissonGenerator(book=base.book), windows, initial=2, window=3, labeler=labeler,
                  schemes=("frozen", "regime"), block=block)
    return {"losses": res["losses"], "regime_models": res["regime_models"]}


def switch_task(order: list, fits: dict, bins: int, seed: int) -> dict:
    gens = [MarketGenerator.from_dict(fits[f]) for f in order]
    seeds = [int(x) for x in np.random.SeedSequence([seed, 13]).generate_state(12)]
    windows = [gens[k // 4].simulate(bins, seeds[k], warmup_bins=100) for k in range(12)]
    from ..inference.adaptive import concat
    train = concat(windows[:4])
    names = ["poisson", "hawkes", "queue_reactive", "regime_switching", "state_dependent"]
    comps = [FAM[n](book=gens[0].book).fit(train) for n in names]
    select = concat(windows[4:6])
    sel_scores = {n: float(np.nanmean(c.loglik(select))) for n, c in zip(names, comps)}
    best = max(sel_scores, key=sel_scores.get)
    evaluation = concat(windows[4:])
    ll = np.column_stack([c.loglik(evaluation) for c in comps])
    dma = dynamic(ll, alpha=0.99)
    eval_mask = np.zeros(len(evaluation), bool)
    eval_mask[2 * bins:] = True
    ok = eval_mask & np.all(np.isfinite(ll), 1)
    best_loss = float(-ll[ok, names.index(best)].mean())
    dma_loss = float(-np.asarray(dma["scores"])[ok].mean())
    return {"best_single": best, "selection_scores": sel_scores, "best_loss": best_loss, "dma_loss": dma_loss,
            "relative_improvement": (best_loss - dma_loss) / abs(best_loss),
            "final_weights": dict(zip(names, dma["path"][-1].tolist()))}


def _rel(losses: dict, a: str, b: str, start: int = 0) -> float:
    x, y = np.asarray(losses[a][start:]), np.asarray(losses[b][start:])
    return float((x.mean() - y.mean()) / abs(x.mean()))


def run(args) -> dict:
    return {"drift": drift, "regime": regime, "switch": switch, "real": real}[args.phase](args)


def _fits(args) -> tuple:
    prun = C.input_run(args, "posterior")
    return prun, C.read_result(prun)["fits"]


def drift(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    prun, fits = _fits(args)
    seeds = C.seeds(args, "adaptive.drift", d["seeds"] * len(SCENARIOS))
    tasks = [Task(f"{s}|{i:03d}", "lob.v08.studies.adaptive:drift_task",
                  {"scenario": s, "fits": fits, "n": d["windows"], "bins": d["bins"], "seed": seeds[j * d["seeds"] + i]})
             for j, s in enumerate(SCENARIOS) for i in range(d["seeds"])]
    rows, _ = run_tasks(tasks, workers=args.workers, checkpoint=out.parent / (out.name + ".ckpt") / "drift",
                        binding={"study": "adaptive.drift"})
    per = {}
    ids = sorted(t.task_id for t in tasks)
    for tid, r in zip(ids, rows):
        s = tid.split("|")[0]
        per.setdefault(s, []).append({"h1_rel": _rel(r["losses"], "frozen", "rolling"),
                                      "post_change_rel": _rel(r["losses"], "frozen", "rolling", CHANGE - 2),
                                      "expanding_rel": _rel(r["losses"], "frozen", "expanding"),
                                      "forgetting_rel": _rel(r["losses"], "frozen", "forgetting"),
                                      "change_triggered_rel": _rel(r["losses"], "frozen", "change_triggered"),
                                      "refits": r["change_refits"], "curve": r["curve"], "baseline": r["baseline"]})
    half = {}
    for s in ("gradual_rates", "gradual_mix"):
        if s in per:
            curves = np.asarray([x["curve"] for x in per[s]], float)
            base = float(np.mean([x["baseline"] for x in per[s]]))
            half[s] = half_life(np.arange(1, curves.shape[1] + 1), curves.mean(0), base)
    result = {"per_scenario": per, "half_life": half,
              "units": {"h1": [x["h1_rel"] for s in SCENARIOS for x in per.get(s, [])],
                        "h18b": [x["post_change_rel"] for s in SCENARIOS for x in per.get(s, [])]},
              "labels": {"evidence": "SYNTHETIC", "template": "state_dependent"}}
    return C.finish(args, out, analysis="adaptive_drift", config={"design": d}, result=C.json_safe(result),
                    inputs=[prun.relative_to(root).as_posix()])


def regime(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    prun, fits = _fits(args)
    truths = [(60.0, 2.0), (60.0, 4.0), (300.0, 2.0), (300.0, 4.0)]
    seeds = C.seeds(args, "adaptive.regime", d["seeds"] * len(truths))
    tasks = [Task(f"{dur:g}-{ratio:g}|{i:03d}", "lob.v08.studies.adaptive:regime_task",
                  {"duration_s": dur, "ratio": ratio, "fits": fits, "n": d["windows"], "bins": d["bins"],
                   "seed": seeds[j * d["seeds"] + i]})
             for j, (dur, ratio) in enumerate(truths) for i in range(d["seeds"])]
    rows, _ = run_tasks(tasks, workers=args.workers, checkpoint=out.parent / (out.name + ".ckpt") / "regime",
                        binding={"study": "adaptive.regime"})
    units = [{"truth": t, "h2_rel": _rel(r["losses"], "frozen", "regime"), "regime_models": r["regime_models"]}
             for t, r in zip(sorted(x.task_id for x in tasks), rows)]
    result = {"units": units, "h2": [u["h2_rel"] for u in units],
              "labels": {"evidence": "SYNTHETIC", "template": "poisson", "labeler": "threshold"}}
    return C.finish(args, out, analysis="adaptive_regime", config={"design": d, "truths": truths},
                    result=C.json_safe(result), inputs=[prun.relative_to(root).as_posix()])


def switch(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    prun, fits = _fits(args)
    orders = ORDERS[: d["orders"]]
    seeds = C.seeds(args, "adaptive.switch", d["seeds"] * len(orders))
    tasks = [Task(f"{'-'.join(o)}|{i:03d}", "lob.v08.studies.adaptive:switch_task",
                  {"order": list(o), "fits": fits, "bins": d["bins"], "seed": seeds[j * d["seeds"] + i]})
             for j, o in enumerate(orders) for i in range(d["seeds"])]
    rows, _ = run_tasks(tasks, workers=args.workers, checkpoint=out.parent / (out.name + ".ckpt") / "switch",
                        binding={"study": "adaptive.switch"})
    result = {"units": [{"order": t, **r} for t, r in zip(sorted(x.task_id for x in tasks), rows)],
              "h3": [r["relative_improvement"] for r in rows],
              "labels": {"evidence": "SYNTHETIC", "dma_forgetting": 0.99}}
    return C.finish(args, out, analysis="adaptive_switch", config={"design": d, "orders": orders},
                    result=C.json_safe(result), inputs=[prun.relative_to(root).as_posix()])


CHRONOLOGY = ["deribit-eth-perp-2020-04-01", "deribit-eth-perp-2020-05-01", "deribit-eth-perp-2020-06-01",
              "deribit-eth-perp-2020-07-01", "deribit-eth-perp-2020-08-01", "deribit-eth-perp-2020-09-01",
              "deribit-eth-perp-2020-10-01", "deribit-eth-perp-2020-11-01", "deribit-eth-perp-2020-12-01",
              "deribit-eth-perp-2021-01-01"]
USE = {"deribit-eth-perp-2020-10-01": "develop", "deribit-eth-perp-2020-11-01": "calibrate",
       "deribit-eth-perp-2020-12-01": "select"}


def real(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    days = CHRONOLOGY[: d["real_days"]]
    windows = realdata.registered_windows()[: d["real_windows"]]
    seq, labels = [], []
    for ds in days:
        frames, _ = realdata.load_frames(ds, use=USE.get(ds, "retrospective"), design="adaptive-real", root=root,
                                         purpose="retrospective chronology", windows=windows)
        seq.extend(frames)
        labels.extend([ds] * len(frames))
    book = PoissonGenerator().fit(seq[0]).book
    res = compare(StateDependentGenerator(book=book), seq, initial=d["real_windows"], window=d["real_windows"],
                  schemes=("frozen", "rolling", "expanding", "forgetting", "change_triggered"))
    names = ["poisson", "hawkes", "queue_reactive", "regime_switching", "state_dependent"]
    from ..inference.adaptive import concat
    comps = [FAM[n](book=book).fit(concat(seq[: d["real_windows"]])) for n in names]
    rest = concat(seq[d["real_windows"]:])
    ll = np.column_stack([c.loglik(rest) for c in comps])
    ok = np.all(np.isfinite(ll), 1)
    dma = dynamic(ll, alpha=0.99)
    first_best = names[int(np.argmax(np.nanmean(ll[: len(seq[d['real_windows']])], 0)))]
    for ds in days:
        realdata.mark_evaluated(ds, use=USE.get(ds, "retrospective"), design="adaptive-real",
                                run=out.relative_to(root).as_posix(), root=root)
    result = {"days": days, "windows_per_day": d["real_windows"], "losses": res["losses"],
              "relative_rolling_vs_frozen": _rel(res["losses"], "frozen", "rolling"),
              "dma_vs_best_single": {"best_single_on_first_future_window": first_best,
                                     "best_loss": float(-ll[ok, names.index(first_best)].mean()),
                                     "dma_loss": float(-np.asarray(dma["scores"])[ok].mean())},
              "labels": {"evidence": "RETROSPECTIVE (consumed days); EXPLORATORY"}}
    return C.finish(args, out, analysis="adaptive_real", config={"design": d}, result=C.json_safe(result),
                    datasets=days)
