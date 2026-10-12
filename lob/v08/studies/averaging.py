"""``model-averaging-v08``: model-class uncertainty and dynamic model averaging (M17).

``synthetic``: for each truth family, five families are fitted on a training frame; on a selection frame the
posterior model probabilities (BIC), information (AIC) weights, stacking weights and regime-conditioned stacking
weights are computed; on an evaluation frame the dynamic-averaging weight path, the mixture log score and the best
single family are compared. Model uncertainty is material when no family holds at least 0.9 weight; no single
generator is forced then.
``real`` (RETROSPECTIVE): the development fits scored on the selection day (weights, dynamic path, materiality).
"""
from __future__ import annotations

import numpy as np

from .. import model_averaging as ma, realdata
from ..generators import HawkesGenerator, PoissonGenerator, QueueReactiveGenerator, RegimeSwitchingGenerator
from ..inference.adaptive import concat
from ..regimes import Labeler, block_features
from ..runner import Task, run_tasks
from . import common as C
from .adaptive import FAM

DEFAULTS = {"tiny": {"seeds": 1, "bins": 1200, "windows": 3}, "small": {"seeds": 3, "bins": 3000, "windows": 8},
            "medium": {"seeds": 8, "bins": 6000, "windows": 24}}
NAMES = ["poisson", "hawkes", "queue_reactive", "regime_switching", "state_dependent"]


def weights_report(comps: list, names: list[str], select, evaluate) -> dict:
    ll_sel = np.column_stack([c.loglik(select) for c in comps])
    ll_eval = np.column_stack([c.loglik(evaluate) for c in comps])
    ok_s, ok_e = np.all(np.isfinite(ll_sel), 1), np.all(np.isfinite(ll_eval), 1)
    n = int(ok_s.sum())
    totals = {k: float(ll_sel[ok_s, i].sum()) for i, k in enumerate(names)}
    pmp = ma.posterior_model_probabilities(totals, {k: c.n_params() for k, c in zip(names, comps)}, n)
    aic = ma.information_weights(totals, {k: c.n_params() for k, c in zip(names, comps)})
    stack = ma.stacking_weights(ll_sel[ok_s])
    lab = Labeler("threshold").fit(block_features(select, 100))
    labels = np.repeat(lab.label(block_features(select, 100)), 100)[: len(select)]
    labels = np.r_[labels, np.full(len(select) - len(labels), -1)]
    regime_w = ma.regime_weights(ll_sel[ok_s], labels[ok_s])
    dyn = ma.dynamic(ll_eval, alpha=0.99)
    best = int(np.argmax(ll_sel[ok_s].mean(0)))
    stack_score = float(ma.mixture_score(ll_eval[ok_e], stack).mean())
    return {"posterior_model_probabilities": pmp, "information_weights": aic,
            "stacking_weights": dict(zip(names, stack.tolist())),
            "regime_stacking_weights": {str(k): dict(zip(names, v)) for k, v in regime_w.items()},
            "dynamic_final_weights": dict(zip(names, dyn["path"][-1].tolist())),
            "dynamic_path_every_600_bins": dyn["path"][::600].tolist(),
            "eval_log_score": {"best_single": names[best], "best_single_mean": float(ll_eval[ok_e, best].mean()),
                               "stacking_mixture_mean": stack_score,
                               "dynamic_mean": float(np.nanmean(np.asarray(dyn["scores"])[ok_e]))},
            "materiality": {"pmp": ma.material(pmp), "stacking": ma.material(stack)}}


def synthetic_task(truth: str, bins: int, seed: int) -> dict:
    g = {"poisson": PoissonGenerator, "hawkes": HawkesGenerator, "queue_reactive": QueueReactiveGenerator,
         "regime_switching": RegimeSwitchingGenerator}[truth]()
    frame = g.simulate(3 * bins, seed)
    train, select, evaluate = frame.slice(0, bins), frame.slice(bins, 2 * bins), frame.slice(2 * bins, 3 * bins)
    comps = [FAM[n](book=g.book).fit(train) for n in NAMES]
    report = weights_report(comps, NAMES, select, evaluate)
    mixture = ma.mixture(comps, list(report["stacking_weights"].values()))
    report["mixture_posterior_predictive"] = {"simulated_bins": 600,
                                              "mean_events_per_bin": float(mixture.simulate(600, seed + 1)
                                                                          .counts.sum(1).mean())}
    return report


def run(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    if args.phase == "synthetic":
        truths = ["poisson", "hawkes", "queue_reactive", "regime_switching"]
        seeds = C.seeds(args, "averaging", d["seeds"] * len(truths))
        tasks = [Task(f"{t}|{i:03d}", "lob.v08.studies.averaging:synthetic_task",
                      {"truth": t, "bins": d["bins"], "seed": seeds[j * d["seeds"] + i]})
                 for j, t in enumerate(truths) for i in range(d["seeds"])]
        rows, _ = run_tasks(tasks, workers=args.workers, checkpoint=out.parent / (out.name + ".ckpt") / "synthetic",
                            binding={"study": "averaging"})
        ids = sorted(t.task_id for t in tasks)
        result = {"units": dict(zip(ids, rows)),
                  "material_fraction": float(np.mean([r["materiality"]["stacking"]["material"] for r in rows])),
                  "dynamic_beats_best_single": float(np.mean([r["eval_log_score"]["dynamic_mean"] >
                                                              r["eval_log_score"]["best_single_mean"] for r in rows])),
                  "labels": {"evidence": "SYNTHETIC"}}
        return C.finish(args, out, analysis="model_averaging_synthetic", config={"design": d},
                        result=C.json_safe(result))
    prun = C.input_run(args, "posterior")
    post = C.read_result(prun)
    comps = [C.generator_from(post["fits"][n]) for n in NAMES]
    frames, meta = realdata.load_frames("deribit-eth-perp-2020-12-01", use="select", design="model-averaging-real",
                                        root=root, purpose="model weights on the selection day",
                                        windows=realdata.registered_windows()[: d["windows"]])
    half = max(1, len(frames) // 2)
    report = weights_report(comps, NAMES, concat(frames[:half]), concat(frames[half:]))
    realdata.mark_evaluated("deribit-eth-perp-2020-12-01", use="select", design="model-averaging-real",
                            run=out.relative_to(root).as_posix(), root=root)
    result = {"report": report, "meta": meta, "labels": {"evidence": "RETROSPECTIVE; EXPLORATORY"}}
    return C.finish(args, out, analysis="model_averaging_real", config={"design": d}, result=C.json_safe(result),
                    datasets=["deribit-eth-perp-2020-12-01"], inputs=[prun.relative_to(root).as_posix()])
