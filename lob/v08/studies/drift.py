"""``change-point-study-v08``: change-point and drift detection (M6) and calibration half-life (M7).

Synthetic: monitoring series of block activity (10 s blocks) from sequences with a known change at block 60 of 120
(market-rate step x3, limit-flow step x2, regime-mix change) and a no-change control. For every detector:
detection delay, detections, misses and false alarms; post-change degradation of a frozen fit (calibration decay).

Real (RETROSPECTIVE, EXPLORATORY): the same detectors on the consumed ETH chronology (no ground truth: alarms are
reported, not scored), and calibration half-lives across days for three targets: observable fit (summary
distance), execution-sensitive fit (simulated vs bounded historical TWAP cost) and policy-ranking utility (Kendall
tau between calibrated-world and bounded historical rankings). A half-life is reported only when the registered
fit is informative; otherwise NOT_ESTABLISHED.
"""
from __future__ import annotations

import numpy as np

from .. import counterfactual as cf, drift as dr, realdata
from ..execution.simulator import run_episode
from ..generators import MixtureGenerator, PoissonGenerator
from ..generators.base import MarketGenerator
from ..inference.adaptive import concat
from ..market.frame import summaries
from ..runner import Task, run_tasks
from ..statistics import kendall_tau
from ..worlds import World
from . import common as C

DEFAULTS = {"tiny": {"seeds": 2, "blocks": 60, "real_days": 3, "real_windows": 2, "policy_seeds": 2},
            "small": {"seeds": 8, "blocks": 120, "real_days": 10, "real_windows": 3, "policy_seeds": 4},
            "medium": {"seeds": 24, "blocks": 120, "real_days": 10, "real_windows": 4, "policy_seeds": 8}}
SCENARIOS = ("market_step", "limit_step", "mix_change", "no_change")
BLOCK = 100
POLICIES = ("twap", "ac", "heuristic", "mpc")


def series_task(scenario: str, base: dict, blocks: int, seed: int) -> dict:
    g0 = MarketGenerator.from_dict(base)
    p = dict(g0.params)
    change = blocks // 2

    def pois(fm, fl):
        q = dict(p)
        q.update(market=p["market"] * fm, add=p["add"] * fl, cancel=p["cancel"] * fl)
        return PoissonGenerator(q, g0.book)

    first, second = pois(1, 1), {"market_step": pois(3, 1), "limit_step": pois(1, 2),
                                 "mix_change": MixtureGenerator.of([pois(0.6, 0.6), pois(1.8, 1.8)], [0.2, 0.8],
                                                                   switch_prob=0.01, book=g0.book),
                                 "no_change": pois(1, 1)}[scenario]
    a = first.simulate(change * BLOCK, seed, warmup_bins=100)
    b = second.simulate((blocks - change) * BLOCK, seed + 1, warmup_bins=100)
    act = [float(f.counts.sum(1).mean()) for f in a.windows(BLOCK) + b.windows(BLOCK)]
    fit = PoissonGenerator(book=g0.book).fit(a)
    pre = float(-np.nanmean(fit.loglik(a.slice(len(a) // 2, len(a)))))
    post = float(-np.nanmean(fit.loglik(b)))
    out = {"series": act, "change": change if scenario != "no_change" else None, "pre_loss": pre, "post_loss": post}
    for m in dr.DETECTORS:
        alarms = dr.detect(np.asarray(act), ref=change // 2, method=m, seed=seed)
        out[m] = dr.evaluate(alarms, [change] if scenario != "no_change" else [], tolerance=20, n=len(act))
    return out


def run(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    prun = C.input_run(args, "posterior")
    post = C.read_result(prun)
    seeds = C.seeds(args, "change_point", d["seeds"] * len(SCENARIOS))
    tasks = [Task(f"{s}|{i:03d}", "lob.v08.studies.drift:series_task",
                  {"scenario": s, "base": post["fits"]["poisson"], "blocks": d["blocks"], "seed": seeds[j * d["seeds"] + i]})
             for j, s in enumerate(SCENARIOS) for i in range(d["seeds"])]
    rows, _ = run_tasks(tasks, workers=args.workers, checkpoint=out.parent / (out.name + ".ckpt") / "synthetic",
                        binding={"study": "change_point"})
    ids = sorted(t.task_id for t in tasks)
    synthetic = {}
    for s in SCENARIOS:
        group = [r for i, r in zip(ids, rows) if i.startswith(s + "|")]
        synthetic[s] = {"post_change_degradation": float(np.mean([r["post_loss"] - r["pre_loss"] for r in group])),
                        "detectors": {m: {k: float(np.mean([r[m][k] for r in group if r[m][k] is not None]))
                                          if any(r[m][k] is not None for r in group) else None
                                          for k in ("detection_delay_mean", "detected", "missed", "false_alarms")}
                                      for m in dr.DETECTORS}}
    real = _real(args, out, d, post)
    result = {"synthetic": synthetic, "real": real,
              "labels": {"synthetic": "SYNTHETIC", "real": "RETROSPECTIVE (consumed days); EXPLORATORY"}}
    return C.finish(args, out, analysis="change_point_study", config={"design": d}, result=C.json_safe(result),
                    datasets=real.get("days", []), inputs=[prun.relative_to(root).as_posix()])


def _real(args, out, d, post) -> dict:
    from .adaptive import CHRONOLOGY, USE
    root = C.root_of(args)
    days = CHRONOLOGY[: d["real_days"]]
    windows = realdata.registered_windows()[: d["real_windows"]]
    frames = {}
    for ds in days:
        frames[ds], _ = realdata.load_frames(ds, use=USE.get(ds, "retrospective"), design="change-point-real",
                                             root=root, purpose="retrospective drift and half-life", windows=windows)
    series, owner = [], []
    for ds in days:
        for f in frames[ds]:
            for blk in f.windows(BLOCK * 3):
                series.append(float(blk.counts.sum(1).mean()))
                owner.append(ds)
    ref = max(4, len(series) // max(len(days), 1))
    alarms = {m: [owner[i] for i in dr.detect(np.asarray(series), ref=ref, method=m)] for m in dr.DETECTORS}
    # calibration half-life across days: calibrate on the first day
    base_frames = frames[days[0]]
    cal = PoissonGenerator().fit(concat(base_frames))
    world = World("day0", cal, cal.book)
    m = C.mandate(root)
    registered = C.registered_policies(root)
    pseeds = C.seeds(args, "change_point.half_life", d["policy_seeds"])
    predicted = {p: float(np.mean([run_episode(world, C.policy(p, registered[p]), m, s)["economic_cost_bps"]
                                   for s in pseeds])) for p in POLICIES}
    sim_summary = summaries(cal.simulate(len(base_frames[0]), pseeds[0]))
    neutral = cf.Assumption("neutral", 0.0, 0.0, "none", 0.0)
    observable, execution, ranking = [], [], []
    for ds in days:
        obs = np.mean([summaries(f) for f in frames[ds]], 0)
        scale = np.abs(sim_summary) * 0.05 + 1e-3
        observable.append(float(np.sqrt((((obs - sim_summary) / scale) ** 2).mean())))
        hist = {}
        for p in POLICIES:
            vals = []
            for f in frames[ds]:
                try:
                    vals.append(cf.replay(f, C.policy(p, registered[p]), m, neutral, start_bin=300)["economic_cost_bps"])
                except ValueError:
                    pass
            hist[p] = float(np.mean(vals)) if vals else np.nan
        execution.append(abs(predicted["twap"] - hist["twap"]))
        ranking.append(kendall_tau([predicted[p] for p in POLICIES], [hist[p] for p in POLICIES]))
    horizons = np.arange(1, len(days))
    halves = {"observable_fit": dr.half_life(horizons, observable[1:], observable[0]),
              "execution_sensitive_fit": dr.half_life(horizons, execution[1:], execution[0]),
              "policy_ranking_utility": dr.half_life(horizons, [-x for x in ranking[1:]], -ranking[0])}
    for ds in days:
        from .adaptive import USE as U
        realdata.mark_evaluated(ds, use=U.get(ds, "retrospective"), design="change-point-real",
                                run=out.relative_to(root).as_posix(), root=root)
    return {"days": days, "alarms": alarms, "series_length": len(series), "half_life": halves,
            "observable_distance": observable, "twap_cost_error_bps": execution, "ranking_tau": ranking,
            "horizon_unit": "days (first-of-month samples; calendar gaps of about one month)"}
