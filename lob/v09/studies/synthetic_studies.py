"""Synthetic decision experiments over world clusters (H11, H13, H20, H21, H24, H28; F4/F6, E1/E2).

A world cluster is one calibration day (strategic ecology calibrated on its registered window frames, plus the
classical families where a hypothesis needs model ambiguity). Pilot: the explore Bybit days with ``explore`` seeds.
Confirm: the design's calibration days and ``confirm`` seeds (sample size fixed at promotion). Per cluster:

* H11: Kendall distance between registered-policy rankings (large mandate) under the endogenous and the exogenous
  strategic ecology (6 seeds each, common random numbers);
* H13: two independent active-calibration recovery replicates (CEM vs random search; budget 120 simulations
  each); the unit effect is their mean and the two replicates form the split halves;
* H20: catastrophic-degradation rate (cost > 25 bps) of the no-fallback choice (best policy in the nominal world)
  minus the contextual fallback ladder, over three registered shifts (liquidity drought, volatility spike, toxic
  flow) x 8 seeds; the ladder uses a pre-episode OOD score (standardized summary distance to nominal) with the
  nominal 75th / 95th percentiles as the low / high thresholds;
* H21: worst-case regret over held-out ambiguity worlds (2 families x 4 fresh perturbations) of the nominal choice
  minus the DRO choice (min-max over the 2 x 3 training ambiguity worlds);
* H24: mean cost of non-adaptive minus adaptive MPC v2 in drift worlds (calm -> volatility spike at 150 s), 8 seeds;
* H28: catastrophic probability (true collapse rate 0.1) of the nominal choice minus the rare-event-aware choice
  (mean + 50 x P(cat) estimated by importance sampling with collapse rate 0.5), 40 evaluation seeds; mean-cost
  non-inferiority reported.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import time

import numpy as np

from .. import protocol as pr, synthetic as SY
from ..generators.strategic import StrategicEcology
from ..runner import Task, run_tasks
from . import common

ANALYSIS = "synthetic-decisions-v09"
LAMBDA_CAT = 50.0
P_COLLAPSE, P_PROPOSAL = 0.1, 0.5


def _seeds(base: int, label: str, n: int) -> list[int]:
    return [int(hashlib.sha256(f"{base}|{label}|{k}".encode()).hexdigest()[:8], 16) for k in range(n)]


class Collapse(StrategicEcology):
    """Strategic ecology with a liquidity collapse (limit flow x 0.1 for 60 s) starting at a registered bin."""

    def __init__(self, base: StrategicEcology, collapse: bool, at_bin: int) -> None:
        super().__init__(deepcopy(base.params), base.book)
        self.collapse, self.at_bin = collapse, at_bin

    def initial_state(self, simulate=False):
        s = super().initial_state(simulate)
        s["n"] = 0
        return s

    def expected(self, s, x):
        r = super().expected(s, x)
        if self.collapse and self.at_bin <= s.get("n", 0) < self.at_bin + 600:
            r = r.copy()
            r[2:] *= 0.1
        return r

    def update(self, s, realized, x, own=None):
        s = super().update(s, realized, x, own)
        s["n"] = s.get("n", 0) + 1
        return s


def _ood_score(g, ref_mu, ref_sd, seed) -> float:
    s = SY._summary(g, seed, n_bins=600)
    return float(np.linalg.norm((s - ref_mu) / ref_sd) / np.sqrt(len(s)))


def cluster_task(day: str, root: str, design: str, base_seed: int, notional: dict, lot: float,
                 hypotheses: list[str]) -> dict:
    cache = common.load_cache(day, root, design, "synthetic world calibration")
    g, tick, touch = SY.calibrate([cache])
    fam = {}
    if "H21" in hypotheses:
        fam["hawkes"] = SY.calibrate([cache], "hawkes")[0]
    del cache
    out = {"day": day}
    t0 = time.time()
    medium = SY.qty_for(g, tick, lot, notional["medium"])
    if "H11" in hypotheses:
        out["H11"] = SY.h11_world(g, tick, lot, notional["large"], _seeds(base_seed, "h11", 6))
    if "H13" in hypotheses:
        reps = [SY.h13_task(g, s, budget=120) for s in _seeds(base_seed, "h13", 2)]
        out["H13"] = {"effect": float(np.mean([r["effect"] for r in reps])), "odd": reps[1]["effect"],
                      "even": reps[0]["effect"], "replicates": reps}
    if "H20" in hypotheses or "H28" in hypotheses or "H21" in hypotheses:
        nominal = SY.costs(g, SY.RANKING, medium, _seeds(base_seed, "nominal", 4))
        best_nominal = SY.mean_order(nominal)[0]
    if "H20" in hypotheses:
        from ...v08.market.frame import summaries  # noqa: F401 - imported for the summary definition used by OOD
        ref = np.vstack([SY._summary(g, s, n_bins=600) for s in _seeds(base_seed, "ref", 12)])
        mu, sd = ref.mean(0), ref.std(0) + 1e-6
        nom_scores = [_ood_score(g, mu, sd, s) for s in _seeds(base_seed, "refscore", 12)]
        low, high = float(np.quantile(nom_scores, 0.75)), float(np.quantile(nom_scores, 0.95))
        from ..execution.selector import fallback_policy
        rows = {"nofallback": [], "ladder": [], "chosen": []}
        for kind in ("liquidity_drought", "volatility_spike", "toxic_flow"):
            w = SY.shifted(g, kind)
            for s in _seeds(base_seed, f"h20|{kind}", 8):
                score = _ood_score(w, mu, sd, s + 7)
                pol = fallback_policy(score, False, low=low, high=high, selected=best_nominal)
                a = SY.costs(w, [best_nominal], medium, [s])[best_nominal][0]
                b = SY.costs(w, [pol], medium, [s])[pol][0]
                rows["nofallback"].append(a)
                rows["ladder"].append(b)
                rows["chosen"].append(pol)
        ra, rb = SY.catastrophic_rate(rows["nofallback"]), SY.catastrophic_rate(rows["ladder"])
        n = len(rows["ladder"])
        odd = list(range(1, n, 2))
        even = list(range(0, n, 2))
        out["H20"] = {"effect": ra - rb, "rate_nofallback": ra, "rate_ladder": rb, "selected": best_nominal,
                      "chosen": rows["chosen"],
                      "odd": SY.catastrophic_rate([rows["nofallback"][i] for i in odd])
                      - SY.catastrophic_rate([rows["ladder"][i] for i in odd]),
                      "even": SY.catastrophic_rate([rows["nofallback"][i] for i in even])
                      - SY.catastrophic_rate([rows["ladder"][i] for i in even])}
    if "H21" in hypotheses:
        train_worlds, test_worlds = {}, {}
        for fname, base in (("strategic", g), ("hawkes", fam["hawkes"])):
            for k, s in enumerate(_seeds(base_seed, f"h21|{fname}|train", 3)):
                train_worlds[f"{fname}-{k}"] = SY.costs(SY.perturb(base, s), SY.RANKING, medium,
                                                        _seeds(base_seed, "h21|ep", 4))
            for k, s in enumerate(_seeds(base_seed, f"h21|{fname}|test", 4)):
                test_worlds[f"{fname}-{k}"] = SY.costs(SY.perturb(base, s), SY.RANKING, medium,
                                                       _seeds(base_seed, "h21|ep2", 4))
        dro = min(SY.RANKING, key=lambda p: SY.worst_case_regret(train_worlds, p))
        a = SY.worst_case_regret(test_worlds, best_nominal)
        b = SY.worst_case_regret(test_worlds, dro)
        keys = sorted(test_worlds)
        sub = lambda ks: {k: test_worlds[k] for k in ks}  # noqa: E731
        out["H21"] = {"effect": a - b, "nominal": best_nominal, "dro": dro, "wc_regret_nominal": a,
                      "wc_regret_dro": b,
                      "odd": SY.worst_case_regret(sub(keys[1::2]), best_nominal) - SY.worst_case_regret(sub(keys[1::2]), dro),
                      "even": SY.worst_case_regret(sub(keys[0::2]), best_nominal) - SY.worst_case_regret(sub(keys[0::2]), dro)}
    if "H24" in hypotheses:
        drift = SY.Switching(g, SY.shifted(g, "volatility_spike"), at_bin=(300 + 1500))
        seeds = _seeds(base_seed, "h24", 8)
        c = SY.costs(drift, ["mpc_stochastic", "mpc_adaptive"], medium, seeds)
        d = [a - b for a, b in zip(c["mpc_stochastic"], c["mpc_adaptive"]) if a is not None and b is not None]
        out["H24"] = {"effect": float(np.mean(d)) if d else None, "odd": float(np.mean(d[1::2])) if d[1::2] else None,
                      "even": float(np.mean(d[0::2])) if d[0::2] else None, "episodes": len(d)}
    if "H28" in hypotheses:
        rng_seeds = _seeds(base_seed, "h28|prop", 16)
        prop = {}
        for p in SY.RANKING:
            vals, weights = [], []
            for s in rng_seeds:
                col = (s % 1000) / 1000.0 < P_PROPOSAL
                w = (P_COLLAPSE / P_PROPOSAL) if col else ((1 - P_COLLAPSE) / (1 - P_PROPOSAL))
                world = Collapse(g, col, at_bin=300 + (s % 2400))
                v = SY.costs(world, [p], medium, [s])[p][0]
                if v is not None:
                    vals.append(v)
                    weights.append(w)
            vals, weights = np.asarray(vals), np.asarray(weights)
            pm = float((weights * vals).sum() / weights.sum())
            pc = float((weights * (vals > SY.CATASTROPHE_BPS)).sum() / weights.sum())
            prop[p] = pm + LAMBDA_CAT * pc
        rare = min(prop, key=prop.get)
        ev = _seeds(base_seed, "h28|eval", 40)

        def evaluate(p):
            out_c = []
            for s in ev:
                col = (s % 1000) / 1000.0 < P_COLLAPSE
                out_c.append(SY.costs(Collapse(g, col, at_bin=300 + (s % 2400)), [p], medium, [s])[p][0])
            return out_c

        a_c, b_c = evaluate(best_nominal), (evaluate(rare) if rare != best_nominal else None)
        b_c = b_c if b_c is not None else a_c
        pa, pb = SY.catastrophic_rate(a_c), SY.catastrophic_rate(b_c)
        out["H28"] = {"effect": pa - pb, "nominal": best_nominal, "rare_aware": rare, "p_cat_nominal": pa,
                      "p_cat_rare": pb, "mean_cost_diff": float(np.mean([x for x in b_c if x is not None])
                                                                - np.mean([x for x in a_c if x is not None])),
                      "odd": SY.catastrophic_rate(a_c[1::2]) - SY.catastrophic_rate(b_c[1::2]),
                      "even": SY.catastrophic_rate(a_c[0::2]) - SY.catastrophic_rate(b_c[0::2])}
    out["seconds"] = time.time() - t0
    return out


def run_for(args, hypotheses: list[str], analysis: str) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    if args.namespace in ("explore", "pilot", "dev"):
        days = common.explore_units(args, protocol)
        design_name = f"explore-{analysis}"
    else:
        design = common.design_document(args)
        days, design_name = list(design["calibration_days"]), args.design
    base = common.seeds(args, analysis, len(days))
    notional = {k: v["notional_usd"] for k, v in protocol["mandates"]["classes"].items()}
    out = common.start(args)
    tasks = []
    for d, s in zip(days, base):
        inst = d.split("-")[2].upper()
        lot = float(next(x for x in protocol["datasets"] if x["instrument"] == inst)["lot_size"])
        tasks.append(Task(d, "lob.v09.studies.synthetic_studies:cluster_task",
                          {"day": d, "root": str(root), "design": design_name, "base_seed": int(s),
                           "notional": {k: float(v[inst]) for k, v in notional.items()}, "lot": lot,
                           "hypotheses": hypotheses}, seed=int(s), namespace=args.namespace))
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root / (args.out + ".checkpoint"), design={"days": days, "h": hypotheses},
                           inputs=days)
    hours = float(np.mean([r["seconds"] for r in rows])) / 3600 if rows else 0.0
    caps = common.caps_for(protocol, ("propensities",))
    pilot = {}
    for h in hypotheses:
        vals = [r[h] for r in rows if h in r]
        key = "kendall_distance" if h == "H11" else "effect"
        pairs = [(v.get("odd"), v.get("even")) for v in vals if v.get("odd") is not None and v.get("even") is not None]
        rec = common.pilot_record([v.get(key) for v in vals], [a for a, _ in pairs], [b for _, b in pairs],
                                  caps=caps, hours=hours)
        if h == "H13":
            rec["reliability_note"] = "split-half = two independent search replicates per task (seeded)"
        pilot[h] = rec
    result = {"namespace": args.namespace, "clusters": {r["day"]: r for r in rows},
              "pilot": pilot if args.namespace in ("explore", "pilot", "dev") else None, "runner": info,
              "label": "SYNTHETIC (E1/E2): calibrated generator worlds; not market evidence"}
    return common.finish(args, out, analysis=analysis, config={"days": days, "hypotheses": hypotheses},
                         result=result, datasets=days)
