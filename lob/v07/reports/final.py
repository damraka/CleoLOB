"""M22 final analysis: a sealed, pure transformation of the registered runs (no new simulation or data access).

Computes risk-sensitive metrics per policy, robust selection criteria, the realism-to-decision association,
per-pair historical determinacy under both fill bounds, the final decision benchmark with certification,
ablations, complexity penalty, overfitting curve, exploratory scaling, compute versus value, the failure-mode
catalogue and the compute table. Every input run is verified before use.
"""
from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import numpy as np

from ...experiments.registry import PROJECT_ROOT
from ...v06.identifiability import clean
from ..evidence.runs import finalize, new_run, verify_run
from ..posterior.study import FREE_PARAMETERS
from ..uncertainty import risk
from . import decision
from .registry import resolve

RUNS = {"execution": "results/v07/m15/execution", "transfer": "results/v07/m16/transfer", "select": "results/v07/m6/select",
        "fit": "results/v07/m5/fit", "drift": "results/v07/m11/drift", "queue": "results/v07/m3/queue",
        "impact": "results/v07/m13/impact", "matrix": "results/v07/m12/matrix", "recovery": "results/v07/m6/recovery",
        "identifiability": "results/v07/m8/identifiability", "holdout": "results/v07/m17", "power": None}
FRESH = ("deribit-eth-perp-2020-11-01", "deribit-eth-perp-2020-12-01", "deribit-btc-perp-2020-11-01",
         "bitmex-xbtusd-2020-11-01")
WALL_S = {  # measured wall-clock of the registered runs (run logs), with the worker count used
    "m5 generator fit": (158, 14), "m6 recovery-2": (1913, 14), "m6 posterior-2": (35874, 8), "m6 select": (199, 8),
    "m7 execution-aware": (89, 8), "m7 surrogate": (1260, 4), "m8 identifiability-3": (168, 4),
    "m10 bank": (1035, 8), "m17 ETH 2020-11-01": (3205, 1), "m17 ETH 2020-12-01": (3273, 1),
    "m17 BTC 2020-11-01": (1367, 1), "m17 BitMEX 2020-11-01": (750, 1), "m15 execution-2": (3538, 4),
    "m15 stress": (1123, 4), "m13 impact": (173, 4), "m13 ecology": (32, 1), "m11 drift": (907, 1),
    "m12 matrix": (1710, 2)}


def _load(root: Path, run: str) -> dict:
    path = resolve(root, run)
    report = verify_run(root / path, root=root)
    if not report["valid"]:
        raise ValueError(f"{path} does not verify: {report['issues']}")
    return json.loads((root / path / "result.json").read_text(encoding="utf-8"))


def historical_pairs(rows: list[dict], agents: list[str], *, samples: int = 2000, seed: int = 0) -> dict:
    """Per pair and fill mode: mean episode difference with a Bonferroni (0.05 / 2*pairs) interval and direction."""
    pairs = list(combinations(agents, 2))
    alpha = 0.05 / (2 * len(pairs))
    rng = np.random.default_rng(seed)
    out = {}
    for a, b in pairs:
        entry = {}
        for mode in ("conservative", "optimistic"):
            x = {r["episode"]: r["mandate_completion_adjusted_cost_bps"] for r in rows
                 if r["agent"] == a and r["fill_mode"] == mode and r["mandate_completion_adjusted_cost_bps"] is not None}
            y = {r["episode"]: r["mandate_completion_adjusted_cost_bps"] for r in rows
                 if r["agent"] == b and r["fill_mode"] == mode and r["mandate_completion_adjusted_cost_bps"] is not None}
            d = np.asarray([x[k] - y[k] for k in sorted(set(x) & set(y))])
            if len(d) < 10:
                entry[mode] = {"direction": 0, "mean": None, "ci": None, "n": int(len(d))}
                continue
            means = d[rng.integers(0, len(d), (samples, len(d)))].mean(1)
            lo, hi = float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
            entry[mode] = {"mean": float(d.mean()), "ci": [lo, hi], "n": int(len(d)),
                           "direction": -1 if hi < 0 else 1 if lo > 0 else 0}
        out[f"{a}|{b}"] = entry
    return {"pairs": out, "alpha": alpha}


def run(out: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    ex = _load(root, RUNS["execution"])
    tr = _load(root, RUNS["transfer"])
    sel = _load(root, RUNS["select"])
    fit = _load(root, RUNS["fit"])
    drift = _load(root, RUNS["drift"])
    queue = _load(root, RUNS["queue"])
    holdouts = {d: _load(root, f"{RUNS['holdout']}/{d}") for d in FRESH}
    out = new_run(root / out)
    episodes = json.loads((root / resolve(root, RUNS["execution"]) / "episodes.json").read_text(encoding="utf-8"))
    agents = ex["agents_primary"]
    pooled = {a: [] for a in agents + ex["agents_exploratory"]}
    completion = {a: [] for a in pooled}
    ttc = {a: [] for a in pooled}
    for cell in episodes:
        for row in cell["rows"]:
            if row.get("mandate_completion_adjusted_cost_bps") is not None:
                pooled[cell["agent"]].append(row["mandate_completion_adjusted_cost_bps"])
                completion[cell["agent"]].append(1.0 if row.get("mandate_within_horizon_completion") else 0.0)
                ttc[cell["agent"]].append(row.get("time_to_completion"))
    world_means = {w: {a: v["mean_cost_bps"] for a, v in cells.items() if a in agents} for w, cells in ex["summary"].items()}
    risk_table = {a: {**risk.risk_metrics(pooled[a], completion=completion[a], time_to_completion=ttc[a]),
                      "worst_plausible_world": risk.worst_plausible({w: world_means[w][a] for w in world_means})
                      if a in agents else None} for a in pooled}
    robust = risk.robust_selection(world_means)
    realism = {f"post_{i:02d}": v for i, v in enumerate(sel["per_draw_objective"]["selection"])}
    realism.update({k: v["selection"]["objective"] for k, v in fit["scores"].items()})
    r2d = decision.realism_to_decision(ex["H8"]["edges"], realism)
    rows = [json.loads(line) for line in (root / resolve(root, RUNS["transfer"]) / "rows.jsonl").read_text(
        encoding="utf-8").splitlines()]
    hist = historical_pairs(rows, agents)
    worst = {k: v["worst_plausible"] for k, v in ex["H8"]["edges"].items()}
    bench = decision.decision_benchmark(ex["H8"]["edges"], margin=1.0, mde=0.924, worst=worst, history=hist["pairs"],
                                        regime_world="G2_regime_switching",
                                        queue_width_bps=queue["bound_width_mean_fill_fraction"])
    for pair, entry in bench["pairs"].items():
        entry["queue_sensitivity_unit"] = "conservative-to-optimistic mean fill-fraction width (M3, development)"
    fresh = {m: float(np.mean([holdouts[d]["contrasts"]["models"][m]["objective"] for d in FRESH]))
             for m in holdouts[FRESH[0]]["contrasts"]["models"]}
    month2 = next(d for d in drift["days"].values() if d["month"] == 2)
    timing = fit["timing_s"]   # measured fit wall-clock; G0 point (v0.6) and G2 (built from it) were not timed in v0.7
    compute = {"G0_point": None, "G0_post": WALL_S["m6 posterior-2"][0], "G1_state_hawkes": timing["G1_fit_s"],
               "G2_regime_switching": None, "G3_conditional_ar": timing["G3_fit_s"], "G4_neural_temporal": timing["G4_fit_s"]}
    models = {}
    for name, sel_name in (("G0_point", "G0_point"), ("G1_state_hawkes", "G1_state_hawkes"),
                           ("G2_regime_switching", "G2_regime_switching"), ("G3_conditional_ar", "G3_conditional_ar"),
                           ("G4_neural_temporal", "G4_neural_temporal")):
        scores = fit["scores"][sel_name]
        models[name] = {"parameters": FREE_PARAMETERS.get(name, FREE_PARAMETERS["G0_post"]), "compute_s": compute[name],
                        "development": scores["development"]["objective"], "selection": scores["selection"]["objective"],
                        "validation": month2["decay"][name]["objective"], "fresh": fresh[name]}
    models["G0_post"] = {"parameters": FREE_PARAMETERS["G0_post"], "compute_s": compute["G0_post"],
                         "development": sel["posterior_predictive_objective"]["development"],
                         "selection": sel["posterior_predictive_objective"]["selection"],
                         "validation": month2["decay"]["G0_post"]["objective"], "fresh": fresh["G0_post"]}
    eth_es = {m: float(np.mean([holdouts[d]["contrasts"]["models"][m]["es_objective"] for d in FRESH[:2]]))
              for m in ("EA", "G0_point")}
    h12 = tr.get("H12", {}).get("members", {})
    ablations = decision.ablations({
        "posterior calibration vs point": {"complex": fresh["G0_post"], "simple": fresh["G0_point"]},
        "neural generator (G4) vs parametric baseline (G0)": {"complex": fresh["G4_neural_temporal"],
                                                               "simple": fresh["G0_point"]},
        "continuous conditioning (G3) vs discrete regimes (G2)": {"complex": fresh["G3_conditional_ar"],
                                                                   "simple": fresh["G2_regime_switching"]},
        "discrete regimes (G2) vs no conditioning (G0)": {"complex": fresh["G2_regime_switching"],
                                                          "simple": fresh["G0_point"]},
        "execution-aware vs generic calibration (ES objective, fresh ETH)": {"complex": eth_es["EA"],
                                                                            "simple": eth_es["G0_point"]},
        "posterior-world vs single-world policy training (PPO, conservative gap)": {
            "complex": (h12.get("ppo@conservative") or {}).get("gap_posterior_bps"),
            "simple": (h12.get("ppo@conservative") or {}).get("gap_single_bps")}})
    catalogue = decision.failure_catalogue({
        "SUPPORT_FAILURE": ["results/v07/m17 (H3: fresh support coverage 0-1.5%)"],
        "TAIL_FAILURE": ["results/v07/m17 (tail family error largest on every fresh day)"],
        "DEPENDENCE_FAILURE": ["results/v07/m17 (dependence family error above every model's real-vs-real margin)"],
        "TEMPORAL_FAILURE": ["results/v07/m11/drift (objective rises with months after calibration)"],
        "REGIME_FAILURE": ["results/v07/m11/drift (latent HMMs below the discrete baseline out of sample)"],
        "QUEUE_FAILURE": ["results/v07/m3/queue (passive fill bounds differ by about half the conservative value)"],
        "IMPACT_FAILURE": ["results/v07/m13/impact (impact-law fits R^2 about 0)"],
        "CALIBRATION_FAILURE": ["results/v07/m6/recovery-2 (ASSUMPTION_DEPENDENT)", "H1 FAILED"],
        "IDENTIFIABILITY_FAILURE": ["results/v07/m8/identifiability-3 (no parameter identified)"],
        "TRANSFER_FAILURE": ["H4 FAILED", "H5 FAILED", "results/v07/m12/matrix"],
        "EXECUTION_INSTABILITY": ["results/v07/m15/execution-2 (21 distinct rankings in 21 worlds)"],
        "MODEL_CLASS_FAILURE": ["G1 objective 4-6.6 on all fresh days", "H2 VACUOUS"],
        "DATA_CAPABILITY_LIMIT": ["aggregate L2: exact FIFO, queue position, hidden liquidity NOT_AVAILABLE"]})
    result = {"risk_metrics": risk_table, "robust_selection": robust, "realism_to_decision": r2d,
              "historical_pairs": hist, "decision_benchmark": bench, "ablations": ablations,
              "complexity_penalty": decision.complexity_penalty(models), "overfitting_curve": decision.overfitting_curve(models),
              "scaling": decision.scaling(models), "compute_value": decision.compute_value(models, "G0_point"),
              "failure_catalogue": catalogue,
              "compute_table": {k: {"wall_s": w, "workers": n, "cpu_upper_bound_s": w * n} for k, (w, n) in WALL_S.items()},
              "label": "M22 synthesis of sealed runs; no new data access"}
    finalize(out, analysis="m22-final-analysis", dataset_ids=[], config={"runs": {k: v for k, v in RUNS.items() if v}},
             result=clean(result), root=root)
    return result
