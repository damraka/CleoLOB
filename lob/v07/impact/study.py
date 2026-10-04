"""M13 impact/TCA study and registered stress study (workstreams 28-30, 40). Simulation only.

Impact: parent orders vary direction, quantity (7, 14, 56, 224 lots; about 0.5-16x median L1 and
0.7-22% of median top-5 depth), horizon (60, 120 s), urgency (uniform, front-loaded) and style
(TWAP children; one POV arm), in the G0 point world, two richer generator worlds and four posterior
draws, on 8 market seeds per cell. Every run gets the additive TCA decomposition. The impact zoo
(linear, square-root, propagator) is fitted per world and transferred from the G0 point world to
the others (out-of-world R^2). Impact here is the simulators' endogenous response only.

Stress: the 11 market stresses and the fee stress, applied to the G0 point world, run every primary
classical agent on 64 market seeds. Labelled SYNTHETIC_STRESS; never part of the plausible set.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

from ...experiments.registry import PROJECT_ROOT
from ...sim_v2 import SimulatorSpec
from ...v06.identifiability import clean
from ..evidence.runs import finalize, new_run
from ..execution import episode
from ..generators.study import load_specs, v06_inputs
from ..posterior.study import posterior_specs
from ..robustness import stress
from . import metaorder as mo

QUANTITIES, HORIZONS, URGENCY, SIDES = (7, 14, 56, 224), (60.0, 120.0), (0.0, 3.0), ("buy", "sell")
SEEDS = 8
SEED_BASE = 790000


def _impact_cell(task):
    world_name, world, order, seeds = task
    rows = []
    for s in seeds:
        r = mo.run_metaorder(world, order, s)
        t = mo.tca(r, side=order.side, quantity=order.quantity)
        rows.append({k: r[k] for k in ("seed", "filled", "completion", "shortfall_bps", "temporary_impact_bps",
                                       "residual_impact_bps", "peak_move_bps", "recovery_fraction", "market_volume")}
                    | {"tca": t["components_bps"], "tca_identity_error": t["identity_error"],
                       "adverse_selection_bps": t["adverse_selection_passive_bps"],
                       "residual_valuation_bps": t["residual_valuation_bps"],
                       "sigma_bps": float(np.std(np.diff(np.log(r["path"]["mid"]))) * 1e4 * np.sqrt(2)),
                       "propagator_input": {"path": r["path"], "fills": r["fills"]} if order.quantity == 224 and
                       s == seeds[0] else None})
    return {"world": world_name, "order": order.__dict__, "rows": rows}


def worlds(generator_run: str, posterior_run: str, root: Path) -> dict:
    inputs = v06_inputs(root)
    selected = inputs["selection"]["selected"]
    gens = load_specs(root / generator_run)
    out = {"G0_point": SimulatorSpec(selected["config"], selected["extensions"])}
    for name in ("G3_conditional_ar", "G4_neural_temporal"):
        if name in gens:
            out[name] = gens[name]
    for i, spec in enumerate(posterior_specs(root / posterior_run, root)[:4]):
        out[f"post_{i:02d}"] = spec
    return out


def run_impact(out: str | Path, *, generator_run: str, posterior_run: str, root: Path = PROJECT_ROOT,
               workers: int = 8) -> dict:
    ws = worlds(generator_run, posterior_run, root)
    seeds = list(range(SEED_BASE, SEED_BASE + SEEDS))
    orders = [mo.MetaOrder(side=s, quantity=q, horizon_s=h, urgency=u)
              for s, q, h, u in product(SIDES, QUANTITIES, HORIZONS, URGENCY)]
    orders.append(mo.MetaOrder(quantity=56, style="pov", participation=0.1))
    out = new_run(root / out)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        cells = list(executor.map(_impact_cell, [(n, w, o, seeds) for n, w in ws.items() for o in orders], chunksize=1))
    zoo, fits = {}, {}
    for name in ws:
        rows = [(c["order"], r) for c in cells if c["world"] == name for r in c["rows"] if r["filled"]]
        q = np.asarray([o["quantity"] for o, _ in rows], float)
        temp = np.asarray([r["temporary_impact_bps"] for _, r in rows])
        sigma = np.asarray([r["sigma_bps"] for _, r in rows])
        vol = np.asarray([r["market_volume"] for _, r in rows])
        lin, sq = mo.fit_linear(q, temp), mo.fit_sqrt(q, temp, sigma, vol)
        fits[name] = {"q": q, "temp": temp, "sigma": sigma, "vol": vol, "lin": lin, "sq": sq}
        props = [r["propagator_input"] for c in cells if c["world"] == name for r in c["rows"] if r["propagator_input"]]
        zoo[name] = {"linear_a": lin["a"], "linear_r2": mo.r2(temp, lin["predict"](q)), "sqrt_Y": sq["Y"],
                     "sqrt_r2": mo.r2(temp, sq["predict"](q, sigma, vol)),
                     "propagator": {k: v for k, v in (mo.fit_propagator(props) or {}).items() if k != "predict"}}
    ref = fits["G0_point"]
    for name, f in fits.items():
        zoo[name]["transfer_from_G0_point"] = {"linear_r2": mo.r2(f["temp"], ref["lin"]["predict"](f["q"])),
                                               "sqrt_r2": mo.r2(f["temp"], ref["sq"]["predict"](f["q"], f["sigma"], f["vol"]))}
    summary = {}
    for c in cells:
        key = f"{c['world']}|{c['order']['side']}|q{c['order']['quantity']}|h{int(c['order']['horizon_s'])}|" \
              f"u{c['order']['urgency']}|{c['order']['style']}"
        rows = c["rows"]
        summary[key] = {m: float(np.mean([r[m] for r in rows])) for m in ("shortfall_bps", "completion",
                                                                          "temporary_impact_bps", "residual_impact_bps")}
        summary[key]["tca_bps"] = {k: float(np.mean([r["tca"][k] for r in rows])) for k in rows[0]["tca"]}
        summary[key]["max_identity_error"] = max(r["tca_identity_error"] for r in rows)
        recov = [r["recovery_fraction"] for r in rows if r["recovery_fraction"] is not None]
        summary[key]["recovery_fraction"] = float(np.mean(recov)) if recov else None
    for c in cells:
        for r in c["rows"]:
            r.pop("propagator_input", None)
    result = {"worlds": list(ws), "orders": [o.__dict__ for o in orders], "seeds": seeds, "summary": summary,
              "impact_zoo": zoo, "label": "SIMULATION (endogenous simulator impact); EXPLORATORY",
              "interpretation": "No empirical impact law is asserted; fits describe these simulators only."}
    finalize(out, analysis="m13-impact-tca", dataset_ids=[], config={"generator_run": generator_run,
                                                                     "posterior_run": posterior_run},
             result=clean(result), root=root, seeds={"seeds": seeds})
    return result


def _stress_cell(task):
    name, spec, agent, mandate, seeds = task
    rows = [episode.run(spec, agent, mandate, s) for s in seeds]
    costs = [r[episode.COST] for r in rows if r.get(episode.COST) is not None]
    return {"stress": name, "agent": agent, "mean_cost_bps": float(np.mean(costs)) if costs else None,
            "sd_cost_bps": float(np.std(costs, ddof=1)) if len(costs) > 1 else None,
            "invalid": sum(r.get(episode.COST) is None for r in rows), "n": len(rows)}


def run_stress(out: str | Path, *, root: Path = PROJECT_ROOT, workers: int = 8, markets: int = 64) -> dict:
    selected = v06_inputs(root)["selection"]["selected"]
    base = SimulatorSpec(selected["config"], selected["extensions"])
    seeds = list(range(SEED_BASE + 50_000, SEED_BASE + 50_000 + markets))
    worlds_ = {"none": (base, episode.MANDATE)}
    worlds_.update({n: (stress.apply(base, n), episode.MANDATE) for n in stress.STRESSES})
    worlds_.update({n: (base, stress.stressed_mandate(episode.MANDATE, n)) for n in stress.MANDATE_STRESSES})
    out = new_run(root / out)
    tasks = [(n, spec, a, mandate, seeds) for n, (spec, mandate) in worlds_.items() for a in episode.PRIMARY]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        cells = list(executor.map(_stress_cell, tasks, chunksize=1))
    table = {}
    for c in cells:
        table.setdefault(c["stress"], {})[c["agent"]] = c
    ranking = {s: sorted((a for a in row if row[a]["mean_cost_bps"] is not None), key=lambda a: row[a]["mean_cost_bps"])
               for s, row in table.items()}
    result = {"table": table, "ranking_by_mean": ranking, "markets": markets,
              "manifests": {n: stress.manifest(n, "G0_point").to_dict() for n in stress.STRESSES},
              "label": "SYNTHETIC_STRESS; stress is not empirical validation"}
    finalize(out, analysis="m15-stress", dataset_ids=[], config={"markets": markets}, result=clean(result), root=root,
             seeds={"seeds": seeds})
    return result
