"""M6 runs: synthetic recovery, the G0 posterior (3 independent SMC-ABC runs) and the sealed G* selection.

recovery  -> results/v07/m6/recovery: posteriors from data simulated at 3 known vectors (128 x 4)
posterior -> results/v07/m6/posterior: 3 runs x 256 particles x 6 generations on the development target;
             pooled particles, H6 clusters per run, marginals, correlations, 16 posterior-predictive draws
select    -> results/v07/m6/select: posterior-predictive selection objective and the G* rule over
             G0_post and G1-G4 (M5 scores); ties go to the family with fewer free parameters.
Only development and selection data are used (via the frozen v0.6 target sketches).
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
import time

import numpy as np

from ...experiments.registry import PROJECT_ROOT
from ...sim_v2 import SimulatorSpec
from ...v06.calibration import NAMES, spec_from_unit, unit_from_spec
from ..evidence.runs import finalize, new_run, write_json
from ..generators.study import v06_inputs
from ..protocol import core as pr
from ..realism.scoring import objective_of, run_tasks, seed_se, simulate_sketches
from ..v06_compat import pool
from . import smc_abc

DEVELOP_RUN = "results/v06/m6/develop"
REGISTERED = {"particles": 256, "generations": 6, "runs": 3, "seeds_per": 2, "seconds": 1800.0}
RECOVERY = {"truths": 3, "particles": 128, "generations": 4, "target_seeds": 4, "target_seconds": 3600.0}
DRAWS = 16
FREE_PARAMETERS = {"G0_post": 14, "G1_state_hawkes": 96, "G2_regime_switching": 30,
                   "G3_conditional_ar": 10**6, "G4_neural_temporal": 4000}   # G3 is nonparametric (stored rows)


def _family_inputs(root: Path) -> dict:
    inputs = v06_inputs(root)
    developed = json.loads((root / DEVELOP_RUN / "result.json").read_text(encoding="utf-8"))
    return {**inputs, "base": developed["base_config"], "tables": developed["size_tables"]}


def record_derived_use(root: Path, dataset: str, use: str, design: str, path: Path, reason: str) -> None:
    """Ledger a use of a dataset through a derived, hash-identified file (v0.6 frozen target sketches)."""
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    pr.append_event(root / pr.LEDGER_PATH, "access", dataset=dataset, role=pr.dataset_declaration(protocol, dataset)[
        "role"], design=design, reason=reason, protocol=protocol, root=root,
        payload={"use": use, "stage": "evaluated", "derived_source": path.relative_to(root).as_posix(),
                 "source_sha256": {path.name: pr.file_sha256(path)}})


def _logger(path: Path):
    def log(entry):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"t": time.strftime("%H:%M:%S"), **entry}) + "\n")
    return log


def recovery(out: str | Path, *, root: Path = PROJECT_ROOT, budget: dict | None = None, label: str = "registered",
             workers: int | None = None) -> dict:
    budget = {**RECOVERY, **(budget or {})}
    seeds = pr.load_protocol(root / pr.PROTOCOL_PATH)["seeds"]
    inputs = _family_inputs(root)
    frozen = inputs["frozen"]
    out = new_run(root / out)
    log = _logger(out.parent / f"{out.name}.log")
    truths = [unit_from_spec(SimulatorSpec(inputs["selection"]["selected"]["config"],
                                           inputs["selection"]["selected"]["extensions"]))]
    rng = np.random.default_rng(seeds["recovery"][0])
    truths += [rng.random(len(NAMES)) for _ in range(budget["truths"] - 1)]
    results = []
    for k, truth in enumerate(truths[: budget["truths"]]):
        spec = spec_from_unit(truth, inputs["base"], inputs["tables"])
        target_seeds = list(range(seeds["recovery"][k] * 100, seeds["recovery"][k] * 100 + budget["target_seeds"]))
        target = pool(simulate_sketches(spec, target_seeds, budget["target_seconds"], frozen["design"]))
        context = {"design": frozen["design"], "scales": frozen["objective_scales"], "targets": {"synthetic": target},
                   "keep_sketches": False}
        evaluator = smc_abc.Evaluator(base=inputs["base"], tables=inputs["tables"], context=context,
                                      target="synthetic", seed_base=seeds["smc_abc_simulation_seed_base"] + 5_000_000
                                      + k * 100_000, workers=workers)
        log({"truth": k, "unit": np.asarray(truth).round(4).tolist()})
        posterior = smc_abc.run(evaluator, particles=budget["particles"], generations=budget["generations"],
                                seed=seeds["recovery"][k], log=log)
        finite = np.isfinite(posterior["distances"])
        cov = smc_abc.coverage(posterior["particles"][finite], np.asarray(truth))
        results.append({"truth": np.asarray(truth).tolist(), "coverage": cov, "tolerance": posterior["tolerance"],
                        "history": posterior["history"], "marginals": smc_abc.marginals(posterior["particles"]),
                        "evaluations": posterior["evaluations"]})
    passed = [r["coverage"]["covered_fraction"] >= 0.8 for r in results]
    result = {"label": label, "budget": budget, "truths": results, "passed": passed,
              "status": "PASS" if all(passed) else "ASSUMPTION_DEPENDENT",
              "rule": "each truth: the 90% marginal intervals cover the true value for >= 80% of the 14 parameters; "
                      "any failure labels the G0 posterior ASSUMPTION_DEPENDENT",
              "note": "Recovery uses 4 x 3600 s synthetic targets (development day: 144 x 600 s blocks)."}
    finalize(out, analysis=f"m6-recovery-{label}", dataset_ids=[], config={"budget": budget}, result=result,
             root=root, seeds={"recovery": seeds["recovery"]})
    return result


def posterior(out: str | Path, *, root: Path = PROJECT_ROOT, budget: dict | None = None, label: str = "registered",
              workers: int | None = None) -> dict:
    budget = {**REGISTERED, **(budget or {})}
    seeds = pr.load_protocol(root / pr.PROTOCOL_PATH)["seeds"]
    inputs = _family_inputs(root)
    frozen = inputs["frozen"]
    out = new_run(root / out)
    log = _logger(out.parent / f"{out.name}.log")
    context = {"design": frozen["design"], "scales": frozen["objective_scales"],
               "targets": {"development": inputs["targets"]["development"]}, "keep_sketches": False}
    record_derived_use(root, "deribit-eth-perp-2020-04-01", "develop", f"m6-posterior-{label}",
                       root / "results/v06/m1/design/development-sketches.npz",
                       "ABC distance target: frozen v0.6 development sketches")
    runs = []
    for k, run_seed in enumerate(seeds["smc_abc_runs"][: budget["runs"]]):
        evaluator = smc_abc.Evaluator(base=inputs["base"], tables=inputs["tables"], context=context,
                                      target="development", seed_base=seeds["smc_abc_simulation_seed_base"]
                                      + k * 1_000_000, seeds_per=budget["seeds_per"], seconds=budget["seconds"],
                                      workers=workers)
        log({"run": k, "seed": run_seed})
        r = smc_abc.run(evaluator, particles=budget["particles"], generations=budget["generations"], seed=run_seed,
                        log=log)
        runs.append(r)
    pooled = np.vstack([r["particles"] for r in runs])
    distances = np.concatenate([r["distances"] for r in runs])
    rng = np.random.default_rng(seeds["posterior_draws"])
    draw_index = np.sort(rng.choice(len(pooled), DRAWS, replace=False))
    draws = pooled[draw_index]
    np.savez_compressed(out / "particles.npz", pooled=pooled, distances=distances, draws=draws,
                        **{f"run{k}": r["particles"] for k, r in enumerate(runs)})
    cluster_runs = [smc_abc.clusters(r["particles"]) for r in runs]
    multimodal = [c["major_components"] >= 2 for c in cluster_runs]
    h6 = ("ESTABLISHED" if sum(multimodal) >= 2 else "NOT_ESTABLISHED")
    result = {
        "label": label, "budget": budget, "parameters": list(NAMES),
        "runs": [{"seed": s, "tolerance": r["tolerance"], "history": r["history"], "evaluations": r["evaluations"],
                  "marginals": smc_abc.marginals(r["particles"]), "clusters": c}
                 for s, r, c in zip(seeds["smc_abc_runs"], runs, cluster_runs)],
        "pooled": {"marginals": smc_abc.marginals(pooled), "correlation": smc_abc.correlation(pooled),
                   "clusters": smc_abc.clusters(pooled), "distance_quantiles": {
                       q: float(np.quantile(distances[np.isfinite(distances)], q)) for q in (0.05, 0.5, 0.95)}},
        "between_run_mean_spread": {name: float(np.ptp([r["particles"][:, i].mean() for r in runs]))
                                    for i, name in enumerate(NAMES)},
        "draws": draws.tolist(), "draw_index": draw_index.tolist(),
        "H6": {"status": h6, "multimodal_runs": multimodal, "rule": "major components (>= 5% mass, single linkage, "
               "Chebyshev separation > 0.25) >= 2 in at least 2 of 3 runs", "kind": "descriptive"},
        "interpretation": "ABC posterior at the final tolerance of each run; not the exact Bayesian posterior."}
    finalize(out, analysis=f"m6-posterior-{label}", dataset_ids=["deribit-eth-perp-2020-04-01"],
             config={"budget": budget, "develop_run": DEVELOP_RUN}, result=result, root=root,
             seeds={"runs": seeds["smc_abc_runs"], "draws": seeds["posterior_draws"]})
    return result


def posterior_specs(run: Path, root: Path = PROJECT_ROOT) -> list[SimulatorSpec]:
    inputs = _family_inputs(root)
    with np.load(run / "particles.npz") as stored:
        draws = stored["draws"]
    return [spec_from_unit(u, inputs["base"], inputs["tables"]) for u in draws]


def select(out: str | Path, *, posterior_run: str, generator_run: str, root: Path = PROJECT_ROOT,
           workers: int | None = None) -> dict:
    """Posterior-predictive selection objective, then the sealed G* rule."""
    from ..evidence.runs import verify_run
    for run in (posterior_run, generator_run):
        report = verify_run(root / run, root=root)
        if not report["valid"]:
            raise ValueError(f"{run} is not a valid v0.7 run: {report['issues']}")
    seeds = pr.load_protocol(root / pr.PROTOCOL_PATH)["seeds"]
    inputs = _family_inputs(root)
    frozen = inputs["frozen"]
    out = new_run(root / out)
    specs = posterior_specs(root / posterior_run, root)
    for dataset, use, name in (("deribit-eth-perp-2020-04-01", "develop", "development"),
                               ("deribit-eth-perp-2020-05-01", "select", "selection")):
        record_derived_use(root, dataset, use, "m6-generator-selection",
                           root / f"results/v06/m1/design/{name}-sketches.npz",
                           "posterior-predictive scoring and the sealed G* rule")
    context = {"design": frozen["design"], "scales": frozen["objective_scales"], "targets": inputs["targets"]}
    tasks = [(f"draw{i}|{t}", spec, seeds["selection_seeds"], 3600.0, t) for i, spec in enumerate(specs)
             for t in ("development", "selection")]
    results = run_tasks(tasks, context, workers=workers)
    sketches = {t: [s for r in results if r["key"].endswith(t) for s in r["sketches"]] for t in ("development",
                                                                                                 "selection")}
    predictive = {t: objective_of(inputs["targets"][t], sketches[t], frozen["design"], frozen["objective_scales"])
                  for t in sketches}
    per_draw = {t: [r.get("objective") for r in results if r["key"].endswith(t)] for t in sketches}
    scores = json.loads((root / generator_run / "result.json").read_text(encoding="utf-8"))["scores"]
    table = {"G0_post": predictive["selection"]}
    table.update({k: v["selection"]["objective"] for k, v in scores.items() if k != "G0_point"})
    eligible = {k: v for k, v in table.items() if v is not None}
    selected = min(eligible, key=lambda k: (eligible[k], FREE_PARAMETERS.get(k, 10**9)))
    result = {"posterior_predictive_objective": predictive, "per_draw_objective": per_draw,
              "per_draw_seed_se": {t: seed_se(per_draw[t]) for t in per_draw},
              "selection_table": table, "G_star": selected, "G0_point_selection": scores["G0_point"]["selection"],
              "rule": "G* = argmin selection objective over G0_post (posterior predictive) and G1-G4; ties to fewer "
                      "free parameters", "free_parameters": FREE_PARAMETERS}
    write_json(out, "predictive-sketches-meta.json", {t: len(v) for t, v in sketches.items()})
    with (out / "predictive-sketches.pkl").open("xb") as handle:
        pickle.dump(sketches, handle)
    finalize(out, analysis="m6-generator-selection", dataset_ids=["deribit-eth-perp-2020-04-01",
                                                                  "deribit-eth-perp-2020-05-01"],
             config={"posterior_run": posterior_run, "generator_run": generator_run}, result=result, root=root,
             seeds={"selection_seeds": seeds["selection_seeds"]})
    return result
