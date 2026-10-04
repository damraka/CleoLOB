"""Execution-aware calibration (workstream 18; hypotheses H10/H11), development data only.

Candidates are the 64 best distinct particles of the pooled G0 posterior (by
their last ABC distance). Each is re-simulated on 3 seeds x 1800 s and compared
with the development target; the G0 point model is re-simulated on the same
seeds. The execution-aware model (EA) minimizes the execution-sensitive (ES)
objective among candidates whose generic objective is at most 1.10 x the G0
point model's (the noninferiority guard used during selection; H11 tests it
again on fresh data with the registered margin). Ties go to the lower generic
objective. Nothing here reads selection, retrospective or fresh data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...experiments.registry import PROJECT_ROOT
from ...sim_v2 import SimulatorSpec
from ...v06.calibration import spec_from_unit
from ...v06.realism import compare
from ..evidence.runs import finalize, new_run
from ..posterior.study import _family_inputs, record_derived_use
from ..realism.contrast import ES_COMPONENTS, component_scales, es_objective
from ..realism.scoring import run_tasks
from ..v06_compat import pool

CANDIDATES = 64
SEEDS = (77101, 77102, 77103)
SECONDS = 1800.0
GUARD = 1.10


def run(out: str | Path, *, posterior_run: str, root: Path = PROJECT_ROOT, workers: int | None = None) -> dict:
    from ..evidence.runs import verify_run
    report = verify_run(root / posterior_run, root=root)
    if not report["valid"]:
        raise ValueError(f"{posterior_run} is not valid: {report['issues']}")
    inputs = _family_inputs(root)
    frozen = inputs["frozen"]
    out = new_run(root / out)
    record_derived_use(root, "deribit-eth-perp-2020-04-01", "develop", "m7-execution-aware",
                       root / "results/v06/m1/design/development-sketches.npz",
                       "execution-aware calibration on the frozen v0.6 development sketches")
    with np.load(root / posterior_run / "particles.npz") as stored:
        pooled, distances = stored["pooled"], stored["distances"]
    order = np.argsort(distances, kind="stable")
    units, seen = [], set()
    for i in order:
        key = tuple(np.round(pooled[i], 10))
        if np.isfinite(distances[i]) and key not in seen:
            seen.add(key)
            units.append(pooled[i])
        if len(units) == CANDIDATES:
            break
    selected = inputs["selection"]["selected"]
    specs = {f"c{i:02d}": spec_from_unit(u, inputs["base"], inputs["tables"]) for i, u in enumerate(units)}
    specs["G0_point"] = SimulatorSpec(selected["config"], selected["extensions"])
    context = {"design": frozen["design"], "scales": frozen["objective_scales"],
               "targets": {"development": inputs["targets"]["development"]}}
    results = run_tasks([(k, spec, list(SEEDS), SECONDS, "development") for k, spec in specs.items()], context,
                        workers=workers)
    es_scales = component_scales(frozen)
    scored = {}
    for r in results:
        if r["error"] or not r["sketches"]:
            scored[r["key"]] = {"generic": None, "es": None, "error": r["error"]}
            continue
        comparison = compare(inputs["targets"]["development"], pool(r["sketches"]), frozen["design"],
                             frozen["objective_scales"])
        scored[r["key"]] = {"generic": comparison["objective"], "es": es_objective(comparison, es_scales),
                            "es_components": {c: comparison["components"][c]["error"] for c in ES_COMPONENTS},
                            "error": None}
    reference = scored["G0_point"]["generic"]
    feasible = {k: v for k, v in scored.items() if k != "G0_point" and v["generic"] is not None and v["es"] is not None
                and v["generic"] <= GUARD * reference}
    if not feasible:
        raise ValueError("no execution-aware candidate satisfies the generic-objective guard")
    best = min(feasible, key=lambda k: (feasible[k]["es"], feasible[k]["generic"]))
    spec = specs[best]
    result = {"selected": {"key": best, "unit": units[int(best[1:])].tolist(), "config": spec.config,
                           "extensions": spec.extensions, **scored[best]},
              "G0_point_development": scored["G0_point"], "guard": GUARD, "candidates": scored,
              "es_components": list(ES_COMPONENTS), "es_scales": es_scales, "seeds": list(SEEDS),
              "seconds": SECONDS, "label": "DEVELOPMENT (selection of the execution-aware model)"}
    finalize(out, analysis="m7-execution-aware", dataset_ids=["deribit-eth-perp-2020-04-01"],
             config={"posterior_run": posterior_run, "candidates": CANDIDATES, "guard": GUARD}, result=result,
             root=root, seeds={"evaluation": list(SEEDS)})
    return result


def spec(run_dir: Path) -> SimulatorSpec:
    import json
    selected = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["selected"]
    return SimulatorSpec(selected["config"], selected["extensions"])

