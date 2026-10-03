"""M8 identifiability v2 study: component-level Jacobian at the posterior's best particle, standardized by
measured seed noise, combined with the G0 posterior (development data only, via the frozen v0.6 target)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np

from ...experiments.registry import PROJECT_ROOT
from ...v06.calibration import NAMES, spec_from_unit
from ...v06.identifiability import clean
from ...v06.observables import COMPONENTS
from ...v06.realism import compare
from ..evidence.runs import finalize, new_run, verify_run
from ..posterior.study import _family_inputs, record_derived_use
from ..realism.scoring import simulate_sketches
from ..v06_compat import pool
from . import analysis

SEEDS = (78101, 78102, 78103)
SECONDS = 1800.0
STEP = 0.05


def _component_errors(task: tuple) -> list:
    unit, base, tables, design, scales, target, seeds = task
    spec = spec_from_unit(np.asarray(unit), base, tables)
    try:
        sketches = simulate_sketches(spec, list(seeds), SECONDS, design)
    except (RuntimeError, ValueError):
        return [None] * len(COMPONENTS)
    result = compare(target, pool(sketches), design, scales)
    return [result["components"][c.name]["error"] for c in COMPONENTS]


def run(out: str | Path, *, posterior_run: str, root: Path = PROJECT_ROOT, workers: int | None = None) -> dict:
    report = verify_run(root / posterior_run, root=root)
    if not report["valid"]:
        raise ValueError(f"{posterior_run} is not valid: {report['issues']}")
    inputs = _family_inputs(root)
    frozen = inputs["frozen"]
    out = new_run(root / out)
    record_derived_use(root, "deribit-eth-perp-2020-04-01", "develop", "m8-identifiability-v2",
                       root / "results/v06/m1/design/development-sketches.npz", "identifiability Jacobian target")
    with np.load(root / posterior_run / "particles.npz") as stored:
        pooled, distances = stored["pooled"], stored["distances"]
    center = pooled[int(np.nanargmin(np.where(np.isfinite(distances), distances, np.inf)))]
    common = (inputs["base"], inputs["tables"], frozen["design"], frozen["objective_scales"],
              inputs["targets"]["development"])
    units = [center]
    steps = []
    for i in range(len(NAMES)):
        step = min(STEP, center[i], 1 - center[i]) or STEP / 2
        up, down = center.copy(), center.copy()
        up[i], down[i] = min(1.0, center[i] + step), max(0.0, center[i] - step)
        units += [up, down]
        steps.append(float(up[i] - down[i]))
    tasks = [(u.tolist(), *common, SEEDS) for u in units]
    noise_tasks = [(center.tolist(), *common, (s,)) for s in SEEDS]
    workers = workers or max(1, min(14, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        values = list(executor.map(_component_errors, tasks + noise_tasks))
    matrix = np.asarray([[np.nan if v is None else v for v in row] for row in values], float)
    center_row, diffs, noise_rows = matrix[0], matrix[1:1 + 2 * len(NAMES)], matrix[1 + 2 * len(NAMES):]
    jac = np.column_stack([(diffs[2 * i] - diffs[2 * i + 1]) / steps[i] for i in range(len(NAMES))])
    usable = np.all(np.isfinite(jac), axis=1) & np.isfinite(center_row)
    names = [c.name for c, ok in zip(COMPONENTS, usable) if ok]
    families = [c.family for c, ok in zip(COMPONENTS, usable) if ok]
    # Rows are scaled by the seed noise of a 3-seed evaluation (single-seed SD / sqrt(3)). Components that are not
    # evaluable in at least two single-seed runs have no noise estimate and are dropped (listed in the result).
    finite_noise = np.sum(np.isfinite(noise_rows), axis=0) >= 2
    usable = usable & finite_noise
    names = [c.name for c, ok in zip(COMPONENTS, usable) if ok]
    families = [c.family for c, ok in zip(COMPONENTS, usable) if ok]
    noise = np.nanstd(noise_rows[:, usable], axis=0, ddof=1) / np.sqrt(len(SEEDS))
    j = analysis.standardize(jac[usable], np.maximum(noise, 1e-6))
    posterior = analysis.posterior_summary(pooled, distances, list(NAMES))
    result = {"center_unit": center.tolist(), "steps": steps, "observables_used": len(names),
              "observables_dropped": [c.name for c, ok in zip(COMPONENTS, usable) if not ok],
              "sloppiness": analysis.sloppiness(j), "posterior": posterior,
              "parameters": analysis.classify(j, posterior, list(NAMES)),
              "observable_design": analysis.observable_design(j, names, families, list(NAMES)),
              "jacobian_standardized": j.round(6).tolist(), "observables": names,
              "v06_comparison": "v0.6 differentiated 9 family errors (rank <= 9); v0.7 uses component errors",
              "label": "DEVELOPMENT; statements about this simulator family, observables and resolution only"}
    finalize(out, analysis="m8-identifiability-v2", dataset_ids=["deribit-eth-perp-2020-04-01"],
             config={"posterior_run": posterior_run, "seeds": list(SEEDS), "seconds": SECONDS, "step": STEP},
             result=clean(result), root=root, seeds={"jacobian": list(SEEDS)})
    return result
