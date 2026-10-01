"""M7/M10: parameter identifiability, sensitivity and equifinality diagnostics for calibration v3.

All evaluations use development data and the calibration fit seeds; nothing
here reads a holdout. Diagnostics:

* near-optimal ranges, correlations and principal (flat) directions of the
  near-optimal parameter vectors in the transformed unit box;
* local one-at-a-time sensitivity of every family error around the selected
  model (common random numbers: same seeds for baseline and perturbation),
  and the eigen-structure of the sensitivity Gram matrix (small eigenvalues are
  locally flat directions);
* profile diagnostics: each parameter fixed on a 5-point grid while the others
  are re-optimized by seeded perturbation draws;
* optional Morris elementary effects (labelled EXPLORATORY).
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from .calibration import NAMES, PARAMETERS, spec_from_unit
from .observables import FAMILIES

GRID = (0.1, 0.3, 0.5, 0.7, 0.9)


def near_optimal_structure(units: dict[str, np.ndarray]) -> dict:
    keys = sorted(units)
    matrix = np.vstack([units[k] for k in keys]) if keys else np.empty((0, len(NAMES)))
    out = {"members": len(keys), "unit_min": None, "unit_max": None, "natural_ranges": None,
           "correlations": None, "principal_variances": None, "principal_directions": None}
    if not keys:
        return out
    out["unit_min"] = dict(zip(NAMES, matrix.min(0).tolist()))
    out["unit_max"] = dict(zip(NAMES, matrix.max(0).tolist()))
    out["natural_ranges"] = {p.name: [p.value(lo), p.value(hi)] for p, lo, hi in
                             zip(PARAMETERS, matrix.min(0), matrix.max(0))}
    if len(keys) >= 3:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(matrix.T)
        out["correlations"] = np.where(np.isfinite(corr), corr, 0.0).round(6).tolist()
        cov = np.cov(matrix.T)
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        out["principal_variances"] = values[order].tolist()
        out["principal_directions"] = [dict(zip(NAMES, vectors[:, i].round(6).tolist())) for i in order[:3]]
    return out


def sensitivity_tasks(selected: np.ndarray, *, step: float, base: dict, tables: dict, seeds: list[int],
                      seconds: float, target: str) -> list[tuple]:
    tasks = []
    centre = spec_from_unit(selected, base, tables)
    tasks.append(("sens-centre", {"config": centre.config, "extensions": centre.extensions}, seeds, seconds, target))
    for i, name in enumerate(NAMES):
        for sign, label in ((1, "up"), (-1, "down")):
            u = selected.copy()
            u[i] = float(np.clip(u[i] + sign * step, 0.0, 1.0))
            spec = spec_from_unit(u, base, tables)
            tasks.append((f"sens-{name}-{label}", {"config": spec.config, "extensions": spec.extensions}, seeds,
                          seconds, target))
    return tasks


def sensitivity_matrix(records: dict[str, dict], selected: np.ndarray, *, step: float) -> dict:
    """Central finite differences of family errors per unit step; Gram eigen-structure."""
    centre = np.asarray(records["sens-centre"]["families"], dtype=float)
    rows, actual_steps = [], []
    for i, name in enumerate(NAMES):
        up = np.asarray(records[f"sens-{name}-up"]["families"], dtype=float)
        down = np.asarray(records[f"sens-{name}-down"]["families"], dtype=float)
        width = min(1.0, selected[i] + step) - max(0.0, selected[i] - step)
        actual_steps.append(width)
        rows.append((up - down) / width if width > 0 else np.full(len(FAMILIES), np.nan))
    matrix = np.vstack(rows).T   # families x parameters
    finite = np.where(np.isfinite(matrix), matrix, 0.0)
    gram = finite.T @ finite
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)
    return {"families": list(FAMILIES), "parameters": list(NAMES), "matrix": matrix.tolist(),
            "centre_family_errors": centre.tolist(), "steps": actual_steps,
            "gram_eigenvalues": values[order].tolist(),
            "flattest_directions": [dict(zip(NAMES, vectors[:, j].round(6).tolist())) for j in order[:3]],
            "most_sensitive_parameter_per_family": {
                f: (NAMES[int(np.nanargmax(np.abs(matrix[k])))] if np.isfinite(matrix[k]).any() else None)
                for k, f in enumerate(FAMILIES)},
            "parameter_influence": dict(zip(NAMES, np.sqrt(np.diag(gram)).tolist()))}


def profile_tasks(selected: np.ndarray, *, draws: int, step: float, seed: int, base: dict, tables: dict,
                  seeds: list[int], seconds: float, target: str) -> list[tuple]:
    rng = np.random.default_rng(seed)
    tasks = []
    for i, name in enumerate(NAMES):
        for g, value in enumerate(GRID):
            for d in range(draws):
                u = np.clip(selected + rng.normal(0, step, len(NAMES)), 0.0, 1.0)
                u[i] = value
                spec = spec_from_unit(u, base, tables)
                tasks.append((f"prof-{name}-{g}-{d:02d}", {"config": spec.config, "extensions": spec.extensions},
                              seeds, seconds, target))
    return tasks


def profiles(records: dict[str, dict], *, draws: int, noise_se: float) -> dict:
    """Best objective per grid value; a profile is flat if its range is within 2 noise SE."""
    out = {}
    for name in NAMES:
        curve = []
        for g, value in enumerate(GRID):
            values = [records[f"prof-{name}-{g}-{d:02d}"]["objective"] for d in range(draws)]
            finite = [v for v in values if v is not None and math.isfinite(v)]
            curve.append(min(finite) if finite else None)
        finite = [v for v in curve if v is not None]
        spread = (max(finite) - min(finite)) if len(finite) >= 2 else None
        out[name] = {"grid": list(GRID), "best_objective": curve, "range": spread,
                     "flat_within_noise": None if spread is None else spread <= 2 * noise_se}
    return out


def morris_tasks(*, trajectories: int, seed: int, base: dict, tables: dict, seeds: list[int], seconds: float,
                 target: str, levels: int = 4) -> tuple[list[tuple], list[list[tuple[int, int]]]]:
    """One-at-a-time trajectories on a 4-level grid with delta = levels / (2 (levels - 1))."""
    rng = np.random.default_rng(seed)
    delta = levels / (2 * (levels - 1))
    starts = np.arange(levels // 2) / (levels - 1)
    tasks, layout = [], []
    for t in range(trajectories):
        u = rng.choice(starts, len(NAMES))
        order = rng.permutation(len(NAMES))
        path = [(t, -1)]
        points = [u.copy()]
        for i in order:
            u = u.copy()
            u[i] = u[i] + delta
            points.append(u)
            path.append((t, int(i)))
        for k, point in enumerate(points):
            spec = spec_from_unit(point, base, tables)
            tasks.append((f"morris-{t:02d}-{k:02d}", {"config": spec.config, "extensions": spec.extensions}, seeds,
                          seconds, target))
        layout.append(path)
    return tasks, layout


def morris_effects(records: dict[str, dict], layout: list[list[tuple[int, int]]], *, levels: int = 4) -> dict:
    delta = levels / (2 * (levels - 1))
    effects: dict[str, list[np.ndarray]] = {n: [] for n in NAMES}
    for path in layout:
        t = path[0][0]
        for k in range(1, len(path)):
            before = np.asarray(records[f"morris-{t:02d}-{k - 1:02d}"]["families"], dtype=float)
            after = np.asarray(records[f"morris-{t:02d}-{k:02d}"]["families"], dtype=float)
            effects[NAMES[path[k][1]]].append((after - before) / delta)
    out = {}
    for name, values in effects.items():
        stacked = np.vstack(values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[name] = {"mu_star": clean(np.nanmean(np.abs(stacked), 0).tolist()),
                         "sigma": clean(np.nanstd(stacked, 0).tolist())}
    return {"families": list(FAMILIES), "effects": out, "status": "EXPLORATORY"}


def clean(value):
    """Recursively replace non-finite floats by None (strict JSON; NaN means NOT_EVALUABLE here)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value
