"""Identifiability v2 and observable design (workstreams 12, 13).

Inputs: a component-error Jacobian ``J`` (observables x parameters, unit-box
parameters, central differences with common random numbers) and the ABC
posterior particles. The v0.6 analysis capped the rank at the nine family
errors; v0.7 differentiates all 85 component errors, removing that cap.

* local sensitivity — column norms of the standardized Jacobian;
* sloppiness — eigenvalues of F = J'J: effective rank (eigenvalues above 1e-3 of the largest),
  participation ratio, log10 spread, weak directions (eigenvectors of the smallest eigenvalues);
* posterior — 90% interval widths, correlations, compensation pairs (|rho| >= 0.7), profiles
  (minimum ABC distance per parameter bin) and profile flatness;
* identification class per parameter — STRUCTURALLY_NOT_IDENTIFIED (no observable responds),
  PRACTICALLY_NOT_IDENTIFIED (responds, but the posterior spans >= 50% of its range or the profile
  is flat), IDENTIFIED otherwise. These are statements about this simulator family, these
  observables and this data resolution only.
* observable design — redundant pairs (|cosine| >= 0.95), weak observables (row norm < 5% of the
  maximum), complementary observables (one dominant parameter), uninformative families.
"""
from __future__ import annotations

import numpy as np

WEAK_ROW = 0.05
REDUNDANT_COS = 0.95
COMPENSATION_RHO = 0.7


def standardize(j: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
    """Scale each observable row by its seed noise (or its own scale) so rows are comparable."""
    scale = noise if noise is not None else np.maximum(np.abs(j).max(1), 1e-12)
    return j / np.maximum(scale, 1e-12)[:, None]


def sloppiness(j: np.ndarray) -> dict:
    if not np.all(np.isfinite(j)):
        raise ValueError("Jacobian contains non-finite entries; drop unevaluable observables first")
    f = j.T @ j
    values, vectors = np.linalg.eigh(f)
    order = np.argsort(values)[::-1]
    values, vectors = np.clip(values[order], 0, None), vectors[:, order]
    top = values[0] if len(values) else 0.0
    rank = int(np.sum(values > 1e-3 * top)) if top > 0 else 0
    participation = float(values.sum() ** 2 / np.sum(values ** 2)) if top > 0 else 0.0
    positive = values[values > 0]
    spread = float(np.log10(positive.max() / positive.min())) if len(positive) > 1 else 0.0
    return {"eigenvalues": values.tolist(), "effective_rank": rank, "participation_ratio": participation,
            "log10_eigen_spread": spread, "weak_directions": vectors[:, -3:].T.tolist()}


def posterior_summary(particles: np.ndarray, distances: np.ndarray, names: list[str], bins: int = 10) -> dict:
    finite = np.isfinite(distances)
    x, d = particles[finite], distances[finite]
    lo, hi = np.quantile(x, 0.05, axis=0), np.quantile(x, 0.95, axis=0)
    corr = np.nan_to_num(np.corrcoef(x.T))
    pairs = [(names[i], names[k], float(corr[i, k])) for i in range(len(names)) for k in range(i + 1, len(names))
             if abs(corr[i, k]) >= COMPENSATION_RHO]
    profiles, flatness = {}, {}
    for i, name in enumerate(names):
        edges = np.linspace(0, 1, bins + 1)
        values = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (x[:, i] >= a) & (x[:, i] <= b)
            values.append(float(d[m].min()) if m.any() else None)
        profiles[name] = values
        seen = [v for v in values if v is not None]
        flatness[name] = (float((max(seen) - min(seen)) / max(min(seen), 1e-9)) if len(seen) > 1 else None)
    return {"interval90_width": dict(zip(names, (hi - lo).tolist())), "correlation": corr.round(4).tolist(),
            "compensation_pairs": pairs, "profiles": profiles, "profile_relative_range": flatness}


def classify(j: np.ndarray, posterior: dict, names: list[str]) -> dict:
    influence = np.linalg.norm(j, axis=0)
    top = influence.max() if len(influence) else 0.0
    table = {}
    for i, name in enumerate(names):
        width = posterior["interval90_width"][name]
        flat = posterior["profile_relative_range"][name]
        if top == 0 or influence[i] < 1e-6 * top:
            status = "STRUCTURALLY_NOT_IDENTIFIED"
        elif width >= 0.5 or (flat is not None and flat < 0.05):
            status = "PRACTICALLY_NOT_IDENTIFIED"
        else:
            status = "IDENTIFIED"
        table[name] = {"influence": float(influence[i]), "relative_influence": float(influence[i] / top) if top else 0.0,
                       "posterior_interval90_width": width, "profile_relative_range": flat, "status": status}
    return table


def observable_design(j: np.ndarray, observables: list[str], families: list[str], params: list[str]) -> dict:
    rows = np.linalg.norm(j, axis=1)
    top = rows.max() if len(rows) else 0.0
    weak = [observables[i] for i in range(len(rows)) if top == 0 or rows[i] < WEAK_ROW * top]
    unit = j / np.maximum(rows, 1e-12)[:, None]
    cos = unit @ unit.T
    redundant = [(observables[a], observables[b], float(cos[a, b])) for a in range(len(rows)) for b in range(a + 1, len(rows))
                 if rows[a] >= WEAK_ROW * top and rows[b] >= WEAK_ROW * top and abs(cos[a, b]) >= REDUNDANT_COS]
    dominant = {}
    for i, name in enumerate(observables):
        if name in weak:
            continue
        share = np.abs(j[i]) / max(np.abs(j[i]).sum(), 1e-12)
        k = int(np.argmax(share))
        if share[k] >= 0.5:
            dominant[name] = {"parameter": params[k], "share": float(share[k])}
    by_family: dict[str, list[str]] = {}
    for name, fam in zip(observables, families):
        by_family.setdefault(fam, []).append(name)
    uninformative = [f for f, names in by_family.items() if all(n in weak for n in names)]
    return {"weak_observables": weak, "redundant_pairs": redundant[:200], "redundant_pair_count": len(redundant),
            "complementary_observables": dominant, "uninformative_families": uninformative,
            "row_norms": dict(zip(observables, rows.tolist()))}


def jacobian(evaluate, center: np.ndarray, *, h: float = 0.05) -> tuple[np.ndarray, list]:
    """Central differences of a vector function on the unit box with common random numbers (``evaluate``
    must use identical seeds for every call). Steps are shrunk near the box boundary."""
    columns, steps = [], []
    for i in range(len(center)):
        step = min(h, center[i], 1 - center[i]) or h / 2
        up, down = center.copy(), center.copy()
        up[i] = min(1.0, center[i] + step)
        down[i] = max(0.0, center[i] - step)
        columns.append((np.asarray(evaluate(up)) - np.asarray(evaluate(down))) / (up[i] - down[i]))
        steps.append(float(up[i] - down[i]))
    return np.column_stack(columns), steps
