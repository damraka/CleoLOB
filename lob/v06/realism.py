"""v0.6 realism comparison: distances, component and family errors, objective, scorecard.

All quantities are computed from pooled sketches (``lob.v06.observables``).
Distribution components are binned on development-frozen bins in a transformed
space (log for positive heavy-tailed quantities); overflow mass is placed at its
mean, so a distance's resolution is one bin width. Reported per distribution:
normalized 1-Wasserstein (the primary component error), Kolmogorov-Smirnov,
1-D energy distance 2*int (F-G)^2, Jensen-Shannon divergence (base 2, frozen
bins) and quantile errors. Scalar components use the error rule declared in the
component registry. Families are never collapsed into a validity score; the
calibration objective is a separately labelled, preregistered optimization target.
"""
from __future__ import annotations

import math

import numpy as np

from .observables import BINS, COMPONENTS, FAMILIES, STATES

ERROR_CAP = 10.0
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
TAIL_QUANTILES = (0.01, 0.99)
_TRADE, _CANCEL = STATES.index("trade"), STATES.index("cancel")


# ----------------------------------------------------------------------------- distributions


def _points(vector: np.ndarray, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Support points (bin centres plus overflow means) and masses of one histogram sketch."""
    counts = vector[:BINS]
    under_n, under_sum, over_n, over_sum = vector[BINS:BINS + 4]
    edges = np.linspace(spec["lo"], spec["hi"], BINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    positions = np.r_[under_sum / under_n if under_n else spec["lo"], centres,
                      over_sum / over_n if over_n else spec["hi"]]
    weights = np.r_[under_n, counts, over_n]
    return positions, weights


def histogram_size(vector: np.ndarray) -> float:
    """Number of transformed samples in a histogram sketch (bins plus both overflow counts)."""
    return float(vector[:BINS].sum() + vector[BINS] + vector[BINS + 2])


def _cdfs(a: tuple, b: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Both CDFs on the merged sorted support."""
    (pa, wa), (pb, wb) = a, b
    support = np.unique(np.r_[pa, pb])
    fa = np.cumsum(np.bincount(np.searchsorted(support, pa), weights=wa, minlength=len(support))) / wa.sum()
    fb = np.cumsum(np.bincount(np.searchsorted(support, pb), weights=wb, minlength=len(support))) / wb.sum()
    return support, fa, fb


def _quantile(points: tuple, q: float) -> float:
    positions, weights = points
    order = np.argsort(positions)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return float(positions[order][min(np.searchsorted(cumulative, q), len(order) - 1)])


def distribution_metrics(h: np.ndarray, s: np.ndarray, spec: dict) -> dict:
    """W1 (primary, normalized), KS, energy distance, JSD and quantile errors on the transformed scale."""
    a, b = _points(h, spec), _points(s, spec)
    support, fa, fb = _cdfs(a, b)
    gaps = np.diff(support)
    diff = (fa - fb)[:-1]
    w1 = float(np.sum(np.abs(diff) * gaps))
    energy = float(2 * np.sum(diff * diff * gaps))
    ks = float(np.max(np.abs(fa - fb)))
    pa = np.r_[h[BINS], h[:BINS], h[BINS + 2]] / h[np.r_[BINS, np.arange(BINS), BINS + 2]].sum()
    pb = np.r_[s[BINS], s[:BINS], s[BINS + 2]] / s[np.r_[BINS, np.arange(BINS), BINS + 2]].sum()
    m = (pa + pb) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        jsd = 0.5 * np.nansum(np.where(pa > 0, pa * np.log2(pa / m), 0.0)) + \
            0.5 * np.nansum(np.where(pb > 0, pb * np.log2(pb / m), 0.0))
    scale = spec["scale"]
    q_err = {f"q{int(q * 100):02d}": (_quantile(b, q) - _quantile(a, q)) / scale for q in QUANTILES + TAIL_QUANTILES}
    return {"w1_normalized": min(ERROR_CAP, w1 / scale), "w1": w1, "ks": ks, "energy": energy, "jsd": float(jsd),
            "quantile_errors_normalized": q_err, "scale": scale}


# ----------------------------------------------------------------------------- scalar statistics


def _corr(m: np.ndarray) -> float:
    n, sx, sy, sxx, syy, sxy = m
    if n < 3:
        return math.nan
    vx, vy = sxx / n - (sx / n) ** 2, syy / n - (sy / n) ** 2
    if vx <= 1e-15 or vy <= 1e-15:
        return 0.0
    return float((sxy / n - sx * sy / n ** 2) / math.sqrt(vx * vy))


def _logratio(a: float, b: float, floor: float) -> float:
    return abs(math.log((a + floor) / (b + floor)))


def _fano(v: np.ndarray) -> float:
    n, s, ss = v
    mean = s / n
    return float((ss / n - mean * mean) / mean) if mean > 0 else math.nan


def _burstiness(v: np.ndarray) -> float:
    n, s, ss = v
    mean = s / n
    sd = math.sqrt(max(ss / n - mean * mean, 0.0))
    return float((sd - mean) / (sd + mean)) if sd + mean > 0 else math.nan


def _mi(table: np.ndarray) -> float:
    table = table.reshape(5, 3)
    total = table.sum()
    p = table / total
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p / (px * py)), 0.0)
    return float(np.nansum(terms))


def _markov(vector: np.ndarray) -> np.ndarray:
    return vector.reshape(len(STATES), len(STATES))


def statistic(component, sketch: dict, design: dict) -> tuple[float, float]:
    """(value, sample size) of a scalar component; value NaN when it cannot be formed."""
    kind, source = component.kind, component.source
    if kind == "acf":
        m = sketch[f"acf:{source}"]
        return _corr(m), m[0]
    if kind == "pacf":
        r1, r2 = _corr(sketch["acf:r1:1"]), _corr(sketch["acf:r1:2"])
        n = min(sketch["acf:r1:1"][0], sketch["acf:r1:2"][0])
        return ((r2 - r1 * r1) / (1 - r1 * r1) if abs(r1) < 1 else math.nan), n
    if kind in {"corr", "pit"}:
        m = sketch[f"{kind}:{source}"]
        return _corr(m), m[0]
    if kind == "prob":
        k, n = sketch[f"prob:{source}"]
        return (k / n if n else math.nan), n
    if kind == "exc":
        k, n = sketch[f"exc:{component.name}"]
        return (k / n if n else math.nan), n
    if kind == "hill":
        k, slog = sketch[f"hill:{component.name}"]
        return (k / slog if k and slog > 0 else math.nan), k
    if kind == "ratio":
        v = sketch[f"ratio:{source}"]
        return (_fano(v) if v[0] else math.nan), v[0]
    if kind == "moment":
        v = sketch[f"moment:{source}"]
        return (_burstiness(v) if v[0] else math.nan), v[0]
    if kind == "markov":
        matrix = _markov(sketch["markov"])
        if ">" in source:
            left, right = source.split(":")[1].split(">")
            row = matrix[STATES.index(left)]
            return (row[STATES.index(right)] / row.sum() if row.sum() else math.nan), row.sum()
        return math.nan, matrix.sum()
    if kind == "mi":
        table = sketch[f"mi:{source}"]
        return (_mi(table) if table.sum() else math.nan), table.sum()
    raise ValueError(f"no scalar statistic for {component.name}")


def _markov_error(component, h: dict, s: dict) -> tuple[float | None, dict]:
    mh, ms = _markov(h["markov"]), _markov(s["markov"])
    if mh.sum() < component.min_n or ms.sum() < component.min_n:
        return None, {}
    ph, ps = mh.sum(1) / mh.sum(), ms.sum(1) / ms.sum()
    if component.error == "tv":
        return float(0.5 * np.abs(ph - ps).sum()), {"hist": ph.tolist(), "sim": ps.tolist()}
    tv, weight = 0.0, 0.0
    for i in range(len(STATES)):
        if mh[i].sum() < 30:
            continue
        rh = mh[i] / mh[i].sum()
        rs = ms[i] / ms[i].sum() if ms[i].sum() >= 30 else None
        tv += ph[i] * (0.5 * np.abs(rh - rs).sum() if rs is not None else 1.0)
        weight += ph[i]
    return (float(tv / weight) if weight else None), {}


def _cond_error(h: np.ndarray, s: np.ndarray, scale: float, min_n: int) -> tuple[float | None, dict]:
    nh, sh, ns, ss = h[0::2], h[1::2], s[0::2], s[1::2]
    keep = (nh >= min_n) & (ns >= min_n)
    if not keep.any():
        return None, {}
    mh, ms = sh[keep] / nh[keep], ss[keep] / ns[keep]
    return float(np.mean(np.abs(mh - ms)) / scale), {"hist_means": mh.tolist(), "sim_means": ms.tolist()}


def component_error(component, h: dict, s: dict, design: dict) -> dict:
    """Error and diagnostics of one component; ``error`` is None when NOT_EVALUABLE."""
    kind = component.kind
    record: dict = {"family": component.family, "kind": kind}
    if kind == "dist":
        spec = design["dist"][component.source]
        vh, vs = h[f"dist:{component.source}"], s[f"dist:{component.source}"]
        nh, ns = histogram_size(vh), histogram_size(vs)
        record.update(n_hist=float(nh), n_sim=float(ns))
        if nh < component.min_n or ns < component.min_n:
            return {**record, "error": None, "reason": "too few samples"}
        metrics = distribution_metrics(vh, vs, spec)
        return {**record, "error": metrics["w1_normalized"], **metrics}
    if kind == "tailq":
        target = next(c for c in COMPONENTS if c.name == component.source)
        spec = design["dist"][target.source]
        vh, vs = h[f"dist:{target.source}"], s[f"dist:{target.source}"]
        if histogram_size(vh) < 200 or histogram_size(vs) < 200:
            return {**record, "error": None, "reason": "fewer than 200 samples for 1%/99% quantiles"}
        a, b = _points(vh, spec), _points(vs, spec)
        errors = [abs(_quantile(b, q) - _quantile(a, q)) / spec["scale"] for q in TAIL_QUANTILES]
        return {**record, "error": min(ERROR_CAP, float(np.mean(errors))), "q01": errors[0], "q99": errors[1]}
    if kind == "markov" and ">" not in component.source:
        error, extra = _markov_error(component, h, s)
        return {**record, "error": None if error is None else min(ERROR_CAP, error), **extra}
    if kind == "cond":
        error, extra = _cond_error(h[f"cond:{component.source}"], s[f"cond:{component.source}"],
                                   design["return_scale_bps"], component.min_n)
        return {**record, "error": None if error is None else min(ERROR_CAP, error), **extra}
    value_h, n_h = statistic(component, h, design)
    value_s, n_s = statistic(component, s, design)
    record.update(hist=None if math.isnan(value_h) else value_h, sim=None if math.isnan(value_s) else value_s,
                  n_hist=float(n_h), n_sim=float(n_s))
    if n_h < component.min_n or n_s < component.min_n or math.isnan(value_h) or math.isnan(value_s):
        return {**record, "error": None, "reason": "too few samples or undefined statistic"}
    if component.error == "abs":
        error = abs(value_s - value_h)
    elif component.error in {"logratio", "mi_logratio"}:
        floor = {"exc": 1e-4, "prob": 1e-4, "ratio": 1e-3, "hill": 1e-3}.get(kind, 1e-4)
        error = _logratio(value_s, value_h, floor)
    else:
        raise ValueError(f"unknown error rule {component.error}")
    return {**record, "error": min(ERROR_CAP, float(error))}


# ----------------------------------------------------------------------------- families and objective


def compare(h: dict, s: dict, design: dict, scales: dict | None = None) -> dict:
    """Component errors, family errors (mean and max) and the calibration objective."""
    components = {c.name: component_error(c, h, s, design) for c in COMPONENTS}
    families = {}
    for family in FAMILIES:
        values = [r["error"] for r in components.values() if r["family"] == family and r["error"] is not None]
        total = sum(1 for c in COMPONENTS if c.family == family)
        families[family] = {"error": float(np.mean(values)) if values else None,
                            "max": float(np.max(values)) if values else None,
                            "evaluable": len(values), "components": total}
    return {"components": components, "families": families, "objective": objective(families, scales)}


def family_vector(result: dict) -> np.ndarray:
    return np.asarray([np.nan if result["families"][f]["error"] is None else result["families"][f]["error"]
                       for f in FAMILIES])


def objective(families: dict, scales: dict | None = None) -> float | None:
    """Mean over evaluable families of error / development scale (scale 1 when not given)."""
    values = []
    for family in FAMILIES:
        error = families[family]["error"]
        if error is not None:
            values.append(error / (max(scales[family], 0.05) if scales else 1.0))
    return float(np.mean(values)) if values else None


def quick_objective(h: dict, s: dict, design: dict, scales: dict | None = None) -> tuple[float, np.ndarray]:
    """Objective and family-error vector without diagnostics (used inside bootstraps and searches)."""
    result = compare(h, s, design, scales)
    value = result["objective"]
    return (math.inf if value is None else value), family_vector(result)


def equivalence_status(upper: float | None, lower: float | None, margin: float | None) -> str:
    """One-sided equivalence of a nonnegative family error to a sealed real-vs-real margin."""
    if upper is None or lower is None or margin is None or not all(map(math.isfinite, (upper, lower, margin))):
        return "NOT_EVALUABLE"
    if upper < margin:
        return "EQUIVALENT_WITHIN_MARGIN"
    if lower > margin:
        return "FAILED_MARGIN"
    return "NOT_ESTABLISHED"


# ----------------------------------------------------------------------------- uncertainty


def bootstrap_realism(hist_blocks: list[dict], sims: dict[str, list[dict]], design: dict, scales: dict | None, *,
                      samples: int, seed: int, alpha: float, contrasts: tuple[tuple[str, str], ...] = (),
                      margins: dict | None = None, equivalence_alpha: float | None = None) -> dict:
    """Block bootstrap of historical blocks with independent seed bootstraps of each simulated model.

    The same historical resample is applied to every model in a draw, so contrasts
    are paired on history. Point estimates use every block and seed once.
    """
    from .inference import bootstrap_pvalue, counts, equivalence_upper, interval
    from .observables import pool, stack, weighted
    if len(hist_blocks) < 2:
        raise ValueError("at least two historical blocks are required")
    if any(len(seeds) < 2 for seeds in sims.values()):
        raise ValueError("at least two simulated seeds per model are required")
    hist_stack = stack(hist_blocks)
    sim_stacks = {name: stack(seeds) for name, seeds in sims.items()}
    h_full = pool(hist_blocks)
    point = {name: compare(h_full, pool(seeds), design, scales) for name, seeds in sims.items()}
    rng = np.random.default_rng(seed)
    draws = {name: {"objective": np.empty(samples), "families": np.empty((samples, len(FAMILIES)))} for name in sims}
    for b in range(samples):
        h = weighted(hist_stack, counts(len(hist_blocks), rng))
        for name, matrix in sim_stacks.items():
            s = weighted(matrix, counts(len(sims[name]), rng))
            value, families = quick_objective(h, s, design, scales)
            draws[name]["objective"][b] = value
            draws[name]["families"][b] = families
    models = {}
    for name, result in point.items():
        low, high = interval(draws[name]["objective"], alpha)
        families = {}
        for j, family in enumerate(FAMILIES):
            column = draws[name]["families"][:, j]
            f_low, f_high = interval(column, alpha)
            entry = {"error": result["families"][family]["error"], "ci_low": f_low, "ci_high": f_high,
                     "evaluable_components": result["families"][family]["evaluable"]}
            if margins is not None and equivalence_alpha is not None:
                lower, upper = equivalence_upper(column, alpha=equivalence_alpha)
                entry.update(margin=margins.get(family), one_sided_lower=lower, one_sided_upper=upper,
                             equivalence=equivalence_status(upper if math.isfinite(upper) else None,
                                                            lower if math.isfinite(lower) else None,
                                                            margins.get(family)))
            families[family] = entry
        models[name] = {"objective": result["objective"], "ci_low": low, "ci_high": high, "families": families,
                        "components": {k: {"error": v["error"], "family": v["family"]}
                                       for k, v in result["components"].items()}}
    contrast_results = {}
    for left, right in contrasts:
        diff = draws[left]["objective"] - draws[right]["objective"]
        low, high = interval(diff, alpha)
        family_diffs = {}
        for j, family in enumerate(FAMILIES):
            column = draws[left]["families"][:, j] - draws[right]["families"][:, j]
            f_low, f_high = interval(column, alpha)
            el, er = point[left]["families"][family]["error"], point[right]["families"][family]["error"]
            family_diffs[family] = {"difference": None if el is None or er is None else el - er,
                                    "ci_low": f_low, "ci_high": f_high}
        estimate = (point[left]["objective"] - point[right]["objective"]
                    if point[left]["objective"] is not None and point[right]["objective"] is not None else None)
        contrast_results[f"{left}-{right}"] = {"difference": estimate, "ci_low": low, "ci_high": high,
                                               "p_value": bootstrap_pvalue(diff), "families": family_diffs}
    return {"samples": samples, "seed": seed, "alpha": alpha, "historical_blocks": len(hist_blocks),
            "seeds": {k: len(v) for k, v in sims.items()}, "models": models, "contrasts": contrast_results,
            "method": "historical block bootstrap (same draw for all models) x independent seed bootstrap per model"}


def improvement_status(contrast: dict) -> str:
    """Preregistered H1-H3/H9 rule: upper bound < 0 ESTABLISHED, lower bound > 0 FAILED."""
    low, high = contrast.get("ci_low"), contrast.get("ci_high")
    if low is None or high is None or not (math.isfinite(low) and math.isfinite(high)):
        return "INVALID"
    if high < 0:
        return "ESTABLISHED"
    if low > 0:
        return "FAILED"
    return "NOT_ESTABLISHED"
