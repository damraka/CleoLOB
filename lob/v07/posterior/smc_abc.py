"""G0_post: adaptive SMC-ABC posterior over the v0.6 14-parameter simulator family (workstreams 9, 27).

Algorithm (Del Moral, Doucet and Jasra 2012, adaptive tolerance, one MCMC move
per particle per generation):

1. Generation 0: ``N`` particles from the uniform prior on the v0.6 transformed
   unit box; each is simulated once and gets an ABC distance = the sealed v0.6
   objective against the development target (2 seeds x 1800 s, 60 s warmup).
2. Generations 1..G-1: tolerance ``eps_g`` = 50th percentile of the current
   distances; weights ``1[d <= eps_g]``; systematic resampling; one Gaussian
   random-walk move per particle with covariance twice the weighted particle
   covariance, reflected back into the unit box (reflection keeps the kernel
   symmetric); a move is accepted iff its fresh distance is ``<= eps_g`` (symmetric
   kernel and uniform prior make the Metropolis-Hastings ratio one).

Failures (exceptions, the 500 events/s implausibility guard) have distance
``+inf`` and are retained. The result approximates the ABC posterior at the final
tolerance; it is not the exact Bayesian posterior of the simulator. Every
simulation seed is a deterministic function of (run seed, generation, particle).
"""
from __future__ import annotations

import math

import numpy as np

from ...v06.calibration import NAMES, spec_from_unit
from ..realism.scoring import run_tasks

DIM = len(NAMES)


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    return np.minimum(np.searchsorted(np.cumsum(weights), positions), len(weights) - 1)


def reflect(x: np.ndarray) -> np.ndarray:
    """Fold values into [0, 1] by reflection at both walls (a symmetric map for random-walk kernels)."""
    y = np.mod(x, 2.0)
    return np.where(y > 1.0, 2.0 - y, y)


def weighted_cov(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w / w.sum()
    mu = w @ x
    d = x - mu
    cov = (d * w[:, None]).T @ d
    return cov + 1e-6 * np.eye(x.shape[1])


class Evaluator:
    """Distances for batches of unit vectors (parallel), with deterministic per-evaluation seeds."""

    def __init__(self, *, base: dict, tables: dict, context: dict, target: str, seed_base: int, seeds_per: int = 2,
                 seconds: float = 1800.0, workers: int | None = None) -> None:
        self.base, self.tables, self.context, self.target = base, tables, context, target
        self.seed_base, self.seeds_per, self.seconds, self.workers = seed_base, seeds_per, seconds, workers
        self.evaluations = 0

    def __call__(self, units: np.ndarray, tag: int) -> tuple[np.ndarray, list[dict]]:
        tasks = []
        for i, u in enumerate(units):
            spec = spec_from_unit(u, self.base, self.tables)
            first = self.seed_base + tag * 10_000 + i * self.seeds_per
            tasks.append((f"{tag}:{i}", spec, list(range(first, first + self.seeds_per)), self.seconds, self.target))
        results = run_tasks(tasks, self.context, workers=self.workers)
        self.evaluations += len(tasks)
        distances = np.asarray([r["objective"] if r.get("objective") is not None else math.inf for r in results])
        records = [{"objective": r.get("objective"), "families": r.get("families"), "error": r["error"]}
                   for r in results]
        return distances, records


def run(evaluator: Evaluator, *, particles: int, generations: int, seed: int, quantile: float = 0.5,
        log=None) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.random((particles, DIM))
    d, records = evaluator(x, tag=0)
    history = [{"generation": 0, "tolerance": None, "acceptance": None, "alive": int(np.isfinite(d).sum()),
                "median_distance": float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else None,
                "failures": int((~np.isfinite(d)).sum())}]
    if log:
        log(history[-1])
    eps = math.inf
    for g in range(1, generations):
        finite = d[np.isfinite(d)]
        if not len(finite):
            break
        eps = float(np.quantile(finite, quantile))
        w = (d <= eps).astype(float)
        index = systematic_resample(w, rng)
        x, d = x[index], d[index]
        records = [records[i] for i in index]
        cov = 2.0 * weighted_cov(x, np.ones(len(x)))
        proposal = reflect(x + rng.multivariate_normal(np.zeros(DIM), cov, size=len(x)))
        inside = np.all((proposal >= 0) & (proposal <= 1), axis=1)
        moved_d = np.full(len(x), math.inf)
        moved_records: list[dict | None] = [None] * len(x)
        if inside.any():
            nd, nrec = evaluator(proposal[inside], tag=g)
            moved_d[inside] = nd
            for j, k in enumerate(np.flatnonzero(inside)):
                moved_records[k] = nrec[j]
        accept = inside & (moved_d <= eps)
        x = np.where(accept[:, None], proposal, x)
        d = np.where(accept, moved_d, d)
        records = [moved_records[i] if accept[i] else records[i] for i in range(len(x))]
        history.append({"generation": g, "tolerance": eps, "acceptance": float(accept.mean()),
                        "unique_particles": int(len(np.unique(x.round(12), axis=0))),
                        "proposals_inside_box": int(inside.sum()), "alive": int((d <= eps).sum()),
                        "median_distance": float(np.median(d)), "failures_in_moves": int(
                            (inside & ~np.isfinite(moved_d)).sum())})
        if log:
            log(history[-1])
    return {"particles": x, "distances": d, "records": records, "tolerance": eps, "history": history,
            "evaluations": evaluator.evaluations}


# ----------------------------------------------------------------------------- posterior summaries


def clusters(x: np.ndarray, *, separation: float = 0.25, min_mass: float = 0.05) -> dict:
    """Single-linkage components under Chebyshev distance > ``separation``; components with >= ``min_mass``."""
    n = len(x)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        close = np.flatnonzero(np.max(np.abs(x[i + 1:] - x[i]), axis=1) <= separation) + i + 1
        for j in close:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    labels = np.asarray([find(i) for i in range(n)])
    sizes = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    major = {k: v for k, v in sizes.items() if v / n >= min_mass}
    centres = {k: x[labels == k].mean(0).tolist() for k in major}
    return {"components": len(sizes), "major_components": len(major), "major_masses": sorted(
        (v / n for v in major.values()), reverse=True), "centres": centres}


def marginals(x: np.ndarray) -> dict:
    return {name: {"mean": float(x[:, i].mean()), "sd": float(x[:, i].std()),
                   "q05": float(np.quantile(x[:, i], 0.05)), "q50": float(np.quantile(x[:, i], 0.5)),
                   "q95": float(np.quantile(x[:, i], 0.95))} for i, name in enumerate(NAMES)}


def correlation(x: np.ndarray) -> list[list[float]]:
    c = np.corrcoef(x.T)
    return np.nan_to_num(c).round(4).tolist()


def coverage(x: np.ndarray, truth: np.ndarray, level: float = 0.90) -> dict:
    lo, hi = np.quantile(x, (1 - level) / 2, axis=0), np.quantile(x, 1 - (1 - level) / 2, axis=0)
    inside = (truth >= lo) & (truth <= hi)
    return {"covered_fraction": float(inside.mean()), "covered": dict(zip(NAMES, inside.tolist())),
            "interval_width_mean": float(np.mean(hi - lo))}
