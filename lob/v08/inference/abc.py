"""Bounded likelihood-free inference (M12).

The prior is a box over flattened generator parameters (``Prior``): uniform, or
log-uniform for strictly positive scales. All proposals live in an unconstrained
transformed space (logit of the box position) so perturbation kernels never
leave the support.

* ``rejection`` — ABC rejection: draw from the prior, keep the closest quantile;
* ``smc`` — adaptive ABC-SMC (tolerance = registered quantile of the previous
  population's distances; Gaussian kernel with twice the weighted covariance;
  importance weights prior / mixture-kernel density);
* ``synthetic_loglik`` and ``metropolis`` — Gaussian synthetic (surrogate)
  likelihood of the observed summaries (Wood 2010) and a seeded random-walk
  Metropolis sampler on it;
* ``reweight`` — posterior importance reweighting of a particle population by
  a synthetic likelihood of new data (used for posterior updating);
* ``regression_adjust`` — local-linear regression adjustment of accepted
  particles (Beaumont et al. 2002).

Every simulation is seed-addressable: ``SeedSequence([seed, stage, index])``.
``map_fn`` (serial by default) lets callers evaluate candidates in parallel;
candidate lists are generated before evaluation, so results do not depend on
scheduling. Distances use summaries standardized by a pilot prior-predictive
scale (median absolute deviation), recorded with the result.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from ..generators.base import MarketGenerator
from ..market.frame import SUMMARY_NAMES, summaries


@dataclass(frozen=True)
class Prior:
    names: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    log: tuple[bool, ...]

    @classmethod
    def from_space(cls, space: dict, log: set[str] | None = None) -> Prior:
        names = tuple(space)
        log = log or set()
        return cls(names, tuple(float(space[n][0]) for n in names), tuple(float(space[n][1]) for n in names),
                   tuple(n in log for n in names))

    def __post_init__(self) -> None:
        for lo, hi, lg in zip(self.low, self.high, self.log):
            if not lo < hi or (lg and lo <= 0):
                raise ValueError("invalid prior box")

    @property
    def dim(self) -> int:
        return len(self.names)

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.asarray([math.log(a) if lg else a for a, lg in zip(self.low, self.log)])
        hi = np.asarray([math.log(b) if lg else b for b, lg in zip(self.high, self.log)])
        return lo, hi

    def to_unit(self, theta: np.ndarray) -> np.ndarray:
        lo, hi = self._bounds()
        x = np.where(np.asarray(self.log), np.log(np.maximum(theta, 1e-300)), theta)
        return (x - lo) / (hi - lo)

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        lo, hi = self._bounds()
        x = lo + np.clip(u, 0.0, 1.0) * (hi - lo)
        return np.where(np.asarray(self.log), np.exp(x), x)

    def to_free(self, theta: np.ndarray) -> np.ndarray:
        u = np.clip(self.to_unit(theta), 1e-9, 1 - 1e-9)
        return np.log(u / (1 - u))

    def from_free(self, z: np.ndarray) -> np.ndarray:
        return self.from_unit(1 / (1 + np.exp(-z)))

    def log_jacobian_free(self, z: np.ndarray) -> np.ndarray:
        """log |du/dz| for the logistic map (prior density in free space up to a constant)."""
        z = np.atleast_2d(z)
        return np.sum(-z - 2 * np.log1p(np.exp(-z)), axis=1)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.from_unit(rng.random((n, self.dim)))

    def to_dict(self) -> dict:
        return {"names": list(self.names), "low": list(self.low), "high": list(self.high), "log": list(self.log)}

    def sd_unit(self) -> float:
        return 1.0 / math.sqrt(12.0)


def simulate_summary(generator_dict: dict, names: list[str], theta: list[float], n_bins: int, seed: int,
                     warmup_bins: int = 200) -> list[float]:
    """Module-level (picklable) simulation of the summary vector for one parameter vector."""
    g = MarketGenerator.from_dict(generator_dict).with_params(dict(zip(names, theta)))
    frame = g.simulate(n_bins, seed, warmup_bins=warmup_bins)
    return summaries(frame).tolist()


def _seed(seed: int, stage: int, index: int) -> int:
    return int(np.random.SeedSequence([seed, stage, index]).generate_state(1, dtype=np.uint32)[0])


class Problem:
    """Observed summaries, a generator template, a prior and a summary scale."""

    def __init__(self, generator: MarketGenerator, prior: Prior, observed: np.ndarray, *, n_bins: int, seed: int,
                 summary_index: list[int] | None = None, map_fn: Callable = map, scale: np.ndarray | None = None,
                 pilot: int = 32, reference: np.ndarray | None = None) -> None:
        self.generator, self.prior, self.n_bins, self.seed = generator, prior, int(n_bins), int(seed)
        self.template = generator.to_dict()
        self.index = list(range(len(SUMMARY_NAMES))) if summary_index is None else list(summary_index)
        self.observed = np.asarray(observed, float)[self.index]
        self.map_fn = map_fn
        self.simulations = 0
        if scale is None:
            # Aleatoric scale: replicate simulations at a reference parameter (default: the prior's centre in
            # unit space). Standardizing by prior-predictive spread would down-weight informative summaries.
            ref = prior.from_unit(np.full(prior.dim, 0.5)) if reference is None else np.asarray(reference, float)
            self.reference = ref
            pilot_s = self.simulate(np.repeat(ref[None, :], pilot, axis=0), stage=0)
            mad = np.median(np.abs(pilot_s - np.median(pilot_s, 0)), 0) * 1.4826
            scale = np.where(mad > 1e-9, mad, np.std(pilot_s, 0) + 1e-9)
            scale = np.maximum(scale, 1e-3 * (np.abs(np.median(pilot_s, 0)) + 1e-3))
        self.scale = np.asarray(scale, float)

    def simulate(self, thetas: np.ndarray, *, stage: int, offset: int = 0) -> np.ndarray:
        args = [(self.template, list(self.prior.names), list(map(float, t)), self.n_bins,
                 _seed(self.seed, stage, offset + i)) for i, t in enumerate(thetas)]
        out = list(self.map_fn(_simulate_args, args))
        self.simulations += len(args)
        return np.asarray(out, float)[:, self.index]

    def distance(self, s: np.ndarray) -> np.ndarray:
        return np.sqrt((((np.atleast_2d(s) - self.observed) / self.scale) ** 2).sum(1))


def _simulate_args(args) -> list[float]:
    return simulate_summary(*args)


# ============================================================================= rejection


def rejection(problem: Problem, *, draws: int, accept: float) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([problem.seed, 1]))
    thetas = problem.prior.sample(draws, rng)
    s = problem.simulate(thetas, stage=1)
    d = problem.distance(s)
    eps = float(np.quantile(d, accept))
    keep = d <= eps
    return _population(problem, thetas[keep], np.ones(keep.sum()), d[keep], "abc_rejection",
                       {"draws": draws, "accept_quantile": accept, "epsilon": eps}, summaries=s[keep])


# ============================================================================= SMC


def smc(problem: Problem, *, particles: int, generations: int, quantile: float = 0.5, oversample: float = 2.0
        ) -> dict:
    prior = problem.prior
    rng = np.random.default_rng(np.random.SeedSequence([problem.seed, 2]))
    theta = prior.sample(particles, rng)
    s = problem.simulate(theta, stage=10)
    d = problem.distance(s)
    w = np.ones(particles) / particles
    history = [{"generation": 0, "epsilon": float(np.max(d)), "ess": float(particles), "acceptance": 1.0}]
    for gen in range(1, generations):
        eps = float(np.quantile(d, quantile))
        z = prior.to_free(theta)
        cov = 2.0 * np.cov(z.T, aweights=w) if prior.dim > 1 else np.atleast_2d(2.0 * np.cov(z.T, aweights=w))
        cov = cov + 1e-6 * np.eye(prior.dim)
        chol = np.linalg.cholesky(cov)
        new_theta, new_s, new_d = [], [], []
        proposed = 0
        batch = int(math.ceil(particles * oversample))
        rounds = 0
        while len(new_theta) < particles and rounds < 20:
            parents = rng.choice(particles, size=batch, p=w)
            cand_z = z[parents] + rng.standard_normal((batch, prior.dim)) @ chol.T
            cand = prior.from_free(cand_z)
            cs = problem.simulate(cand, stage=10 + gen, offset=proposed)
            cd = problem.distance(cs)
            proposed += batch
            ok = np.nonzero(cd <= eps)[0]
            for i in ok[: particles - len(new_theta)]:
                new_theta.append(cand[i])
                new_s.append(cs[i])
                new_d.append(cd[i])
            rounds += 1
        if len(new_theta) < particles:
            history.append({"generation": gen, "epsilon": eps, "status": "STOPPED_LOW_ACCEPTANCE",
                            "accepted": len(new_theta), "proposed": proposed})
            break
        new_theta, new_s, new_d = np.asarray(new_theta), np.asarray(new_s), np.asarray(new_d)
        nz = prior.to_free(new_theta)
        inv = np.linalg.inv(cov)
        logdet = float(np.linalg.slogdet(cov)[1])
        diff = nz[:, None, :] - z[None, :, :]
        kern = np.exp(-0.5 * np.einsum("ijk,kl,ijl->ij", diff, inv, diff) - 0.5 * logdet)
        denominator = kern @ w
        prior_free = np.exp(prior.log_jacobian_free(nz))
        w = prior_free / np.maximum(denominator, 1e-300)
        w = w / w.sum()
        theta, s, d = new_theta, new_s, new_d
        history.append({"generation": gen, "epsilon": eps, "ess": float(1 / np.sum(w * w)),
                        "acceptance": particles / proposed})
    return _population(problem, theta, w, d, "abc_smc", {"particles": particles, "generations": generations,
                                                           "quantile": quantile, "history": history}, summaries=s)


# ============================================================================= synthetic likelihood


def synthetic_loglik(problem: Problem, theta: np.ndarray, *, replicates: int, stage: int, offset: int = 0,
                     observed: np.ndarray | None = None) -> float:
    reps = np.repeat(np.atleast_2d(theta), replicates, axis=0)
    s = problem.simulate(reps, stage=stage, offset=offset) / problem.scale
    obs = (problem.observed if observed is None else observed) / problem.scale
    mu = s.mean(0)
    cov = np.cov(s.T) + 1e-3 * np.eye(s.shape[1])
    diff = obs - mu
    sign, logdet = np.linalg.slogdet(cov)
    return float(-0.5 * (diff @ np.linalg.solve(cov, diff) + logdet + len(diff) * math.log(2 * math.pi)))


def metropolis(problem: Problem, *, steps: int, replicates: int, step_size: float = 0.3, start=None) -> dict:
    prior = problem.prior
    rng = np.random.default_rng(np.random.SeedSequence([problem.seed, 3]))
    z = prior.to_free(np.asarray(start, float)) if start is not None else np.zeros(prior.dim)
    ll = synthetic_loglik(problem, prior.from_free(z), replicates=replicates, stage=30, offset=0)
    lp = float(prior.log_jacobian_free(z)[0])
    chain, accepted = [], 0
    for k in range(1, steps + 1):
        zc = z + step_size * rng.standard_normal(prior.dim)
        llc = synthetic_loglik(problem, prior.from_free(zc), replicates=replicates, stage=30, offset=k * replicates)
        lpc = float(prior.log_jacobian_free(zc)[0])
        if math.log(rng.random()) < (llc + lpc) - (ll + lp):
            z, ll, lp = zc, llc, lpc
            accepted += 1
        chain.append(prior.from_free(z))
    chain = np.asarray(chain)[steps // 4:]
    return _population(problem, chain, np.ones(len(chain)), np.full(len(chain), np.nan), "synthetic_likelihood_mcmc",
                       {"steps": steps, "replicates": replicates, "acceptance": accepted / steps,
                        "burn_in": steps // 4})


def reweight(problem: Problem, posterior: dict, new_observed: np.ndarray, *, replicates: int = 8,
             forgetting: float = 1.0) -> dict:
    """Importance reweighting of a particle posterior by the synthetic likelihood of new observed summaries.

    ``forgetting`` in (0, 1] tempers the old weights (w_old ** forgetting) before the update.
    """
    theta = np.asarray(posterior["particles"], float)
    w = np.asarray(posterior["weights"], float) ** forgetting
    obs = np.asarray(new_observed, float)[problem.index]
    ll = np.asarray([synthetic_loglik(problem, t, replicates=replicates, stage=40, offset=i * replicates,
                                      observed=obs) for i, t in enumerate(theta)])
    logw = np.log(np.maximum(w, 1e-300)) + ll
    w = np.exp(logw - logw.max())
    return _population(problem, theta, w / w.sum(), np.full(len(theta), np.nan), "importance_reweighting",
                       {"replicates": replicates, "forgetting": forgetting})


def regression_adjust(posterior: dict, problem: Problem) -> dict:
    """Beaumont local-linear regression adjustment of accepted particles (in free space)."""
    theta = np.asarray(posterior["particles"], float)
    s = np.asarray(posterior["summaries"], float)
    d = np.asarray(posterior["distances"], float)
    eps = float(np.max(d)) or 1.0
    kw = np.maximum(1 - (d / eps) ** 2, 1e-6)
    z = problem.prior.to_free(theta)
    x = np.column_stack([np.ones(len(s)), (s - problem.observed) / problem.scale])
    sw = np.sqrt(kw)[:, None]
    beta = np.linalg.lstsq(x * sw, z * sw, rcond=None)[0]
    adjusted = z - x[:, 1:] @ beta[1:]
    out = dict(posterior)
    out["particles"] = problem.prior.from_free(adjusted).tolist()
    out["weights"] = (kw / kw.sum()).tolist()
    out["method"] = posterior["method"] + "+regression_adjustment"
    out["summary"] = summarize(np.asarray(out["particles"]), np.asarray(out["weights"]), problem.prior)
    return out


# ============================================================================= summaries


def _population(problem: Problem, theta, w, d, method: str, info: dict, summaries=None) -> dict:
    w = np.asarray(w, float)
    w = w / w.sum()
    return {"method": method, "names": list(problem.prior.names), "particles": np.asarray(theta).tolist(),
            "weights": w.tolist(), "distances": np.asarray(d).tolist(),
            "summaries": None if summaries is None else np.asarray(summaries).tolist(),
            "prior": problem.prior.to_dict(), "summary_index": problem.index, "scale": problem.scale.tolist(),
            "simulations": problem.simulations, "info": info,
            "summary": summarize(np.asarray(theta), w, problem.prior)}


def _wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][min(np.searchsorted(cw, q * cw[-1]), len(x) - 1)])


def summarize(theta: np.ndarray, w: np.ndarray, prior: Prior) -> dict:
    theta = np.atleast_2d(theta)
    w = np.asarray(w, float) / np.sum(w)
    u = prior.to_unit(theta)
    out = {"ess": float(1 / np.sum(w * w)), "parameters": {}}
    for j, name in enumerate(prior.names):
        x = theta[:, j]
        mean = float(np.dot(w, x))
        sd = float(math.sqrt(max(np.dot(w, (x - mean) ** 2), 0.0)))
        su = float(math.sqrt(max(np.dot(w, (u[:, j] - np.dot(w, u[:, j])) ** 2), 0.0)))
        out["parameters"][name] = {"mean": mean, "sd": sd, "q05": _wquantile(x, w, 0.05), "q50": _wquantile(x, w, 0.5),
                                   "q95": _wquantile(x, w, 0.95), "contraction": 1 - su / prior.sd_unit(),
                                   "modes": count_modes(u[:, j], w)}
    if theta.shape[1] > 1:
        cov = np.cov(u.T, aweights=w)
        sd = np.sqrt(np.diag(cov)) + 1e-12
        out["correlation"] = (cov / np.outer(sd, sd)).tolist()
    return out


def count_modes(u: np.ndarray, w: np.ndarray, bins: int = 20, min_mass: float = 0.05) -> int:
    """Number of separated weighted-histogram peaks holding at least ``min_mass`` each (unit scale)."""
    hist, _ = np.histogram(np.clip(u, 0, 1), bins=bins, range=(0, 1), weights=w)
    smooth = np.convolve(hist, np.ones(3) / 3, mode="same")
    peaks = [i for i in range(bins) if smooth[i] >= min_mass / 3 and smooth[i] == max(smooth[max(0, i - 2):i + 3])]
    merged = []
    for p in peaks:
        if not merged or p - merged[-1] > 2:
            merged.append(p)
    return max(1, len(merged))


def credible_interval(posterior: dict, name: str, level: float = 0.9) -> tuple[float, float]:
    j = posterior["names"].index(name)
    x = np.asarray(posterior["particles"], float)[:, j]
    w = np.asarray(posterior["weights"], float)
    a = (1 - level) / 2
    return _wquantile(x, w, a), _wquantile(x, w, 1 - a)


def draw(posterior: dict, n: int, seed: int) -> list[dict]:
    """Systematic resampling of ``n`` parameter dicts from a weighted particle posterior."""
    w = np.asarray(posterior["weights"], float)
    rng = np.random.default_rng(seed)
    idx = np.minimum(np.searchsorted(np.cumsum(w / w.sum()), (rng.random() + np.arange(n)) / n), len(w) - 1)
    particles = np.asarray(posterior["particles"], float)
    return [dict(zip(posterior["names"], map(float, particles[i]))) for i in idx]
