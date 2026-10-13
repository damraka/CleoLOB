"""Synthetic world builder and the synthetic decision experiments (F4, F6; H11, H13, H20, H21, H24, H28).

A *world* is a generator calibrated on the registered window frames of one explore day (classical family or the
strategic ecology) with a seeded multiplicative perturbation of its rate scales (LogNormal(0, 0.2), registered). A
*world cluster* (the inferential unit of the synthetic hypotheses) is one calibration day with its seed block. Worlds
never read confirm or vault data; confirmatory runs use the ``confirm`` seed namespace for perturbations and episode
seeds (fresh synthetic data under the frozen design).

Mandates follow the protocol (USD notional / calibrated mid price, in lots of the instrument). Costs are the v0.8
simulator's ``optimization_cost_bps`` (gross + registered fees + priced residual + penalty), the synthetic analogue of
``is_net_bps``. Everything here is SYNTHETIC (E1) or multi-model synthetic (E2).
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from . import twin
from .execution import policies as P
from .generators import classical as G
from .generators.strategic import StrategicEcology

CATASTROPHE_BPS = 25.0
RANKING = ("twap", "vwap", "pov", "static_opt", "sweep", "passive_deadline", "liquidity_pov", "mpc_stochastic",
           "mpc_cvar")


def calibrate(caches: list[dict], family: str = "strategic_ecology"):
    frame = G.training_frame(caches)
    book = G.book_model(frame)
    if family == "strategic_ecology":
        g = StrategicEcology(None, book).fit(frame)
        g.book = book
    else:
        g = G.fit_family(family, frame, book)
    return g, float(caches[0]["tick"]), G.median_touch(frame)


def perturb(generator, seed: int, sd: float = 0.2):
    g = deepcopy(generator)
    rng = np.random.default_rng(np.random.SeedSequence([9, 1001, int(seed)]))
    if isinstance(g, StrategicEcology):
        g.params["market_scale"] *= float(np.exp(rng.normal(0, sd)))
        g.params["limit_scale"] *= float(np.exp(rng.normal(0, sd)))
        for a in g.params["agents"]:
            g.params["agents"][a] *= float(np.exp(rng.normal(0, sd)))
    else:
        theta = g.get_flat()
        g = g.with_params({k: float(v * np.exp(rng.normal(0, sd))) for k, v in theta.items()})
    return g


def qty_for(generator, tick: float, lot: float, notional: float) -> int:
    return max(1, int(notional / (generator.book.mid_ticks * tick) / lot))


def costs(generator, policies, qty: int, seeds: list[int]) -> dict[str, list]:
    out = {}
    for p in policies:
        vals = []
        for k, s in enumerate(seeds):
            r = twin.synthetic_episode(generator, P.make(p) if isinstance(p, str) else p(), qty=qty, seed=s,
                                       side=1 if k % 2 == 0 else -1)
            vals.append(r["is_net_bps"] if r["status"] == "OK" else None)
        out[p if isinstance(p, str) else getattr(p, "name", str(p))] = vals
    return out


def mean_order(c: dict[str, list]) -> list[str]:
    return sorted(c, key=lambda p: float(np.mean([v for v in c[p] if v is not None] or [np.inf])))


# ----------------------------------------------------------------------------- H11 self-impact


def h11_world(generator, tick, lot, notional_large, seeds) -> dict:
    endo = generator
    exo = generator.exogenous() if isinstance(generator, StrategicEcology) else generator
    q = qty_for(endo, tick, lot, notional_large)
    ce, cx = costs(endo, RANKING, q, seeds), costs(exo, RANKING, q, seeds)
    d = twin.kendall_distance(mean_order(ce), mean_order(cx))
    half = len(seeds) // 2
    d_odd = twin.kendall_distance(mean_order({k: v[1::2] for k, v in ce.items()}),
                                  mean_order({k: v[1::2] for k, v in cx.items()}))
    d_even = twin.kendall_distance(mean_order({k: v[0::2] for k, v in ce.items()}),
                                   mean_order({k: v[0::2] for k, v in cx.items()}))
    del half
    return {"kendall_distance": d, "odd": d_odd, "even": d_even, "qty_lots": q,
            "endogenous_order": mean_order(ce), "exogenous_order": mean_order(cx),
            "mean_cost": {"endogenous": {k: _m(v) for k, v in ce.items()}, "exogenous": {k: _m(v) for k, v in cx.items()}}}


def _m(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


# ----------------------------------------------------------------------------- H13 active calibration


def _summary(g, seed, n_bins=3000):
    from ..v08.market.frame import summaries
    return summaries(g.simulate(n_bins, seed, warmup_bins=300))


def h13_task(generator, seed: int, budget: int = 240, batch: int = 12) -> dict:
    """Recover 4 scales of a strategic ecology from simulated summaries: random search vs cross-entropy (active)."""
    rng = np.random.default_rng(np.random.SeedSequence([9, 13, int(seed)]))
    names = ("market_scale", "limit_scale", "momentum", "toxicity_proxy")
    lo, hi = np.log([0.3, 0.3, 0.02, 0.02]), np.log([3.0, 3.0, 1.0, 1.0])
    truth = np.exp(rng.uniform(lo, hi))

    def make(theta):
        g = deepcopy(generator)
        g.params["market_scale"], g.params["limit_scale"] = float(theta[0]), float(theta[1])
        g.params["agents"]["momentum"], g.params["agents"]["toxicity_proxy"] = float(theta[2]), float(theta[3])
        return g

    obs = _summary(make(truth), seed)
    noise = [_summary(make(truth), seed + 1000 + k) for k in range(4)]
    scale = np.std(np.vstack(noise + [obs]), 0) + 1e-6
    tol = 1.5 * float(np.mean([np.linalg.norm((n - obs) / scale) for n in noise]))

    def dist(theta, s):
        return float(np.linalg.norm((_summary(make(theta), s) - obs) / scale))

    def random_search():
        for k in range(budget):
            if dist(np.exp(rng.uniform(lo, hi)), seed + 2000 + k) <= tol:
                return k + 1
        return budget

    def cem():
        mu, sd = (lo + hi) / 2, (hi - lo) / 2
        used = 0
        while used < budget:
            cand = np.clip(rng.normal(mu, sd, (batch, 4)), lo, hi)
            ds = []
            for c in cand:
                used += 1
                d = dist(np.exp(c), seed + 3000 + used)
                if d <= tol:
                    return used
                ds.append(d)
            elite = cand[np.argsort(ds)[: max(2, batch // 4)]]
            mu, sd = elite.mean(0), np.maximum(elite.std(0), 0.05)
        return budget

    n_random, n_active = random_search(), cem()
    return {"truth": dict(zip(names, truth.tolist())), "tolerance": tol, "n_random": n_random, "n_active": n_active,
            "effect": 1.0 - n_active / max(n_random, 1), "budget": budget}


# ----------------------------------------------------------------------------- shifted / stressed worlds


def shifted(generator, kind: str):
    g = deepcopy(generator)
    if not isinstance(g, StrategicEcology):
        return g
    if kind == "liquidity_drought":
        g.params["limit_scale"] *= 0.3
    elif kind == "volatility_spike":
        g.params["market_scale"] *= 3.0
    elif kind == "toxic_flow":
        g.params["agents"]["toxicity_proxy"] *= 5.0
        g.params["agents"]["momentum"] *= 3.0
    return g


class Switching:
    """Regime drift: the wrapped generator switches to ``after`` once ``at_bin`` bins have been simulated."""

    def __init__(self, before, after, at_bin: int) -> None:
        self.before, self.after, self.at_bin = before, after, at_bin
        self.book, self.name = before.book, "regime_switch"

    def initial_state(self, simulate=False):
        return {"n": 0, "a": self.before.initial_state(simulate), "b": self.after.initial_state(simulate)}

    def _g(self, s):
        return self.after if s["n"] >= self.at_bin else self.before

    def expected(self, s, x):
        return self._g(s).expected(s["a"] if s["n"] < self.at_bin else s["b"], x)

    def draw(self, s, x, rng):
        return self._g(s).draw(s["a"] if s["n"] < self.at_bin else s["b"], x, rng)

    def update(self, s, realized, x, own=None):
        key = "a" if s["n"] < self.at_bin else "b"
        s[key] = self._g(s).update(s[key], realized, x, own=own)
        s["n"] += 1
        return s


def catastrophic_rate(vals: list) -> float:
    v = [x for x in vals if x is not None]
    return float(np.mean([x > CATASTROPHE_BPS for x in v])) if v else 0.0


def worst_case_regret(cost_table: dict[str, dict[str, list]], policy: str) -> float:
    """max over worlds of (mean cost of the policy - best mean cost in that world)."""
    out = []
    for world, c in cost_table.items():
        means = {p: _m(v) for p, v in c.items()}
        ok = {p: m for p, m in means.items() if m is not None}
        if policy in ok:
            out.append(ok[policy] - min(ok.values()))
    return float(max(out)) if out else math.inf
