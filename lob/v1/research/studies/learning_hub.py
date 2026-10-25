"""Learning hub: registered unit values of H41 (safe offline policy selection) and H43 (decision-focused calibration).

* H41 = harm rate(ungated) - harm rate(conservatively gated). Offline learners (v0.9 BC, CQL, IQL, TD3+BC and the
  return-conditioned policy) are trained on logged behaviour episodes in the F4 world calibrated on the training
  units (corrected book, propensities recorded); OPE uses separate logged episodes and an FQE critic. Ungated
  deployment = every learner whose DR mean improvement over TWAP is positive; gated = the v1 ``rl.learning.gate``
  (bootstrap LCB95 > 0, support, OOD rate and importance-weight ESS). A deployment is harmful when its true mean
  cost in the F4 world calibrated on the *held-out* unit exceeds TWAP's by more than 1 bps (registered endpoint).
  Rates are over the learner set, so the unit value lies in [-1, 1]. Labelled F4: outcomes are synthetic, never fills.
* H43 = F2 regret(policy chosen in the likelihood-calibrated F4) - F2 regret(policy chosen in the decision-calibrated
  F4) on the evaluation unit. Likelihood calibration = the Poisson maximum-likelihood fit on the training units;
  decision calibration = ``calibration.api.calibrate(objective="decision")`` over multipliers (0.5-2) of its
  market/add/cancel intensities, minimizing the mean F2 central-assumption regret of the chosen policy on the
  *training* units. The evaluation unit is never seen by either calibration.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np

from ....v09.studies.bounds import RANKING_POLICIES

N_TRAIN, N_OPE, N_EVAL = 200, 60, 16
HARM_BPS = 1.0


def _seeds(base: int, label: str, n: int) -> list[int]:
    return [int(hashlib.sha256(f"{base}|{label}|{k}".encode()).hexdigest()[:8], 16) for k in range(n)]


def _qty(book, tick: float, lot: float, notional: float) -> int:
    return max(1, int(notional / (book.mid_ticks * tick) / lot))


def _costs(g, policy, qty: int, seeds: list[int]) -> list:
    from ...twins import classical as TC
    out = []
    for k, s in enumerate(seeds):
        r = TC.synthetic_episode(g, policy, qty=qty, seed=s, side=1 if k % 2 == 0 else -1)
        out.append(r["is_net_bps"] if r["status"] == "OK" else None)
    return out


def _mean(x) -> float:
    v = [a for a in x if a is not None]
    return float(np.mean(v)) if v else float("nan")


def h41(held_path: str, train_paths: list[str], *, seed: int, notional: float, family: str = "hawkes") -> dict:
    from ....v09 import rl as R9
    from ....v09.execution import policies as P
    from ...data import daycache as DC
    from ...rl import learning as L
    from ...runtime import cap_threads
    from ...twins import classical as TC
    cap_threads()
    t0 = time.time()
    train = [DC.load(Path(p)) for p in train_paths]
    tick, lot = float(train[0]["tick"]), float(train[0]["lot"])
    g = TC.fit(train, book="v1", families=[family])["generators"][family]
    del train
    truth_g = TC.fit([DC.load(Path(held_path))], book="v1", families=[family])["generators"][family]
    qty = _qty(g.book, tick, lot, notional)
    logged = L.collect(g, episodes=N_TRAIN, qty=qty, seed=_seeds(seed, "train", 1)[0])
    ope_eps = L.collect(g, episodes=N_OPE, qty=qty, seed=_seeds(seed, "ope", 1)[0])
    learned = R9.learners(logged)
    S, A, Pm, R, nxt, done, rtg = R9._stack(logged)
    fqe = R9.fit_q(S, A, R, nxt, done, kind="fqe")
    base_value = -_mean(_costs(g, P.make("twap"), qty, _seeds(seed, "base", 8)))
    states = np.asarray([s for e in ope_eps for s in e["s"]], float)
    sup = R9.support(logged, states)
    eval_seeds = _seeds(seed, "eval", N_EVAL)
    twap_true = _mean(_costs(truth_g, P.make("twap"), qty, eval_seeds))
    est, harmful, ungated, gated = {}, {}, [], []
    for name, fn in learned.items():
        o = L.ope(ope_eps, fn, fqe)
        imp = np.asarray(o["dr_episodes"]) - base_value
        gd = L.gate(np.asarray(o["dr_episodes"]), np.asarray([base_value]), support=sup,
                    ood_rate=o["ood_action_rate"], ess=o["effective_sample_size"], seed=seed)
        true_cost = _mean(_costs(truth_g, R9.Greedy(fn, name), qty, eval_seeds))
        harmful[name] = bool(np.isfinite(true_cost) and true_cost - twap_true > HARM_BPS)
        if imp.mean() > 0:
            ungated.append(name)
        if gd["deploy"]:
            gated.append(name)
        est[name] = {"dr": o["dr"], "improvement_mean": float(imp.mean()), "gate": gd, "true_cost": true_cost,
                     "ess": o["effective_sample_size"], "ood_action_rate": o["ood_action_rate"]}
    n = len(learned)
    rate_u = sum(harmful[x] for x in ungated) / n
    rate_g = sum(harmful[x] for x in gated) / n
    return {"H41": rate_u - rate_g, "harm_rate_ungated": rate_u, "harm_rate_gated": rate_g, "deployed_ungated": ungated,
            "deployed_gated": gated, "twap_true_cost": twap_true, "estimates": est, "support": sup, "qty_lots": qty,
            "label": "F4 synthetic outcomes in a world calibrated on the held-out unit; never historical fills",
            "seconds": time.time() - t0}


def h43(eval_costs: dict[str, list], train_costs: list[dict[str, list]], train_paths: list[str], *, seed: int,
        notional: float, budget: int = 10, refine: int = 4, seeds_per_policy: int = 6) -> dict:
    """``eval_costs`` / ``train_costs``: F2 central-assumption window costs per policy (``multifidelity_hub``)."""
    from ....v08.generators.families import PoissonGenerator
    from ....v09.execution import policies as P
    from ...calibration import api as CA
    from ...data import daycache as DC
    from ...inference.methods import Prior
    from ...runtime import cap_threads
    from ...twins import classical as TC
    from .multifidelity_hub import regret
    cap_threads()
    train = [DC.load(Path(p)) for p in train_paths]
    tick, lot = float(train[0]["tick"]), float(train[0]["lot"])
    mle = TC.fit(train, book="v1", families=["poisson"])["generators"]["poisson"]
    del train
    qty = _qty(mle.book, tick, lot, notional)
    ep_seeds = _seeds(seed, "h43", seeds_per_policy)

    def factory(theta):
        p = dict(mle.params)
        for k in ("market", "add", "cancel"):
            p[k] = p[k] * theta[k]
        return PoissonGenerator(p, mle.book)

    def choose(g) -> str:
        m = {pol: _mean(_costs(g, P.make(pol), qty, ep_seeds)) for pol in RANKING_POLICIES}
        return min(m, key=lambda k: m[k] if np.isfinite(m[k]) else np.inf)

    def train_regret(g) -> float:
        c = choose(g)
        r = [regret(tc, c) for tc in train_costs]
        r = [x for x in r if x is not None]
        return float(np.mean(r)) if r else 1e6

    prior = Prior.from_space({"market": (0.5, 2.0), "add": (0.5, 2.0), "cancel": (0.5, 2.0)})
    res = CA.calibrate(factory, prior, "decision", scorer=train_regret, train=("train",), test=("eval",),
                       budget=budget, refine=refine, seed=seed)
    c_lik = choose(mle)
    c_dec = choose(factory(res.parameters))
    r_lik, r_dec = regret(eval_costs, c_lik), regret(eval_costs, c_dec)
    return {"H43": (r_lik - r_dec) if None not in (r_lik, r_dec) else None, "choice_likelihood": c_lik,
            "choice_decision": c_dec, "decision_parameters": res.parameters, "train_regret_decision": res.objective,
            "train_regret_likelihood": train_regret(mle)}


__all__ = ["h41", "h43", "HARM_BPS"]
