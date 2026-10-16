"""Multi-fidelity market-twin API (M33-M37): one episode interface across fidelities, capability matrix,
cross-fidelity runner, fidelity-specific uncertainty and consensus.

Fidelities (protocol ``fidelities``):

* F1 historical replay (observational; no agent) - ``observational``;
* F2 bounded counterfactual replay - ``bounded_episode`` (interval over the registered assumption grid);
* F3 hybrid twin - ``hybrid_episode`` (self-impact + stochastic resilience on a real backbone; seed spread);
* F4/F6 synthetic - ``synthetic_episode`` (calibrated classical or strategic generators, v0.8 simulator);
* F5 neural generators model trades/paths only and do not host execution (CAPABILITY_MISSING for execution).

Uncertainty per fidelity (M36): F2 = width of the assumption interval; F3 = spread over seeds; F4/F6 = spread over
worlds and seeds. Consensus (M37): a pairwise direction is reported by each fidelity as +1 / -1 / 0 (indeterminate);
the consensus is the sign of the sum of determinate votes, 0 on ties; the raw votes are always reported.
"""
from __future__ import annotations

import math

import numpy as np

from .replay import engine as E

CAPABILITY_MATRIX = {
    "F1": {"execution": False, "passive_fills": False, "self_impact": False, "uncertainty": "none (observed)"},
    "F2": {"execution": True, "passive_fills": "bounded", "self_impact": "assumption grid", "uncertainty": "interval"},
    "F3": {"execution": True, "passive_fills": "neutral queue on the backbone", "self_impact": "fitted response",
           "uncertainty": "seed spread"},
    "F4": {"execution": True, "passive_fills": "simulated", "self_impact": "simulated book", "uncertainty": "worlds"},
    "F5": {"execution": False, "passive_fills": False, "self_impact": False,
           "uncertainty": "n/a (trades / paths only)"},
    "F6": {"execution": True, "passive_fills": "simulated", "self_impact": "endogenous ecology", "uncertainty": "worlds"},
}


def bounded_episode(cache: dict, policy_factory, *, window: int, notional: float, grid: list[E.Assumption]) -> dict:
    book = E.DayBook(cache)
    rows = [E.episode(book, policy_factory(), window=window, notional_usd=notional, assumption=a) for a in grid]
    return {"fidelity": "F2", "rows": rows, "bounds": E.bounds(rows)}


def hybrid_episode(cache: dict, policy, *, window: int, notional: float, impact_model, persistent: float,
                   seed: int, assumption: E.Assumption = E.Assumption("neutral")) -> dict:
    from .generators.hybrid import HybridReplay
    book = E.DayBook(cache)
    qty = E.window_quantity(book, window, notional, float(cache["lot"]))
    t0 = float(E.WINDOW_STARTS_S[window])
    if qty <= 0 or not cache["valid"][book.bin_of(t0 - E.WARMUP_S):book.bin_of(t0 + E.HORIZON_S)].all():
        return {"fidelity": "F3", "status": "NOT_EVALUABLE"}
    rep = HybridReplay(cache, policy, impact_model=impact_model, persistent=persistent, seed=seed, window=window,
                       qty=qty, side=E.side_of(window), assumption=assumption, context={"seed": seed})
    res = rep.run()
    m = E.metrics(res, rep.book)
    forced = sum(f[2] for f in res.fills if abs(f[0] - (t0 + E.HORIZON_S)) < 1e-9)
    return {"fidelity": "F3", "status": "OK", **m, "policy_completion": 1.0 - forced / max(qty, 1e-9),
            "half_life_s": rep.half_life}


def synthetic_episode(generator, policy, *, qty: int, seed: int, side: int = 1) -> dict:
    from ..v08.execution.simulator import Fees, Mandate, run_episode
    from ..v08.worlds import World
    mandate = Mandate(side=side, qty=int(qty), horizon_s=300.0, decision_dt_s=5.0, warmup_s=30.0, completion="hard",
                      fees=Fees(taker_bps=5.5, maker_bps=2.0))
    row = run_episode(World(getattr(generator, "name", "g"), generator, generator.book), policy, mandate, seed,
                      context={"seed": seed})
    net = row.get("optimization_cost_bps") if row.get("status") == "VALID" else None
    return {"fidelity": "F4", "status": "OK" if net is not None else "NOT_EVALUABLE", "is_net_bps": net,
            "completion": row.get("fill_frac"), "within_horizon_completion": row.get("within_horizon_completion"),
            "raw_status": row.get("status")}


def direction(a: list[float], b: list[float], *, min_n: int = 4) -> int:
    """+1 if a is cheaper than b determinately (|mean diff| > 2 SE), -1 if more expensive, 0 otherwise."""
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d[np.isfinite(d)]
    if len(d) < min_n:
        return 0
    se = float(d.std(ddof=1) / math.sqrt(len(d))) or 1e-12
    m = float(d.mean())
    return 1 if m > 2 * se else -1 if m < -2 * se else 0


def consensus(votes: dict[str, int]) -> int:
    s = sum(v for v in votes.values() if v)
    return 1 if s > 0 else -1 if s < 0 else 0


def agreement(pred: dict[str, int], truth: dict[str, int]) -> tuple[float | None, int]:
    """Pairwise sign agreement over pairs where both directions are determinate."""
    pairs = [k for k in truth if truth[k] != 0 and pred.get(k, 0) != 0]
    if not pairs:
        return None, 0
    return float(np.mean([pred[k] == truth[k] for k in pairs])), len(pairs)


def kendall_distance(order_a: list[str], order_b: list[str]) -> float:
    """Normalized Kendall distance between two rankings of the same items (0 identical, 1 reversed)."""
    pos = {x: i for i, x in enumerate(order_b)}
    items = [x for x in order_a if x in pos]
    n = len(items)
    if n < 2:
        return 0.0
    disc = sum(1 for i in range(n) for j in range(i + 1, n) if pos[items[i]] > pos[items[j]])
    return disc / (n * (n - 1) / 2)
