"""Cross-fidelity scenario and policy comparison (M68) and synthetic-hybrid-historical ranking agreement (M134).

``compare(twins, policies, run_kwargs)`` runs every policy at every fidelity on matched scenarios (same seeds /
windows) and returns a cost table. ``directions`` turns per-policy cost samples into pairwise directions (+1 a
cheaper than b determinately, -1 dearer, 0 indeterminate, |mean| > 2 SE rule of v0.9); F2 directions come from the
bound intervals (IDENTIFIED only when the interval excludes zero). ``agreement`` is the share of pairs where two
fidelities give the same determinate sign; ``kendall`` the rank correlation of mean costs; ``consensus`` the sign of
the sum of determinate votes. ``reality_gap`` is the per-conclusion difference of the synthetic and bounded effects.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def compare(twins: dict, policies: dict, run_kwargs: list[dict]) -> dict:
    """{fidelity: {policy: [cost per scenario]}} with NaN for non-evaluable episodes."""
    out = {}
    for fid, twin in twins.items():
        out[fid] = {}
        for pname, factory in policies.items():
            vals = []
            for kw in run_kwargs:
                r = twin.episode(factory(), **kw)
                v = r.get("is_net_bps", r.get("is_net_bps_lower"))
                vals.append(float(v) if r.get("status") == "OK" and v is not None else math.nan)
            out[fid][pname] = vals
    return out


def direction(a, b, *, min_n: int = 4) -> int:
    d = np.asarray(b, float) - np.asarray(a, float)
    d = d[np.isfinite(d)]
    if len(d) < min_n:
        return 0
    se = float(d.std(ddof=1) / math.sqrt(len(d))) or 1e-12
    m = float(d.mean())
    return 1 if m > 2 * se else -1 if m < -2 * se else 0


def directions(costs: dict[str, list]) -> dict[str, int]:
    return {f"{a}>{b}": direction(costs[a], costs[b]) for a, b in itertools.combinations(sorted(costs), 2)}


def bounded_directions(intervals: dict[str, tuple[float, float]]) -> dict[str, int]:
    """F2: pair -> +1 / -1 when the oriented interval (b - a) excludes zero, else 0."""
    return {k: (1 if lo > 0 else -1 if hi < 0 else 0) for k, (lo, hi) in intervals.items()}


def agreement(pred: dict[str, int], truth: dict[str, int]) -> tuple[float | None, int]:
    pairs = [k for k in truth if truth[k] != 0 and pred.get(k, 0) != 0]
    if not pairs:
        return None, 0
    return float(np.mean([pred[k] == truth[k] for k in pairs])), len(pairs)


def consensus(votes: list[dict[str, int]]) -> dict[str, int]:
    keys = set().union(*[v.keys() for v in votes]) if votes else set()
    out = {}
    for k in keys:
        s = sum(v.get(k, 0) for v in votes)
        out[k] = 1 if s > 0 else -1 if s < 0 else 0
    return out


def kendall(costs_a: dict[str, list], costs_b: dict[str, list]) -> float | None:
    common = sorted(set(costs_a) & set(costs_b))
    ma = [np.nanmean(costs_a[p]) for p in common]
    mb = [np.nanmean(costs_b[p]) for p in common]
    n = len(common)
    if n < 3:
        return None
    s = 0
    for i, j in itertools.combinations(range(n), 2):
        s += np.sign(ma[i] - ma[j]) * np.sign(mb[i] - mb[j])
    return float(s / (n * (n - 1) / 2))


def reality_gap(synthetic_effect: dict[str, float], bounded_effect: dict[str, float]) -> dict[str, float]:
    return {c: float(synthetic_effect[c] - bounded_effect[c]) for c in synthetic_effect if c in bounded_effect}


__all__ = ["compare", "direction", "directions", "bounded_directions", "agreement", "consensus", "kendall",
           "reality_gap"]
