"""Model-risk analysis of execution costs across plausible worlds (H7, H8, H9; workstreams 35-45).

Input: a cost cube ``cost[world][agent] -> array over identical market seeds``
(bps; lower is better). Everything here is a pure function of that cube, so it
is unit-testable without simulation.

* H7 — per agent, ratio R = between-world SD of world means / pooled within-world
  SE of a world mean. Two-stage bootstrap (worlds, then market seeds within each
  resampled world); p = P*(R* <= 1); Holm over agents; ESTABLISHED for an agent if
  Holm rejects and the between-world SD >= 0.5 bps.
* H8 — per agent pair, pooled mean paired difference across worlds; two-stage
  bootstrap interval at alpha/28. ROBUST (``ROBUSTLY_BETTER``/``ROBUSTLY_WORSE``)
  only if: the interval excludes zero; |pooled| >= 1 bps; no world has a
  determinate (alpha/28) difference of the opposite sign; and the worst plausible
  world (adversarial search inside the 90% posterior region) keeps the sign.
  A determinate opposite world makes the pair MODEL_DEPENDENT; a worst plausible
  world with the opposite sign makes it REVERSED; an interval inside (-1, 1) bps is
  EQUIVALENT_WITHIN_MARGIN (descriptive); otherwise INDETERMINATE.
* H9 — pairs determinate in the G0 point world alone whose H8 edge is not robust.
"""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np

MATERIAL_BPS = 1.0
H7_MIN_SD = 0.5


def _finite(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(x)))


def world_means(cube: dict, agent: str) -> np.ndarray:
    return np.asarray([float(np.mean(cube[w][agent])) for w in cube])


def h7_ratio(cube: dict, agent: str) -> tuple[float, float, float]:
    means = world_means(cube, agent)
    within = np.asarray([np.var(cube[w][agent], ddof=1) / len(cube[w][agent]) for w in cube])
    se = float(np.sqrt(np.mean(within)))
    sd = float(np.std(means, ddof=1))
    return sd / se if se > 0 else math.inf, sd, se


def h7(cube: dict, agents: tuple[str, ...], *, alpha: float, samples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    worlds = list(cube)
    out, pvalues = {}, {}
    for agent in agents:
        ratio, sd, se = h7_ratio(cube, agent)
        draws = np.empty(samples)
        for b in range(samples):
            picked = rng.integers(0, len(worlds), len(worlds))
            sub = {}
            for j, w in enumerate(picked):
                values = cube[worlds[w]][agent]
                sub[f"{j}"] = {agent: values[rng.integers(0, len(values), len(values))]}
            draws[b] = h7_ratio(sub, agent)[0]
        p = float(np.mean(draws <= 1.0))
        pvalues[agent] = p
        out[agent] = {"ratio": ratio, "between_world_sd_bps": sd, "within_world_se_bps": se, "p_value": p,
                      "ratio_quantiles": {q: float(np.quantile(draws, q)) for q in (0.025, 0.5, 0.975)}}
    order = sorted(pvalues, key=lambda a: pvalues[a])
    stop = False
    for i, agent in enumerate(order):
        threshold = alpha / (len(order) - i)
        rejected = not stop and pvalues[agent] <= threshold
        stop = stop or not rejected
        material = out[agent]["between_world_sd_bps"] >= H7_MIN_SD
        upper_below_one = out[agent]["ratio_quantiles"][0.975] < 1.0
        out[agent].update(holm_threshold=threshold, holm_rejected=rejected, material=material,
                          status="ESTABLISHED" if rejected and material else "FAILED" if upper_below_one
                          else "NOT_ESTABLISHED")
    summary = "ESTABLISHED" if any(v["status"] == "ESTABLISHED" for v in out.values()) else "NOT_ESTABLISHED"
    return {"agents": out, "status": summary, "alpha": alpha, "samples": samples, "worlds": len(cube),
            "rule": "Holm over agents on p = P*(R <= 1); ESTABLISHED if rejected and between-world SD >= 0.5 bps"}


def _world_diff(cube: dict, w: str, a: str, b: str) -> np.ndarray:
    return np.asarray(cube[w][a], float) - np.asarray(cube[w][b], float)


def paired_interval(diff: np.ndarray, *, alpha: float, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(diff), (samples, len(diff)))
    means = diff[idx].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def pooled_interval(cube: dict, a: str, b: str, *, alpha: float, samples: int, rng) -> tuple[float, float, float]:
    worlds = list(cube)
    diffs = [_world_diff(cube, w, a, b) for w in worlds]
    point = float(np.mean([d.mean() for d in diffs]))
    draws = np.empty(samples)
    for s in range(samples):
        picked = rng.integers(0, len(worlds), len(worlds))
        draws[s] = np.mean([diffs[w][rng.integers(0, len(diffs[w]), len(diffs[w]))].mean() for w in picked])
    return point, float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def h8(cube: dict, agents: tuple[str, ...], *, alpha: float, samples: int, seed: int,
       worst: dict | None = None, reference: str = "G0_point") -> dict:
    """``worst``: {(a, b): {"candidate": name, "difference": mean diff of the most adverse plausible world}}."""
    pairs = list(combinations(agents, 2))
    adjusted = alpha / len(pairs)
    rng = np.random.default_rng(seed)
    edges = {}
    for a, b in pairs:
        point, low, high = pooled_interval(cube, a, b, alpha=adjusted, samples=samples, rng=rng)
        sign = -1 if high < 0 else 1 if low > 0 else 0
        per_world = {}
        opposite = []
        for w in cube:
            d = _world_diff(cube, w, a, b)
            lo, hi = paired_interval(d, alpha=adjusted, samples=samples, rng=rng)
            ws = -1 if hi < 0 else 1 if lo > 0 else 0
            per_world[w] = {"mean": float(d.mean()), "ci": [lo, hi], "direction": ws}
            if sign and ws == -sign:
                opposite.append(w)
        adverse = (worst or {}).get(f"{a}|{b}")
        keeps = adverse is None or sign == 0 or np.sign(adverse["difference"]) == sign
        material = abs(point) >= MATERIAL_BPS
        if sign and material and not opposite and keeps and adverse is not None:
            edge = "ROBUSTLY_BETTER" if sign < 0 else "ROBUSTLY_WORSE"   # a is cheaper when the difference < 0
        elif opposite:
            edge = "MODEL_DEPENDENT"
        elif sign and adverse is not None and not keeps:
            edge = "REVERSED"
        elif -MATERIAL_BPS < low and high < MATERIAL_BPS:
            edge = "EQUIVALENT_WITHIN_MARGIN"
        else:
            edge = "INDETERMINATE"
        edges[f"{a}|{b}"] = {"pooled_mean_bps": point, "ci": [low, high], "direction": sign, "material": material,
                             "opposite_worlds": opposite, "worst_plausible": adverse, "edge": edge,
                             "reference_direction": per_world.get(reference, {}).get("direction"),
                             "per_world": per_world}
    robust = [k for k, v in edges.items() if v["edge"] in {"ROBUSTLY_BETTER", "ROBUSTLY_WORSE"}]
    return {"edges": edges, "robust_pairs": robust, "adjusted_alpha": adjusted, "family_size": len(pairs),
            "status": "ESTABLISHED" if robust else "NOT_ESTABLISHED",
            "edge_counts": {e: sum(v["edge"] == e for v in edges.values()) for e in
                            ("ROBUSTLY_BETTER", "ROBUSTLY_WORSE", "EQUIVALENT_WITHIN_MARGIN", "INDETERMINATE",
                             "MODEL_DEPENDENT", "REVERSED")}}


def h9(h8_result: dict) -> dict:
    flagged = [k for k, v in h8_result["edges"].items()
               if v["reference_direction"] not in (None, 0) and v["edge"] in {"MODEL_DEPENDENT", "INDETERMINATE",
                                                                             "REVERSED"}]
    determinate = [k for k, v in h8_result["edges"].items() if v["reference_direction"] not in (None, 0)]
    status = "INCONCLUSIVE" if not determinate else "ESTABLISHED" if flagged else "NOT_ESTABLISHED"
    return {"status": status, "single_world_determinate_pairs": len(determinate), "flagged_pairs": flagged,
            "kind": "descriptive"}


def worst_plausible(candidate_cube: dict, pooled_signs: dict, agents: tuple[str, ...]) -> dict:
    """For each pair, the candidate world whose mean difference is most adverse to the pooled sign.

    A pooled sign of -1 (``a`` cheaper) is most threatened by the largest difference, +1 by the smallest.
    """
    out = {}
    if not candidate_cube:
        return out
    for a, b in combinations(agents, 2):
        sign = pooled_signs.get(f"{a}|{b}", 0)
        diffs = {w: float(np.mean(_world_diff(candidate_cube, w, a, b))) for w in candidate_cube}
        name = min(diffs, key=diffs.get) if sign > 0 else max(diffs, key=diffs.get)
        out[f"{a}|{b}"] = {"candidate": name, "difference": diffs[name], "candidates": len(diffs)}
    return out


def ranking_topology(cube: dict, agents: tuple[str, ...]) -> dict:
    rankings = {w: tuple(sorted(agents, key=lambda a: float(np.mean(cube[w][a])))) for w in cube}
    distinct = sorted(set(rankings.values()))

    def kendall(r1, r2):
        pos1, pos2 = {a: i for i, a in enumerate(r1)}, {a: i for i, a in enumerate(r2)}
        return sum((pos1[a] - pos1[b]) * (pos2[a] - pos2[b]) < 0 for a, b in combinations(agents, 2))
    worlds = list(cube)
    distances = [kendall(rankings[x], rankings[y]) for x, y in combinations(worlds, 2)]
    winners = {}
    for r in rankings.values():
        winners[r[0]] = winners.get(r[0], 0) + 1
    return {"distinct_rankings": len(distinct), "worlds": len(worlds), "winner_counts": winners,
            "kendall_distance": {"mean": float(np.mean(distances)) if distances else 0.0,
                                 "max": int(max(distances)) if distances else 0,
                                 "pairs_total": len(agents) * (len(agents) - 1) // 2},
            "rankings": {w: list(r) for w, r in rankings.items()}}


def decomposition(cube: dict, agents: tuple[str, ...], groups: dict[str, str]) -> dict:
    """Variance of costs split into market-seed, within-group world (parameter) and between-group (model class)."""
    out = {}
    for agent in agents:
        means = {w: float(np.mean(cube[w][agent])) for w in cube}
        seed_var = float(np.mean([np.var(cube[w][agent], ddof=1) for w in cube]))
        by_group: dict[str, list[float]] = {}
        for w, m in means.items():
            by_group.setdefault(groups.get(w, w), []).append(m)
        within = [np.var(v, ddof=1) for v in by_group.values() if len(v) > 1]
        group_means = [np.mean(v) for v in by_group.values()]
        out[agent] = {"market_seed_sd_bps": math.sqrt(seed_var),
                      "parameter_sd_bps": math.sqrt(float(np.mean(within))) if within else None,
                      "model_class_sd_bps": float(np.std(group_means, ddof=1)) if len(group_means) > 1 else None,
                      "world_mean_range_bps": float(max(means.values()) - min(means.values()))}
    return out
