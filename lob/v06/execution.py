"""M10/M11/M14/M15/M16: execution outcomes across simulator worlds.

Pure analysis over episode rows (one row per world x agent x training seed x
market seed). Costs are the mandate completion-adjusted cost (bps; lower is
better); completion is within-horizon mandate completion. Learned agents are
averaged over their training seeds per market seed before any comparison.
INVALID rows keep their error text; a cell with more than 5% INVALID rows is
INVALID, and any planned comparison touching a missing or INVALID value is
WITHHELD (it counts as not passing and never shrinks its family).

No result here is a historical, live or profitability statement: every world is
a simulator configuration.
"""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np

from .inference import (difference_interval, direction, kendall_tau, mean_interval, spearman_interval)

COST = "mandate_completion_adjusted_cost_bps"
COMPLETION = "mandate_within_horizon_completion"
CLASSICAL = ("twap", "vwap", "pov", "ac")
PRIMARY = CLASSICAL + ("ppo:single", "dqn:single")
LEARNED = ("ppo:single", "dqn:single", "ppo:ensemble", "dqn:ensemble")
INVALID_CELL_FRACTION = 0.05
MATERIAL_BPS = 1.0


def _cost(row: dict) -> float:
    value = row.get(COST)
    ok = row.get("status") in {"VALID", "WARNING"} and value is not None
    return float(value) if ok and math.isfinite(float(value)) else math.nan


def cells(rows: list[dict], markets: list[int]) -> dict[tuple[str, str], dict]:
    """Per (world, agent): cost and completion vectors over market seeds (learned: training-seed means)."""
    grouped: dict[tuple[str, str], dict[int, list[dict]]] = {}
    for row in rows:
        grouped.setdefault((row["world"], row["agent"]), {}).setdefault(int(row["seed"]), []).append(row)
    out = {}
    for key, by_market in grouped.items():
        costs, completions, invalid, total = [], [], 0, 0
        training: dict[str, list[float]] = {}
        for market in markets:
            items = by_market.get(market, [])
            values = [_cost(r) for r in items]
            invalid += sum(not math.isfinite(v) for v in values) + (0 if items else 1)
            total += max(1, len(items))
            costs.append(float(np.mean(values)) if values and all(map(math.isfinite, values)) else math.nan)
            flags = [r.get(COMPLETION) for r in items]
            completions.append(float(np.mean([f is True for f in flags])) if flags and None not in flags else math.nan)
            for r in items:
                training.setdefault(str(r.get("training_seed")), []).append(_cost(r))
        fraction = invalid / total if total else 1.0
        out[key] = {"cost": np.asarray(costs), "completion": np.asarray(completions), "invalid_fraction": fraction,
                    "status": "INVALID" if fraction > INVALID_CELL_FRACTION else "VALID",
                    "training_seed_means": {s: (float(np.mean(v)) if all(map(math.isfinite, v)) else None)
                                            for s, v in training.items()}}
    return out


def summarize_cell(cell: dict, *, alpha: float, samples: int, seed: int) -> dict:
    cost, completion = cell["cost"], cell["completion"]
    finite = cost[np.isfinite(cost)]
    summary = {"status": cell["status"], "invalid_fraction": cell["invalid_fraction"], "markets": int(len(cost)),
               "mean_cost_bps": float(finite.mean()) if len(finite) else None,
               "sd_cost_bps": float(finite.std(ddof=1)) if len(finite) > 1 else None,
               "within_horizon_completion_rate": float(np.nanmean(completion)) if np.isfinite(completion).any() else None,
               "training_seed_means": cell["training_seed_means"]}
    if cell["status"] == "VALID" and np.isfinite(cost).all():
        summary["mean_ci"] = mean_interval(cost, alpha=alpha, samples=samples, seed=seed)
    return summary


def paired(cell_map: dict, world: str, left: str, right: str, *, alpha: float, samples: int, seed: int) -> dict:
    a, b = cell_map.get((world, left)), cell_map.get((world, right))
    if a is None or b is None or a["status"] != "VALID" or b["status"] != "VALID":
        return {"status": "WITHHELD", "reason": "missing or INVALID cell"}
    if not (np.isfinite(a["cost"]).all() and np.isfinite(b["cost"]).all()):
        return {"status": "WITHHELD", "reason": "missing or INVALID market outcome"}
    return difference_interval(a["cost"], b["cost"], alpha=alpha, samples=samples, seed=seed, paired=True)


def pair_seed(base: int, *parts: str) -> int:
    """Deterministic, collision-resistant sub-seed for one comparison."""
    import hashlib
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return (base * 1_000_003 + int.from_bytes(digest[:4], "little")) % (2**32)


# ----------------------------------------------------------------------------- H5 equifinality


def equifinality(cell_map: dict, members: list[str], *, alpha: float, samples: int, seed: int) -> dict:
    pairs = list(combinations(members, 2))
    family = len(CLASSICAL) * len(pairs)
    if len(members) < 2:
        return {"status": "NOT_AVAILABLE", "reason": "fewer than two ensemble members", "family_size": 0}
    adjusted = alpha / family
    contrasts = []
    for agent in CLASSICAL:
        for left, right in pairs:
            a, b = cell_map.get((left, agent)), cell_map.get((right, agent))
            if a is None or b is None or a["status"] != "VALID" or b["status"] != "VALID" \
                    or not (np.isfinite(a["cost"]).all() and np.isfinite(b["cost"]).all()):
                result = {"status": "WITHHELD"}
            else:
                result = difference_interval(a["cost"], b["cost"], alpha=adjusted, samples=samples,
                                             seed=pair_seed(seed, agent, left, right), paired=False)
            material = result.get("status") == "AVAILABLE" and direction(result) != 0 and \
                abs(result["mean"]) >= MATERIAL_BPS
            contrasts.append({"agent": agent, "left": left, "right": right, **result, "material": bool(material)})
    passing = [c for c in contrasts if c["material"]]
    return {"status": "ESTABLISHED" if passing else "NOT_ESTABLISHED", "family_size": family,
            "adjusted_alpha": adjusted, "material_bps": MATERIAL_BPS, "material_contrasts": len(passing),
            "withheld": sum(c.get("status") == "WITHHELD" for c in contrasts), "contrasts": contrasts,
            "max_abs_difference_bps": max((abs(c["mean"]) for c in contrasts if c.get("status") == "AVAILABLE"),
                                          default=None)}


# ----------------------------------------------------------------------------- H6 / M15 rank stability


def conclusions(cell_map: dict, worlds: list[str], agents: tuple[str, ...], *, alpha: float, samples: int,
                seed: int) -> dict:
    """Paired interval and direction per (world, pair); direction +1 means left is costlier."""
    out = {}
    for world in worlds:
        for left, right in combinations(agents, 2):
            result = paired(cell_map, world, left, right, alpha=alpha, samples=samples,
                            seed=pair_seed(seed, world, left, right))
            out[(world, left, right)] = {**result, "direction": direction(result)}
    return out


def rank_stability(cell_map: dict, members: list[str], agents: tuple[str, ...], *, alpha: float, samples: int,
                   seed: int, reference: str) -> dict:
    pairs = list(combinations(agents, 2))
    if len(members) < 2:
        return {"status": "NOT_AVAILABLE", "reason": "fewer than two ensemble members"}
    family = len(pairs) * len(members)
    adjusted = alpha / family
    found = conclusions(cell_map, members, agents, alpha=adjusted, samples=samples, seed=seed)
    per_pair, reversals = {}, []
    for left, right in pairs:
        dirs = [found[(w, left, right)]["direction"] for w in members]
        means = [found[(w, left, right)].get("mean") for w in members]
        determinate = [d for d in dirs if d in (1, -1)]
        certified = 1 in determinate and -1 in determinate
        signs = [np.sign(m) for m in means if m is not None]
        ref_sign = np.sign(found[(reference, left, right)].get("mean") or 0.0) if reference in members else None
        world_pairs = list(combinations([d for d in dirs if d in (1, -1)], 2))
        per_pair[f"{left}|{right}"] = {
            "directions": dict(zip(members, dirs)), "mean_differences_bps": dict(zip(members, means)),
            "certified_reversal": certified,
            "fraction_left_costlier": float(np.mean([d == 1 for d in dirs])),
            "fraction_right_costlier": float(np.mean([d == -1 for d in dirs])),
            "indeterminate_fraction": float(np.mean([d == 0 for d in dirs])),
            "withheld_fraction": float(np.mean([d is None for d in dirs])),
            "sign_stability": (float(np.mean([s == ref_sign for s in signs])) if ref_sign is not None and signs else None),
            "reversal_frequency": (float(np.mean([a != b for a, b in world_pairs])) if world_pairs else 0.0)}
        if certified:
            reversals.append(f"{left}|{right}")
    rankings = {}
    for world in members:
        means = {a: (float(np.mean(cell_map[(world, a)]["cost"])) if (world, a) in cell_map and
                     cell_map[(world, a)]["status"] == "VALID" and np.isfinite(cell_map[(world, a)]["cost"]).all()
                     else None) for a in agents}
        rankings[world] = sorted((a for a in means if means[a] is not None), key=lambda a: means[a])
    taus = [kendall_tau(rankings[a], rankings[b]) for a, b in combinations(members, 2)]
    taus = [t for t in taus if t is not None]
    return {"status": "ESTABLISHED" if reversals else "NOT_ESTABLISHED", "family_size": family,
            "adjusted_alpha": adjusted, "certified_reversals": reversals, "pairs": per_pair, "rankings": rankings,
            "kendall_tau_mean": float(np.mean(taus)) if taus else None,
            "kendall_tau_min": float(np.min(taus)) if taus else None,
            "interpretation": ("ESTABLISHED means some pairwise conclusion reverses with certified intervals across "
                               "plausible worlds; NOT_ESTABLISHED does not establish invariance.")}


# ----------------------------------------------------------------------------- H7 / M16 execution-sensitive realism


def disagreement(cell_map: dict, worlds: list[str], reference: str, agents: tuple[str, ...], *, alpha: float,
                 samples: int, seed: int) -> dict[str, dict]:
    """Per world: fraction of pairwise conclusions differing from the reference world, and mean |delta| shift."""
    adjusted = alpha / len(list(combinations(agents, 2)))
    found = conclusions(cell_map, [reference, *[w for w in worlds if w != reference]], agents, alpha=adjusted,
                        samples=samples, seed=seed)
    out = {}
    for world in worlds:
        flags, shifts = [], []
        for left, right in combinations(agents, 2):
            ref, here = found[(reference, left, right)], found[(world, left, right)]
            if ref["direction"] is None or here["direction"] is None:
                continue
            flags.append(ref["direction"] != here["direction"])
            shifts.append(abs(here["mean"] - ref["mean"]))
        out[world] = {"conclusion_disagreement": float(np.mean(flags)) if flags else None,
                      "mean_abs_shift_bps": float(np.mean(shifts)) if shifts else None, "evaluable_pairs": len(flags)}
    return out


def execution_sensitive_realism(family_errors: dict[str, dict[str, float | None]], instability: dict[str, dict], *,
                                families: tuple[str, ...], alpha: float, samples: int, seed: int,
                                minimum_worlds: int = 8, threshold: float = 0.5) -> dict:
    worlds = sorted(w for w in instability if instability[w]["conclusion_disagreement"] is not None
                    and w in family_errors)
    if len(worlds) < minimum_worlds:
        return {"status": "INCONCLUSIVE", "reason": f"fewer than {minimum_worlds} worlds", "worlds": len(worlds)}
    y = [instability[w]["conclusion_disagreement"] for w in worlds]
    shift = [instability[w]["mean_abs_shift_bps"] for w in worlds]
    rows = {}
    for j, family in enumerate(families):
        x = [family_errors[w].get(family) for w in worlds]
        x = [math.nan if v is None else v for v in x]
        rows[family] = {"disagreement": spearman_interval(x, y, alpha=alpha, samples=samples, seed=seed + j),
                        "shift": spearman_interval(x, shift, alpha=alpha, samples=samples, seed=seed + 100 + j)}
    pvalues = [rows[f]["disagreement"].get("p_one_sided", 1.0) if rows[f]["disagreement"]["status"] == "AVAILABLE"
               else 1.0 for f in families]
    pvalues = [1.0 if p is None or not math.isfinite(p) else p for p in pvalues]
    from .inference import holm
    adjusted = holm(pvalues)
    passing = []
    for family, p in zip(families, adjusted):
        entry = rows[family]["disagreement"]
        entry["holm_adjusted_p"] = p
        if entry["status"] == "AVAILABLE" and entry["rho"] >= threshold and p < alpha:
            passing.append(family)
    return {"status": "ESTABLISHED" if passing else "NOT_ESTABLISHED", "families": rows, "passing": passing,
            "worlds": worlds, "threshold_rho": threshold, "correction": "holm", "family_size": len(families),
            "interpretation": "Within-simulator association across synthetic worlds; not a real-market causal claim."}


# ----------------------------------------------------------------------------- M14 model-risk decomposition


def decomposition(cell_map: dict, *, selected: str, members: list[str], interventions: list[str],
                  regime_worlds: list[str], agents: tuple[str, ...], fill_bounds: dict | None = None) -> dict:
    """Separate standard deviations (bps) of each uncertainty source; never summed into one number."""
    def world_mean(world: str, agent: str) -> float | None:
        cell = cell_map.get((world, agent))
        if cell is None or cell["status"] != "VALID" or not np.isfinite(cell["cost"]).all():
            return None
        return float(np.mean(cell["cost"]))

    def sd(values: list) -> float | None:
        finite = [v for v in values if v is not None]
        return float(np.std(finite, ddof=1)) if len(finite) >= 2 else None

    out = {}
    for agent in agents:
        cell = cell_map.get((selected, agent))
        seed_sd = float(np.std(cell["cost"], ddof=1)) if cell is not None and np.isfinite(cell["cost"]).all() else None
        training = cell["training_seed_means"] if cell is not None else {}
        out[agent] = {
            "market_seed_sd_bps": seed_sd,
            "market_seed_se_bps": seed_sd / math.sqrt(len(cell["cost"])) if seed_sd is not None else None,
            "calibration_ensemble_sd_bps": sd([world_mean(w, agent) for w in members]),
            "structural_intervention_sd_bps": sd([world_mean(w, agent) for w in interventions]),
            "regime_model_sd_bps": sd([world_mean(w, agent) for w in [selected, *regime_worlds]]),
            "training_seed_sd_bps": sd(list(training.values())) if len(training) > 1 else None,
            "historical_fill_bound_width_bps": (fill_bounds or {}).get(agent)}
    return {"agents": out, "units": "bps of mandate completion-adjusted cost",
            "rule": "each component is a separate standard deviation (or width); components are not additive"}
