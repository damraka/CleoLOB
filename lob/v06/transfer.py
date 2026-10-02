"""M24: historical execution transfer v2 under bounded fills; H11 and H12.

Episodes replay displayed historical liquidity through the unchanged
``lob.historical_sim.HistoricalSimulator`` under the conservative and optimistic
fill modes (separate self-consistent paths). v0.6 worlds are calibrated in real
seconds, so the clock ratio is 1 and lots per native unit are 1 / development
depth scale. No exact historical fill, exact FIFO, profitability or live claim
follows. Predictions come from simulator worlds (sources A-D); every comparison
keeps its uncertainty and is classified, never collapsed into a point verdict.
"""
from __future__ import annotations

import csv
import gzip
from itertools import combinations
import math

import numpy as np

from ..historical_sim import FILL_MODES, HistoricalEpisode
from ..replay.l2 import L2Replay
from .execution import COST, PRIMARY, cells
from .inference import counts, difference_interval, direction, interval

BOOK_LEVELS = 20
EPISODES = 144
PAIRS = list(combinations(PRIMARY, 2))


def extract_episodes(files: dict, *, tick: float, lots_per_native: float, window_s: float,
                     count: int = EPISODES, label: str = "") -> tuple[list[HistoricalEpisode], dict]:
    """Episodes at first capture + 300 s + 600 s * k (the v0.5 rule) with clock ratio 1."""
    replay = L2Replay(files["l2"], depth=BOOK_LEVELS, max_rows=80_000_000, max_expanded_bytes=6 * 1024**3,
                      max_file_bytes=512 * 1024**2)
    starts, windows, first, previous = None, None, None, None
    for state in replay:
        t = state.local_timestamp_us / 1e6
        if starts is None:
            first = t
            starts = [first + 300.0 + 600.0 * k for k in range(count)]
            windows = [[] for _ in starts]
        snapshot = (t, {int(round(float(p) / tick)): float(q) * lots_per_native for p, q in state.bids},
                    {int(round(float(p) / tick)): float(q) * lots_per_native for p, q in state.asks})
        k = int((t - first - 300.0) // 600.0)
        for j in (k, k + 1):
            if 0 <= j < len(starts) and starts[j] - 5.0 <= t <= starts[j] + window_s:
                if not windows[j] and previous is not None:
                    windows[j].append(previous)
                windows[j].append(snapshot)
        previous = snapshot
    if not replay.stats["complete"]:
        raise ValueError("L2 source incomplete")
    prints = [[] for _ in starts]
    opener = gzip.open if str(files["trades"]).endswith(".gz") else open
    with opener(files["trades"], "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            side = row["side"].lower()
            if side not in {"buy", "sell"}:
                continue
            t = int(row["local_timestamp"]) / 1e6
            k = int((t - first - 300.0) // 600.0)
            if 0 <= k < len(starts) and starts[k] < t <= starts[k] + window_s:
                prints[k].append((t, int(round(float(row["price"]) / tick)), float(row["amount"]) * lots_per_native,
                                  "BUY" if side == "buy" else "SELL"))
    episodes, skipped = [], []
    for k, start in enumerate(starts):
        updates = [u for u in windows[k] if u[0] <= start + window_s]
        if not updates or updates[0][0] > start:
            skipped.append(k)
            continue
        head = [u for u in updates if u[0] <= start][-1]
        updates = [(start, head[1], head[2])] + [u for u in updates if u[0] > start]
        episodes.append(HistoricalEpisode(start, updates, prints[k], tick, 1.0, label=f"{label}#{k}",
                                          meta={"index": k}))
    return episodes, {"episodes": len(episodes), "planned": len(starts), "skipped": skipped, "tick": tick,
                      "lots_per_native": lots_per_native, "clock_ratio": 1.0, "window_s": window_s}


# ----------------------------------------------------------------------------- historical analysis


def episode_matrix(rows: list[dict], mode: str, agent: str) -> dict[int, float]:
    """Per episode: cost (learned agents averaged over training seeds; NaN if any seed is INVALID)."""
    values: dict[int, list[float]] = {}
    for r in rows:
        if r["fill_mode"] == mode and r["agent"] == agent:
            ok = r.get("status") in {"VALID", "WARNING"} and r.get(COST) is not None
            values.setdefault(int(r["episode"]), []).append(float(r[COST]) if ok else math.nan)
    return {k: (float(np.mean(v)) if all(map(math.isfinite, v)) else math.nan) for k, v in values.items()}


def historical_pairs(rows: list[dict], *, alpha: float, samples: int, seed: int) -> dict:
    out = {}
    for mode_index, mode in enumerate(FILL_MODES):
        for pair_index, (left, right) in enumerate(PAIRS):
            a, b = episode_matrix(rows, mode, left), episode_matrix(rows, mode, right)
            keys = sorted(set(a) & set(b))
            delta = np.asarray([a[k] - b[k] for k in keys])
            if len(delta) < 10 or not np.isfinite(delta).all():
                result = {"status": "WITHHELD", "reason": "fewer than 10 episodes or INVALID rows"}
            else:
                from .inference import mean_interval
                result = mean_interval(delta, alpha=alpha, samples=samples, seed=seed + 100 * mode_index + pair_index)
            out[f"{mode}|{left}|{right}"] = {**result, "direction": direction(result)}
    return out


def fill_semantics(pairs: dict) -> dict:
    """H12: stable if each pair has the same determinate direction in both modes or is indeterminate in both."""
    per_pair, opposite, one_mode = {}, [], []
    for left, right in PAIRS:
        c = pairs[f"conservative|{left}|{right}"]["direction"]
        o = pairs[f"optimistic|{left}|{right}"]["direction"]
        if c is None or o is None:
            state = "not_evaluable"
        elif c == o:
            state = "stable_determinate" if c != 0 else "stable_indeterminate"
        elif c != 0 and o != 0:
            state = "opposite"
            opposite.append(f"{left}|{right}")
        else:
            state = "assumption_dependent"
            one_mode.append(f"{left}|{right}")
        per_pair[f"{left}|{right}"] = {"conservative": c, "optimistic": o, "state": state}
    evaluable = [p for p in per_pair.values() if p["state"] != "not_evaluable"]
    status = ("FAILED" if opposite else "NOT_ESTABLISHED" if one_mode else
              "ESTABLISHED" if evaluable else "NOT_AVAILABLE")
    return {"status": status, "pairs": per_pair, "opposite": opposite, "assumption_dependent": one_mode,
            "evaluable": len(evaluable)}


# ----------------------------------------------------------------------------- predictions (sources A-D)


def predicted_difference(cell_map: dict, worlds: list[str], left: str, right: str, *, alpha: float, samples: int,
                         seed: int) -> dict:
    """Two-stage bootstrap: resample worlds, then market seeds within each world, of the paired difference."""
    deltas = []
    for world in worlds:
        a, b = cell_map.get((world, left)), cell_map.get((world, right))
        if a is None or b is None or a["status"] != "VALID" or b["status"] != "VALID":
            return {"status": "WITHHELD", "reason": f"missing or INVALID cell in {world}"}
        d = a["cost"] - b["cost"]
        if not np.isfinite(d).all():
            return {"status": "WITHHELD", "reason": f"INVALID market outcome in {world}"}
        deltas.append(d)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for s in range(samples):
        w = counts(len(deltas), rng) if len(deltas) > 1 else np.ones(1)
        total, weight = 0.0, 0.0
        for d, c in zip(deltas, w):
            if c:
                total += c * d[rng.integers(0, len(d), len(d))].mean()
                weight += c
        draws[s] = total / weight
    low, high = interval(draws, alpha)
    return {"status": "AVAILABLE", "mean": float(np.mean([d.mean() for d in deltas])), "ci_low": low, "ci_high": high,
            "worlds": len(deltas)}


def classify(predicted: dict, historical: dict[str, dict]) -> str:
    p = direction(predicted)
    h = {mode: historical[mode]["direction"] for mode in FILL_MODES}
    if p is None or any(v is None for v in h.values()):
        return "not_evaluable"
    if p == 0:
        return "indeterminate"
    if all(v == p for v in h.values()):
        return "agrees"
    if all(v == -p for v in h.values()):
        return "reverses"
    if any(v == p for v in h.values()):
        return "assumption_dependent"
    return "indeterminate"


def source_comparison(cell_map: dict, sources: dict[str, list[str]], pairs_hist: dict, *, alpha: float, samples: int,
                      seed: int) -> dict:
    out = {}
    family = len(PAIRS) * len(sources)
    for s_index, (source, worlds) in enumerate(sources.items()):
        entries, tally = {}, {}
        for p_index, (left, right) in enumerate(PAIRS):
            predicted = predicted_difference(cell_map, worlds, left, right, alpha=alpha / family, samples=samples,
                                             seed=seed + 1000 * s_index + p_index)
            historical = {mode: pairs_hist[f"{mode}|{left}|{right}"] for mode in FILL_MODES}
            label = classify(predicted, historical)
            tally[label] = tally.get(label, 0) + 1
            entries[f"{left}|{right}"] = {"predicted": predicted, "classification": label}
        determinate = [e for e in entries.values() if e["classification"] in {"agrees", "reverses"}]
        out[source] = {"worlds": worlds, "pairs": entries, "counts": tally,
                       "agreement_rate_among_determinate": (sum(e["classification"] == "agrees" for e in determinate)
                                                            / len(determinate)) if determinate else None}
    return {"family_size": family, "adjusted_alpha": alpha / family, "sources": out}


# ----------------------------------------------------------------------------- H11


def transfer_gap(rows: list[dict], cell_map: dict, *, algorithm: str, single_worlds: list[str],
                 ensemble_worlds: list[str], alpha: float, samples: int, seed: int, reference: str = "twap") -> dict:
    """|historical - simulated| (policy - TWAP) for the ensemble variant minus the single variant, per fill mode."""
    out = {}
    for mode_index, mode in enumerate(FILL_MODES):
        ref = episode_matrix(rows, mode, reference)
        series = {}
        for variant in ("single", "ensemble"):
            values = episode_matrix(rows, mode, f"{algorithm}:{variant}")
            keys = sorted(set(values) & set(ref))
            series[variant] = np.asarray([values[k] - ref[k] for k in keys])
        if any(len(v) < 10 or not np.isfinite(v).all() for v in series.values()) or \
                len(series["single"]) != len(series["ensemble"]):
            out[mode] = {"status": "WITHHELD", "reason": "missing or INVALID historical episodes"}
            continue
        sims = {}
        for variant, worlds in (("single", single_worlds), ("ensemble", ensemble_worlds)):
            deltas = []
            for world in worlds:
                a, b = cell_map.get((world, f"{algorithm}:{variant}")), cell_map.get((world, reference))
                if a is None or b is None or not (np.isfinite(a["cost"]).all() and np.isfinite(b["cost"]).all()):
                    deltas = None
                    break
                deltas.append(a["cost"] - b["cost"])
            sims[variant] = deltas
        if any(v is None for v in sims.values()):
            out[mode] = {"status": "WITHHELD", "reason": "missing or INVALID simulated cells"}
            continue
        rng = np.random.default_rng(seed + mode_index)
        draws = np.empty(samples)
        n = len(series["single"])
        for s in range(samples):
            pick = rng.integers(0, n, n)
            gaps = {}
            for variant in ("single", "ensemble"):
                deltas = sims[variant]
                w = counts(len(deltas), rng) if len(deltas) > 1 else np.ones(1)
                predicted = sum(c * d[rng.integers(0, len(d), len(d))].mean() for d, c in zip(deltas, w) if c) / w.sum()
                gaps[variant] = abs(series[variant][pick].mean() - predicted)
            draws[s] = gaps["ensemble"] - gaps["single"]
        low, high = interval(draws, alpha)
        point = {v: abs(series[v].mean() - float(np.mean([d.mean() for d in sims[v]]))) for v in series}
        out[mode] = {"status": "AVAILABLE", "gap_single_bps": point["single"], "gap_ensemble_bps": point["ensemble"],
                     "difference": point["ensemble"] - point["single"], "ci_low": low, "ci_high": high,
                     "historical_single_bps": float(series["single"].mean()),
                     "historical_ensemble_bps": float(series["ensemble"].mean())}
    dirs = {m: direction(r) if r.get("status") == "AVAILABLE" else None for m, r in out.items()}
    if any(d is None for d in dirs.values()):
        status = "NOT_AVAILABLE"
    elif all(d == -1 for d in dirs.values()):
        status = "ESTABLISHED"
    elif all(d == 1 for d in dirs.values()):
        status = "FAILED"
    elif any(d == -1 for d in dirs.values()):
        status = "ASSUMPTION_DEPENDENT"
    else:
        status = "NOT_ESTABLISHED"
    return {"status": status, "modes": out, "directions": dirs}


__all__ = ["extract_episodes", "historical_pairs", "fill_semantics", "source_comparison", "transfer_gap", "cells",
           "difference_interval"]
