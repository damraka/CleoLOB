"""Objective market regimes on fixed 5-minute blocks (M6).

Regimes are defined from observable block statistics with thresholds fitted on
the development period only, before any holdout is read. They are never chosen
from policy or calibration outcomes. A regime label describes observable
conditions; it is not a causal state of the market.
"""
from __future__ import annotations

import numpy as np

from .observables_v2 import SAMPLE_DT, Tape

BLOCK_S = 300.0
DIMENSIONS = ("volatility", "spread", "depth", "activity", "imbalance", "trend")
STRESS_QUANTILE = 0.9


def block_statistics(tape: Tape, block_s: float = BLOCK_S) -> list[dict]:
    """One row per complete block with enough valid samples (>= 80%)."""
    rows = []
    step = int(round(1 / SAMPLE_DT))
    start = float(tape.t[0])
    stop = float(tape.t[-1])
    while start + block_s <= stop + SAMPLE_DT:
        block = tape.block(start, start + block_s)
        valid = block.valid
        if len(valid) and valid.mean() >= 0.8:
            mid = block.mid
            idx = np.arange(0, len(block.t), step)
            ok = valid[idx]
            one = idx[ok]
            returns = np.log(mid[one][1:] / mid[one][:-1]) * 1e4
            spread = ((block.ap[:, 0] - block.bp[:, 0]) / mid * 1e4)[valid]
            d5b, d5a = block.bq.sum(1)[valid], block.aq.sum(1)[valid]
            first, last = np.flatnonzero(valid)[[0, -1]]
            rows.append({"start_s": start, "volatility": float(returns.std()) if len(returns) > 1 else 0.0,
                         "spread": float(np.mean(spread)), "depth": float(np.mean(d5b + d5a)),
                         "activity": float(len(block.trade_t)),
                         "imbalance": float(np.mean(np.abs((d5b - d5a) / (d5b + d5a)))),
                         "trend": float(abs(np.log(mid[last] / mid[first]) * 1e4))})
        start += block_s
    return rows


def fit_thresholds(development_blocks: list[dict]) -> dict:
    """Median split per dimension; stress = volatility and spread both above the 90th percentile."""
    if len(development_blocks) < 20:
        raise ValueError("at least 20 development blocks are required to fix regime thresholds")
    thresholds = {}
    for name in DIMENSIONS:
        values = np.asarray([b[name] for b in development_blocks])
        thresholds[name] = {"median": float(np.median(values)),
                            "p90": float(np.quantile(values, STRESS_QUANTILE))}
    return thresholds


def label(block: dict, thresholds: dict) -> dict:
    labels = {name: ("high" if block[name] > thresholds[name]["median"] else "low") for name in DIMENSIONS}
    labels["stress"] = ("stress" if block["volatility"] > thresholds["volatility"]["p90"]
                        and block["spread"] > thresholds["spread"]["p90"] else "normal")
    return labels


def regime_counts(blocks: list[dict], thresholds: dict) -> dict:
    counts: dict[str, dict[str, int]] = {}
    for block in blocks:
        for name, value in label(block, thresholds).items():
            counts.setdefault(name, {}).setdefault(value, 0)
            counts[name][value] += 1
    return counts


def blocks_in(blocks: list[dict], thresholds: dict, dimension: str, value: str) -> list[float]:
    return [b["start_s"] for b in blocks if label(b, thresholds)[dimension] == value]
