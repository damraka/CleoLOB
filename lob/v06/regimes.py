"""M12/M17/M18: regime-conditioned realism and regime-transition realism.

Regime labels use the v0.5 M6 development-only thresholds unchanged and the v0.5
labelling code (``lob.regimes``) on a top-5 view of the v0.6 tape, so v0.5 and
v0.6 regimes are identical by construction. Labels describe observable
conditions of 5-minute blocks; they are not causal market states.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..observables_v2 import Tape as V05Tape
from ..regimes import BLOCK_S, block_statistics, label
from .observables import measure, sketch
from .tape import BookTape

THRESHOLDS_RUN = "results/v05/m6/thresholds/result.json"
DIMENSIONS = ("volatility", "spread", "activity", "stress")


def v05_view(tape: BookTape) -> V05Tape:
    return V05Tape(tape.t, tape.bp[:, :5], tape.bq[:, :5], tape.ap[:, :5], tape.aq[:, :5], tape.trade_t,
                   tape.trade_p, tape.trade_q, tape.trade_s)


def load_thresholds(root: Path) -> tuple[dict, str]:
    text = (root / THRESHOLDS_RUN).read_text(encoding="utf-8")
    from . import protocol as pr
    return json.loads(text)["thresholds"], pr.text_sha256(root / THRESHOLDS_RUN)


def labelled_blocks(tape: BookTape, thresholds: dict) -> list[dict]:
    """v0.5 block statistics and labels for every complete 5-minute block with >= 80% valid samples."""
    rows = []
    for stats in block_statistics(v05_view(tape)):
        rows.append({**stats, "labels": label(stats, thresholds)})
    return rows


def regime_sketches(tape: BookTape, design: dict, thresholds: dict, *, dimension: str = "volatility") -> dict:
    """Sketches of 5-minute blocks grouped by regime label (blocks lacking statistics are skipped)."""
    groups: dict[str, list[dict]] = {}
    starts = {round(b["start_s"], 6): b["labels"][dimension] for b in labelled_blocks(tape, thresholds)}
    for block in tape.blocks(BLOCK_S):
        if not len(block.t):
            continue
        value = starts.get(round(float(block.t[0]), 6))
        if value is None:
            continue
        groups.setdefault(value, []).append(sketch(measure(block), design))
    return groups


def transitions(paths: list[list[str]], states: tuple[str, ...]) -> dict:
    """Transition counts/matrix, dwell times (block runs) and occupancy over separate label paths."""
    index = {s: i for i, s in enumerate(states)}
    matrix = np.zeros((len(states), len(states)))
    runs: dict[str, list[int]] = {s: [] for s in states}
    occupancy = {s: 0 for s in states}
    total = 0
    for labels in paths:
        for a, b in zip(labels[:-1], labels[1:]):
            matrix[index[a], index[b]] += 1
        current, length = None, 0
        for value in labels:
            occupancy[value] += 1
            total += 1
            if value == current:
                length += 1
            else:
                if current is not None:
                    runs[current].append(length)
                current, length = value, 1
        if current is not None:
            runs[current].append(length)
    rows = matrix.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probabilities = np.where(rows > 0, matrix / np.where(rows > 0, rows, 1), np.nan)
    return {"states": list(states), "counts": matrix.tolist(), "probabilities": probabilities.tolist(),
            "occupancy": {s: (occupancy[s] / total if total else None) for s in states},
            "mean_dwell_blocks": {s: (float(np.mean(r)) if r else None) for s, r in runs.items()},
            "blocks": total, "paths": len(paths)}


def transition_comparison(hist: dict, sim: dict) -> dict:
    """Row-wise total variation of transition probabilities, occupancy TV and dwell-time log ratios."""
    ph, ps = np.asarray(hist["probabilities"], float), np.asarray(sim["probabilities"], float)
    rows = [0.5 * float(np.nansum(np.abs(ph[i] - ps[i]))) for i in range(len(ph))
            if np.isfinite(ph[i]).all() and np.isfinite(ps[i]).all()]
    occupancy = 0.5 * sum(abs((hist["occupancy"][s] or 0) - (sim["occupancy"][s] or 0)) for s in hist["states"])
    dwell = {}
    for s in hist["states"]:
        a, b = hist["mean_dwell_blocks"][s], sim["mean_dwell_blocks"][s]
        dwell[s] = float(abs(np.log(b / a))) if a and b else None
    return {"transition_row_tv_mean": float(np.mean(rows)) if rows else None, "occupancy_tv": float(occupancy),
            "dwell_log_ratio": dwell}


def label_sequence(blocks: list[dict], dimension: str) -> list[str]:
    return [b["labels"][dimension] for b in blocks]


STATES = {"volatility": ("low", "high"), "spread": ("low", "high"), "activity": ("low", "high"),
          "stress": ("normal", "stress")}


def transition_profile(paths: list[list[dict]]) -> dict:
    """Per dimension transition statistics over separate paths of labelled blocks."""
    out = {}
    for dimension in DIMENSIONS:
        stats = transitions([label_sequence(blocks, dimension) for blocks in paths], STATES[dimension])
        if dimension == "stress":
            counts = np.asarray(stats["counts"])
            normal = counts[0].sum()
            stress = counts[1].sum()
            stats["stress_entry_per_normal_block"] = float(counts[0, 1] / normal) if normal else None
            stats["stress_recovery_per_stress_block"] = float(counts[1, 0] / stress) if stress else None
        out[dimension] = stats
    return out
