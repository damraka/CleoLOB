"""v0.7 historical tape builder with registered reader limits and a data-quality report.

The algorithm is the v0.6 measurement grid (``lob.v06.tape.tape_from_tardis``,
frozen and not edited): causal 100 ms samples of the top 10 levels, 5 s
staleness, crossed books invalid, prints grouped by source timestamp and
aggressor. The only differences are explicit reader limits (BitMEX files are far
larger than Deribit's) and the quality report returned with the tape. A
differential test checks that both builders produce identical tapes.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import math
from pathlib import Path
import pickle

import numpy as np

from ...observables_v2 import group_prints
from ...replay.l2 import L2Replay
from ..v06_compat import BookTape, LEVELS, MAX_STALENESS_S, SAMPLE_DT, TAPE_FIELDS
from .quality import QualityThresholds, assess

CACHE_DIR = "data/v07/cache"
READER_LIMITS = {"max_file_bytes": 2 * 1024**3, "max_expanded_bytes": 32 * 1024**3, "max_rows": 1_500_000_000,
                 "max_levels": 200_000, "max_group_rows": 200_000}


def build_tape(updates: str | Path, trades: str | Path, *, tick: float, levels: int = LEVELS,
               limits: dict | None = None) -> tuple[BookTape, dict]:
    limits = {**READER_LIMITS, **(limits or {})}
    replay = L2Replay(Path(updates), depth=levels, **limits)
    step_us = int(round(SAMPLE_DT * 1e6))
    stale_us = int(MAX_STALENESS_S * 1e6)
    rows_t: list[float] = []
    book = {name: [] for name in ("bp", "bq", "ap", "aq")}
    empty_p, empty_q = [math.nan] * levels, [0.0] * levels
    stale = crossed = 0
    previous = None
    next_us = None

    def emit(state, at_us: int) -> None:
        nonlocal stale, crossed
        rows_t.append(at_us / 1e6)
        is_stale = at_us - state.local_timestamp_us > stale_us
        stale += int(is_stale)
        crossed += int(state.crossed)
        ok = not is_stale and not state.crossed
        for side, p_name, q_name in ((state.bids, "bp", "bq"), (state.asks, "ap", "aq")):
            if not ok:
                book[p_name].append(empty_p)
                book[q_name].append(empty_q)
                continue
            top = side[:levels]
            book[p_name].append([float(x[0]) for x in top] + [math.nan] * (levels - len(top)))
            book[q_name].append([float(x[1]) for x in top] + [0.0] * (levels - len(top)))

    for state in replay:
        now = state.local_timestamp_us
        if next_us is None:
            next_us = (now // step_us + 1) * step_us
        while previous is not None and next_us < now:
            emit(previous, next_us)
            next_us += step_us
        previous = state
    l2_stats = replay.stats
    if not l2_stats["complete"]:
        raise ValueError("L2 source incomplete")
    if not rows_t:
        raise ValueError("L2 source produced no samples")
    t0 = rows_t[0]
    times, prices, sizes, sides, keys = [], [], [], [], []
    trade_stats = {"rows": 0, "unknown_side": 0, "outside_book_window": 0, "duplicate_ids": 0}
    seen: set[str] = set()
    opener = gzip.open if str(trades).endswith(".gz") else open
    with opener(trades, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            trade_stats["rows"] += 1
            if row.get("id"):
                trade_stats["duplicate_ids"] += int(row["id"] in seen)
                seen.add(row["id"])
            side = row["side"].lower()
            if side not in {"buy", "sell"}:
                trade_stats["unknown_side"] += 1
                continue
            local = int(row["local_timestamp"]) / 1e6
            if local < t0 or local >= rows_t[-1] + SAMPLE_DT:
                trade_stats["outside_book_window"] += 1
                continue
            times.append(local)
            prices.append(float(row["price"]))
            sizes.append(float(row["amount"]))
            sides.append(1 if side == "buy" else -1)
            keys.append(int(row["timestamp"]) * 4 + (1 if side == "buy" else 2))
    et, ep, eq, es = group_prints(np.asarray(times, float), np.asarray(prices, float), np.asarray(sizes, float),
                                  np.asarray(sides, float), np.asarray(keys, dtype=np.int64))
    arrays = {name: np.asarray(values, dtype=float) for name, values in book.items()}
    tape = BookTape(np.asarray(rows_t) - t0, arrays["bp"], arrays["bq"], arrays["ap"], arrays["aq"],
                    np.asarray(et, float) - t0, np.asarray(ep, float), np.asarray(eq, float), np.asarray(es, float),
                    float(tick))
    samples = {"samples": len(rows_t), "stale_samples": stale, "crossed_samples": crossed}
    return tape, assess(tape, l2_stats, trade_stats, samples, QualityThresholds())


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cached_build(files: dict[str, Path], *, tick: float, root: Path, levels: int = LEVELS) -> tuple[BookTape, dict]:
    """Tape plus quality report, cached in the ignored ``data/v07/cache`` under the exact source hashes."""
    key = hashlib.sha256(("|".join(_digest(files[k]) for k in ("l2", "trades")) + f"|{levels}|{tick}|v07"
                          ).encode()).hexdigest()[:24]
    cache = root / CACHE_DIR / f"{key}.tape.pkl"
    if cache.is_file():
        stored = pickle.loads(cache.read_bytes())
        return BookTape(**stored["tape"]), stored["quality"]
    tape, quality = build_tape(files["l2"], files["trades"], tick=tick, levels=levels)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".part")
    temporary.write_bytes(pickle.dumps({"tape": {n: getattr(tape, n) for n in TAPE_FIELDS + ("tick",)},
                                        "quality": quality}))
    temporary.replace(cache)
    return tape, quality
