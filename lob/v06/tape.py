"""Ten-level book tapes: the single measurement grid for history and simulation.

A tape holds causal 100 ms samples of the top ``LEVELS`` displayed levels per
side and aggressor-grouped trade events (prints sharing a source timestamp and
aggressor side, or one simulated taker order, form one event; the v0.5 rule).
Prices are native currency units, quantities are lots (native size divided by
the development depth scale), and ``tick`` is the instrument tick in native
units. Historical samples whose last update is older than ``MAX_STALENESS_S``
or whose book is crossed are invalid (NaN prices, zero quantities).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import gzip
import hashlib
import math
from pathlib import Path
import pickle

import numpy as np

from ..observables_v2 import group_prints
from ..replay.l2 import L2Replay

SAMPLE_DT = 0.1
LEVELS = 10
MAX_STALENESS_S = 5.0
WARMUP_S = 60.0
CACHE_DIR = "data/v06/cache"
TAPE_FIELDS = ("t", "bp", "bq", "ap", "aq", "trade_t", "trade_p", "trade_q", "trade_s")


@dataclass
class BookTape:
    t: np.ndarray        # (n,) seconds on the regular 100 ms grid
    bp: np.ndarray       # (n, L) bid prices (native), NaN where absent
    bq: np.ndarray       # (n, L) bid quantities (lots), 0 where absent
    ap: np.ndarray
    aq: np.ndarray
    trade_t: np.ndarray  # aggressive event times (s)
    trade_p: np.ndarray  # event VWAP price (native)
    trade_q: np.ndarray  # event size (lots)
    trade_s: np.ndarray  # +1 buy aggressor, -1 sell aggressor
    tick: float          # native price units per tick

    def __post_init__(self) -> None:
        n = len(self.t)
        for name in ("bp", "bq", "ap", "aq"):
            array = getattr(self, name)
            if array.ndim != 2 or array.shape[0] != n:
                raise ValueError(f"{name} must have shape (n, levels)")
        if len({getattr(self, name).shape[1] for name in ("bp", "bq", "ap", "aq")}) != 1:
            raise ValueError("all book arrays must have the same number of levels")
        if n > 1 and not np.allclose(np.diff(self.t), SAMPLE_DT, atol=1e-6):
            raise ValueError("tape samples must lie on the regular 100 ms grid")
        lengths = {len(getattr(self, name)) for name in ("trade_t", "trade_p", "trade_q", "trade_s")}
        if len(lengths) != 1:
            raise ValueError("trade arrays must have equal length")
        if not (math.isfinite(self.tick) and self.tick > 0):
            raise ValueError("tick must be positive")

    @property
    def levels(self) -> int:
        return self.bp.shape[1]

    @property
    def duration(self) -> float:
        return len(self.t) * SAMPLE_DT

    @property
    def valid(self) -> np.ndarray:
        return (np.isfinite(self.bp[:, 0]) & np.isfinite(self.ap[:, 0]) & (self.bq[:, 0] > 0)
                & (self.aq[:, 0] > 0) & (self.bp[:, 0] < self.ap[:, 0]))

    @property
    def mid(self) -> np.ndarray:
        return (self.bp[:, 0] + self.ap[:, 0]) / 2

    def block(self, start: float, stop: float) -> BookTape:
        s = (self.t >= start - 1e-9) & (self.t < stop - 1e-9)
        e = (self.trade_t >= start - 1e-9) & (self.trade_t < stop - 1e-9)
        return BookTape(self.t[s], self.bp[s], self.bq[s], self.ap[s], self.aq[s], self.trade_t[e],
                        self.trade_p[e], self.trade_q[e], self.trade_s[e], self.tick)

    def blocks(self, seconds: float) -> list[BookTape]:
        """Complete consecutive blocks of ``seconds`` from the first sample."""
        if not len(self.t):
            return []
        count = int((self.duration + 1e-9) // seconds)
        start = float(self.t[0])
        return [self.block(start + k * seconds, start + (k + 1) * seconds) for k in range(count)]

    def rescaled(self, scale: float) -> BookTape:
        """Divide every quantity by ``scale`` (native units per lot)."""
        if not (math.isfinite(scale) and scale > 0):
            raise ValueError("scale must be positive")
        return replace(self, bq=self.bq / scale, aq=self.aq / scale, trade_q=self.trade_q / scale)


def depth_scale(tape: BookTape, lots: float = 1000.0) -> float:
    """The v0.5 rule: development median top-5 side depth (native) / 1000 lots."""
    valid = tape.valid
    depth = np.r_[tape.bq[valid, :5].sum(1), tape.aq[valid, :5].sum(1)]
    return float(np.median(depth)) / lots


def tick_bps(tape: BookTape) -> float:
    valid = tape.valid
    return float(tape.tick / np.median(tape.mid[valid]) * 1e4)


# ----------------------------------------------------------------------------- history


def tape_from_tardis(updates: str | Path, trades: str | Path, *, tick: float, levels: int = LEVELS) -> BookTape:
    """Causal 100 ms samples of the top ``levels`` plus grouped aggressive events (native units)."""
    replay = L2Replay(Path(updates), depth=levels, max_rows=80_000_000, max_expanded_bytes=6 * 1024**3,
                      max_file_bytes=512 * 1024**2)
    step_us = int(round(SAMPLE_DT * 1e6))
    stale_us = int(MAX_STALENESS_S * 1e6)
    rows_t: list[float] = []
    book = {name: [] for name in ("bp", "bq", "ap", "aq")}
    empty_p, empty_q = [math.nan] * levels, [0.0] * levels
    previous = None
    next_us = None

    def emit(state, at_us: int) -> None:
        rows_t.append(at_us / 1e6)
        ok = at_us - state.local_timestamp_us <= stale_us and not state.crossed
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
    if not replay.stats["complete"]:
        raise ValueError("L2 source incomplete")
    if not rows_t:
        raise ValueError("L2 source produced no samples")
    t0 = rows_t[0]
    times, prices, sizes, sides, keys = [], [], [], [], []
    opener = gzip.open if str(trades).endswith(".gz") else open
    with opener(trades, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            side = row["side"].lower()
            if side not in {"buy", "sell"}:
                continue
            local = int(row["local_timestamp"]) / 1e6
            if local < t0 or local >= rows_t[-1] + SAMPLE_DT:
                continue
            times.append(local)
            prices.append(float(row["price"]))
            sizes.append(float(row["amount"]))
            sides.append(1 if side == "buy" else -1)
            keys.append(int(row["timestamp"]) * 4 + (1 if side == "buy" else 2))
    et, ep, eq, es = group_prints(np.asarray(times, float), np.asarray(prices, float), np.asarray(sizes, float),
                                  np.asarray(sides, float), np.asarray(keys, dtype=np.int64))
    arrays = {name: np.asarray(values, dtype=float) for name, values in book.items()}
    return BookTape(np.asarray(rows_t) - t0, arrays["bp"], arrays["bq"], arrays["ap"], arrays["aq"],
                    np.asarray(et, float) - t0, np.asarray(ep, float), np.asarray(eq, float), np.asarray(es, float),
                    float(tick))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cached_tape(files: dict[str, Path], *, tick: float, root: Path, levels: int = LEVELS) -> BookTape:
    """Native-unit tape cached locally (ignored directory) under the exact source hashes."""
    key = hashlib.sha256(("|".join(_file_digest(files[k]) for k in ("l2", "trades")) + f"|{levels}|{tick}"
                          ).encode()).hexdigest()[:24]
    cache = root / CACHE_DIR / f"{key}.tape.pkl"
    if cache.is_file():
        stored = pickle.loads(cache.read_bytes())
        return BookTape(**stored)
    tape = tape_from_tardis(files["l2"], files["trades"], tick=tick, levels=levels)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".part")
    temporary.write_bytes(pickle.dumps({name: getattr(tape, name) for name in TAPE_FIELDS + ("tick",)}))
    temporary.replace(cache)
    return tape


# ----------------------------------------------------------------------------- simulation


def tape_from_simulator(spec, seed: int, *, seconds: float, warmup: float = WARMUP_S,
                        levels: int = LEVELS) -> BookTape:
    """Same grid and grouping for a simulator built from ``spec`` (a ``SimulatorSpec``)."""
    sim = spec.build(seed)
    sim.step(warmup)
    tick = sim.cfg.tick_size
    first_trade = len(sim.book.trades)
    n = int(round(seconds / SAMPLE_DT))
    bp, bq = np.full((n, levels), np.nan), np.zeros((n, levels))
    ap, aq = np.full((n, levels), np.nan), np.zeros((n, levels))
    start = sim.t
    for i in range(n):
        sim.step(SAMPLE_DT)
        bids, asks = sim.book.depth(levels)
        for j, (p, q) in enumerate(bids):
            bp[i, j], bq[i, j] = p * tick, q
        for j, (p, q) in enumerate(asks):
            ap[i, j], aq[i, j] = p * tick, q
    trades = sim.book.trades[first_trade:]
    if trades:
        t = np.asarray([x.time for x in trades]) - start
        keep = t <= n * SAMPLE_DT
        keys = np.asarray([(x.taker_order_id or 0) for x in trades], dtype=np.int64)
        et, ep, eq, es = group_prints(t[keep], np.asarray([x.price * tick for x in trades])[keep],
                                      np.asarray([float(x.qty) for x in trades])[keep],
                                      np.asarray([1.0 if x.taker_side.value == 1 else -1.0 for x in trades])[keep],
                                      keys[keep])
    else:
        et = ep = eq = es = np.asarray([], dtype=float)
    # Sample i is the book at the end of interval i; shift so sample 0 is at time 0.
    return BookTape(np.arange(n) * SAMPLE_DT, bp, bq, ap, aq, np.asarray(et, float) - SAMPLE_DT,
                    np.asarray(ep, float), np.asarray(eq, float), np.asarray(es, float), float(tick))
