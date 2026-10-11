"""Book model and the bin-level simulation driver shared by every v0.8 generator.

``BookModel`` holds the parameters a count generator does not: where additions
and cancellations land (geometric level distributions, inside-spread
improvement probability), event sizes, cancellation fractions, refill depth,
the background-cancel position rule and the hidden-liquidity knob. It is fitted
from book-model samples (``EventFrame.samples``) by moments.

``MarketSim`` advances an ``AggBook`` in 100 ms bins. At each bin start the
generator receives the book features and returns event counts; the events are
placed uniformly inside the bin, merged in time order with any strategy
messages (``schedule``), and realized against the live book. Randomness is
seed-addressable: every bin uses its own generator derived from
``SeedSequence([seed, stream, bin])``, so twin runs (with and without a
strategy) share their background draws for as long as the generator's rates do
not diverge, and results never depend on scheduling.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Callable

import numpy as np

from .book import BUY, SELL, AggBook
from .frame import BIN_S, FEATURES, MARKS, VISIBLE, EventFrame, features

STREAM_MARKET = 8_008_001


@dataclass(frozen=True)
class BookModel:
    add_level_p: float = 0.45        # P(level k) ~ (1 - p)^k for additions at or behind the touch
    improve_prob: float = 0.25       # P(addition improves inside the spread | spread > 1)
    cancel_level_p: float = 0.35
    market_size_mean: float = 2.0    # lots (geometric)
    limit_size_mean: float = 3.0
    cancel_frac_mean: float = 0.35   # mean fraction of the level's background removed by one cancellation
    refill_mean: float = 12.0        # mean depth of a newly exposed level (geometric)
    cancel_position: str = "uniform"
    hidden_fraction: float = 0.0
    levels: int = 40
    mid_ticks: int = 4000

    def __post_init__(self) -> None:
        for name in ("add_level_p", "cancel_level_p"):
            if not 0.01 <= getattr(self, name) <= 0.99:
                raise ValueError(f"{name} must lie in [0.01, 0.99]")
        if not 0.0 <= self.improve_prob <= 1.0 or not 0.01 <= self.cancel_frac_mean <= 1.0:
            raise ValueError("probabilities/fractions out of range")
        if min(self.market_size_mean, self.limit_size_mean, self.refill_mean) < 1.0:
            raise ValueError("size means must be at least one lot")

    def refill(self, level: int, rng: np.random.Generator) -> int:
        return int(rng.geometric(1.0 / self.refill_mean))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BookModel:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def replace(self, **changes) -> BookModel:
        return BookModel.from_dict({**self.to_dict(), **changes})

    @classmethod
    def fit(cls, frame: EventFrame, **fixed) -> BookModel:
        """Moment estimates from book-model samples; missing samples keep the defaults (reported in meta)."""
        s, d = frame.samples, cls().to_dict()

        def geo_p(levels) -> float | None:
            levels = np.asarray(levels, float)
            levels = levels[levels >= 0]
            return None if len(levels) < 20 else float(np.clip(1.0 / (1.0 + levels.mean()), 0.01, 0.99))

        for key, name in (("add_level", "add_level_p"), ("cancel_level", "cancel_level_p")):
            if key in s and (p := geo_p(s[key])) is not None:
                d[name] = p
        if "add_level" in s and len(s["add_level"]) >= 20:
            d["improve_prob"] = float(np.clip(np.mean(np.asarray(s["add_level"]) < 0) * 2.0, 0.0, 1.0))
        for key, name in (("market_size", "market_size_mean"), ("add_size", "limit_size_mean")):
            if key in s and len(s[key]) >= 20:
                d[name] = float(max(1.0, np.mean(s[key])))
        if "cancel_frac" in s and len(s["cancel_frac"]) >= 20:
            d["cancel_frac_mean"] = float(np.clip(np.mean(s["cancel_frac"]), 0.01, 1.0))
        if frame.valid.any():
            depth = frame.depth[frame.valid][:, :, 1:].astype(float)
            d["refill_mean"] = float(max(1.0, np.median(depth[depth > 0]) if (depth > 0).any() else 1.0))
        d.update(fixed)
        return cls.from_dict(d)


def _geometric(u: float, p: float) -> int:
    """Inverse-CDF geometric on {1, 2, ...} with success probability ``p`` (``u`` in (0, 1))."""
    if p >= 1.0:
        return 1
    return 1 + int(math.log(max(u, 1e-300)) / math.log(1.0 - p))


class MarketSim:
    """Bin-level driver: ``generator`` decides counts, ``book_model`` decides where they land."""

    def __init__(self, generator, book_model: BookModel, seed: int, *, record: bool = True,
                 visible: int = VISIBLE) -> None:
        self.generator, self.model, self.seed = generator, book_model, int(seed)
        self.visible = visible
        root = np.random.default_rng(np.random.SeedSequence([self.seed, STREAM_MARKET, 0]))
        self.book = AggBook(refill=book_model.refill, rng=root, levels=book_model.levels,
                            mid_ticks=book_model.mid_ticks, cancel_position=book_model.cancel_position,
                            hidden_fraction=book_model.hidden_fraction)
        self.mstate = generator.initial_state(simulate=True)
        self.bin = 0
        self.t = 0.0
        self._mids: deque = deque([self.book.mid()], maxlen=11)
        self.record = record
        self.rows: dict[str, list] = {k: [] for k in ("counts", "state", "bid_px", "ask_px", "depth", "volume")}
        self.mids: list[float] = [self.book.mid()]
        self.samples: dict[str, list] = {k: [] for k in ("add_level", "add_size", "cancel_level", "cancel_frac",
                                                          "market_size")}
        self.strategy_flow = np.zeros(2)   # own market volume (buy, sell) in the current bin
        self._cancel_decay = [(1.0 - book_model.cancel_level_p) ** k for k in range(visible)]

    # ------------------------------------------------------------------ one background event
    def _apply(self, mark: int, u: tuple[float, float, float]) -> None:
        """Realize one background event from three pre-drawn uniforms (level/side choice, size, position)."""
        book, m = self.book, self.model
        if mark < 2:
            size = _geometric(u[1], 1.0 / m.market_size_mean)
            book.market(BUY if mark == 0 else SELL, size, self.t)
            if self.record:
                self.samples["market_size"].append(size)
            return
        side = BUY if mark in (2, 4) else SELL
        if mark < 4:
            if book.spread() > 1 and u[2] < m.improve_prob:
                level = -1
            else:
                level = min(_geometric(u[0], m.add_level_p) - 1, self.visible - 1)
            size = _geometric(u[1], 1.0 / m.limit_size_mean)
            book.add(side, level, size)
            if self.record:
                self.samples["add_level"].append(level)
                self.samples["add_size"].append(size)
            return
        q = book.queues(side)
        decay = self._cancel_decay
        weights = [decay[k] if q[k] > 0 else 0.0 for k in range(self.visible)]
        total = sum(weights)
        if total <= 0:
            return
        target, acc, level = u[0] * total, 0.0, self.visible - 1
        for k, w in enumerate(weights):
            acc += w
            if target < acc:
                level = k
                break
        frac = min(1.0, u[1] * 2.0 * m.cancel_frac_mean)
        background = q[level] - book.own_volume(side, book.price(side, level))
        size = max(1, int(round(frac * background))) if background > 0 else 0
        if size > 0:
            book.cancel(side, level, size)
            if self.record:
                self.samples["cancel_level"].append(level)
                self.samples["cancel_frac"].append(frac)

    # ------------------------------------------------------------------ one bin
    def features(self) -> np.ndarray:
        b = self.book
        return features(b.bid_px, b.ask_px, b.bq, b.aq, list(self._mids))

    def run_bin(self, schedule: list[tuple[float, int, Callable[[], None]]] | None = None,
                own_counts: np.ndarray | None = None, external_flow: np.ndarray | None = None) -> np.ndarray:
        """Advance one bin. ``schedule``: (exchange time, tie-break, action) strategy messages inside this bin.

        Returns the realized counts (background plus strategy events reported by ``own_counts``).
        ``external_flow`` (buy, sell lots) is added to the strategy flow the generator responds to; it carries
        cross-impact from another instrument or venue in multi-book simulations.
        """
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, STREAM_MARKET, self.bin + 1]))
        self.book.rng = rng
        x = self.features()
        counts = np.asarray(self.generator.draw(self.mstate, x, rng), dtype=np.int64)
        counts = np.clip(counts, 0, 200)
        if self.record:
            b = self.book
            self.rows["state"].append(x)
            self.rows["bid_px"].append(b.bid_px)
            self.rows["ask_px"].append(b.ask_px)
            self.rows["depth"].append((b.bq[:self.visible], b.aq[:self.visible]))
        marks = np.repeat(np.arange(len(MARKS)), counts)
        times = self.t + rng.uniform(0.0, BIN_S, size=len(marks))
        order = np.argsort(times, kind="stable")
        events = [(float(times[i]), 1, int(marks[i]), None) for i in order]
        if schedule:
            events += [(float(at), 0, -1, action) for at, _, action in schedule]
            events.sort(key=lambda e: (e[0], e[1]))
        tape_start = len(self.book.tape)
        uniforms = rng.random((len(marks), 3)).tolist()
        k = 0
        for at, _, mark, action in events:
            self.t = at
            if action is not None:
                action()
            else:
                self._apply(mark, uniforms[k])
                k += 1
        self.t = (self.bin + 1) * BIN_S
        realized = counts.copy()
        if own_counts is not None:
            realized = realized + own_counts
        volume = [0, 0]
        own_flow = np.zeros(2)
        for _, _, qty, side, strategy in self.book.tape[tape_start:]:
            volume[0 if side == BUY else 1] += qty
            if strategy:
                own_flow[0 if side == BUY else 1] += qty
        self.strategy_flow = own_flow
        response_flow = own_flow if external_flow is None else own_flow + np.asarray(external_flow, float)
        self.mstate = self.generator.update(self.mstate, realized, x, own=response_flow)
        self.bin += 1
        mid = self.book.mid()
        self._mids.append(mid)
        if self.record:
            self.rows["counts"].append(realized)
            self.rows["volume"].append(volume)
            self.mids.append(mid)
        return realized

    def run(self, n_bins: int) -> EventFrame:
        for _ in range(int(n_bins)):
            self.run_bin()
        return self.frame()

    def frame(self, start: int = 0) -> EventFrame:
        if not self.record:
            raise RuntimeError("recording is disabled")
        n = len(self.rows["counts"])
        depth = np.zeros((n, 2, self.visible), dtype=np.int64)
        for i, (bq, aq) in enumerate(self.rows["depth"][:n]):
            depth[i, 0, :len(bq)] = bq
            depth[i, 1, :len(aq)] = aq
        frame = EventFrame(np.asarray(self.rows["counts"], dtype=np.int64).reshape(n, len(MARKS)),
                           np.asarray(self.rows["state"][:n], float).reshape(n, len(FEATURES)),
                           np.asarray(self.rows["bid_px"][:n], dtype=np.int64),
                           np.asarray(self.rows["ask_px"][:n], dtype=np.int64), depth,
                           np.asarray(self.rows["volume"], dtype=np.int64).reshape(n, 2),
                           np.asarray(self.mids[:n + 1], float), np.ones(n, dtype=bool),
                           samples={**{k: np.asarray(v, float) for k, v in self.samples.items()},
                                    **self._trade_samples()},
                           meta={"source": "simulation", "generator": getattr(self.generator, "name", "?"),
                                 "seed": self.seed, "tick_bps": 1e4 / self.model.mid_ticks})
        return frame.slice(start, n) if start else frame

    def _trade_samples(self) -> dict:
        tape = self.book.tape
        return {"trade_t": np.asarray([x[0] for x in tape], float), "trade_px": np.asarray([x[1] for x in tape], float),
                "trade_q": np.asarray([x[2] for x in tape], float),
                "trade_s": np.asarray([x[3] for x in tape], float),
                "trade_strategy": np.asarray([x[4] for x in tape], float)}


def simulate(generator, book_model: BookModel, n_bins: int, seed: int, *, warmup_bins: int = 0) -> EventFrame:
    sim = MarketSim(generator, book_model, seed)
    sim.run(warmup_bins + n_bins)
    return sim.frame(start=warmup_bins)


def tick_bps(book_model: BookModel) -> float:
    return 1e4 / book_model.mid_ticks


__all__ = ["BookModel", "MarketSim", "simulate", "tick_bps", "BIN_S", "MARKS", "FEATURES", "math"]
