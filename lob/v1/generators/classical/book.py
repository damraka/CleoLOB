"""Corrected, state-dependent book-placement calibration (M70; audit Issue A).

The defect (v0.9, frozen): ``lob.v09.generators.classical.book_samples`` inferred additions only between
consecutive 100 ms frames whose touch price was unchanged (``same = px[1:] == px[:-1]``), and indexed levels
relative to the touch. An addition that *improves* the touch moves the touch price, so it was never observed; every
``add_level`` sample was >= 0 and ``lob.v08.market.sim.BookModel.fit`` estimated ``improve_prob = mean(add_level < 0)
* 2 = 0``. With improvement impossible, a spread opened by a depleted touch can only close by the opposite touch
being depleted towards it, so every generator family drifted to spreads of tens of ticks (v0.9 realism benchmark:
mean spread 45 ticks generated vs 1.006 observed; ``P(spread > 1)`` 1.0 vs 0.0008).

The correction aligns consecutive frames by *absolute price*, not by level:

* for each side, the visible size at every price of the previous and the current frame is compared; prices inside
  the previous spread had size 0 before, so an increase there is an inside-spread addition whose level is
  ``-(ticks inside the previous touch)``;
* additions at or behind the previous touch keep their level relative to the previous touch;
* a decrease at the previous touch is first explained by passive-side prints in the bin (v0.9 rule), the remainder
  is a cancellation; decreases deeper in the book are cancellations.

From these events ``fit_book_model`` estimates, besides the v0.8 moments, the **state-dependent improvement
probability** ``P(addition lands inside the spread | spread = s)`` for s = 2, 3 and >= 4 ticks, and the improvement
depth distribution. Empty spread classes get a Jeffreys Beta(1/2, 1/2) posterior mean pooled with the other classes
and are flagged ``identified=False``; no value is invented silently.

Every inferred event is an inference from aggregate 100 ms L2 snapshots (several messages can merge within a bin);
the labels say so. ``MarketSimV1`` uses the state-dependent table when it places background additions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ....v08.market.sim import BookModel, MarketSim, _geometric

SPREAD_CLASSES = (2, 3, 4)            # 4 means ">= 4 ticks"
MIN_EVENTS = 20


def _spread_class(spread: int) -> int:
    return min(int(spread), SPREAD_CLASSES[-1])


def infer_book_events(bid_px, ask_px, depth, volume, valid, *, max_samples: int = 400_000) -> dict:
    """Book events inferred between consecutive valid frames, aligned by absolute price.

    ``depth`` is (n, 2, L) visible sizes at contiguous prices from each touch (bid side index 0), ``volume`` is
    (n, 2) traded volume per aggressor side (buy, sell). Returns arrays of add levels/sizes (negative level =
    inside the previous spread), the previous-frame spread of every addition, cancel levels/fractions, and counts.
    """
    bid_px, ask_px = np.asarray(bid_px, np.int64), np.asarray(ask_px, np.int64)
    depth = np.asarray(depth, float)
    volume = np.asarray(volume, float)
    valid = np.asarray(valid, bool)
    n, _, L = depth.shape
    add_level, add_size, add_spread, add_side = [], [], [], []
    cancel_level, cancel_frac = [], []
    pairs = np.flatnonzero(valid[1:] & valid[:-1]) + 1
    for i in pairs:
        if len(add_level) >= max_samples:
            break
        b0, a0, b1, a1 = int(bid_px[i - 1]), int(ask_px[i - 1]), int(bid_px[i]), int(ask_px[i])
        if not (b0 < a0 and b1 < a1):
            continue
        spread0 = a0 - b0
        for side in (0, 1):
            sign = 1 if side == 0 else -1           # bid prices decrease with level, ask prices increase
            touch0, touch1 = (b0, b1) if side == 0 else (a0, a1)
            prev = {touch0 - sign * k: depth[i - 1, side, k] for k in range(L)}
            new = {touch1 - sign * k: depth[i, side, k] for k in range(L)}
            # passive-side prints of this bin: sells hit bids (aggressor sell = column 1) and buys lift asks
            traded = volume[i, 1 - side]
            # comparable window: prices visible in both frames, plus prices inside the previous spread
            deepest0 = touch0 - sign * (L - 1)
            deepest1 = touch1 - sign * (L - 1)
            far = min(deepest0, deepest1) if side == 0 else max(deepest0, deepest1)
            for price, size in new.items():
                beyond_far = price < far if side == 0 else price > far
                if beyond_far:
                    continue
                inside = (price > touch0) if side == 0 else (price < touch0)
                if inside:
                    # prices strictly inside the previous spread were empty on this side
                    if (side == 0 and price >= a0) or (side == 1 and price <= b0):
                        continue      # a crossed-through price; never an addition on this side
                    if size > 0:
                        add_level.append(-abs(price - touch0))
                        add_size.append(float(size))
                        add_spread.append(spread0)
                        add_side.append(side)
                    continue
                before = prev.get(price, 0.0)
                diff = size - before
                level = abs(touch0 - price)
                if diff > 0:
                    add_level.append(level)
                    add_size.append(float(diff))
                    add_spread.append(spread0)
                    add_side.append(side)
                elif diff < 0:
                    removed = -diff
                    if level == 0:
                        removed = max(removed - traded, 0.0)
                    if removed > 0 and before > 0:
                        cancel_level.append(level)
                        cancel_frac.append(min(removed / before, 1.0))
    return {"add_level": np.asarray(add_level, float), "add_size": np.asarray(add_size, float),
            "add_spread": np.asarray(add_spread, float), "add_side": np.asarray(add_side, float),
            "cancel_level": np.asarray(cancel_level, float), "cancel_frac": np.asarray(cancel_frac, float),
            "pairs": int(len(pairs)),
            "label": "INFERRED from aggregate 100 ms L2 snapshots aligned by absolute price; merged messages possible"}


@dataclass(frozen=True)
class ImprovementTable:
    """P(addition inside the spread | spread class) with counts and identifiability flags."""
    probs: dict[int, float]
    counts: dict[int, tuple[int, int]]          # class -> (improving additions, all additions)
    identified: dict[int, bool]
    depth_p: float                              # geometric parameter of the improvement depth (ticks >= 1)

    def prob(self, spread: int) -> float:
        if spread <= 1:
            return 0.0
        return self.probs[_spread_class(spread)]

    def to_dict(self) -> dict:
        return {"probs": {str(k): v for k, v in self.probs.items()},
                "counts": {str(k): list(v) for k, v in self.counts.items()},
                "identified": {str(k): v for k, v in self.identified.items()}, "depth_p": self.depth_p}


def improvement_table(events: dict) -> ImprovementTable:
    lv, sp = events["add_level"], events["add_spread"]
    counts, probs, ident = {}, {}, {}
    total_k = int(((lv < 0) & (sp >= 2)).sum())
    total_n = int((sp >= 2).sum())
    pooled = (total_k + 0.5) / (total_n + 1.0)
    for c in SPREAD_CLASSES:
        m = (sp >= c) if c == SPREAD_CLASSES[-1] else (sp == c)
        k, nn = int((lv[m] < 0).sum()), int(m.sum())
        counts[c] = (k, nn)
        ident[c] = nn >= MIN_EVENTS
        # Jeffreys posterior mean; an unidentified class shrinks to the pooled rate (flagged, never hidden)
        probs[c] = (k + 0.5) / (nn + 1.0) if ident[c] else pooled
    deep = -lv[lv < 0]
    depth_p = float(np.clip(1.0 / deep.mean(), 0.01, 0.99)) if len(deep) >= MIN_EVENTS else 0.99
    return ImprovementTable(probs, counts, ident, depth_p)


@dataclass(frozen=True)
class BookModelV1(BookModel):
    """v0.8 ``BookModel`` plus the state-dependent improvement table (M70)."""
    improve_by_spread: tuple = ((2, 0.0), (3, 0.0), (4, 0.0))
    improve_depth_p: float = 0.99
    calibration: dict = field(default_factory=dict, compare=False, hash=False)

    def improve(self, spread: int) -> float:
        if spread <= 1:
            return 0.0
        return float(dict(self.improve_by_spread)[_spread_class(spread)])


def fit_book_model(events: dict, *, market_sizes, refill_depths, mid_ticks: int, **fixed) -> BookModelV1:
    """Moment estimates from corrected events (levels >= 0 for placement, negative levels for improvement)."""
    d = BookModel().to_dict()
    lv = events["add_level"]
    behind = lv[lv >= 0]
    if len(behind) >= MIN_EVENTS:
        d["add_level_p"] = float(np.clip(1.0 / (1.0 + behind.mean()), 0.01, 0.99))
    cl = events["cancel_level"]
    if len(cl) >= MIN_EVENTS:
        d["cancel_level_p"] = float(np.clip(1.0 / (1.0 + cl.mean()), 0.01, 0.99))
    if len(events["add_size"]) >= MIN_EVENTS:
        d["limit_size_mean"] = float(max(1.0, np.mean(events["add_size"])))
    ms = np.asarray(market_sizes, float)
    if len(ms) >= MIN_EVENTS:
        d["market_size_mean"] = float(max(1.0, ms.mean()))
    if len(events["cancel_frac"]) >= MIN_EVENTS:
        d["cancel_frac_mean"] = float(np.clip(np.mean(events["cancel_frac"]), 0.01, 1.0))
    rd = np.asarray(refill_depths, float)
    rd = rd[rd > 0]
    if len(rd):
        d["refill_mean"] = float(max(1.0, np.median(rd)))
    table = improvement_table(events)
    # the v0.8 scalar is kept consistent (probability for a 2-tick spread) for code that only reads it
    d["improve_prob"] = float(table.probs[2])
    d["mid_ticks"] = int(max(mid_ticks, 100))
    d.update(fixed)
    calibration = {"improvement": table.to_dict(),
                   "events": {k: int(len(v)) for k, v in events.items() if isinstance(v, np.ndarray)},
                   "label": events.get("label", "")}
    return BookModelV1(**{k: v for k, v in d.items() if k in BookModel.__dataclass_fields__},
                       improve_by_spread=tuple(sorted(table.probs.items())), improve_depth_p=table.depth_p,
                       calibration=calibration)


def fit_from_frame(frame) -> BookModelV1:
    """Fit from a v0.8 ``EventFrame`` (historical or simulated)."""
    ev = infer_book_events(frame.bid_px, frame.ask_px, frame.depth, frame.volume, frame.valid)
    mid = frame.mid[frame.mid > 0]
    depth = frame.depth[frame.valid][:, :, 1:].ravel()
    market = frame.samples.get("market_size", frame.samples.get("trade_q", np.zeros(0)))
    return fit_book_model(ev, market_sizes=market, refill_depths=depth,
                          mid_ticks=int(np.median(mid)) if len(mid) else 4000)


class MarketSimV1(MarketSim):
    """``MarketSim`` whose background additions use the state-dependent improvement table of ``BookModelV1``."""

    def _apply(self, mark: int, u: tuple[float, float, float]) -> None:
        if mark not in (2, 3) or not isinstance(self.model, BookModelV1):
            return super()._apply(mark, u)
        book, m = self.book, self.model
        side = 1 if mark == 2 else -1
        spread = book.spread()
        if spread > 1 and u[2] < m.improve(spread):
            # improvement depth: geometric on {1, ...}, at most spread - 1 ticks (re-uses u[0] like the v0.8 path)
            ticks = min(_geometric(min(max(u[0], 1e-12), 1 - 1e-12), m.improve_depth_p), spread - 1)
            level = -ticks
        else:
            level = min(_geometric(u[0], m.add_level_p) - 1, self.visible - 1)
        size = _geometric(u[1], 1.0 / m.limit_size_mean)
        book.add(side, level, size)
        if self.record:
            self.samples["add_level"].append(level)
            self.samples["add_size"].append(size)


def simulate_v1(generator, model: BookModelV1, n_bins: int, seed: int, *, warmup_bins: int = 0):
    sim = MarketSimV1(generator, model, seed)
    sim.run(warmup_bins + n_bins)
    return sim.frame(start=warmup_bins)


def spread_summary(frame) -> dict:
    v = frame.valid.astype(bool)
    s = (frame.ask_px - frame.bid_px)[v].astype(float)
    if len(s) == 0:
        return {"mean": math.nan, "p_gt1": math.nan, "p_ge3": math.nan}
    return {"mean": float(s.mean()), "p_gt1": float((s > 1).mean()), "p_ge3": float((s >= 3).mean()),
            "max": float(s.max())}


__all__ = ["infer_book_events", "improvement_table", "ImprovementTable", "BookModelV1", "fit_book_model",
           "fit_from_frame", "MarketSimV1", "simulate_v1", "spread_summary", "SPREAD_CLASSES"]
