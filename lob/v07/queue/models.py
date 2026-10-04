"""Queue-position uncertainty for hypothetical passive orders on aggregate L2 (workstreams 5, 79).

A passive order of size ``q`` joins the back of its price level at ``t0``. The
level's subsequent history is a sequence of ``LevelEvent``: prints at the price
(``trade_at``), prints through it (``trade_through``) and displayed-size
observations (``size``). A size decrease not explained by prints at the price is
a cancellation of unknown position. Models differ only in where they put those
cancellations and in what fills the order:

* ``conservative`` — only prints through the price fill (v0.5 lower bound);
* ``fifo_lower`` — cancellations are behind the order;
* ``probabilistic`` — the expected share ``F(a)`` of each cancellation is ahead,
  where ``a`` is the fraction of the level ahead of the order and ``F`` the CDF
  of cancellation queue positions (uniform ``F(a) = a`` is pro-rata);
* ``fifo_upper`` — cancellations are ahead of the order;
* ``optimistic`` — the order is at the front; every print at or through fills.

Filled quantity is ordered ``conservative <= fifo_lower <= probabilistic <=
fifo_upper <= optimistic`` for any monotone ``F`` (property-tested). ``draw``
samples cancellation positions instead of taking expectations, giving a fill
distribution. All models assume the hypothetical order changes nobody's
behaviour, no hidden liquidity and price-time priority; aggregate L2 establishes
none of these, so results are ASSUMPTION_DEPENDENT bounds, never exact fills.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

MODELS = ("conservative", "fifo_lower", "probabilistic", "fifo_upper", "optimistic")


@dataclass(frozen=True)
class LevelEvent:
    t: float
    kind: str      # trade_at | trade_through | size
    qty: float     # print size, or the new displayed level size for ``size``


def uniform_cdf(a: float) -> float:
    return min(max(a, 0.0), 1.0)


def _cancel_share(model: str, ahead: float, level: float, cancelled: float, cdf: Callable[[float], float],
                  rng: np.random.Generator | None) -> float:
    if model in {"conservative", "fifo_lower", "optimistic"}:
        return 0.0
    if model == "fifo_upper":
        return min(cancelled, ahead)
    fraction = ahead / level if level > 0 else 0.0
    if rng is None:
        return min(cancelled * cdf(fraction), ahead)
    return min(float(rng.binomial(int(round(cancelled)), cdf(fraction))) if cancelled >= 1 else
               cancelled * float(rng.random() < cdf(fraction)), ahead)


def fill(model: str, events: Iterable[LevelEvent], *, qty: float, level_at_join: float,
         cdf: Callable[[float], float] = uniform_cdf, rng: np.random.Generator | None = None) -> dict:
    """Filled quantity and fill time of one hypothetical passive order under ``model``."""
    if model not in MODELS:
        raise ValueError(f"unknown queue model {model!r}")
    ahead = 0.0 if model == "optimistic" else float(level_at_join)
    level = float(level_at_join)          # displayed level excluding our order
    filled, first_fill = 0.0, None
    for event in events:
        if filled >= qty:
            break
        if event.kind == "trade_through":
            # Volume printed through the price exhausted the level; it fills the order (v0.5 conservative rule).
            take = min(qty - filled, event.qty)
            ahead = 0.0
            level = 0.0
        elif event.kind == "trade_at":
            if model == "conservative":
                take = 0.0
            else:
                consumed_ahead = min(ahead, event.qty)
                ahead -= consumed_ahead
                take = min(qty - filled, event.qty - consumed_ahead)
            # Historically the whole print removed displayed volume (no-impact assumption), wherever we sit.
            level = max(level - event.qty, 0.0)
        elif event.kind == "size":
            take = 0.0
            new_level = max(float(event.qty), 0.0)
            if new_level < level:
                cancelled = level - new_level
                ahead -= _cancel_share(model, ahead, level, cancelled, cdf, rng)
            level = new_level
            ahead = min(ahead, level)
        else:
            raise ValueError(f"unknown level event {event.kind!r}")
        if take > 0:
            filled += take
            first_fill = event.t if first_fill is None else first_fill
    return {"model": model, "filled": filled, "complete": filled >= qty - 1e-12, "first_fill_t": first_fill,
            "remaining_ahead": ahead}


def all_models(events: list[LevelEvent], *, qty: float, level_at_join: float,
               cdf: Callable[[float], float] = uniform_cdf) -> dict:
    return {m: fill(m, events, qty=qty, level_at_join=level_at_join, cdf=cdf) for m in MODELS}


def fill_distribution(events: list[LevelEvent], *, qty: float, level_at_join: float, draws: int, seed: int,
                      cdf: Callable[[float], float] = uniform_cdf) -> dict:
    rng = np.random.default_rng(seed)
    filled = np.asarray([fill("probabilistic", events, qty=qty, level_at_join=level_at_join, cdf=cdf,
                              rng=rng)["filled"] for _ in range(draws)])
    return {"mean": float(filled.mean()), "p_complete": float(np.mean(filled >= qty - 1e-12)),
            "quantiles": {f"p{q}": float(np.quantile(filled, q / 100)) for q in (5, 50, 95)}, "draws": draws}


def level_events(tape, *, side: int, price: float, start: float, stop: float, tol: float = 1e-9) -> tuple[float, list]:
    """Extract the level history of ``price`` from a 100 ms ``BookTape`` (bid side ``+1``, ask ``-1``).

    The resting order is on the bid for ``side=+1`` and is hit by sell aggressors.
    Returns the displayed size at join and the event list. Displayed size is 0 while the
    price is outside the top 10 levels (it then counts as unobserved, not cancelled).
    """
    prices, sizes = (tape.bp, tape.bq) if side > 0 else (tape.ap, tape.aq)
    i0 = int(np.searchsorted(tape.t, start))
    i1 = int(np.searchsorted(tape.t, stop))

    def size_at(i: int) -> float | None:
        row = np.where(np.abs(prices[i] - price) <= tol)[0]
        if len(row):
            return float(sizes[i, row[0]])
        inside = prices[i][np.isfinite(prices[i])]
        if not len(inside):
            return None
        # Price better than the whole visible side: level is empty; worse than the deepest: unobserved.
        deepest = inside.min() if side > 0 else inside.max()
        return None if (side > 0 and price < deepest) or (side < 0 and price > deepest) else 0.0

    join = size_at(i0) or 0.0
    trades = [(t, p, q, s) for t, p, q, s in zip(tape.trade_t, tape.trade_p, tape.trade_q, tape.trade_s)
              if start <= t < stop and s == -side]
    events, k = [], 0
    for i in range(i0 + 1, i1):
        while k < len(trades) and trades[k][0] <= tape.t[i]:
            t, p, q, _ = trades[k]
            through = p < price - tol if side > 0 else p > price + tol
            at = abs(p - price) <= tol
            if through or at:
                events.append(LevelEvent(t, "trade_through" if through else "trade_at", q))
            k += 1
        size = size_at(i)
        if size is not None:
            events.append(LevelEvent(float(tape.t[i]), "size", size))
    return join, events
