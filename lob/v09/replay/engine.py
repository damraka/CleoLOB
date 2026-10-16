"""Bounded counterfactual execution replay on reconstructed L2 day caches (M26-M31, F1/F2).

A hypothetical strategy is injected into a recorded day under one explicit ``Assumption``. Everything it produces
is HYPOTHETICAL; ``bounds`` runs the registered assumption grid and reports intervals (BOUNDED,
ASSUMPTION_DEPENDENT). Nothing here is an actual historical fill.

Registered semantics (protocol ``counterfactual_assumptions`` and ``cost_endpoints``):

* **windows** - 24 per day at HH:10:00 UTC, 30 s warmup, 300 s horizon, decisions every 5 s; side alternates
  (buy on even windows); quantity = notional / arrival mid rounded down to the lot.
* **observation** - at decision time t the policy sees the book at ``t - latency`` (touch and contiguous depth from
  the 100 ms frame for the first ten ticks and from the 1 s deep snapshot beyond), traded volume since the last
  decision and recent mids. Policies are the unchanged v0.8 ``Policy`` classes plus the v0.9 additions.
* **market orders** arrive at ``t + latency`` and walk the absolute-price ladder of that time (frame levels within
  ten ticks of the touch, deep-snapshot levels beyond), net of liquidity the strategy already consumed that the
  impact rule says is not yet replenished (``none``: replenished immediately; ``transient``: exponential refill with
  a 10 s half-life; ``persistent``: never within the episode). Quantity beyond the 200 recorded levels is filled at
  the worst recorded level plus 1 tick per additional percent of the mandate (recorded as ``beyond_book``).
* **passive orders** rest at the own touch (or ``improve`` ticks inside) from arrival. Queue ahead = visible size at
  that price (``x (1 + hidden_fraction)`` under the conservative rule; inside-spread orders have none). Passive-side
  prints at the price consume ahead first, then fill the order; a print strictly through the price fills it
  completely. Visible decreases not explained by prints are cancellations credited to ahead entirely
  (``optimistic``), proportionally (``neutral``) or not at all (``conservative``).
* **completion** - hard: whatever remains at the horizon is sent as a market order; any quantity beyond the recorded
  book is valued at the worst recorded level plus the registered 25 bps penalty.
* **costs** - ``is_gross_bps`` and ``is_net_bps`` (registered fees: taker on aggressive, maker on passive fills, plus
  the penalty) relative to the arrival mid; ``markout5s_bps`` of passive fills is fill-weighted, positive =
  favourable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import itertools
import math

import numpy as np

from ...v08.execution.simulator import Observation
from ..data.daycache import BIN_MS, HORIZON_S, WARMUP_S, WINDOW_STARTS_S

QUEUE_RULES = ("optimistic", "neutral", "conservative")
IMPACT_RULES = ("none", "transient", "persistent")
TRANSIENT_HALF_LIFE_S = 10.0
DECISION_DT_S = 5.0
LADDER_TICKS = 200
LABEL = "HYPOTHETICAL"


@dataclass(frozen=True)
class Assumption:
    queue: str = "neutral"
    latency_s: float = 0.0
    hidden_fraction: float = 0.0
    impact: str = "none"

    def __post_init__(self) -> None:
        if self.queue not in QUEUE_RULES or self.impact not in IMPACT_RULES:
            raise ValueError("unknown queue or impact rule")
        if self.latency_s < 0 or not 0 <= self.hidden_fraction <= 5:
            raise ValueError("invalid latency or hidden fraction")

    def key(self) -> str:
        return f"{self.queue}|lat={self.latency_s:g}|hidden={self.hidden_fraction:g}|impact={self.impact}"


def grid(latencies=(0.0, 0.05, 0.25), hidden=(0.0, 0.5), impacts=IMPACT_RULES) -> list[Assumption]:
    return [Assumption(q, lat, h, i) for q, lat, h, i in itertools.product(QUEUE_RULES, latencies, hidden, impacts)]


@dataclass(frozen=True)
class Fees:
    taker_bps: float = 5.5
    maker_bps: float = 2.0
    penalty_bps: float = 25.0


@dataclass
class _Order:
    oid: int
    price: int
    remaining: float
    ahead: float
    placed_bin: int


@dataclass
class EpisodeResult:
    window: int
    side: int
    qty: float
    arrival_mid: float
    fills: list = field(default_factory=list)          # (time s, price ticks, qty, "maker"|"taker")
    unfilled: float = 0.0
    beyond_book: float = 0.0
    decisions: int = 0
    completion_time_s: float | None = None
    traded_volume_window: float = 0.0
    unidentified: bool = False                          # an L1-only book could not price a child


class DayBook:
    """Read access to a day cache at arbitrary times (frames at 100 ms, deep snapshots at 1 s)."""

    def __init__(self, cache: dict) -> None:
        self.c = cache
        self.tick = float(cache["tick"])
        self.deep_index = {int(s): i for i, s in enumerate(cache["deep_s"].tolist())}
        self.trade_t = cache["trade_t"]
        self.trade_bin = np.floor(cache["trade_t"] * 1000 / BIN_MS).astype(np.int64)
        self.levels = int(cache.get("levels", 200))        # 1 = L1-only source (Binance bookTicker)

    def bin_of(self, t: float) -> int:
        return int(min(max(math.floor(t * 1000 / BIN_MS), 0), len(self.c["valid"]) - 1))

    def touch(self, t: float) -> tuple[int, int, bool]:
        b = self.bin_of(t)
        return int(self.c["bid_px"][b]), int(self.c["ask_px"][b]), bool(self.c["valid"][b])

    def mid(self, t: float) -> float:
        return float(self.c["mid"][self.bin_of(t)])

    def ladder(self, t: float, side_taken: int) -> tuple[np.ndarray, np.ndarray]:
        """Absolute (prices, sizes) a market order on ``side_taken`` (+1 = lifts asks) would walk at time t."""
        b = self.bin_of(t)
        k = 1 if side_taken > 0 else 0          # asks for buys
        touch = int(self.c["ask_px"][b] if side_taken > 0 else self.c["bid_px"][b])
        step = 1 if side_taken > 0 else -1
        vis = self.c["depth"][b, k].astype(float)
        prices = [touch + step * i for i in range(len(vis))]
        sizes = list(vis)
        i = self.deep_index.get(int(math.floor(t)))
        if i is not None and self.c["deep_ok"][i]:
            dp, dq = self.c["deep_px"][i, k], self.c["deep_q"][i, k]
            edge = touch + step * (len(vis) - 1)
            beyond = (dp > edge) if side_taken > 0 else ((dp < edge) & (dp > 0))
            for p, q in zip(dp[beyond & (dq > 0)].tolist(), dq[beyond & (dq > 0)].tolist()):
                prices.append(int(p))
                sizes.append(float(q))
        order = np.argsort(prices) if side_taken > 0 else np.argsort(prices)[::-1]
        return np.asarray(prices, np.int64)[order], np.asarray(sizes, float)[order]

    def contiguous(self, t: float, side_taken: int, n: int = LADDER_TICKS) -> list:
        prices, sizes = self.ladder(t, side_taken)
        if not len(prices):
            return [0] * n
        out = np.zeros(n)
        step = 1 if side_taken > 0 else -1
        off = (prices - prices[0]) * step
        keep = off < n
        out[off[keep]] = sizes[keep]
        return out.tolist()


class Replay:
    """One hypothetical episode of ``policy`` in window ``w`` of a recorded day under ``assumption``."""

    def __init__(self, book: DayBook, policy, *, window: int, qty: float, side: int, assumption: Assumption,
                 fees: Fees = Fees(), horizon_s: float = HORIZON_S, warmup_s: float = WARMUP_S,
                 dt_s: float = DECISION_DT_S, context: dict | None = None) -> None:
        self.book, self.policy, self.a, self.fees = book, policy, assumption, fees
        self.w, self.side, self.qty = window, side, float(qty)
        self.t0 = float(WINDOW_STARTS_S[window])
        self.horizon, self.warmup, self.dt = horizon_s, warmup_s, dt_s
        self.consumed: dict[int, list[tuple[float, float]]] = {}
        self.orders: list[_Order] = []
        self.next_oid = 1
        self.filled = 0.0
        self.res = EpisodeResult(window, side, qty, arrival_mid=book.mid(self.t0))
        self.context = dict(context or {})

    # ----------------------------------------------------------------- impact bookkeeping
    def _remaining_consumed(self, price: int, t: float) -> float:
        total = 0.0
        for at, q in self.consumed.get(price, ()):
            if self.a.impact == "none":
                continue
            if self.a.impact == "persistent":
                total += q
            else:
                total += q * 0.5 ** ((t - at) / TRANSIENT_HALF_LIFE_S)
        return total

    def _market(self, qty: float, t: float) -> None:
        if qty <= 0:
            return
        prices, sizes = self.book.ladder(t, self.side)
        rem = qty
        last = None
        for p, s in zip(prices.tolist(), sizes.tolist()):
            avail = max(0.0, s - self._remaining_consumed(p, t))
            take = min(avail, rem)
            if take > 0:
                self.res.fills.append((t, p, take, "taker"))
                self.consumed.setdefault(p, []).append((t, take))
                rem -= take
                last = p
            if rem <= 1e-9:
                break
        if rem > 1e-9 and self.book.levels == 1:
            self.res.unidentified = True
        if rem > 1e-9:
            worst = last if last is not None else (int(prices[-1]) if len(prices) else
                                                   int(self.book.touch(t)[1 if self.side > 0 else 0]))
            extra = max(1, int(math.ceil(100 * rem / max(self.qty, 1e-9))))
            price = worst + (extra if self.side > 0 else -extra)
            self.res.fills.append((t, price, rem, "taker"))
            self.res.beyond_book += rem
        self.filled += qty

    def _place(self, qty: float, improve: int, t: float) -> None:
        if qty <= 0:
            return
        bid, ask, ok = self.book.touch(t)
        if not ok:
            return
        own = bid if self.side > 0 else ask
        price = own + (improve if self.side > 0 else -improve)
        if (self.side > 0 and price >= ask) or (self.side < 0 and price <= bid):
            self._market(qty, t)            # a marketable limit executes as a market order
            return
        ahead = 0.0
        if improve == 0:
            k = 0 if self.side > 0 else 1
            ahead = float(self.book.c["depth"][self.book.bin_of(t), k, 0])
            if self.a.queue == "conservative":
                ahead *= 1.0 + self.a.hidden_fraction
        self.orders.append(_Order(self.next_oid, price, float(qty), ahead, self.book.bin_of(t)))
        self.next_oid += 1

    def _cancel_all(self) -> None:
        self.orders.clear()

    def _advance_passive(self, t_from: float, t_to: float) -> None:
        """Process prints and visible decreases between t_from and t_to against resting orders."""
        if not self.orders:
            return
        c, bk = self.book.c, self.book
        lo, hi = np.searchsorted(bk.trade_t, [t_from, t_to])
        passive_sign = -1 if self.side > 0 else 1          # our buy rests on bids: hit by sell prints
        for j in range(lo, hi):
            if c["trade_s"][j] != passive_sign:
                continue
            px, q, tj = int(c["trade_px"][j]), float(c["trade_q"][j]), float(c["trade_t"][j])
            for o in list(self.orders):
                through = px < o.price if self.side > 0 else px > o.price
                at = px == o.price
                if through:
                    self._fill_passive(o, o.remaining, tj)
                elif at:
                    use = min(o.ahead, q)
                    o.ahead -= use
                    left = q - use
                    if left > 0:
                        self._fill_passive(o, min(o.remaining, left), tj)
        if self.a.queue != "conservative":
            b0, b1 = bk.bin_of(t_from), bk.bin_of(t_to)
            k = 0 if self.side > 0 else 1
            touch = c["bid_px"] if self.side > 0 else c["ask_px"]
            for o in self.orders:
                if o.ahead <= 0:
                    continue
                prev = None
                for b in range(b0, b1 + 1):
                    off = (touch[b] - o.price) if self.side > 0 else (o.price - touch[b])
                    size = float(c["depth"][b, k, off]) if 0 <= off < c["depth"].shape[2] else None
                    if prev is not None and size is not None and size < prev:
                        dec = prev - size
                        traded = float(c["volume"][b, 1 if self.side > 0 else 0])
                        cancel = max(0.0, dec - traded)
                        credit = cancel if self.a.queue == "optimistic" else cancel * o.ahead / max(prev, 1e-9)
                        o.ahead = max(0.0, o.ahead - credit)
                    prev = size if size is not None else prev
        self.orders = [o for o in self.orders if o.remaining > 1e-9]

    def _fill_passive(self, o: _Order, q: float, t: float) -> None:
        q = min(q, o.remaining, self.qty - self.filled)
        if q <= 0:
            o.remaining = 0.0
            return
        o.remaining -= q
        self.filled += q
        self.res.fills.append((t, o.price, q, "maker"))

    # ----------------------------------------------------------------- observations and the loop
    def _observe(self, t: float, step: int, steps: int, last: float) -> Observation:
        seen = max(t - self.a.latency_s, 0.0)
        bid, ask, _ = self.book.touch(seen)
        bdepth = self.book.contiguous(seen, -1)
        adepth = self.book.contiguous(seen, 1)
        b0, b1 = self.book.bin_of(last), self.book.bin_of(seen)
        vol = float(self.book.c["volume"][b0:b1].sum()) if b1 > b0 else 0.0
        mids = self.book.c["mid"][max(0, b1 - 600):b1 + 1:1].tolist()
        resting = [(o.oid, int(round(o.remaining))) for o in self.orders]
        tick_bps = 1e4 / max(self.res.arrival_mid, 1e-9)
        ctx = {**self.context, "pending_limit": 0, "in_flight": 0, "window": self.w,
               "imbalance5": float(self.book.c["state"][b1, 1]), "spread": ask - bid,
               "resting_detail": [(o.oid, o.price, o.remaining) for o in self.orders],
               "own_touch": bid if self.side > 0 else ask}
        return Observation(t=t, elapsed=t - self.t0, remaining_time=self.t0 + self.horizon - t, horizon=self.horizon,
                           step=step, steps=steps, observed_at=seen, bid=bid, ask=ask, bid_depth=bdepth,
                           ask_depth=adepth, volume_since_last=vol, recent_mids=mids,
                           target=int(round(self.qty)), side=self.side, filled=int(round(self.filled)),
                           resting=resting, tick_bps=tick_bps, context=ctx)

    def run(self) -> EpisodeResult:
        from ...v08.execution.simulator import EpisodeContext, Mandate
        steps = int(math.ceil(self.horizon / self.dt - 1e-9))
        mandate = Mandate(side=self.side, qty=max(1, int(round(self.qty))), horizon_s=self.horizon,
                          decision_dt_s=self.dt, warmup_s=self.warmup, completion="hard")
        self.policy.reset(EpisodeContext(mandate=mandate, start=self.t0,
                                         tick_bps=1e4 / max(self.res.arrival_mid, 1e-9),
                                         rng=np.random.default_rng(int(self.context.get("seed", 0))),
                                         context=self.context))
        last = self.t0 - self.dt
        t = self.t0
        for step in range(steps):
            t = self.t0 + step * self.dt
            self._advance_passive(last + self.a.latency_s, t + self.a.latency_s)
            if self.filled >= self.qty - 1e-9:
                break
            obs = self._observe(t, step, steps, last)
            actions = self.policy.decide(obs) or []
            self.res.decisions += 1
            arrive = t + self.a.latency_s
            for action in actions:
                if action[0] == "market":
                    self._market(min(float(action[1]), self.qty - self.filled - self._resting()), arrive)
                elif action[0] == "limit":
                    self._place(min(float(action[1]), self.qty - self.filled - self._resting()), int(action[2]),
                                arrive)
                elif action[0] == "cancel_all":
                    self._cancel_all()
                elif action[0] == "keep":
                    pass
            last = t
        end = self.t0 + self.horizon
        self._advance_passive(last + self.a.latency_s, end)
        self._cancel_all()
        if self.filled < self.qty - 1e-9:
            self._market(self.qty - self.filled, end)
        if self.res.fills:
            self.res.completion_time_s = max(f[0] for f in self.res.fills) - self.t0
        b0, b1 = self.book.bin_of(self.t0), self.book.bin_of(end)
        self.res.traded_volume_window = float(self.book.c["volume"][b0:b1].sum())
        return self.res

    def _resting(self) -> float:
        return sum(o.remaining for o in self.orders)


def metrics(res: EpisodeResult, book: DayBook, fees: Fees = Fees()) -> dict:
    """Registered cost endpoints of one episode (bps of the arrival mid)."""
    mid0 = res.arrival_mid
    if not res.fills or mid0 <= 0:
        return {"is_gross_bps": None, "is_net_bps": None, "completion": 0.0}
    q = np.asarray([f[2] for f in res.fills], float)
    p = np.asarray([f[1] for f in res.fills], float)
    maker = np.asarray([f[3] == "maker" for f in res.fills])
    vwap = float((p * q).sum() / q.sum())
    gross = res.side * (vwap - mid0) / mid0 * 1e4
    fee = float((q * np.where(maker, fees.maker_bps, fees.taker_bps)).sum() / q.sum())
    penalty = fees.penalty_bps * res.beyond_book / max(res.qty, 1e-9)
    mk = []
    for (t, price, qq, kind) in res.fills:
        if kind == "maker":
            later = book.mid(t + 5.0)
            mk.append((res.side * (later - price) / mid0 * 1e4, qq))
    markout = float(sum(m * w for m, w in mk) / sum(w for _, w in mk)) if mk else None
    passive_q = float(q[maker].sum())
    return {"is_gross_bps": gross, "is_net_bps": gross + fee + penalty, "fees_bps": fee, "penalty_bps": penalty,
            "completion": 1.0 if res.beyond_book <= 1e-9 else 1.0 - res.beyond_book / res.qty,
            "maker_fraction": passive_q / float(q.sum()), "markout5s_bps": markout,
            "completion_time_s": res.completion_time_s, "decisions": res.decisions,
            "participation": float(q.sum()) / max(res.traded_volume_window + q.sum(), 1e-9),
            "identified": not res.unidentified, "label": LABEL}


def window_quantity(book: DayBook, window: int, notional_usd: float, lot: float) -> float:
    mid = book.mid(float(WINDOW_STARTS_S[window])) * book.tick
    if mid <= 0:
        return 0.0
    return math.floor(notional_usd / mid / lot)


def side_of(window: int) -> int:
    return 1 if window % 2 == 0 else -1


def episode(book: DayBook, policy, *, window: int, notional_usd: float, assumption: Assumption,
            fees: Fees = Fees(), context: dict | None = None) -> dict:
    qty = window_quantity(book, window, notional_usd, float(book.c["lot"]))
    t0 = float(WINDOW_STARTS_S[window])
    valid = book.c["valid"][book.bin_of(t0 - WARMUP_S):book.bin_of(t0 + HORIZON_S)]
    if qty <= 0 or not valid.all():
        return {"window": window, "status": "NOT_EVALUABLE", "reason": "invalid book in the window" if qty > 0
                else "no arrival price"}
    res = Replay(book, policy, window=window, qty=qty, side=side_of(window), assumption=assumption, fees=fees,
                 context=context).run()
    return {"window": window, "status": "OK", "qty": qty, "side": res.side, "assumption": assumption.key(),
            **metrics(res, book, fees)}


def bounds(rows: list[dict], fields=("is_net_bps", "is_gross_bps", "markout5s_bps", "completion")) -> dict:
    """Interval of each endpoint over the assumption grid (ignoring NOT_EVALUABLE rows)."""
    out = {}
    for f in fields:
        vals = [r[f] for r in rows if r.get("status") == "OK" and r.get(f) is not None]
        out[f] = [min(vals), max(vals)] if vals else None
    return out


def assumption_dict(a: Assumption) -> dict:
    return asdict(a)
