"""v0.8 execution episodes with separated exchange-truth, observed and decision state (M22, M30, section 11).

Three states are kept apart:

* **exchange truth** — the live ``AggBook`` inside ``MarketSim``;
* **observed market data** — bin-end snapshots; a strategy decides on the most
  recent snapshot older than the feed (dissemination) latency, never on the
  live book when a feed delay is configured;
* **strategy decision state** — the strategy's knowledge of its own fills and
  orders, updated only after the acknowledgement latency.

Messages leave the strategy after the decision latency and reach the exchange
after the send latency (cancels use the cancel latency). They are merged in
time order with background events inside a bin, so cancels can lose races to
fills and market orders can execute against a book that moved after the
observation (stale-book execution). Latencies are simulator settings; none is a
measurement of a real venue.

Completion modes: ``none`` (the policy alone), ``urgency`` (catch-up with
marketable orders after ``urgency_fraction`` of the horizon) and ``hard``
(cancel resting orders at the horizon and, once each cancel is confirmed, send a
market order for the remainder within the settlement window). Every episode
reports actual fills, terminal inventory, hypothetical residual liquidation,
completion penalty, economic cost and optimization objective as separate
fields; none is folded into another.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import heapq
import math

import numpy as np

from ...v07.exchange.latency import LatencyModel
from ..market.book import BUY, SELL
from ..market.frame import BIN_S, MARKS
from ..market.sim import MarketSim

COMPLETION_MODES = ("none", "urgency", "hard")
ZERO = LatencyModel("fixed", 0.0, 0.0)
STREAM_LATENCY = 8_008_101
STREAM_POLICY = 8_008_201


@dataclass(frozen=True)
class Fees:
    taker_bps: float = 1.0
    maker_bps: float = -0.25          # negative = rebate (reported as rebate, never netted silently)
    per_lot_ticks: float = 0.0        # fixed charge per executed lot, in ticks of price

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Mandate:
    side: int = BUY
    qty: int = 40
    horizon_s: float = 120.0
    decision_dt_s: float = 5.0
    warmup_s: float = 30.0
    settlement_s: float = 5.0
    completion: str = "none"
    urgency_fraction: float = 0.8
    penalty_bps: float = 25.0
    participation_cap: float | None = None
    fees: Fees = field(default_factory=Fees)

    def __post_init__(self) -> None:
        if self.side not in (BUY, SELL) or self.qty <= 0:
            raise ValueError("mandate side must be +1/-1 and qty positive")
        if self.completion not in COMPLETION_MODES:
            raise ValueError(f"completion must be one of {COMPLETION_MODES}")
        if not 0 < self.decision_dt_s <= self.horizon_s or self.warmup_s < 0 or self.settlement_s < 0:
            raise ValueError("invalid mandate timing")
        if self.participation_cap is not None and not 0 < self.participation_cap <= 1:
            raise ValueError("participation_cap must lie in (0, 1]")

    @property
    def steps(self) -> int:
        return max(1, int(math.ceil(self.horizon_s / self.decision_dt_s - 1e-9)))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fees"] = self.fees.to_dict()
        return d

    def replace(self, **changes) -> Mandate:
        d = {**asdict(self), **changes}
        d["fees"] = changes.get("fees", self.fees)
        return Mandate(**d)


@dataclass(frozen=True)
class LatencyProfile:
    feed: LatencyModel = ZERO
    decision: LatencyModel = ZERO
    send: LatencyModel = ZERO
    ack: LatencyModel = ZERO
    cancel: LatencyModel = ZERO

    def to_dict(self) -> dict:
        return {k: asdict(getattr(self, k)) for k in ("feed", "decision", "send", "ack", "cancel")}

    @classmethod
    def from_dict(cls, d: dict) -> LatencyProfile:
        return cls(**{k: LatencyModel(**v) for k, v in d.items()})

    @classmethod
    def uniform(cls, seconds: float) -> LatencyProfile:
        m = LatencyModel("fixed", float(seconds), 0.0)
        return cls(feed=m, decision=ZERO, send=m, ack=m, cancel=m)


@dataclass
class Observation:
    """What a policy may see at a decision (observed market data + strategy decision state)."""

    t: float
    elapsed: float
    remaining_time: float
    horizon: float
    step: int
    steps: int
    observed_at: float
    bid: int
    ask: int
    bid_depth: list
    ask_depth: list
    volume_since_last: float          # visible traded volume (both sides) since the previous decision
    recent_mids: list                 # visible bin-end mids, oldest first
    target: int
    side: int
    filled: int                       # acknowledged fills
    resting: list                     # acknowledged resting orders: (order id, remaining known)
    tick_bps: float
    context: dict = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return max(0, self.target - self.filled)

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> int:
        return self.ask - self.bid

    def touch_depth(self) -> int:
        """Depth at the touch the strategy would take from."""
        return int(self.ask_depth[0] if self.side == BUY else self.bid_depth[0])

    def walk(self, qty: int) -> tuple[float, int]:
        """Notional (ticks) and fillable quantity of a market order on the *observed* snapshot."""
        depth, base, step = (self.ask_depth, self.ask, 1) if self.side == BUY else (self.bid_depth, self.bid, -1)
        rem, notional, got = qty, 0.0, 0
        for i, size in enumerate(depth):
            take = min(rem, int(size))
            notional += take * (base + step * i)
            got += take
            rem -= take
            if rem <= 0:
                break
        return notional, got


# Actions a policy may return
def market(qty: int) -> tuple:
    return ("market", int(qty))


def limit(qty: int, *, improve: int = 0, level: int = 0) -> tuple:
    return ("limit", int(qty), int(improve), int(level))


def cancel_all() -> tuple:
    return ("cancel_all",)


class Episode:
    """One execution episode in one world for one seed.

    Strategy knowledge per child order: ``sent_t``; ``exec_t`` (exchange time it arrived); ``ack_t`` (when the
    strategy learns it arrived); fills carry their own acknowledgement time; ``cancel_sent``/``cancel_ack_t``.
    """

    def __init__(self, world, policy, mandate: Mandate, seed: int, *, latency: LatencyProfile | None = None,
                 context: dict | None = None, record_tape: bool = False) -> None:
        self.world, self.policy, self.mandate, self.seed = world, policy, mandate, int(seed)
        self.latency = latency or getattr(world, "latency", None) or LatencyProfile()
        self.context = dict(context or {})
        self.sim = MarketSim(world.generator, world.book, seed, record=record_tape)
        self.lat_rng = np.random.default_rng(np.random.SeedSequence([self.seed, STREAM_LATENCY]))
        self.policy_rng = np.random.default_rng(np.random.SeedSequence([self.seed, STREAM_POLICY]))
        self._pending: list = []
        self._seq = 0
        self.fills: list[tuple] = []          # (exchange t, price, qty, maker, ack t, child id)
        self.orders: dict[int, dict] = {}
        self.children = 0
        self.decisions = 0
        self.cancel_races_lost = 0
        self.snapshots: list[tuple] = []
        self.latency_slippage = 0.0
        self._fill_cursor: dict[int, int] = {}
        self._next_coid = 1
        self._volume_total = 0.0
        self._own_bin = np.zeros(len(MARKS), dtype=np.int64)

    # ------------------------------------------------------------------ messaging
    def _push(self, at: float, kind: str, payload) -> None:
        heapq.heappush(self._pending, (at, self._seq, kind, payload))
        self._seq += 1

    def _send(self, now: float, action: tuple, observed_touch: int | None) -> None:
        kind = action[0]
        leave = now + self.latency.decision.draw(self.lat_rng)
        if kind in ("cancel_all", "cancel"):
            targets = [c for c, o in self.orders.items() if o["kind"] == "limit" and not o["cancel_sent"]
                       and (kind == "cancel_all" or c == action[1])]
            for coid in targets:
                self.orders[coid]["cancel_sent"] = True
                self._push(leave + self.latency.cancel.draw(self.lat_rng), "cancel", coid)
            return
        qty = int(action[1])
        if qty <= 0:
            return
        coid = self._next_coid
        self._next_coid += 1
        self.children += 1
        self.orders[coid] = {"kind": kind, "qty": qty, "sent_t": now, "exec_t": None, "ack_t": None, "oid": None,
                             "cancel_sent": False, "cancel_ack_t": None, "observed_touch": observed_touch}
        self._push(leave + self.latency.send.draw(self.lat_rng), kind, (coid, action))

    def _ack(self, t: float) -> float:
        return t + self.latency.ack.draw(self.lat_rng)

    def _execute(self, kind: str, payload) -> None:
        book, side, t = self.sim.book, self.mandate.side, self.sim.t
        if kind == "market":
            coid, action = payload
            filled, notional, fills = book.market(side, action[1], t, strategy=True)
            self._own_bin[0 if side == BUY else 1] += 1
            info = self.orders[coid]
            ack = self._ack(t)
            for price, qty in fills:
                self.fills.append((t, price, qty, False, ack, coid))
            if info["observed_touch"] is not None and filled:
                self.latency_slippage += side * (notional - info["observed_touch"] * filled)
            info.update(exec_t=t, ack_t=ack)
        elif kind == "limit":
            coid, action = payload
            _, qty, improve, level = action
            info = self.orders[coid]
            oid = book.submit_limit(side, qty, t, improve=improve, level=level)
            self._own_bin[2 if side == BUY else 3] += 1
            info.update(exec_t=t, ack_t=self._ack(t), oid=oid)
            self._fill_cursor[oid] = 0
        elif kind == "cancel":
            info = self.orders[payload]
            if info["oid"] is None:
                self._push(t + 1e-6, "cancel", payload)     # cancel overtook its order: retry when it lands
                return
            if book.cancel_own(info["oid"]) == 0:
                self.cancel_races_lost += 1                  # already filled: the fill stands
            else:
                self._own_bin[4 if side == BUY else 5] += 1
            info["cancel_ack_t"] = self._ack(t)
        elif kind == "complete":
            # strategy-side event: cancels are confirmed; buy what the strategy knows is still missing
            known = sum(f[2] for f in self.fills if f[4] <= t)
            self._send(t, market(self.mandate.qty - known - self._in_flight(t)), None)

    def _in_flight(self, now: float) -> int:
        return sum(o["qty"] for o in self.orders.values()
                   if o["kind"] == "market" and (o["ack_t"] is None or o["ack_t"] > now))

    def _collect_passive(self) -> None:
        book = self.sim.book
        for coid, info in self.orders.items():
            oid = info["oid"]
            if oid is None or info["kind"] != "limit":
                continue
            order = book.own[oid]
            start = self._fill_cursor[oid]
            for t, price, qty in order.fills[start:]:
                self.fills.append((t, price, qty, True, self._ack(t), coid))
            self._fill_cursor[oid] = len(order.fills)

    # ------------------------------------------------------------------ observation
    def _snapshot(self) -> None:
        b = self.sim.book
        self.snapshots.append((self.sim.t, b.bid_px, b.ask_px, list(b.bq[:10]), list(b.aq[:10]), self._volume_total,
                               b.mid()))

    def _observe(self, now: float, t0: float, step: int, last_obs_vol: float) -> Observation:
        delay = self.latency.feed.draw(self.lat_rng)
        visible_until = now - delay
        index = len(self.snapshots) - 1
        while index > 0 and self.snapshots[index][0] > visible_until + 1e-12:
            index -= 1
        snap = self.snapshots[index]
        known = [f for f in self.fills if f[4] <= now]
        by_child: dict[int, int] = {}
        for f in known:
            by_child[f[5]] = by_child.get(f[5], 0) + f[2]
        resting = []
        for coid, o in self.orders.items():
            if o["kind"] != "limit" or o["ack_t"] is None or o["ack_t"] > now:
                continue
            if o["cancel_ack_t"] is not None and o["cancel_ack_t"] <= now:
                continue
            left = o["qty"] - by_child.get(coid, 0)
            if left > 0:
                resting.append((coid, left))
        mids = [s[6] for s in self.snapshots[max(0, index - 600):index + 1]]
        ctx = dict(self.context)
        ctx["in_flight"] = self._in_flight(now)
        ctx["pending_limit"] = sum(o["qty"] for o in self.orders.values()
                                   if o["kind"] == "limit" and (o["ack_t"] is None or o["ack_t"] > now))
        return Observation(t=now, elapsed=now - t0, remaining_time=max(0.0, t0 + self.mandate.horizon_s - now),
                           horizon=self.mandate.horizon_s, step=step, steps=self.mandate.steps, observed_at=snap[0],
                           bid=snap[1], ask=snap[2], bid_depth=snap[3], ask_depth=snap[4],
                           volume_since_last=max(0.0, snap[5] - last_obs_vol), recent_mids=mids,
                           target=self.mandate.qty, side=self.mandate.side, filled=sum(f[2] for f in known),
                           resting=resting, tick_bps=1e4 / self.world.book.mid_ticks, context=ctx)

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        m, sim = self.mandate, self.sim
        self._snapshot()
        for _ in range(int(round(m.warmup_s / BIN_S))):
            self._advance_bin()
        t0 = sim.t
        arrival = sim.book.mid()
        tape_start = len(sim.book.tape)
        self.policy.reset(EpisodeContext(mandate=m, start=t0, tick_bps=1e4 / self.world.book.mid_ticks,
                                         rng=self.policy_rng, context=self.context))
        decision_bins = max(1, int(round(m.decision_dt_s / BIN_S)))
        horizon_bins = int(round(m.horizon_s / BIN_S))
        settle_bins = int(round(m.settlement_s / BIN_S))
        last_obs_vol = self.snapshots[-1][5]
        horizon_end = t0 + m.horizon_s
        within_filled = None
        for b in range(horizon_bins + settle_bins):
            now = sim.t
            if b < horizon_bins and b % decision_bins == 0:
                obs = self._observe(now, t0, b // decision_bins, last_obs_vol)
                last_obs_vol = obs.volume_since_last + last_obs_vol
                touch = obs.ask if m.side == BUY else obs.bid
                for action in self._decide(obs):
                    self._send(now, action, touch if action[0] == "market" else None)
                self.decisions += 1
            if b == horizon_bins:
                within_filled = sum(f[2] for f in self.fills if f[0] <= horizon_end + 1e-12)
                self._send(now, cancel_all(), None)
                if m.completion == "hard":
                    cancels = [at for at, _, kind, _ in self._pending if kind == "cancel"]
                    confirm = (max(cancels) if cancels else now) + self.latency.ack.draw(self.lat_rng)
                    self._push(confirm, "complete", None)
            self._advance_bin()
        if within_filled is None:
            within_filled = sum(f[2] for f in self.fills if f[0] <= horizon_end + 1e-12)
        return self._report(arrival, t0, tape_start, within_filled)

    def _decide(self, obs: Observation) -> list:
        m = self.mandate
        in_flight = obs.context.get("in_flight", 0) + obs.context.get("pending_limit", 0)
        if m.completion == "urgency" and obs.elapsed >= m.urgency_fraction * m.horizon_s - 1e-9:
            resting = sum(r for _, r in obs.resting)
            actions = [cancel_all()] if obs.resting else []
            need = obs.remaining - in_flight - resting
            if need > 0:
                actions.append(market(need))
            return actions
        actions = list(self.policy.decide(obs))
        if m.participation_cap is not None:
            cap = max(1, int(m.participation_cap * max(obs.volume_since_last, 1.0)))
            actions = [("market", min(a[1], cap)) if a[0] == "market" else a for a in actions]
        return actions

    def _advance_bin(self) -> None:
        sim = self.sim
        end = sim.t + BIN_S
        schedule = []
        while self._pending and self._pending[0][0] < end - 1e-12:
            at, seq, kind, payload = heapq.heappop(self._pending)
            schedule.append((max(at, sim.t), seq, lambda k=kind, p=payload: self._execute(k, p)))
        tape_start = len(sim.book.tape)
        self._own_bin = np.zeros(len(MARKS), dtype=np.int64)
        sim.run_bin(schedule, own_counts=self._own_bin)
        self._volume_total += sum(x[2] for x in sim.book.tape[tape_start:])
        self._collect_passive()
        self._snapshot()

    # ------------------------------------------------------------------ report
    def _report(self, arrival: float, t0: float, tape_start: int, within_filled: int) -> dict:
        m, book = self.mandate, self.sim.book
        side, q = m.side, m.qty
        fills = self.fills
        filled = sum(f[2] for f in fills)
        notional = sum(f[1] * f[2] for f in fills)
        maker_notional = sum(f[1] * f[2] for f in fills if f[3])
        taker_notional = notional - maker_notional
        bps = 1e4 / (arrival * q)
        gross = side * (notional - arrival * filled) * bps
        fees_paid = (taker_notional * m.fees.taker_bps + maker_notional * max(m.fees.maker_bps, 0.0)) * 1e-4
        rebates = maker_notional * max(-m.fees.maker_bps, 0.0) * 1e-4
        fee_bps = (fees_paid - rebates + m.fees.per_lot_ticks * filled) * bps
        residual = max(0, q - filled)
        hypo, unpriceable = 0.0, False
        if residual > 0:
            walk_notional, fillable = book.walk_cost(side, residual)
            if fillable < residual:
                hypo, unpriceable = None, True
            else:
                hypo = (side * (walk_notional - arrival * residual) + walk_notional * m.fees.taker_bps * 1e-4) * bps
        penalty = m.penalty_bps * residual / q
        economic = gross + fee_bps
        horizon_end = t0 + m.horizon_s
        tape = book.tape[tape_start:]
        market_volume = sum(x[2] for x in tape if x[0] <= horizon_end)
        own_taken = sum(x[2] for x in tape if x[4] and x[0] <= horizon_end)
        own_passive = sum(f[2] for f in fills if f[3] and f[0] <= horizon_end)
        return {
            "actual_filled_qty": int(filled),
            "fill_frac": filled / q,
            "actual_completion": bool(filled >= q),
            "within_horizon_filled_qty": int(within_filled),
            "within_horizon_completion": bool(within_filled >= q),
            "post_horizon_filled_qty": int(filled - within_filled),
            "terminal_inventory": int(residual),
            "outstanding_qty": int(sum(o.remaining for o in book.resting())),
            "overfill_qty": int(max(0, filled - q)),
            "implementation_shortfall_bps": float(side * (notional / filled - arrival) / arrival * 1e4) if filled
            else None,
            "gross_cost_bps": float(gross),
            "fee_bps": float(fee_bps),
            "rebate_bps": float(rebates * bps),
            "economic_cost_bps": float(economic),
            "hypothetical_residual_cost_bps": None if hypo is None else float(hypo),
            "residual_unpriceable": bool(unpriceable),
            "completion_penalty_bps": float(penalty),
            "optimization_cost_bps": None if hypo is None else float(economic + hypo + penalty),
            "participation": float((own_taken + own_passive) / market_volume) if market_volume else 0.0,
            "children": int(self.children),
            "decision_count": int(self.decisions),
            "latency_cost_proxy_bps": float(self.latency_slippage * bps),
            "cancel_races_lost": int(self.cancel_races_lost),
            "maker_filled_qty": int(sum(f[2] for f in fills if f[3])),
            "taker_filled_qty": int(sum(f[2] for f in fills if not f[3])),
            "arrival_mid_ticks": float(arrival),
        }


@dataclass
class EpisodeContext:
    mandate: Mandate
    start: float
    tick_bps: float
    rng: np.random.Generator
    context: dict = field(default_factory=dict)


SEPARATE_FIELDS = ("actual_filled_qty", "fill_frac", "actual_completion", "terminal_inventory", "outstanding_qty",
                   "implementation_shortfall_bps", "gross_cost_bps", "fee_bps", "hypothetical_residual_cost_bps",
                   "completion_penalty_bps", "optimization_cost_bps", "participation", "children", "decision_count",
                   "latency_cost_proxy_bps")


def run_episode(world, policy, mandate: Mandate, seed: int, *, latency: LatencyProfile | None = None,
                context: dict | None = None) -> dict:
    """One episode row; engineering exceptions become INVALID rows (retained, never dropped)."""
    from ..market.book import BookError
    try:
        row = Episode(world, policy, mandate, seed, latency=latency, context=context).run()
        row["status"] = "VALID"
    except (BookError, ValueError, ZeroDivisionError, OverflowError) as exc:
        row = {"status": "INVALID", "error": f"{type(exc).__name__}: {exc}"}
    row.update({"policy": policy.name, "seed": int(seed), "world_id": getattr(world, "world_id", None),
                "generator_family": getattr(world.generator, "name", None),
                "posterior_sample_id": getattr(world, "posterior_sample_id", None),
                "regime_id": getattr(world, "regime_id", None)})
    for key in ("ood_score", "abstention", "fallback_used"):
        if context and key in context:
            row[key] = context[key]
    return row
