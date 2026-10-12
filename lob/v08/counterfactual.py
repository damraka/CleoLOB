"""Counterfactual historical execution bounds on aggregate L2 (M29, brief sections 12 and 38).

Historical aggregate L2 cannot determine where a hypothetical order would have
queued or exactly when it would have filled. ``replay`` therefore runs an
unchanged policy against a recorded ``EventFrame`` under one explicit
``Assumption`` and labels every number HYPOTHETICAL; ``bounds`` runs the
registered assumption grid and reports lower/upper bounds, labelled BOUNDED and
ASSUMPTION_DEPENDENT. Nothing here is an actual historical fill.

Semantics of one replay (all registered):

* **impact**: the recorded market is otherwise unchanged by the hypothetical
  orders; liquidity the strategy takes is either immediately replenished
  (``none``), replenished with a 10 s half-life (``transient``) or never
  replenished at that price (``persistent``). The synthetic benchmark measures
  how well this bounds true impact. ``permanent_ticks_per_lot`` additionally
  shifts the prices of later market orders and of the residual valuation by
  that many ticks per lot already executed (a linear permanent-impact
  assumption; passive orders are not shifted);
* **latency**: decisions see the snapshot ``latency_s`` old; orders reach the
  book ``latency_s`` after the decision (both rounded up to whole bins);
* **market orders** walk the visible depth of the arrival snapshot; any
  quantity beyond the visible ten levels prices at the last visible level plus
  ``beyond_visible_ticks`` (recorded);
* **passive orders** rest at the own touch with ``ahead`` = visible size there
  (plus ``hidden_fraction`` of it under the conservative rule). Opposite-side
  prints at the order price consume ``ahead`` first and then fill the order; a
  print through the order price fills it completely. Visible decreases not
  explained by prints are cancellations, credited to ``ahead`` entirely
  (``optimistic``), proportionally (``neutral``) or not at all
  (``conservative``);
* **residual**: valued by walking the final visible snapshot; unpriceable when
  the visible depth is insufficient.

``synthetic_coverage`` validates the method where truth is known: a simulated
world is run twice with the same seed, once with the strategy (ground truth,
including impact) and once without (the "historical" aggregate tape the bounds
see); coverage is the fraction of true outcomes inside the bounds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math

import numpy as np

from .execution.simulator import Mandate, Observation
from .market.book import BUY
from .market.frame import BIN_S, EventFrame

QUEUE_RULES = ("optimistic", "neutral", "conservative")
IMPACT_RULES = ("none", "transient", "persistent")
TRANSIENT_HALF_LIFE_S = 10.0
LABEL = "HYPOTHETICAL"
BEYOND_VISIBLE_TICKS = 10


@dataclass(frozen=True)
class Assumption:
    queue: str = "neutral"
    latency_s: float = 0.0
    hidden_fraction: float = 0.0
    impact: str = "none"
    permanent_ticks_per_lot: float = 0.0

    def __post_init__(self) -> None:
        if self.queue not in QUEUE_RULES:
            raise ValueError(f"queue rule must be one of {QUEUE_RULES}")
        if self.impact not in IMPACT_RULES:
            raise ValueError(f"impact rule must be one of {IMPACT_RULES}")
        if self.permanent_ticks_per_lot < 0:
            raise ValueError("permanent impact must be nonnegative")
        if self.latency_s < 0 or not 0 <= self.hidden_fraction <= 5:
            raise ValueError("invalid latency or hidden fraction")

    def key(self) -> str:
        return (f"{self.queue}|lat={self.latency_s:g}|hidden={self.hidden_fraction:g}|impact={self.impact}"
                f"|perm={self.permanent_ticks_per_lot:g}")


def grid(latencies: tuple[float, float], hidden: float, impacts: tuple[str, ...] = IMPACT_RULES,
         permanent: tuple[float, ...] = (0.0,)) -> list[Assumption]:
    return [Assumption(q, lat, h, i, k) for q, lat, h, i, k in itertools.product(
        QUEUE_RULES, sorted(set(latencies)), sorted({0.0, hidden}), impacts, sorted(set(permanent)))]


class _Order:
    __slots__ = ("price", "remaining", "ahead", "cancel_row", "child")

    def __init__(self, price, remaining, ahead, child):
        self.price, self.remaining, self.ahead, self.cancel_row, self.child = price, remaining, ahead, None, child


def _level_size(frame: EventFrame, row: int, side: int, price: int) -> float | None:
    touch = frame.bid_px[row] if side == BUY else frame.ask_px[row]
    k = (touch - price) if side == BUY else (price - touch)
    if 0 <= k < frame.depth.shape[2]:
        return float(frame.depth[row, 0 if side == BUY else 1, k])
    if (side == BUY and price > touch) or (side != BUY and price < touch):
        return 0.0          # inside the recorded spread: the level is empty in history
    return None             # beyond the visible window: unobserved


def _walk(frame: EventFrame, row: int, side: int, qty: float, consumed: dict | None = None, shift: float = 0.0
          ) -> tuple[float, float, bool]:
    """(notional ticks, filled qty, used beyond-visible pricing) of a market order on snapshot ``row``.

    ``consumed`` (price -> lots) is liquidity the hypothetical strategy already took and that the impact
    assumption says has not been replenished; it is subtracted from the recorded sizes and updated.
    """
    depth = frame.depth[row, 1 if side == BUY else 0]
    touch = frame.ask_px[row] if side == BUY else frame.bid_px[row]
    step = 1 if side == BUY else -1
    rem, notional, filled = float(qty), 0.0, 0.0
    for k, size in enumerate(depth):
        price = int(touch + step * k)
        available = max(0.0, float(size) - (consumed.get(price, 0.0) if consumed is not None else 0.0))
        take = min(rem, available)
        notional += take * (price + shift)
        filled += take
        rem -= take
        if consumed is not None and take > 0:
            consumed[price] = consumed.get(price, 0.0) + take
        if rem <= 1e-12:
            return notional, filled, False
    price = touch + step * (len(depth) - 1 + BEYOND_VISIBLE_TICKS) + shift
    return notional + rem * price, filled + rem, True


def replay(frame: EventFrame, policy, mandate: Mandate, assumption: Assumption, *, start_bin: int,
           seed: int = 0, context: dict | None = None) -> dict:
    from .execution.simulator import EpisodeContext
    side = mandate.side
    lat = int(math.ceil(assumption.latency_s / BIN_S - 1e-9))
    decision_bins = max(1, int(round(mandate.decision_dt_s / BIN_S)))
    horizon_bins = int(round(mandate.horizon_s / BIN_S))
    settle_bins = int(round(mandate.settlement_s / BIN_S))
    end = start_bin + horizon_bins + settle_bins
    if end + lat >= len(frame) or start_bin < 1:
        raise ValueError("frame too short for the registered horizon, settlement and latency")
    if not frame.valid[start_bin:end].all():
        raise ValueError("invalid bins inside the replay span (episode marked INVALID, retained)")
    tt, tp, tq, ts = frame.trades()
    rows = np.floor((tt - frame.t0) / BIN_S + 1e-9).astype(int)
    arrival = 0.5 * (frame.bid_px[start_bin] + frame.ask_px[start_bin])
    rng = np.random.default_rng(seed)
    policy.reset(EpisodeContext(mandate=mandate, start=start_bin * BIN_S, tick_bps=1e4 / arrival, rng=rng,
                                context=dict(context or {})))
    fills: list[tuple[int, float, float, bool]] = []     # (row, price ticks, qty, maker)
    orders: list[_Order] = []
    pending: list[tuple[int, tuple]] = []                 # (arrival row, action)
    children = decisions = 0
    beyond = False
    last_volume_row = max(start_bin - lat, 0)
    consumed: dict | None = None if assumption.impact == "none" else {}
    decay = math.exp(-math.log(2) * BIN_S / TRANSIENT_HALF_LIFE_S)
    for b in range(start_bin, end):
        if assumption.impact == "transient" and consumed:
            for price in list(consumed):
                consumed[price] *= decay
        # ---------------------------------------------------------------- decisions
        if b < start_bin + horizon_bins and (b - start_bin) % decision_bins == 0:
            ob = b - lat
            filled_known = sum(f[2] for f in fills if f[0] + lat <= b)
            resting = [(i, o.remaining) for i, o in enumerate(orders) if o.remaining > 1e-12 and o.cancel_row is None]
            in_flight = sum(a[1] for r, a in pending if a[0] == "market")
            pending_limit = sum(a[1] for r, a in pending if a[0] == "limit")
            obs = Observation(t=b * BIN_S, elapsed=(b - start_bin) * BIN_S,
                              remaining_time=(start_bin + horizon_bins - b) * BIN_S, horizon=mandate.horizon_s,
                              step=(b - start_bin) // decision_bins, steps=mandate.steps, observed_at=ob * BIN_S,
                              bid=int(frame.bid_px[ob]), ask=int(frame.ask_px[ob]),
                              bid_depth=frame.depth[ob, 0].tolist(), ask_depth=frame.depth[ob, 1].tolist(),
                              volume_since_last=float(frame.volume[last_volume_row:ob].sum()),
                              recent_mids=frame.mid[max(0, ob - 600):ob + 1].tolist(), target=mandate.qty,
                              side=side, filled=int(round(filled_known)), resting=resting,
                              tick_bps=1e4 / arrival,
                              context={**(context or {}), "in_flight": in_flight, "pending_limit": pending_limit})
            last_volume_row = ob
            actions = list(policy.decide(obs))
            if mandate.completion == "urgency" and obs.elapsed >= mandate.urgency_fraction * mandate.horizon_s - 1e-9:
                need = obs.remaining - in_flight - sum(r for _, r in resting)
                actions = ([("cancel_all",)] if resting else []) + ([("market", need)] if need > 0 else [])
            for action in actions:
                if action[0] in ("market", "limit") and action[1] <= 0:
                    continue
                if action[0] in ("market", "limit"):
                    children += 1
                pending.append((b + lat, action))
            decisions += 1
        if b == start_bin + horizon_bins:
            pending.append((b + lat, ("cancel_all",)))
            if mandate.completion == "hard":
                pending.append((b + 2 * lat, ("complete",)))
        # ---------------------------------------------------------------- arrivals at this row
        for r, action in [p for p in pending if p[0] == b]:
            pending.remove((r, action))
            if action[0] == "market":
                shift = side * assumption.permanent_ticks_per_lot * sum(f[2] for f in fills)
                notional, got, over = _walk(frame, b, side, action[1], consumed, shift)
                beyond |= over
                if got > 0:
                    fills.append((b, notional / got, got, False))
            elif action[0] == "limit":
                price = int(frame.bid_px[b] if side == BUY else frame.ask_px[b])
                visible = _level_size(frame, b, side, price) or 0.0
                ahead = visible * (1 + assumption.hidden_fraction) if assumption.queue == "conservative" else visible
                orders.append(_Order(price, float(action[1]), ahead, children))
            elif action[0] == "cancel_all":
                for o in orders:
                    if o.cancel_row is None:
                        o.cancel_row = b
            elif action[0] == "complete":
                done = sum(f[2] for f in fills)
                need = mandate.qty - done
                if need > 1e-9:
                    shift = side * assumption.permanent_ticks_per_lot * done
                    notional, got, over = _walk(frame, b, side, need, consumed, shift)
                    beyond |= over
                    if got > 0:
                        fills.append((b, notional / got, got, False))
                    children += 1
        # ---------------------------------------------------------------- passive fills during this bin
        lo, hi = np.searchsorted(rows, b), np.searchsorted(rows, b, side="right")
        for o in orders:
            if o.remaining <= 1e-12 or (o.cancel_row is not None and o.cancel_row <= b):
                continue
            opposite = -1 if side == BUY else 1                  # aggressor sign that can fill us
            at_price = through = 0.0
            for j in range(lo, hi):
                if ts[j] != opposite:
                    continue
                p = tp[j]
                if (side == BUY and p < o.price - 1e-9) or (side != BUY and p > o.price + 1e-9):
                    through += tq[j]
                elif abs(p - o.price) <= 1e-9:
                    at_price += tq[j]
            if through > 0:
                fills.append((b, float(o.price), o.remaining, True))
                o.remaining = 0.0
                continue
            printed = at_price
            if printed > o.ahead:
                got = min(o.remaining, printed - o.ahead)
                fills.append((b, float(o.price), got, True))
                o.remaining -= got
                o.ahead = 0.0
            else:
                o.ahead -= printed
            before = _level_size(frame, b, side, o.price)
            after = _level_size(frame, b + 1, side, o.price)
            if before is not None and after is not None:
                cancels = max(0.0, before - after - at_price)
                if assumption.queue == "optimistic":
                    o.ahead = max(0.0, o.ahead - cancels)
                elif assumption.queue == "neutral" and before > 0:
                    o.ahead = max(0.0, o.ahead - cancels * min(1.0, o.ahead / before))
    # ---------------------------------------------------------------- accounting
    q = mandate.qty
    filled = sum(f[2] for f in fills)
    notional = sum(f[1] * f[2] for f in fills)
    maker = sum(f[1] * f[2] for f in fills if f[3])
    bps = 1e4 / (arrival * q)
    fee_bps = ((notional - maker) * mandate.fees.taker_bps + maker * max(mandate.fees.maker_bps, 0.0)
               - maker * max(-mandate.fees.maker_bps, 0.0)) * 1e-4 * bps + mandate.fees.per_lot_ticks * filled * bps
    gross = side * (notional - arrival * filled) * bps
    residual = max(0.0, q - filled)
    hypo, unpriceable = 0.0, False
    if residual > 1e-9:
        walk_notional, got, over = _walk(frame, end, side, residual, dict(consumed) if consumed is not None else None,
                                         side * assumption.permanent_ticks_per_lot * filled)
        if over:
            hypo, unpriceable = None, True
        else:
            hypo = (side * (walk_notional - arrival * residual) + walk_notional * mandate.fees.taker_bps * 1e-4) * bps
    penalty = mandate.penalty_bps * residual / q
    economic = gross + fee_bps
    within = sum(f[2] for f in fills if f[0] < start_bin + horizon_bins)
    return {"label": LABEL, "assumption": asdict(assumption), "actual_filled_qty": None,
            "hypothetical_filled_qty": float(filled), "fill_frac": float(filled / q),
            "completion": bool(filled >= q - 1e-9), "within_horizon_completion": bool(within >= q - 1e-9),
            "terminal_inventory": float(residual), "gross_cost_bps": float(gross), "fee_bps": float(fee_bps),
            "economic_cost_bps": float(economic),
            "hypothetical_residual_cost_bps": None if hypo is None else float(hypo),
            "residual_unpriceable": bool(unpriceable), "completion_penalty_bps": float(penalty),
            "optimization_cost_bps": None if hypo is None else float(economic + hypo + penalty),
            "children": int(children), "decision_count": int(decisions), "beyond_visible_pricing": bool(beyond),
            "maker_filled_qty": float(sum(f[2] for f in fills if f[3]))}


BOUND_FIELDS = ("economic_cost_bps", "optimization_cost_bps", "fill_frac", "completion")


def bounds(frame: EventFrame, policy_factory, mandate: Mandate, assumptions: list[Assumption], *, start_bin: int,
           seed: int = 0) -> dict:
    """Replay under every assumption; per-field [min, max] with the assumption attaining each bound."""
    rows = {}
    for a in assumptions:
        try:
            rows[a.key()] = replay(frame, policy_factory(), mandate, a, start_bin=start_bin, seed=seed)
        except ValueError as exc:
            return {"status": "INVALID", "reason": str(exc), "label": "BOUNDED"}
    out = {"status": "VALID", "label": "BOUNDED", "assumption_dependent": True, "rows": rows}
    for f in BOUND_FIELDS:
        values = {k: r[f] for k, r in rows.items() if r[f] is not None}
        if not values:
            out[f] = {"lower": None, "upper": None, "status": "NOT_EVALUABLE"}
            continue
        lo_k, hi_k = min(values, key=values.get), max(values, key=values.get)
        out[f] = {"lower": float(values[lo_k]), "upper": float(values[hi_k]), "at_lower": lo_k, "at_upper": hi_k,
                  "width": float(values[hi_k]) - float(values[lo_k])}
    return out


def pair_bound(bounds_a: list[dict], bounds_b: list[dict], *, field: str = "economic_cost_bps",
               alpha: float = 0.05, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bound on the mean difference A - B over episodes, with assumptions shared between A and B.

    For every assumption the mean difference over episodes is computed; the bound is [min, max] over
    assumptions, widened by bootstrap sampling uncertainty at each extreme. Determinate only if the whole
    widened interval excludes zero; otherwise INCONCLUSIVE.
    """
    keys = sorted(set.intersection(*[set(b["rows"]) for b in bounds_a + bounds_b if b.get("status") == "VALID"]))
    valid = [(a, b) for a, b in zip(bounds_a, bounds_b) if a.get("status") == "VALID" and b.get("status") == "VALID"]
    if len(valid) < 3 or not keys:
        return {"status": "NOT_EVALUABLE", "reason": "fewer than three valid paired episodes", "n": len(valid)}
    rng = np.random.default_rng(seed)
    per_key = {}
    for k in keys:
        d = np.asarray([a["rows"][k][field] - b["rows"][k][field] for a, b in valid
                        if a["rows"][k][field] is not None and b["rows"][k][field] is not None], float)
        if len(d) < 3:
            continue
        boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
        per_key[k] = {"mean": float(d.mean()), "low": float(np.quantile(boots, alpha / 2)),
                      "high": float(np.quantile(boots, 1 - alpha / 2)), "n": len(d)}
    if not per_key:
        return {"status": "NOT_EVALUABLE", "reason": "no priceable paired values"}
    lower = min(v["low"] for v in per_key.values())
    upper = max(v["high"] for v in per_key.values())
    point_lo = min(v["mean"] for v in per_key.values())
    point_hi = max(v["mean"] for v in per_key.values())
    status = "A_LOWER" if upper < 0 else "B_LOWER" if lower > 0 else "INCONCLUSIVE"
    return {"status": status, "field": field, "lower": lower, "upper": upper, "assumption_span": [point_lo, point_hi],
            "n": len(valid), "assumptions": per_key, "label": "BOUNDED, ASSUMPTION_DEPENDENT"}


def synthetic_coverage(world, policy_factory, mandate: Mandate, seeds: list[int], assumptions: list[Assumption],
                       *, latency_profile=None) -> dict:
    """True (simulated, with impact) outcomes versus bounds computed on the strategy-free twin tape."""
    from .execution.simulator import Episode
    from .market.sim import MarketSim
    rows = []
    warm = int(round(mandate.warmup_s / BIN_S))
    total = warm + int(round((mandate.horizon_s + mandate.settlement_s) / BIN_S)) + 40
    for seed in seeds:
        truth = Episode(world, policy_factory(), mandate, seed, latency=latency_profile).run()
        sim = MarketSim(world.generator, world.book, seed)
        sim.run(total)
        frame = sim.frame()
        b = bounds(frame, policy_factory, mandate, assumptions, start_bin=warm, seed=seed)
        if b["status"] != "VALID":
            rows.append({"seed": seed, "status": "INVALID", "reason": b.get("reason")})
            continue
        row = {"seed": seed, "status": "VALID"}
        truth_map = {"economic_cost_bps": truth["economic_cost_bps"], "fill_frac": truth["fill_frac"],
                     "completion": float(truth["actual_completion"]),
                     "optimization_cost_bps": truth["optimization_cost_bps"]}
        for f, true in truth_map.items():
            lo, hi = b[f]["lower"], b[f]["upper"]
            if lo is None or true is None:
                row[f] = {"covered": None}
                continue
            row[f] = {"truth": float(true), "lower": float(lo), "upper": float(hi),
                      "covered": bool(lo - 1e-9 <= float(true) <= hi + 1e-9), "width": float(hi - lo)}
        rows.append(row)
    summary = {}
    for f in ("economic_cost_bps", "fill_frac", "completion", "optimization_cost_bps"):
        cov = [r[f]["covered"] for r in rows if r["status"] == "VALID" and r[f].get("covered") is not None]
        width = [r[f]["width"] for r in rows if r["status"] == "VALID" and r[f].get("covered") is not None]
        summary[f] = {"coverage": float(np.mean(cov)) if cov else None, "n": len(cov),
                      "mean_width": float(np.mean(width)) if width else None}
    return {"rows": rows, "summary": summary, "label": "SYNTHETIC validation of the bounding method"}
