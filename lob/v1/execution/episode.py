"""Execution episodes at every fidelity with one result schema (M60/M68 support, M110-M119 evaluation).

* ``f6_episode`` - a registered policy trades inside the order-level ecology (``lob.v1.ecology``): decisions every
  ``decision_s`` on the stale gateway view, child orders through the latency gateway, hard completion at the horizon
  (remaining quantity sent as a market order), costs from the exact fee accounting;
* ``f4_episode`` - the v0.8 execution simulator on a calibrated generator with the corrected book
  (``twins.classical.synthetic_episode``);
* ``f2_episode`` - bounded counterfactual replay of a real day window (v0.9 engine) under one assumption.

Result keys: ``is_net_bps`` (side x (VWAP incl. fees - arrival mid) / arrival mid x 1e4), ``completion`` (fraction
filled before the forced completion), ``fills``, ``fidelity``, ``status``.
"""
from __future__ import annotations

from ...v08.execution.simulator import Observation
from ..ecology.agents import default_ecology
from ..ecology.runtime import Ecology

TAKER_BPS, MAKER_BPS = 5.5, 2.0


class _F6Strategy:
    def __init__(self, policy, *, side: int, qty: int, horizon_s: float, decision_s: float, tick_bps: float):
        self.policy, self.side, self.qty = policy, side, int(qty)
        self.horizon_s, self.decision_s, self.tick_bps = horizon_s, decision_s, tick_bps
        self.steps = int(round(horizon_s / decision_s))
        self.step = 0
        self.arrival = None
        self.mids: list[float] = []
        self.policy.reset(None)
        self.forced = 0

    def __call__(self, t, view, ctx):
        if view is None or view["bid"] is None or view["ask"] is None:
            return []
        mid = 0.5 * (view["bid"] + view["ask"])
        if self.arrival is None:
            self.arrival = mid
        self.mids.append(mid)
        filled = abs(ctx.inventory)
        cmds = []
        if self.step == self.steps:                 # forced completion, phase 1: withdraw resting orders
            cmds = [("cancel", oid) for oid in list(ctx.live)]
            self.step += 1
            return cmds
        if self.step == self.steps + 1:             # phase 2 (fills and cancels settled): market the remainder
            left = self.qty - filled - sum(r for (_, _, r) in ctx.live.values())
            self.step += 1
            if left > 0:
                self.forced += left
                return [("submit", {"side": self.side, "qty": left, "kind": "market"})]
            return []
        if self.step > self.steps + 1:
            return []
        obs = Observation(t=t, elapsed=t, remaining_time=self.horizon_s - t, horizon=self.horizon_s, step=self.step,
                          steps=self.steps, observed_at=t, bid=view["bid"], ask=view["ask"],
                          bid_depth=[q for _, q in view["bids"]] + [0] * (10 - len(view["bids"])),
                          ask_depth=[q for _, q in view["asks"]] + [0] * (10 - len(view["asks"])),
                          volume_since_last=0.0, recent_mids=list(self.mids[-300:]), target=self.qty, side=self.side,
                          filled=filled, resting=[(o, r) for o, (_, _, r) in ctx.live.items()],
                          tick_bps=self.tick_bps,
                          context={"seed": 0, "resting_detail": [(o, p, r) for o, (_, p, r) in ctx.live.items()],
                                   "own_touch": view["bid"] if self.side > 0 else view["ask"]})
        resting_q = sum(r for (_, _, r) in ctx.live.values())
        for a in self.policy.decide(obs):
            if a[0] == "market" and a[1] > 0:
                cap = self.qty - filled - resting_q          # never commit beyond the target as far as it is known
                cmds.append(("submit", {"side": self.side, "qty": int(min(a[1], cap)), "kind": "market"}))
            elif a[0] == "limit" and a[1] > 0:
                touch = view["bid"] if self.side > 0 else view["ask"]
                price = touch + self.side * int(a[2]) - self.side * int(a[3])
                if self.side > 0 and price >= view["ask"]:
                    price = view["ask"] - 1
                if self.side < 0 and price <= view["bid"]:
                    price = view["bid"] + 1
                cmds.append(("submit", {"side": self.side, "qty": int(a[1]), "kind": "limit", "price": int(price)}))
            elif a[0] == "cancel_all":
                cmds += [("cancel", oid) for oid in ctx.live]
        self.step += 1
        return [c for c in cmds if c[0] != "submit" or c[1]["qty"] > 0]


def f6_episode(policy, *, side: int = 1, qty: int = 200, horizon_s: float = 300.0, decision_s: float = 5.0,
               seed: int = 0, agents=None, warmup_s: float = 30.0, tick_bps: float = 1.0) -> dict:
    eco = Ecology(agents or default_ecology(), seed=seed)
    strat = _F6Strategy(policy, side=side, qty=qty, horizon_s=horizon_s, decision_s=decision_s, tick_bps=tick_bps)
    def strategy(t, view, ctx):
        return [] if t < warmup_s else strat(t - warmup_s, view, ctx)

    out = eco.run(warmup_s + horizon_s + 3 * decision_s, strategy=strategy, decision_s=decision_s)
    fills = [r for r in eco.strategy_reports if r.kind == "fill"]
    qty_f = sum(r.qty for r in fills)
    if qty_f <= 0 or strat.arrival is None:
        return {"fidelity": "F6", "status": "NOT_EVALUABLE", "reason": "no strategy fills"}
    vwap = sum(r.qty * r.price for r in fills) / qty_f
    maker = sum(r.qty for r in fills if r.liquidity == "maker")
    fee_bps = ((qty_f - maker) * TAKER_BPS + maker * MAKER_BPS) / qty_f
    is_bps = side * (vwap - strat.arrival) / strat.arrival * 1e4 + fee_bps
    return {"fidelity": "F6", "status": "OK", "is_net_bps": float(is_bps),
            "completion": float(1.0 - strat.forced / max(qty, 1)), "filled": int(qty_f),
            "overfill": int(max(0, qty_f - qty)),
            "maker_fraction": float(maker / qty_f), "market": out["engine"],
            "label": "F6 strategic multi-agent simulation (assumed participant behaviour); never a real fill"}


def f4_episode(generator, policy, *, qty: int, seed: int, side: int = 1) -> dict:
    from ..twins import classical as TC
    r = TC.synthetic_episode(generator, policy, qty=qty, seed=seed, side=side)
    return {**r, "fidelity": "F4"}


def f2_episode(cache: dict, policy, *, window: int, notional_usd: float, assumption=None) -> dict:
    from ...v09.replay import engine as E
    book = E.DayBook(cache)
    r = E.episode(book, policy, window=window, notional_usd=notional_usd, assumption=assumption or E.Assumption("neutral"))
    return {**r, "fidelity": "F2"}


__all__ = ["f6_episode", "f4_episode", "f2_episode", "TAKER_BPS", "MAKER_BPS"]
