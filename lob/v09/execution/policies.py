"""v0.9 execution policy zoo (§20 of the brief).

The v0.8 policies (TWAP, VWAP, POV, Almgren-Chriss, adaptive TWAP/POV/VWAP, dynamic AC, v0.8 MPC) are reused
unchanged through a separate v0.9 registry (the v0.8 registry is never mutated). v0.9 adds:

* ``sweep`` - the whole mandate as one market order at the first decision (the immediate-execution control);
* ``passive_naive`` - rest the whole remaining quantity at the own touch, re-pegging every decision (the engine's
  hard completion crosses whatever is left at the horizon);
* ``passive_deadline`` - passive at the touch until a registered fraction of the horizon, then aggressive TWAP of
  the remainder (a deadline controller);
* ``imbalance_passive`` - passive while the five-level imbalance favours a fill, otherwise a TWAP-sized market child;
* ``toxicity_passive`` - passive unless the flow-risk proxy (recent signed mid drift against the order plus
  adverse imbalance) is high, in which case it steps aside (cancels) or crosses when behind schedule;
* ``liquidity_pov`` - participation scaled by visible touch liquidity relative to its running median.

Every policy sees only the observation (never the future, never exchange truth).
"""
from __future__ import annotations

import math

import numpy as np

from ...v08.execution import mpc as _v08_mpc  # noqa: F401 - registers the v0.8 MPC policies
from ...v08.execution import policies as v08p
from ...v08.execution.simulator import Observation, cancel_all, limit, market

REGISTRY: dict[str, type] = dict(v08p.REGISTRY)


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def make(name: str, **params):
    if name not in REGISTRY:
        _load_v09_extensions()
    if name not in REGISTRY:
        raise ValueError(f"unknown v0.9 policy {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


class _Base(v08p.Policy):
    family = "v09"


def rest(obs: Observation, qty: int) -> list:
    """Keep resting orders still at the own touch (queue priority kept); re-peg only when the touch moved.

    Returns the actions that leave ``qty`` resting at the current own touch. Without price detail (the v0.8
    simulator) it falls back to cancel-and-replace.
    """
    detail = obs.context.get("resting_detail")
    if detail is None:
        return ([cancel_all()] if obs.resting else []) + ([limit(qty)] if qty > 0 else [])
    touch = obs.context.get("own_touch")
    at_touch = sum(r for _, price, r in detail if price == touch)
    if detail and any(price != touch for _, price, _ in detail):
        return [cancel_all()] + ([limit(qty)] if qty > 0 else [])
    extra = int(qty - at_touch)
    return [limit(extra)] if extra > 0 else []


@register
class Sweep(_Base):
    name, label = "sweep", "Immediate sweep"

    def decide(self, obs: Observation) -> list:
        return [market(obs.remaining)] if obs.step == 0 and obs.remaining > 0 else []


@register
class PassiveNaive(_Base):
    name, label = "passive_naive", "Passive at touch (re-peg)"

    def decide(self, obs: Observation) -> list:
        return rest(obs, obs.remaining)


@register
class PassiveDeadline(_Base):
    name, label = "passive_deadline", "Passive then deadline TWAP"
    defaults = {"switch_fraction": 0.7}

    def decide(self, obs: Observation) -> list:
        left = obs.remaining
        if left <= 0:
            return [cancel_all()] if obs.resting else []
        tau = obs.step / obs.steps
        if tau < self.params["switch_fraction"]:
            return rest(obs, left)
        steps_left = max(1, obs.steps - obs.step)
        return ([cancel_all()] if obs.resting else []) + [market(int(math.ceil(left / steps_left)))]


def _schedule_gap(obs: Observation) -> float:
    """Fraction behind a linear schedule (positive = behind)."""
    return (obs.step / obs.steps) - obs.filled / max(obs.target, 1)


@register
class ImbalancePassive(_Base):
    name, label = "imbalance_passive", "Imbalance-aware passive"
    defaults = {"threshold": 0.2}

    def decide(self, obs: Observation) -> list:
        out = [cancel_all()] if obs.resting else []
        left = obs.remaining
        if left <= 0:
            return out
        imb = float(obs.context.get("imbalance5", 0.0))
        # a buyer resting on the bid fills faster when sellers press (imbalance < 0); price may then fall
        favourable = (imb < -self.params["threshold"]) if obs.side > 0 else (imb > self.params["threshold"])
        steps_left = max(1, obs.steps - obs.step)
        child = int(math.ceil(left / steps_left))
        if favourable or _schedule_gap(obs) < 0.1:
            return rest(obs, left)
        return out + [market(child), limit(max(0, left - child))]


def flow_risk(obs: Observation, lookback: int = 50) -> float:
    """Adverse-selection proxy in [-1, 1]: signed recent mid drift against the order and adverse imbalance."""
    mids = np.asarray(obs.recent_mids[-lookback:], float)
    drift = 0.0
    if len(mids) > 5 and np.std(np.diff(mids)) > 0:
        drift = float((mids[-1] - mids[0]) / (np.std(np.diff(mids)) * math.sqrt(len(mids)) + 1e-9))
    imb = float(obs.context.get("imbalance5", 0.0))
    # for a buy, risk is high when the price is falling (resting bids get run over) and sellers dominate
    signed = -obs.side * drift * 0.5 - obs.side * imb * 0.5
    return float(np.tanh(signed))


@register
class ToxicityPassive(_Base):
    name, label = "toxicity_passive", "Toxicity-proxy-aware passive"
    defaults = {"risk_threshold": 0.3, "behind_threshold": 0.15}

    def decide(self, obs: Observation) -> list:
        out = [cancel_all()] if obs.resting else []
        left = obs.remaining
        if left <= 0:
            return out
        risk = flow_risk(obs)
        steps_left = max(1, obs.steps - obs.step)
        if risk > self.params["risk_threshold"]:
            if _schedule_gap(obs) > self.params["behind_threshold"]:
                return out + [market(int(math.ceil(left / steps_left)))]
            return out                       # step aside: do not rest into adverse flow
        return rest(obs, left)


@register
class LiquidityPOV(_Base):
    name, label = "liquidity_pov", "Liquidity-aware POV"
    defaults = {"participation": 0.1, "low": 0.5, "high": 2.0}

    def reset(self, ctx):
        super().reset(ctx)
        self.touch_hist: list[float] = []

    def decide(self, obs: Observation) -> list:
        left = obs.remaining
        if left <= 0:
            return []
        touch = float(obs.touch_depth())
        self.touch_hist.append(touch)
        med = float(np.median(self.touch_hist)) if self.touch_hist else touch
        scale = min(self.params["high"], max(self.params["low"], touch / med if med > 0 else 1.0))
        child = int(self.params["participation"] * scale * max(obs.volume_since_last, 0.0))
        steps_left = max(1, obs.steps - obs.step)
        floor_ = int(math.ceil(left / steps_left)) if _schedule_gap(obs) > 0.2 else 0
        return [market(min(left, max(child, floor_)))]


def _load_v09_extensions() -> None:
    from . import mpc  # noqa: F401 - registers MPC v2 and the static optimizer


def describe_all() -> dict:
    _load_v09_extensions()
    return {name: cls().describe() for name, cls in sorted(REGISTRY.items())}
