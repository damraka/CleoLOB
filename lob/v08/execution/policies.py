"""Typed execution-policy plugin interface, registered baselines and adaptive variants (section 10, M25, M26).

Each policy declares ``PolicyRequirements``: observations it reads, market
capabilities it needs, whether it needs order identity (queue position), whether
it supports delayed observations, whether it needs a model posterior, its
completion behaviour and whether it is stochastic. ``check_capabilities`` fails
closed: a policy that needs a capability the dataset or world does not provide
is refused, never silently degraded.

Baselines (``BASELINES``) are fixed rules with declared constants. The v0.5
originals in ``lob.execution`` are untouched; these are v0.8 implementations on
the v0.8 simulator with their own labels. Adaptive variants (``ADAPTIVE``) are
separate classes with separate labels and configs. Parameters are never re-fitted
per world (a world comparison must not become a tuning comparison).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

import numpy as np

from .simulator import Observation, cancel_all, limit, market

OBSERVATIONS = ("touch", "depth", "volume", "mids", "own_fills", "regime", "ood_score", "posterior")
MARKET_CAPABILITIES = ("aggregate_depth", "trade_prints", "order_identity", "queue_position", "exchange_timestamps",
                       "receive_timestamps")
COMPLETION_BEHAVIOURS = ("schedule", "catch_up", "best_effort", "constrained")


class CapabilityError(RuntimeError):
    """A policy needs a capability the data or world does not provide (fail closed)."""


@dataclass(frozen=True)
class PolicyRequirements:
    observations: frozenset = frozenset({"touch", "depth", "volume", "own_fills"})
    market_capabilities: frozenset = frozenset({"aggregate_depth"})
    queue_identity: bool = False
    delayed_observations: bool = True
    model_posterior: bool = False
    completion_behaviour: str = "schedule"
    stochastic: bool = False

    def __post_init__(self) -> None:
        if not set(self.observations) <= set(OBSERVATIONS):
            raise ValueError(f"unknown observations {sorted(set(self.observations) - set(OBSERVATIONS))}")
        if not set(self.market_capabilities) <= set(MARKET_CAPABILITIES):
            raise ValueError("unknown market capabilities")
        if self.completion_behaviour not in COMPLETION_BEHAVIOURS:
            raise ValueError(f"completion_behaviour must be one of {COMPLETION_BEHAVIOURS}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["observations"] = sorted(self.observations)
        d["market_capabilities"] = sorted(self.market_capabilities)
        return d


def check_capabilities(policy: Policy, available: set[str] | frozenset, *, delayed: bool = False,
                       context: set[str] | frozenset = frozenset()) -> None:
    req = policy.requirements
    missing = sorted(set(req.market_capabilities) - set(available))
    if req.queue_identity and "order_identity" not in available:
        missing.append("order_identity (queue identity)")
    if missing:
        raise CapabilityError(f"{policy.name}: missing market capabilities {missing}")
    if delayed and not req.delayed_observations:
        raise CapabilityError(f"{policy.name}: does not support delayed observations")
    if req.model_posterior and "posterior" not in context:
        raise CapabilityError(f"{policy.name}: requires a model posterior in its context")
    needs = {"regime", "ood_score"} & set(req.observations)
    if needs - set(context):
        raise CapabilityError(f"{policy.name}: requires context {sorted(needs - set(context))}")


REGISTRY: dict[str, type] = {}


def register(cls):
    if cls.name in REGISTRY:
        raise ValueError(f"duplicate policy {cls.name}")
    REGISTRY[cls.name] = cls
    return cls


def make(name: str, **params) -> Policy:
    if name not in REGISTRY:
        raise KeyError(f"unknown policy {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


class Policy:
    name = "base"
    label = "base"
    family = "baseline"
    requirements = PolicyRequirements()
    defaults: dict = {}

    def __init__(self, **params) -> None:
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameters {sorted(unknown)}")
        self.params = {**self.defaults, **params}
        self.ctx = None

    def reset(self, ctx) -> None:
        self.ctx = ctx

    def decide(self, obs: Observation) -> list:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "label": self.label, "family": self.family, "params": dict(self.params),
                "requirements": self.requirements.to_dict()}

    # ------------------------------------------------------------------ shared helpers
    @staticmethod
    def committed(obs: Observation) -> int:
        """Quantity filled, in flight or resting (as the strategy knows it)."""
        return (obs.filled + obs.context.get("in_flight", 0) + obs.context.get("pending_limit", 0)
                + sum(r for _, r in obs.resting))

    def to_schedule(self, obs: Observation, fraction_done: float) -> int:
        """Child quantity that brings committed volume to ``fraction_done`` of the target."""
        target = int(math.ceil(obs.target * min(max(fraction_done, 0.0), 1.0) - 1e-9))
        return max(0, min(target - self.committed(obs), obs.remaining))

    @staticmethod
    def tau_next(obs: Observation) -> float:
        return min(1.0, (obs.step + 1) / obs.steps)


def ac_fraction(tau: float, kappa_t: float) -> float:
    """Almgren-Chriss fraction executed by normalized time ``tau`` with urgency ``kappa T`` (0 = linear)."""
    if abs(kappa_t) < 1e-6:
        return tau
    return 1.0 - math.sinh(kappa_t * (1.0 - tau)) / math.sinh(kappa_t)


def realized_sigma(obs: Observation, window: int = 300) -> float:
    """Tick volatility per sqrt(second) from visible bin-end mids."""
    mids = np.asarray(obs.recent_mids[-window:], float)
    if len(mids) < 20:
        return 1.0
    return float(np.std(np.diff(mids)) / math.sqrt(0.1)) or 1e-3


# ============================================================================= baselines


@register
class TWAP(Policy):
    name, label = "twap", "TWAP"
    requirements = PolicyRequirements(observations=frozenset({"own_fills"}), market_capabilities=frozenset())

    def decide(self, obs):
        return [market(self.to_schedule(obs, self.tau_next(obs)))]


@register
class VWAP(Policy):
    name, label = "vwap", "VWAP"
    defaults = {"profile": None}
    requirements = PolicyRequirements(observations=frozenset({"own_fills"}), market_capabilities=frozenset())

    def decide(self, obs):
        profile = self.params["profile"] or obs.context.get("volume_profile") or [1.0] * obs.steps
        profile = np.asarray(profile, float)
        if len(profile) != obs.steps:
            profile = np.interp(np.linspace(0, 1, obs.steps), np.linspace(0, 1, len(profile)), profile)
        cumulative = np.cumsum(profile) / profile.sum()
        return [market(self.to_schedule(obs, float(cumulative[obs.step])))]


@register
class POV(Policy):
    name, label = "pov", "POV"
    defaults = {"participation": 0.1}
    requirements = PolicyRequirements(observations=frozenset({"volume", "own_fills"}),
                                      market_capabilities=frozenset({"trade_prints"}), completion_behaviour="best_effort")

    def decide(self, obs):
        p = self.params["participation"]
        want = int(round(p / (1.0 - p) * obs.volume_since_last))
        return [market(min(want, obs.remaining - self.committed(obs) + obs.filled))]


@register
class AlmgrenChriss(Policy):
    name, label = "ac", "Almgren-Chriss"
    defaults = {"kappa_t": 1.5}
    requirements = PolicyRequirements(observations=frozenset({"own_fills"}), market_capabilities=frozenset())

    def decide(self, obs):
        return [market(self.to_schedule(obs, ac_fraction(self.tau_next(obs), self.params["kappa_t"])))]


@register
class Heuristic(Policy):
    """Spread-aware: passive at its own touch while on schedule and the spread is wide, crosses otherwise."""

    name, label = "heuristic", "Heuristic (spread-aware)"
    defaults = {"lag_slices": 1.0}
    requirements = PolicyRequirements(observations=frozenset({"touch", "depth", "own_fills"}))

    def decide(self, obs):
        slice_ = max(1, math.ceil(obs.target / obs.steps))
        target = int(math.ceil(obs.target * self.tau_next(obs)))
        behind = target - obs.filled - obs.context.get("in_flight", 0)
        actions = [cancel_all()] if obs.resting else []
        if behind <= 0:
            return actions
        if obs.spread <= 1 or behind > self.params["lag_slices"] * slice_ + slice_:
            return actions + [market(min(behind, obs.remaining))]
        return actions + [limit(min(behind, obs.remaining))]


@register
class RandomPolicy(Policy):
    """Seeded random schedule (control): random child fractions and random passive/aggressive choice."""

    name, label = "random", "Random (control)"
    requirements = PolicyRequirements(observations=frozenset({"own_fills"}), market_capabilities=frozenset(),
                                      completion_behaviour="best_effort", stochastic=True)

    def decide(self, obs):
        rng = self.ctx.rng
        left = obs.remaining - obs.context.get("in_flight", 0)
        if left <= 0:
            return []
        steps_left = max(1, obs.steps - obs.step)
        qty = int(rng.binomial(left, min(1.0, 2.0 / steps_left)))
        actions = [cancel_all()] if obs.resting else []
        if qty <= 0:
            return actions
        return actions + [market(qty) if rng.random() < 0.5 else limit(qty)]


BASELINES = ("twap", "vwap", "pov", "ac", "heuristic", "random")


# ============================================================================= adaptive variants


@register
class AdaptiveTWAP(Policy):
    """Passive-first TWAP that re-spreads the remainder; crosses when behind by more than ``lag_slices``."""

    name, label, family = "adaptive_twap", "Adaptive TWAP", "adaptive"
    defaults = {"lag_slices": 1.0}
    requirements = PolicyRequirements(observations=frozenset({"touch", "own_fills"}))

    def decide(self, obs):
        steps_left = max(1, obs.steps - obs.step)
        actions = [cancel_all()] if obs.resting else []
        left = obs.remaining - obs.context.get("in_flight", 0)
        if left <= 0:
            return actions
        slice_ = max(1, math.ceil(left / steps_left))
        behind = int(math.ceil(obs.target * obs.step / obs.steps)) - obs.filled
        if steps_left == 1 or behind > self.params["lag_slices"] * max(1, obs.target / obs.steps):
            return actions + [market(min(left, max(slice_, behind)))]
        return actions + [limit(slice_)]


@register
class LiquidityAwareTWAP(Policy):
    name, label, family = "liquidity_twap", "Liquidity-aware TWAP", "adaptive"
    defaults = {"low": 0.5, "high": 2.0}
    requirements = PolicyRequirements(observations=frozenset({"depth", "own_fills"}))

    def reset(self, ctx):
        super().reset(ctx)
        self.depths: list[float] = []

    def decide(self, obs):
        depth = float(obs.touch_depth())
        self.depths.append(depth)
        ref = float(np.mean(self.depths))
        base = self.to_schedule(obs, self.tau_next(obs))
        scale = min(max(depth / ref if ref > 0 else 1.0, self.params["low"]), self.params["high"])
        steps_left = obs.steps - obs.step
        child = base if steps_left <= 1 else int(round(base * scale))
        return [market(min(child, obs.remaining - self.committed(obs) + obs.filled))]


@register
class AdaptiveVWAP(Policy):
    """VWAP whose remaining-volume forecast is updated online from the observed traded volume rate."""

    name, label, family = "adaptive_vwap", "Adaptive VWAP", "adaptive"
    requirements = PolicyRequirements(observations=frozenset({"volume", "own_fills"}),
                                      market_capabilities=frozenset({"trade_prints"}))

    def reset(self, ctx):
        super().reset(ctx)
        self.volume = 0.0

    def decide(self, obs):
        self.volume += obs.volume_since_last
        steps_left = max(1, obs.steps - obs.step)
        if steps_left == 1:
            return [market(obs.remaining - obs.context.get("in_flight", 0))]
        rate = self.volume / max(obs.elapsed, 1e-9) if obs.elapsed > 0 else None
        if not rate:
            return [market(self.to_schedule(obs, self.tau_next(obs)))]
        expected_now = rate * obs.horizon / obs.steps
        expected_rest = rate * obs.remaining_time
        child = int(round((obs.remaining - obs.context.get("in_flight", 0)) * expected_now / max(expected_rest, 1e-9)))
        return [market(max(0, child))]


@register
class VolatilityAware(Policy):
    """AC trajectory whose urgency scales with observed volatility relative to its running mean."""

    name, label, family = "volatility_aware", "Volatility-aware schedule", "adaptive"
    defaults = {"kappa_t": 1.5, "max_kappa_t": 4.0}
    requirements = PolicyRequirements(observations=frozenset({"mids", "own_fills"}))

    def reset(self, ctx):
        super().reset(ctx)
        self.sigmas: list[float] = []

    def decide(self, obs):
        sigma = realized_sigma(obs)
        self.sigmas.append(sigma)
        ratio = sigma / float(np.mean(self.sigmas))
        kappa = min(self.params["kappa_t"] * ratio, self.params["max_kappa_t"])
        return [market(self.to_schedule(obs, ac_fraction(self.tau_next(obs), kappa)))]


@register
class AdaptivePOV(Policy):
    """POV whose participation grows with the schedule deficit (capped), so completion is pursued."""

    name, label, family = "adaptive_pov", "Adaptive POV", "adaptive"
    defaults = {"participation": 0.1, "max_participation": 0.5}
    requirements = PolicyRequirements(observations=frozenset({"volume", "own_fills"}),
                                      market_capabilities=frozenset({"trade_prints"}), completion_behaviour="catch_up")

    def decide(self, obs):
        slice_ = obs.target / obs.steps
        deficit = obs.target * obs.step / obs.steps - obs.filled
        p = min(self.params["participation"] * (1.0 + max(deficit, 0.0) / slice_), self.params["max_participation"])
        want = int(round(p / (1.0 - p) * obs.volume_since_last))
        if obs.steps - obs.step <= 1:
            want = obs.remaining
        return [market(min(want, obs.remaining - obs.context.get("in_flight", 0)))]


@register
class DynamicAC(Policy):
    """Re-solves the AC trajectory each step with online volatility and a depth-based impact estimate."""

    name, label, family = "dynamic_ac", "Dynamic Almgren-Chriss", "adaptive"
    defaults = {"risk_aversion": 1e-4}
    requirements = PolicyRequirements(observations=frozenset({"mids", "depth", "touch", "own_fills"}))

    def decide(self, obs):
        sigma = realized_sigma(obs)
        depth = max(float(np.mean(obs.ask_depth[:3] if obs.side > 0 else obs.bid_depth[:3])), 1.0)
        eta = 0.5 * obs.spread + 1.0 / depth                    # ticks per lot of temporary cost (proxy)
        kappa = math.sqrt(self.params["risk_aversion"] * sigma * sigma / eta) * obs.remaining_time
        steps_left = max(1, obs.steps - obs.step)
        tau = 1.0 / steps_left
        left = obs.remaining - obs.context.get("in_flight", 0)
        frac = ac_fraction(tau, min(kappa, 6.0))
        return [market(max(0, int(math.ceil(left * frac - 1e-9))))]


ADAPTIVE = ("adaptive_twap", "liquidity_twap", "adaptive_vwap", "volatility_aware", "adaptive_pov", "dynamic_ac")


# ============================================================================= robust parametric family


@register
class ParamSchedule(Policy):
    """Parametric family searched by robust optimization (M24).

    ``urgency`` = AC kappa*T of the schedule (0 linear, > 0 front-loaded); ``passive`` = fraction of each child
    posted at the own touch; ``catchup`` = deficit (in slices) beyond which the deficit is crossed.
    """

    name, label, family = "param_schedule", "Parametric schedule", "robust"
    defaults = {"urgency": 0.0, "passive": 0.0, "catchup": 1.0, "label": None}
    requirements = PolicyRequirements(observations=frozenset({"touch", "own_fills"}))

    def decide(self, obs):
        p = self.params
        actions = [cancel_all()] if obs.resting else []
        left = obs.remaining - obs.context.get("in_flight", 0)
        if left <= 0:
            return actions
        slice_ = max(1.0, obs.target / obs.steps)
        target = int(math.ceil(obs.target * ac_fraction(self.tau_next(obs), p["urgency"]) - 1e-9))
        child = max(0, min(target - obs.filled - obs.context.get("in_flight", 0), left))
        behind = obs.target * ac_fraction(obs.step / obs.steps, p["urgency"]) - obs.filled
        if obs.steps - obs.step <= 1 or behind > p["catchup"] * slice_:
            return actions + [market(child)]
        passive = int(round(child * p["passive"]))
        out = actions + ([limit(passive)] if passive > 0 else [])
        return out + ([market(child - passive)] if child - passive > 0 else [])


def describe_all() -> dict:
    return {name: cls().describe() for name, cls in sorted(REGISTRY.items())}


__all__ = ["Policy", "PolicyRequirements", "CapabilityError", "check_capabilities", "make", "REGISTRY", "BASELINES",
           "ADAPTIVE", "ac_fraction", "realized_sigma", "field"]
