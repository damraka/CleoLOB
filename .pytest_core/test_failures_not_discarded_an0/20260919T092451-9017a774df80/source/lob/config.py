"""Strict immutable research configuration and safe, bounded composition.

Precedence: inherited files (left to right), current file, CLEO__ environment
overrides, explicit dotted-key overrides. Hashes refer to fully resolved values.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .engine import SimConfig

PositiveInt = Annotated[int, Field(gt=0)]
Seed = Annotated[int, Field(ge=0, le=2**32 - 1)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
AgentName = Literal["twap", "vwap", "pov", "ac", "heuristic", "random", "ppo"]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class MarketSettings(Settings):
    tick_size: Positive = Field(0.01, description="Currency per integer price tick.")
    lot_size: PositiveInt = Field(1, description="Minimum quantity increment in shares.")
    initial_mid_ticks: PositiveInt = Field(10_000, description="Initial midpoint in ticks.")
    limit_rate: Nonnegative = Field(25.0, description="Limit arrival rate per side per second.")
    market_rate: Nonnegative = Field(6.0, description="Market arrival rate per side per second.")
    cancel_rate: Nonnegative = Field(0.15, description="Per-background-order cancellation hazard per second.")
    offset_p: Annotated[float, Field(gt=0, le=1)] = Field(0.25, description="Geometric limit offset probability.")
    limit_qty_mean: Positive = Field(35.0, description="Mean exponential limit size before lot rounding.")
    market_qty_mean: Positive = Field(35.0, description="Mean exponential market size before lot rounding.")
    target_level_vol: PositiveInt = Field(300, description="Initial quantity per price level; must align to lot.")
    resilience: Nonnegative = Field(0.8, description="Scale for endogenous depth replenishment.")
    resilience_levels: PositiveInt = Field(5, description="Top levels used for refill deficit.")
    latency_base: Nonnegative = Field(0.005, description="Fixed message transport latency, seconds.")
    latency_jitter: Nonnegative = Field(0.005, description="Mean exponential message jitter, seconds.")

    @model_validator(mode="after")
    def mechanics(self) -> MarketSettings:
        if self.initial_mid_ticks <= 20:
            raise ValueError("initial_mid_ticks must exceed the 20-level seeded ladder")
        if self.target_level_vol % self.lot_size:
            raise ValueError("target_level_vol must align to lot_size")
        return self

    def engine_config(self, seed: int) -> SimConfig:
        return SimConfig(**self.model_dump(), seed=seed, check_invariants=True)


class ExecutionSettings(Settings):
    side: Literal["buy", "sell"] = Field("sell", description="Parent execution direction; sells begin with endowed inventory.")
    quantity: PositiveInt = Field(2_000, description="Parent quantity in shares; lot aligned.")
    horizon: Annotated[float, Field(gt=0, le=3600)] = Field(10.0, description="Execution horizon, seconds.")
    decision_dt: Positive = Field(0.5, description="Decision interval, seconds; final step ends at horizon.")
    warmup_seconds: Annotated[float, Field(ge=0, le=3600)] = Field(5.0, description="Background simulator warmup before arrival-price observation.")
    risk_aversion: Nonnegative = Field(1e-6, description="AC quadratic inventory-risk coefficient.")
    temp_impact: Positive = Field(1.3e-3, description="AC temporary impact slope, currency seconds per quantity unit; calibrate on simulator diagnostics.")
    sigma: Nonnegative = Field(1.5, description="AC arithmetic price volatility, currency per square-root second; calibrate on simulator diagnostics.")
    participation: Annotated[float, Field(gt=0, lt=1)] = Field(0.3, description="POV target participation.")
    terminal_penalty_bps: Nonnegative = Field(25.0, description="Noncompletion objective penalty; separate from economic costs.")
    settlement_timeout: Annotated[float, Field(ge=0, le=3600)] = Field(5.0, description="Maximum post-decision seconds to cancel/drain outstanding strategy orders.")
    settlement_poll_dt: Positive = Field(0.01, description="Settlement polling interval, seconds; no further strategy actions.")

    @model_validator(mode="after")
    def interval(self) -> ExecutionSettings:
        if self.decision_dt > self.horizon:
            raise ValueError("decision_dt cannot exceed horizon")
        if self.settlement_timeout / self.settlement_poll_dt > 100_000:
            raise ValueError("settlement exceeds 100000 polling steps")
        return self


class FeeSettings(Settings):
    maker_bps: Annotated[float, Field(ge=-100, le=1000)] = Field(0.0, description="Maker fee in bps; negative means rebate.")
    taker_bps: Annotated[float, Field(ge=0, le=1000)] = Field(1.0, description="Taker fee in bps of fill notional.")
    per_share: Nonnegative = Field(0.0, description="Additional currency fee per executed share.")


class RiskSettings(Settings):
    max_order_qty: PositiveInt | None = Field(None, description="Maximum child quantity; null uses parent budget.")
    max_order_notional: Positive | None = Field(None, description="Maximum currency notional for a child order.")
    max_position: PositiveInt | None = Field(None, description="Maximum absolute signed filled position, shares.")
    max_gross_notional: Positive | None = Field(None, description="Maximum marked gross position plus reserved order notional.")
    max_loss: Positive | None = Field(None, description="Maximum mark-to-market loss in currency before stop.")
    kill_switch: bool = Field(False, description="Start with new orders disabled; cancellations remain possible.")


class EvaluationSettings(Settings):
    agents: tuple[AgentName, ...] = Field(("twap", "vwap", "pov", "ac"), description="Frozen candidate list; PPO requires a checkpoint.")
    reference: AgentName = Field("twap", description="Baseline for all paired tests.")
    seeds: tuple[Seed, ...] = Field((101, 102, 103), description="Unique paired episode seeds; common across agents.")
    bootstrap_samples: Annotated[int, Field(ge=100, le=100_000)] = Field(2_000, description="Bootstrap replicates per mean/paired interval.")
    confidence_alpha: Annotated[float, Field(gt=0, lt=1)] = Field(0.05, description="Two-sided interval error probability.")
    correction: Literal["holm", "bonferroni", "fdr_bh"] = Field("holm", description="Correction across every candidate-versus-reference test in this run.")
    statistics_seed: Seed = Field(0, description="Separate deterministic bootstrap/design RNG seed.")
    model_path: str | None = Field(None, description="Explicit local trusted PPO checkpoint; its bytes are hashed for provenance.")

    @field_validator("agents", "seeds", mode="before")
    @classmethod
    def arrays(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def design(self) -> EvaluationSettings:
        if not self.agents or len(self.agents) != len(set(self.agents)):
            raise ValueError("agents must be nonempty and unique")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be nonempty and unique")
        if self.reference not in self.agents:
            raise ValueError("reference must be included in agents")
        if "ppo" in self.agents and not self.model_path:
            raise ValueError("PPO research requires an explicit model_path; no heuristic substitution")
        return self


class ResourceSettings(Settings):
    max_episodes: Annotated[int, Field(ge=1, le=100_000)] = Field(200, description="Hard preflight bound on agent × seed episodes.")
    max_runtime_seconds: Positive = Field(120.0, description="Wall-clock stop checked between episodes; not a per-event timeout.")
    max_decisions_per_episode: PositiveInt = Field(20_000, description="Reject horizon/decision interval above this count.")
    max_estimated_events: PositiveInt = Field(2_000_000, description="Conservative arrival-work estimate across the design; not a memory cap.")
    max_events_per_episode: PositiveInt = Field(100_000, description="Hard exchange event cap including warmup and each auxiliary VWAP path.")


class ResearchConfig(Settings):
    schema_version: Literal[1] = Field(1, description="Resolved configuration schema version.")
    name: Annotated[str, Field(min_length=1, max_length=100)] = Field("execution_foundations", description="Human-readable study label; not an output path.")
    market: MarketSettings = Field(default_factory=MarketSettings, description="Synthetic market mechanics.")
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings, description="Parent-order objective and decision grid.")
    fees: FeeSettings = Field(default_factory=FeeSettings, description="Costs booked on every fill.")
    risk: RiskSettings = Field(default_factory=RiskSettings, description="Active execution risk limits; parent reservations are always enforced.")
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings, description="Frozen agent comparison and statistical analysis.")
    resources: ResourceSettings = Field(default_factory=ResourceSettings, description="Bounded run admission and interruption settings.")

    @model_validator(mode="after")
    def research_constraints(self) -> ResearchConfig:
        if self.execution.quantity % self.market.lot_size:
            raise ValueError("execution.quantity must align to market.lot_size")
        if self.risk.max_order_qty is not None and self.risk.max_order_qty < self.market.lot_size:
            raise ValueError("max_order_qty is smaller than one lot")
        if self.episode_count > self.resources.max_episodes:
            raise ValueError(f"design requires {self.episode_count} episodes, exceeding max_episodes")
        if math.ceil(self.execution.horizon / self.execution.decision_dt) > self.resources.max_decisions_per_episode:
            raise ValueError("decision count exceeds max_decisions_per_episode")
        if self.estimated_events > self.resources.max_estimated_events:
            raise ValueError("estimated event work exceeds max_estimated_events")
        return self

    @property
    def episode_count(self) -> int:
        return len(self.evaluation.agents) * len(self.evaluation.seeds)

    @property
    def estimated_events(self) -> int:
        # Arrival bound estimate includes warmup, seeded cancellation clocks and
        # refill proposals. VWAP historical-profile fits add five market episodes.
        rate = 2 * (self.market.limit_rate + self.market.market_rate)
        rate += 20 * self.market.resilience * self.market.resilience_levels
        rate = 2 * rate + 40 * self.market.cancel_rate
        paths = self.episode_count + (5 * len(self.evaluation.seeds) if "vwap" in self.evaluation.agents else 0)
        return math.ceil((self.execution.horizon + self.execution.settlement_timeout + 5) * max(1, rate) * paths)

    def runner_params(self, seed: int) -> dict[str, Any]:
        if seed not in self.evaluation.seeds:
            raise ValueError("episode seed is not in the frozen design")
        return {
            "qty": self.execution.quantity, "horizon": self.execution.horizon, "side": self.execution.side,
            "dt": self.execution.decision_dt, "seed": seed,
            "warmup_seconds": self.execution.warmup_seconds,
            "latency_ms": 1000 * (self.market.latency_base + self.market.latency_jitter),
            "resilience": self.market.resilience, "market_rate": self.market.market_rate,
            "risk_aversion": self.execution.risk_aversion,
            "temp_impact": self.execution.temp_impact, "sigma": self.execution.sigma,
            "pov_rate": self.execution.participation,
            "sim": {**self.market.engine_config(seed).__dict__, "max_events": self.resources.max_events_per_episode},
            "fees": self.fees.model_dump(), "risk": self.risk.model_dump(),
            "terminal_penalty_bps": self.execution.terminal_penalty_bps,
            "settlement_timeout": self.execution.settlement_timeout,
            "settlement_poll_dt": self.execution.settlement_poll_dt,
            "model_path": self.evaluation.model_path, "strict_model": True,
        }

    def warnings(self) -> list[str]:
        out = ["Synthetic Poisson market is uncalibrated; results cannot establish live alpha."]
        if len(self.evaluation.seeds) < 20:
            out.append("Fewer than 20 seeds: uncertainty estimates and hypothesis tests have low resolution.")
        if len(self.evaluation.agents) < 2:
            out.append("Only one agent: no baseline comparison is possible.")
        if self.fees.maker_bps == self.fees.taker_bps == self.fees.per_share == 0:
            out.append("All fees are zero; net performance is conditional on a zero-fee assumption.")
        if self.risk.kill_switch:
            out.append("Kill switch is enabled: this experiment exercises disabled-order behavior.")
        return out


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def config_hash(config: ResearchConfig) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise ValueError("configuration mapping keys must be strings")
        if key in result:
            raise ValueError(f"duplicate configuration key: {key}")
        result[key] = value
    return result


class _ConfigLoader(yaml.SafeLoader):
    _depth = 0

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            raise ValueError("YAML aliases are not supported; use extends for composition")
        if self._depth >= 32:
            raise ValueError("YAML nesting exceeds 32 levels")
        self._depth += 1
        try:
            return super().compose_node(parent, index)
        finally:
            self._depth -= 1


def _yaml_mapping(loader: _ConfigLoader, node: Any) -> dict[str, Any]:
    return _unique_pairs([(loader.construct_object(k), loader.construct_object(v)) for k, v in node.value])


_ConfigLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping)


def merge_settings(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        result[key] = merge_settings(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def _read_config(path: Path, ancestors: tuple[Path, ...] = (), budget: list[int] | None = None) -> dict[str, Any]:
    budget = [64] if budget is None else budget
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("configuration composition exceeds 64 file reads")
    path = path.resolve(strict=True)
    if path in ancestors or len(ancestors) >= 16:
        raise ValueError("configuration inheritance cycle or depth greater than 16")
    if path.stat().st_size > 1_048_576:
        raise ValueError("configuration file exceeds 1 MiB")
    content = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        data = json.loads(content, object_pairs_hook=_unique_pairs)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.load(content, Loader=_ConfigLoader)
    else:
        raise ValueError("configuration must be .json, .yaml or .yml")
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    parents = data.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(p, str) for p in parents) or len(parents) > 16:
        raise ValueError("extends must be a file path or a list of at most 16 paths")
    result: dict[str, Any] = {}
    for parent in parents:
        result = merge_settings(result, _read_config(path.parent / parent, (*ancestors, path), budget))
    return merge_settings(result, data)


def _set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    if not all(part and part.isidentifier() for part in parts):
        raise ValueError(f"invalid override path: {key}")
    current = data
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        if not isinstance(current[part], dict):
            raise ValueError(f"override traverses a scalar: {key}")
        current = current[part]
    current[parts[-1]] = value


def load_config(path: str | Path | None = None, overrides: tuple[str, ...] = (),
                environ: dict[str, str] | None = None) -> ResearchConfig:
    data = _read_config(Path(path)) if path is not None else {}
    env = os.environ if environ is None else environ
    for key in sorted(env):
        if key.startswith("CLEO__"):
            dotted = key[6:].lower().replace("__", ".")
            _set_dotted(data, dotted, yaml.load(env[key], Loader=_ConfigLoader))
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError("override must be dotted.key=value")
        _set_dotted(data, key, yaml.load(value, Loader=_ConfigLoader))
    return ResearchConfig.model_validate(data)


def config_diff(left: dict[str, Any], right: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key in sorted(left.keys() | right.keys()):
        path = f"{prefix}.{key}" if prefix else key
        a, b = left.get(key), right.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            result.extend(config_diff(a, b, path))
        elif key not in left or key not in right or a != b:
            result.append({"field": path, "before": a, "after": b})
    return result
