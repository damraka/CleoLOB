"""Simulator worlds and the frozen v0.6 execution mandate.

A *world* is one complete simulator configuration (base config plus flow
extensions) with a name and a provenance label (selected, ensemble member,
structural intervention, regime model, v0.5 control). Execution episodes run
through the unchanged v0.5 runner and environment; only the market changes
between worlds. The mandate (side, quantity, horizon, decision interval,
warmup, settlement, fees, completion rule) is frozen in the environment freeze.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

from ..config import ResearchConfig, canonical_json
from ..core_study import runner_parameters
from ..sim_v2 import SimulatorSpec

MARKET_FIELDS = ("tick_size", "lot_size", "initial_mid_ticks", "limit_rate", "market_rate", "cancel_rate", "offset_p",
                 "limit_qty_mean", "market_qty_mean", "target_level_vol", "resilience", "resilience_levels",
                 "latency_base", "latency_jitter")
MAX_EVENTS_PER_EPISODE = 3_000_000


@dataclass(frozen=True)
class World:
    name: str
    kind: str                     # selected | member | intervention | regime | control
    config: dict
    extensions: dict = field(default_factory=dict)
    note: str = ""

    @property
    def spec(self) -> SimulatorSpec:
        return SimulatorSpec(self.config, self.extensions)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json({"config": self.config, "extensions": self.extensions}).encode()).hexdigest()

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "config": self.config, "extensions": self.extensions,
                "note": self.note, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, d: dict) -> World:
        world = cls(d["name"], d["kind"], d["config"], d.get("extensions", {}), d.get("note", ""))
        if "sha256" in d and d["sha256"] != world.sha256:
            raise ValueError(f"world {d['name']} differs from its recorded hash")
        return world


def research_config(world: World, mandate: dict) -> ResearchConfig:
    market = {k: world.config[k] for k in MARKET_FIELDS if k in world.config}
    market["target_level_vol"] = int(market["target_level_vol"])
    document = {
        "name": f"v06_{world.kind}",
        "market": market,
        "execution": {"side": mandate["side"], "quantity": mandate["quantity"], "horizon": mandate["horizon_s"],
                      "decision_dt": mandate["decision_dt_s"], "warmup_seconds": mandate["warmup_s"],
                      "terminal_penalty_bps": mandate["terminal_penalty_bps"],
                      "settlement_timeout": mandate["settlement_timeout_s"],
                      "participation": mandate["pov_participation"]},
        "fees": mandate["fees"],
        "resources": {"max_episodes": 100_000, "max_estimated_events": 2_000_000_000, "check_invariants": False,
                      "max_events_per_episode": MAX_EVENTS_PER_EPISODE, "max_runtime_seconds": 1e9},
    }
    return ResearchConfig.model_validate(json.loads(json.dumps(document)))


def episode_params(world: World, mandate: dict, seed: int, *, normalization: dict | None = None,
                   ac: dict | None = None) -> dict:
    config = research_config(world, mandate)
    params = runner_parameters(config, seed, mandate["terminal_penalty_bps"])
    params.update(completion={"enabled": True, "urgency_fraction": mandate["completion_urgency_fraction"]},
                  observation_version="v04", observation_normalization=normalization,
                  flow_extensions=dict(world.extensions) or None)
    if ac:
        params.update(ac)
    return params
