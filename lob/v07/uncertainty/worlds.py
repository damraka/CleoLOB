"""World manifests and scenario provenance (workstreams 8, 41).

A plausible world is the combination of a simulator family, a parameter or
posterior sample, a regime model, a queue model, a structural assumption and a
market seed. ``WorldManifest.id`` is the SHA-256 of its canonical JSON, so the
same combination always has the same identifier. Every manifest carries exactly
one scenario label from ``configs/v07/scenario-taxonomy.json``; the plausible
set admits only CALIBRATED and POSTERIOR_SAMPLE worlds (the registered rule), so
synthetic stresses and structural interventions can never enter a robustness claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from ..protocol.taxonomy import SCENARIO_LABELS, TaxonomyError

PLAUSIBLE_LABELS = frozenset({"CALIBRATED", "POSTERIOR_SAMPLE"})
QUEUE_MODELS = ("conservative", "fifo_lower", "probabilistic", "fifo_upper", "optimistic", "engine_fifo")


@dataclass(frozen=True)
class WorldManifest:
    family: str
    parameters: str                  # parameter vector hash or posterior draw identifier
    scenario: str                    # one label from the scenario taxonomy
    regime_model: str = "none"
    queue_model: str = "engine_fifo"
    structural_assumption: str = "none"
    market_seed: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIO_LABELS:
            raise TaxonomyError(f"unknown scenario label {self.scenario!r}")
        if self.queue_model not in QUEUE_MODELS:
            raise TaxonomyError(f"unknown queue model {self.queue_model!r}")
        if self.scenario in {"STRUCTURAL_INTERVENTION", "SYNTHETIC_STRESS"} and self.structural_assumption == "none":
            raise TaxonomyError(f"{self.scenario} worlds must name their structural assumption or stress")

    @property
    def id(self) -> str:
        values = {k: v for k, v in asdict(self).items() if k != "note"}
        return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {**asdict(self), "id": self.id}


def spec_hash(spec) -> str:
    if hasattr(spec, "extensions"):
        payload = {"config": spec.config, "extensions": spec.extensions}
    elif hasattr(spec, "describe"):
        payload = spec.describe()
    else:
        payload = repr(spec)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def plausible_set(manifests: list[WorldManifest]) -> list[WorldManifest]:
    """Admit only CALIBRATED and POSTERIOR_SAMPLE worlds; duplicate identifiers are refused."""
    bad = [m.id for m in manifests if m.scenario not in PLAUSIBLE_LABELS]
    if bad:
        raise TaxonomyError(f"worlds {bad} are not admissible in the plausible set")
    ids = [m.id for m in manifests]
    if len(set(ids)) != len(ids):
        raise TaxonomyError("duplicate world identifiers in the plausible set")
    return manifests


def composition(manifests: list[WorldManifest]) -> dict:
    out: dict = {"worlds": len(manifests), "by_family": {}, "by_scenario": {}, "by_queue_model": {}}
    for m in manifests:
        for key, value in (("by_family", m.family), ("by_scenario", m.scenario), ("by_queue_model", m.queue_model)):
            out[key][value] = out[key].get(value, 0) + 1
    return out
