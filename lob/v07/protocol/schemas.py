"""Research-schema versioning, read-only migration and configuration validation v2 (workstreams 80, 81).

``SCHEMAS`` lists every versioned v0.7 document type. ``schema_of`` identifies a document; ``read_v06_run``
is a read-only migration utility that presents a sealed v0.6 run in the v0.7 result vocabulary without
writing anything (v0.6 artifacts are never mutated).

``validate_study_config`` fails early on incompatible combinations: exact FIFO / queue position on
aggregate L2, unsupported venue rules, missing fee schedule, impossible latency, illegal horizon,
incompatible queue model and impossible data capability. It returns all problems at once.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMAS = {
    "cleolob-v07-protocol-1": "protocol", "cleolob-v07-ledger-1": "ledger entry",
    "cleolob-v07-binding-1": "evidence binding", "cleolob-v07-datasets-1": "dataset registry",
    "cleolob-v07-hypotheses-1": "hypotheses", "cleolob-v07-families-1": "statistical families",
    "cleolob-v07-compute-1": "compute budget", "cleolob-v07-result-taxonomy-1": "result taxonomy",
    "cleolob-v07-claims-1": "claim registry", "cleolob-v07-scenarios-1": "scenario taxonomy",
    "cleolob-v07-power-1": "power design", "cleolob-v07-holdout-design-1": "holdout design",
    "cleolob-v07-transfer-design-1": "transfer design", "cleolob-v07-policy-registration-1": "policy registration",
    "cleolob-v07-replay-checkpoint-1": "replay checkpoint", "cleolob-v07-canonical-1": "canonical market data",
    "cleolob-v07-result-registry-1": "result registry", "cleolob-v07-claim-graph-1": "claim graph",
    "cleolob-v07-study-template-1": "study template", "cleolob-v07-benchmark-tasks-1": "benchmark tasks",
    "cleolob-v07-capsule-1": "reproducibility capsule",
}


class ConfigError(ValueError):
    """A study configuration is internally incompatible."""


def schema_of(document: dict) -> str:
    schema = document.get("schema")
    if schema in SCHEMAS:
        return SCHEMAS[schema]
    if isinstance(schema, str) and schema.startswith("cleolob-v06"):
        return "v0.6 document (read-only)"
    raise ConfigError(f"unknown or missing schema {schema!r}")


def read_v06_run(run: Path) -> dict:
    """Read-only view of a sealed v0.6 run; nothing is written or modified."""
    before = {p.name: p.stat().st_mtime_ns for p in Path(run).iterdir()}
    result = json.loads((Path(run) / "result.json").read_text(encoding="utf-8"))
    binding = json.loads((Path(run) / "binding.json").read_text(encoding="utf-8"))
    view = {"source_schema": binding.get("schema"), "analysis": binding.get("analysis"),
            "protocol_sha256": binding.get("protocol_sha256"), "result_keys": sorted(result),
            "migration": "read-only presentation in v0.7 vocabulary; v0.6 statuses are kept verbatim"}
    after = {p.name: p.stat().st_mtime_ns for p in Path(run).iterdir()}
    if before != after:
        raise RuntimeError("v0.6 run changed during a read-only migration")
    return view


def validate_study_config(config: dict) -> list[str]:
    from ..data.schema import venue_spec
    from ..exchange.venue import rules
    from ..queue.models import MODELS
    problems = []
    venue, instrument = config.get("venue"), config.get("instrument")
    spec = None
    try:
        spec = venue_spec(venue, instrument)
    except ValueError as exc:
        problems.append(f"impossible data capability: {exc}")
    try:
        rules(venue, instrument)
    except ValueError as exc:
        problems.append(f"unsupported venue rule: {exc}")
    queue = config.get("queue_model", "conservative")
    if queue in {"exact_fifo", "exact_queue_position"} and spec is not None and spec.capability_level == "aggregate_l2":
        problems.append("L2 + exact FIFO: aggregate L2 does not establish queue position")
    elif queue not in MODELS and queue not in {"engine_fifo", "exact_fifo", "exact_queue_position"}:
        problems.append(f"incompatible queue model {queue!r}")
    if queue in {"exact_fifo", "exact_queue_position"} and spec is not None and not spec.contract.supports(
            "exact_fifo_position"):
        problems.append("requested exact FIFO is NOT_AVAILABLE for this source")
    fees = config.get("fees")
    if not isinstance(fees, dict) or not {"maker_bps", "taker_bps"} <= set(fees):
        problems.append("missing fee schedule (maker_bps and taker_bps required)")
    horizon, dt = config.get("horizon_s"), config.get("decision_dt_s")
    if not isinstance(horizon, (int, float)) or not isinstance(dt, (int, float)) or horizon <= 0 or dt <= 0 or dt > horizon:
        problems.append("illegal horizon: need 0 < decision interval <= horizon")
    latency = config.get("latency_s", 0.0)
    if not isinstance(latency, (int, float)) or latency < 0 or (isinstance(horizon, (int, float)) and latency >= horizon):
        problems.append("impossible latency: must be nonnegative and shorter than the horizon")
    for capability in config.get("requires", []):
        if spec is not None and not spec.contract.supports(capability):
            problems.append(f"impossible data capability: {capability} is NOT_AVAILABLE from {spec.capability_level}")
    return problems


def require_valid(config: dict) -> dict:
    problems = validate_study_config(config)
    if problems:
        raise ConfigError("; ".join(problems))
    return config
