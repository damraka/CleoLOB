"""v0.8 controlled result vocabulary.

Every registered v0.8 result has exactly one primary status from ``STATUSES``
plus optional qualifiers. Generator realism gates use their own vocabulary
(``GATE_STATUSES``): a family can be ``REJECTED``, a gate can end with
``NO_ACCEPTABLE_MODEL``; the least-bad model is never relabelled as validated.
Equivalence and non-inferiority outcomes are separate from superiority: "not
significant" is never "equivalent".
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

STATUSES = ("ESTABLISHED", "NOT_ESTABLISHED", "FAILED", "INVALID", "INCONCLUSIVE", "NOT_AVAILABLE", "EXPLORATORY",
            "ASSUMPTION_DEPENDENT", "NOT_EVALUABLE")
GATE_STATUSES = ("ACCEPTED_UNDER_GATE", "REJECTED", "NOT_ESTABLISHED", "NO_ACCEPTABLE_MODEL", "NOT_EVALUABLE")
EQUIVALENCE = ("EQUIVALENT_WITHIN_MARGIN", "NOT_EQUIVALENT", "NOT_ESTABLISHED", "NOT_EVALUABLE")
NONINFERIORITY = ("NONINFERIOR_WITHIN_MARGIN", "INFERIOR_BEYOND_MARGIN", "NOT_ESTABLISHED", "NOT_EVALUABLE")
QUALIFIERS = ("SYNTHETIC", "RETROSPECTIVE", "FRESH_TEMPORAL", "FRESH_CROSS_INSTRUMENT", "FRESH_CROSS_VENUE",
              "FINAL_HOLDOUT", "SYNTHETIC_HOLDOUT", "CONFIRMATORY", "DESCRIPTIVE", "EXPLORATORY", "POST_HOC",
              "STRESS", "UNDERPOWERED", "VACUOUS", "BOUNDED", "HYPOTHETICAL", "MODEL_DEPENDENT",
              "DATA_ACQUISITION_BLOCKED")
# Labels that describe fills / costs on aggregate data. "ACTUAL HISTORICAL FILL" is deliberately absent.
FILL_LABELS = ("HYPOTHETICAL", "BOUNDED", "ASSUMPTION_DEPENDENT", "SIMULATED_TRUTH")


class TaxonomyError(ValueError):
    """A result or claim violates the registered vocabulary."""


def _finite(value: Any) -> bool:
    return value is None or (isinstance(value, (int, float)) and math.isfinite(float(value)))


@dataclass(frozen=True)
class Result:
    """One registered result: effect first, then interval, then status."""

    result_id: str
    hypothesis_id: str | None
    status: str
    estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    threshold: float | None = None
    margin: float | None = None
    alpha: float | None = None
    n: int | None = None
    qualifiers: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""

    def __post_init__(self) -> None:
        validate(self)

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def validate(result: Result) -> Result:
    if result.status not in STATUSES:
        raise TaxonomyError(f"unknown status {result.status!r}")
    bad = [q for q in result.qualifiers if q not in QUALIFIERS]
    if bad:
        raise TaxonomyError(f"unknown qualifiers {bad}")
    for name in ("estimate", "ci_low", "ci_high", "threshold", "margin", "alpha"):
        if not _finite(getattr(result, name)):
            raise TaxonomyError(f"{name} must be finite or None")
    if result.ci_low is not None and result.ci_high is not None and result.ci_low > result.ci_high + 1e-12:
        raise TaxonomyError("ci_low exceeds ci_high")
    if result.status == "ESTABLISHED" and result.estimate is None and "DESCRIPTIVE" not in result.qualifiers:
        raise TaxonomyError("an ESTABLISHED inferential result must report its effect estimate")
    if result.status in {"NOT_AVAILABLE", "INVALID", "NOT_EVALUABLE", "INCONCLUSIVE"} and not result.reason:
        raise TaxonomyError(f"{result.status} requires a reason")
    if "RETROSPECTIVE" in result.qualifiers and result.status == "ESTABLISHED":
        raise TaxonomyError("retrospective (consumed-data) evidence is capped below ESTABLISHED")
    return result


def worst(statuses: list[str]) -> str:
    """Conjunction of statuses (all members must hold); missing evidence dominates."""
    order = ("INVALID", "NOT_AVAILABLE", "NOT_EVALUABLE", "FAILED", "INCONCLUSIVE", "NOT_ESTABLISHED",
             "ASSUMPTION_DEPENDENT", "EXPLORATORY", "ESTABLISHED")
    if not statuses:
        return "NOT_AVAILABLE"
    for status in order:
        if status in statuses:
            return status
    raise TaxonomyError(f"unknown statuses {statuses}")
