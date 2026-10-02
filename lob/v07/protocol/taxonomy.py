"""v0.7 machine-readable result taxonomy, multiplicity registry and equivalence enforcement.

Every registered result has exactly one primary status plus optional qualifiers.
``ROBUST_ACROSS_MODEL_UNCERTAINTY`` may only be produced by a registered
robustness rule; ``EQUIVALENT_WITHIN_MARGIN`` only with a registered margin.
Unregistered confirmatory comparisons fail validation, and abstention
(``INCONCLUSIVE`` / ``MODEL_DEPENDENT``) is a first-class outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

STATUSES = ("ESTABLISHED", "NOT_ESTABLISHED", "FAILED", "INVALID", "INCONCLUSIVE", "NOT_AVAILABLE", "EXPLORATORY",
            "ASSUMPTION_DEPENDENT", "MODEL_DEPENDENT", "ROBUST_ACROSS_MODEL_UNCERTAINTY")
EQUIVALENCE_STATUSES = ("EQUIVALENT_WITHIN_MARGIN", "NOT_ESTABLISHED", "FAILED_MARGIN", "NOT_EVALUABLE")
QUALIFIERS = ("RETROSPECTIVE", "FRESH_TEMPORAL", "FRESH_CROSS_INSTRUMENT", "FRESH_CROSS_VENUE", "FINAL_TRANSFER",
              "DEVELOPMENT", "SELECTION", "VALIDATION", "CONFIRMATORY", "DESCRIPTIVE", "EXPLORATORY", "POST_HOC",
              "STRESS", "SYNTHETIC", "VACUOUS", "UNDERPOWERED", "SMALL_WORLD_COUNT")
SCENARIO_LABELS = ("EMPIRICAL", "CALIBRATED", "POSTERIOR_SAMPLE", "STRUCTURAL_INTERVENTION", "SYNTHETIC_STRESS",
                   "EXPLORATORY")
EDGE_TYPES = ("ROBUSTLY_BETTER", "ROBUSTLY_WORSE", "EQUIVALENT_WITHIN_MARGIN", "INDETERMINATE", "MODEL_DEPENDENT",
              "REVERSED")
FAILURE_MODES = ("SUPPORT_FAILURE", "TAIL_FAILURE", "DEPENDENCE_FAILURE", "TEMPORAL_FAILURE", "REGIME_FAILURE",
                 "QUEUE_FAILURE", "IMPACT_FAILURE", "CALIBRATION_FAILURE", "IDENTIFIABILITY_FAILURE",
                 "TRANSFER_FAILURE", "EXECUTION_INSTABILITY", "MODEL_CLASS_FAILURE", "DATA_CAPABILITY_LIMIT")


class TaxonomyError(ValueError):
    """A result, comparison or claim violates the registered taxonomy."""


@dataclass(frozen=True)
class Result:
    """One registered result: effect size first, then status (requirement 63)."""

    result_id: str
    hypothesis_id: str | None
    status: str
    estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    margin: float | None = None
    mde: float | None = None
    n: int | None = None
    alpha: float | None = None
    adjusted_alpha: float | None = None
    family: str | None = None
    qualifiers: tuple[str, ...] = field(default_factory=tuple)
    reason: str | None = None

    def __post_init__(self) -> None:
        validate_result(self)

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def validate_result(result: Result) -> Result:
    if result.status not in STATUSES:
        raise TaxonomyError(f"{result.result_id}: unknown primary status {result.status!r}")
    unknown = [q for q in result.qualifiers if q not in QUALIFIERS]
    if unknown:
        raise TaxonomyError(f"{result.result_id}: unknown qualifiers {unknown}")
    if result.status in {"NOT_AVAILABLE", "INVALID"} and not result.reason:
        raise TaxonomyError(f"{result.result_id}: {result.status} requires an explicit reason")
    for name in ("estimate", "ci_low", "ci_high"):
        value = getattr(result, name)
        if value is not None and not math.isfinite(value):
            raise TaxonomyError(f"{result.result_id}: {name} must be finite or None")
    if result.ci_low is not None and result.ci_high is not None and result.ci_low > result.ci_high:
        raise TaxonomyError(f"{result.result_id}: interval bounds are reversed")
    if result.status == "ESTABLISHED" and "CONFIRMATORY" in result.qualifiers and result.estimate is None:
        raise TaxonomyError(f"{result.result_id}: a confirmatory ESTABLISHED result must report its estimate")
    return result


# ----------------------------------------------------------------------------- multiplicity (req 61)


@dataclass(frozen=True)
class Family:
    name: str
    kind: str              # confirmatory | descriptive | exploratory
    size: int
    correction: str        # bonferroni | holm | none
    alpha: float = 0.05
    members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"confirmatory", "descriptive", "exploratory"}:
            raise TaxonomyError(f"family {self.name}: unknown kind {self.kind}")
        if self.kind == "confirmatory" and self.correction not in {"bonferroni", "holm"}:
            raise TaxonomyError(f"family {self.name}: confirmatory families require Bonferroni or Holm")
        if self.size < 1 or not 0 < self.alpha < 1:
            raise TaxonomyError(f"family {self.name}: invalid size or alpha")

    @property
    def adjusted_alpha(self) -> float:
        """Per-comparison threshold (Bonferroni; Holm's first step)."""
        return self.alpha / self.size if self.correction in {"bonferroni", "holm"} else self.alpha

    def holm_thresholds(self) -> list[float]:
        return [self.alpha / (self.size - i) for i in range(self.size)]


class MultiplicityRegistry:
    """Machine-computed family sizes and thresholds; unregistered confirmatory comparisons fail."""

    def __init__(self, families: Iterable[Family]) -> None:
        self.families = {f.name: f for f in families}

    @classmethod
    def from_config(cls, document: dict) -> MultiplicityRegistry:
        return cls(Family(name, f["kind"], int(f["size"]), f["correction"], float(f["alpha"]),
                          tuple(f.get("comparisons", ()))) for name, f in document["families"].items())

    def check(self, family: str, comparison: str, *, confirmatory: bool = True) -> Family:
        if family not in self.families:
            if confirmatory:
                raise TaxonomyError(f"comparison {comparison!r} belongs to unregistered family {family!r}")
            raise TaxonomyError(f"unknown family {family!r}")
        registered = self.families[family]
        if confirmatory and registered.kind != "confirmatory":
            raise TaxonomyError(f"{comparison!r}: family {family} is {registered.kind}, not confirmatory")
        if registered.members and comparison not in registered.members:
            raise TaxonomyError(f"{comparison!r} is not a registered member of family {family}")
        return registered

    def holm(self, family: str, pvalues: dict[str, float]) -> dict[str, bool]:
        """Holm step-down decisions for the registered members of a family."""
        registered = self.families[family]
        if len(pvalues) > registered.size:
            raise TaxonomyError(f"family {family}: more comparisons than its registered size")
        ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
        rejected, stop = {}, False
        for i, (name, p) in enumerate(ordered):
            if not stop and p <= registered.alpha / (registered.size - i):
                rejected[name] = True
            else:
                stop = True
                rejected[name] = False
        return rejected


# ----------------------------------------------------------------------------- equivalence (req 62)


def equivalence(upper: float | None, lower: float | None, margin: float | None) -> str:
    """One-sided equivalence of a nonnegative distance; refuses to run without a registered margin."""
    if margin is None:
        raise TaxonomyError("equivalence requires a preregistered margin; none is registered")
    if upper is None or lower is None or not (math.isfinite(upper) and math.isfinite(lower)):
        return "NOT_EVALUABLE"
    if upper < margin:
        return "EQUIVALENT_WITHIN_MARGIN"
    if lower > margin:
        return "FAILED_MARGIN"
    return "NOT_ESTABLISHED"


def two_sided_equivalence(ci_low: float | None, ci_high: float | None, margin: float | None) -> str:
    """TOST-style decision for a signed difference: equivalent only if the interval lies inside (-margin, margin)."""
    if margin is None:
        raise TaxonomyError("equivalence requires a preregistered margin; none is registered")
    if ci_low is None or ci_high is None:
        return "NOT_EVALUABLE"
    if -margin < ci_low and ci_high < margin:
        return "EQUIVALENT_WITHIN_MARGIN"
    if ci_low >= margin or ci_high <= -margin:
        return "FAILED_MARGIN"
    return "NOT_ESTABLISHED"


def interval_status(ci_low: float | None, ci_high: float | None, *, direction: int = -1,
                    margin: float | None = None) -> str:
    """Directional confirmatory rule: ESTABLISHED if the interval excludes zero in ``direction`` (and clears
    a registered materiality margin when one is given); FAILED if it excludes zero the other way."""
    if ci_low is None or ci_high is None or not (math.isfinite(ci_low) and math.isfinite(ci_high)):
        return "INVALID"
    material = margin or 0.0
    if direction < 0:
        if ci_high < -material if margin else ci_high < 0:
            return "ESTABLISHED"
        if ci_low > 0:
            return "FAILED"
    else:
        if ci_low > material if margin else ci_low > 0:
            return "ESTABLISHED"
        if ci_high < 0:
            return "FAILED"
    return "NOT_ESTABLISHED"
