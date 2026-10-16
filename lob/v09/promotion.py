"""Hypothesis promotion protocol (M2): candidate -> feasibility -> promoted / not promoted -> confirmatory freeze.

Every v0.9 hypothesis starts as a CANDIDATE in ``configs/v09/hypotheses-candidates.json`` with its endpoint,
estimator, inferential unit, direction, practical threshold (or non-inferiority margin), statistical family,
fidelity, required capabilities and target evidence level, all fixed at the protocol freeze before any data access.

Exploratory pilots (``explore``/``pilot`` namespaces and explore-role days only) supply, per candidate:
data availability, capability, measurement reliability, the number and clustering of pilot units, the pilot effect
and the between-unit standard deviation, and the compute cost per unit. ``decide`` applies the registered gate in a
fixed order and never looks at confirmatory data:

1. ``NOT_PROMOTED_DATA_CAPABILITY`` - required data unavailable or a required capability missing/REJECTED;
2. ``NOT_PROMOTED_MEASUREMENT`` - measurement reliability below the registered minimum;
3. ``NOT_PROMOTED_EFFECT_TOO_SMALL`` - the planning effect does not exceed the practical threshold (superiority) or
   the non-inferiority margin is already violated in the pilot;
4. ``NOT_PROMOTED_INSUFFICIENT_POWER`` - power >= the registered minimum needs more units than the registered
   confirmatory ceiling;
5. ``NOT_PROMOTED_COMPUTE`` - the required units exceed the registered compute budget;
6. otherwise ``PROMOTED`` with the minimal confirmatory sample size that reaches the registered power.

Planning power uses the same decision rule as the final analysis. For superiority the rule is "Holm-adjusted lower
bound > 0 and point estimate >= threshold", i.e. ``estimate > max(z * se, threshold)``, so a planning effect at or
below the threshold can never reach power above 0.5. Planning alpha is the family alpha divided by the number of
candidates in the family (conservative for Holm). The planning effect is the pilot estimate shrunk by one pilot
standard error toward zero (a registered guard against the winner's curse).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..preregistration import ProtocolError, document_sha256
from ..v08.statistics import zq
from . import protocol as pr

TESTS = ("superiority", "noninferiority", "agreement", "transfer")
DIRECTIONS = ("increase", "decrease")
REQUIRED = ("title", "question", "domain", "endpoint", "estimator", "unit", "direction", "test", "threshold",
            "threshold_units", "family", "fidelity", "required_capabilities", "evidence_level_target", "data_roles",
            "pool", "max_units", "pilot", "decision_rule", "headline")
CONFIRMATORY_SCHEMA = "cleolob-v09-confirmatory-1"
PROMOTION_SCHEMA = "cleolob-v09-promotion-1"


def validate_candidates(candidates: dict, protocol: dict, families: dict) -> dict:
    if candidates.get("schema") != "cleolob-v09-candidates-1":
        raise ProtocolError("unsupported candidate registry schema")
    hyps = candidates.get("hypotheses", {})
    for k in range(1, 37):
        if f"H{k}" not in hyps:
            raise ProtocolError(f"candidate H{k} is not registered")
    fams = families.get("families", {})
    for hid, h in hyps.items():
        missing = [k for k in REQUIRED if k not in h]
        if missing:
            raise ProtocolError(f"{hid} missing {missing}")
        if h["test"] not in TESTS or h["direction"] not in DIRECTIONS:
            raise ProtocolError(f"{hid}: unknown test {h['test']} or direction {h['direction']}")
        if h["family"] not in fams:
            raise ProtocolError(f"{hid} references unregistered family {h['family']}")
        if not isinstance(h["threshold"], (int, float)) or not math.isfinite(h["threshold"]) or h["threshold"] < 0:
            raise ProtocolError(f"{hid}: the practical threshold must be a finite nonnegative number")
        if h["evidence_level_target"] not in pr.EVIDENCE_LEVELS:
            raise ProtocolError(f"{hid}: unknown evidence level")
        unknown_roles = set(h["data_roles"]) - set(pr.ROLE_USES) - {"synthetic"}
        if unknown_roles:
            raise ProtocolError(f"{hid}: unknown data roles {sorted(unknown_roles)}")
        if h["pool"] is not None and h["pool"] not in protocol["pools"]:
            raise ProtocolError(f"{hid}: unknown pool {h['pool']}")
        if not isinstance(h["max_units"], int) or h["max_units"] < 2:
            raise ProtocolError(f"{hid}: max_units must be an integer >= 2")
    for name, fam in fams.items():
        if fam["correction"] not in ("holm", "bonferroni", "benjamini_hochberg", "none"):
            raise ProtocolError(f"family {name}: unknown correction {fam['correction']}")
        if not 0 < fam["alpha"] < 0.5:
            raise ProtocolError(f"family {name}: alpha out of range")
        if fam.get("confirmatory", True) and fam["correction"] == "benjamini_hochberg":
            raise ProtocolError(f"family {name}: confirmatory families must control FWER")
    return candidates


# ============================================================================= power


def planning_alpha(family: dict, family_size: int) -> float:
    return family["alpha"] / max(1, family_size)


def power(test: str, effect: float, sd: float, n: int, *, alpha: float, threshold: float) -> float:
    """Probability that the registered decision rule establishes the hypothesis (normal approximation).

    ``effect`` is oriented so that positive is the hypothesised direction. Superiority: estimate > max(z*se, threshold).
    Non-inferiority: estimate - z*se > -margin (``threshold`` is the margin). Agreement/transfer: the same rule as
    superiority applied to (agreement - registered level), where ``threshold`` is 0 and the level is subtracted
    by the caller.
    """
    if n < 2 or not math.isfinite(effect) or not math.isfinite(sd):
        return 0.0
    se = max(sd, 1e-12) / math.sqrt(n)
    z = zq(1 - alpha)
    if test == "noninferiority":
        return _phi((effect + threshold) / se - z)
    bar = max(z * se, threshold)
    return 1.0 - _phi((bar - effect) / se)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def required_units(test: str, effect: float, sd: float, *, alpha: float, threshold: float, target: float,
                   ceiling: int) -> int | None:
    """Smallest n in [2, ceiling] with power >= target, or None."""
    lo, hi = 2, ceiling
    if power(test, effect, sd, hi, alpha=alpha, threshold=threshold) < target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if power(test, effect, sd, mid, alpha=alpha, threshold=threshold) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ============================================================================= gate


def decide(hid: str, candidate: dict, pilot: dict | None, *, rules: dict, family: dict, family_size: int) -> dict:
    """Registered promotion decision for one candidate from its exploratory pilot record."""
    record = {"hypothesis": hid, "lifecycle": ["CANDIDATE"], "pilot": pilot, "family": candidate["family"]}

    def out(outcome: str, why: str, **extra) -> dict:
        feasible = outcome not in ("NOT_PROMOTED_DATA_CAPABILITY", "NOT_PROMOTED_MEASUREMENT",
                                   "NOT_PROMOTED_COMPUTE")
        record["lifecycle"].append("FEASIBILITY_PASS" if feasible else "FEASIBILITY_FAIL")
        record["lifecycle"].append("PROMOTED" if outcome == "PROMOTED" else "NOT_PROMOTED")
        return {**record, "outcome": outcome, "reason": why, **extra}

    if pilot is None:
        return out("NOT_PROMOTED_OTHER_REGISTERED_REASON", "no exploratory pilot record exists for this candidate")
    if not pilot.get("data_available", False):
        return out("NOT_PROMOTED_DATA_CAPABILITY", pilot.get("data_note") or "required data not available")
    missing = [c for c in candidate["required_capabilities"] if not pilot.get("capabilities", {}).get(c, False)]
    if missing:
        return out("NOT_PROMOTED_DATA_CAPABILITY", f"required capabilities missing or rejected: {missing}")
    if rules.get("reliability_rule", "spearman_brown_split_half") == "split_half_consistency":
        ok, why = consistency(pilot)
        record["consistency"] = why
        if not ok:
            return out("NOT_PROMOTED_MEASUREMENT", f"split-half measurement inconsistency: {why}")
    else:
        reliability = pilot.get("reliability")
        if reliability is None or not math.isfinite(reliability) or reliability < rules["min_reliability"]:
            return out("NOT_PROMOTED_MEASUREMENT",
                       f"measurement reliability {reliability} below the registered {rules['min_reliability']}")
    n_pilot, effect, sd = int(pilot.get("units", 0)), pilot.get("effect"), pilot.get("sd")
    if n_pilot < rules["min_pilot_units"] or effect is None or sd is None or not math.isfinite(effect) \
            or not math.isfinite(sd):
        return out("NOT_PROMOTED_MEASUREMENT",
                   f"pilot has {n_pilot} units (registered minimum {rules['min_pilot_units']}) or no finite estimate")
    test, threshold = candidate["test"], float(candidate["threshold"])
    level = float(candidate.get("agreement_level", 0.0)) if test in ("agreement", "transfer") else 0.0
    se_pilot = sd / math.sqrt(max(n_pilot, 1))
    planning = (effect - level) - rules["shrinkage_se"] * se_pilot
    alpha = planning_alpha(family, family_size)
    record.update({"planning_effect": planning, "planning_alpha": alpha, "pilot_se": se_pilot})
    if test == "noninferiority":
        if planning <= -threshold:
            return out("NOT_PROMOTED_EFFECT_TOO_SMALL", "the pilot already violates the non-inferiority margin")
    elif planning <= threshold:
        return out("NOT_PROMOTED_EFFECT_TOO_SMALL",
                   f"planning effect {planning:.4g} does not exceed the practical threshold {threshold:g}")
    n = required_units(test, planning, sd, alpha=alpha, threshold=threshold, target=rules["min_power"],
                       ceiling=candidate["max_units"])
    if n is None:
        best = power(test, planning, sd, candidate["max_units"], alpha=alpha, threshold=threshold)
        return out("NOT_PROMOTED_INSUFFICIENT_POWER",
                   f"power {best:.3f} at the {candidate['max_units']}-unit ceiling is below {rules['min_power']}",
                   power_at_ceiling=best)
    n = max(n, rules["min_confirm_units"])
    hours = float(pilot.get("compute_hours_per_unit", 0.0)) * n
    if hours > float(candidate.get("compute_hours_ceiling", rules["compute_hours_ceiling"])):
        return out("NOT_PROMOTED_COMPUTE", f"{hours:.1f} compute hours exceed the registered ceiling",
                   required_units=n)
    achieved = power(test, planning, sd, n, alpha=alpha, threshold=threshold)
    return out("PROMOTED", "all registered promotion conditions pass", required_units=n, expected_power=achieved,
               expected_compute_hours=hours)


def consistency(pilot: dict) -> tuple[bool, str]:
    """Amendment 3 measurement screen: the odd- and even-half pilot means must not differ significantly.

    With paired halves (o_i, e_i) over pilot units: d_i = o_i - e_i; the halves are inconsistent when
    |mean(d)| > 2 * sd(d) / sqrt(n) (and n >= 4). Missing halves fail the screen (no evidence of consistency).
    """
    odd, even = pilot.get("odd_values"), pilot.get("even_values")
    if not odd or not even or len(odd) != len(even) or len(odd) < 4:
        return False, "no paired split-half values (at least 4 units) were recorded"
    d = np.asarray(odd, float) - np.asarray(even, float)
    d = d[np.isfinite(d)]
    if len(d) < 4:
        return False, "fewer than 4 finite split-half pairs"
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0
    m = float(d.mean())
    if abs(m) > 2 * se and se > 0:
        return False, f"halves differ: mean difference {m:.4g} > 2 SE {2 * se:.4g}"
    return True, f"halves consistent: mean difference {m:.4g} within 2 SE {2 * se:.4g} (n={len(d)})"


def promote(candidates: dict, pilots: dict[str, dict], *, rules: dict, families: dict) -> dict:
    """Decisions for all candidates (deterministic, order-independent)."""
    hyps = candidates["hypotheses"]
    sizes: dict[str, int] = {}
    for h in hyps.values():
        sizes[h["family"]] = sizes.get(h["family"], 0) + 1
    decisions = {hid: decide(hid, h, pilots.get(hid), rules=rules, family=families["families"][h["family"]],
                             family_size=sizes[h["family"]])
                 for hid, h in sorted(hyps.items(), key=lambda kv: int(kv[0][1:]))}
    return {"schema": PROMOTION_SCHEMA, "rules": rules, "decisions": decisions,
            "promoted": [h for h, d in decisions.items() if d["outcome"] == "PROMOTED"],
            "not_promoted": {h: d["outcome"] for h, d in decisions.items() if d["outcome"] != "PROMOTED"}}


def confirmatory_registry(candidates: dict, promotion: dict, *, extra: dict | None = None) -> dict:
    """The frozen confirmatory registry: promoted candidates with their sample sizes and decision rules."""
    hyps = {}
    for hid in promotion["promoted"]:
        c, d = candidates["hypotheses"][hid], promotion["decisions"][hid]
        hyps[hid] = {**{k: c[k] for k in REQUIRED if k in c},
                     **{k: c[k] for k in ("agreement_level", "conclusions", "margin_units") if k in c},
                     "confirm_units": d["required_units"], "planning": {
                         "effect": d["planning_effect"], "sd": d["pilot"]["sd"], "alpha": d["planning_alpha"],
                         "expected_power": d["expected_power"]}}
    return {"schema": CONFIRMATORY_SCHEMA, "hypotheses": hyps,
            "not_promoted": promotion["not_promoted"], **(extra or {})}


def freeze(root: Path, promotion_run: Path, *, reason: str = "v0.9 promotion freeze") -> dict:
    """Write configs/v09/hypotheses-confirmatory.json and append the ledger promote event (exactly once)."""
    from .artifacts import read_run, verify_run
    report = verify_run(promotion_run, root=root)
    if not report["valid"]:
        raise ProtocolError(f"promotion run does not verify: {report['issues']}")
    run = read_run(promotion_run)
    if run["binding"]["analysis"] != "hypothesis-promotion-v09":
        raise ProtocolError("the promotion freeze needs a hypothesis-promotion-v09 run")
    protocol, state = pr.state_of(root)
    if state.promotion is not None:
        raise ProtocolError("the confirmatory registry is already promoted")
    candidates = json.loads((root / pr.CANDIDATES_PATH).read_text(encoding="utf-8"))
    registry = run["result"]["confirmatory_registry"]
    target = root / pr.CONFIRMATORY_PATH
    if target.exists():
        raise ProtocolError("hypotheses-confirmatory.json already exists without a promotion entry")
    entry = pr.append_event(root / pr.LEDGER_PATH, "promote", reason=reason, protocol=protocol, root=root,
                            payload={"confirmatory_sha256": document_sha256(registry),
                                     "candidates_sha256": document_sha256(candidates),
                                     "promotion_run": promotion_run.resolve().relative_to(root.resolve()).as_posix(),
                                     "promotion_run_result_sha256": run["binding"]["result_sha256"],
                                     "promoted": run["result"]["promotion"]["promoted"],
                                     "decisions": run["result"]["promotion"]["not_promoted"]})
    target.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return entry
