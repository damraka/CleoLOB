"""Study templates (workstream 84): a reusable, validated skeleton for a registered v0.7 study."""
from __future__ import annotations

import json
from pathlib import Path

REQUIRED = ("question", "hypotheses", "statistical_family", "power", "datasets", "holdouts", "compute_budget", "seeds",
            "configs", "expected_artifacts", "figures", "claim_skeleton")
ROLES = {"development", "selection", "validation", "retrospective", "fresh_temporal", "fresh_cross_instrument",
         "fresh_cross_venue", "final_transfer", "mbo_retrospective"}


def validate(study: dict) -> list[str]:
    problems = [f"missing {k}" for k in REQUIRED if k not in study]
    family = study.get("statistical_family", {})
    if family.get("kind") == "confirmatory" and family.get("correction") not in {"bonferroni", "holm"}:
        problems.append("confirmatory family needs Bonferroni or Holm")
    for d in study.get("datasets", []):
        if d.get("role") not in ROLES:
            problems.append(f"unknown dataset role {d.get('role')!r}")
    if study.get("compute_budget", {}).get("early_stopping") not in {"prohibited", "registered sequential method"}:
        problems.append("early stopping must be prohibited or a registered sequential method")
    for h in study.get("hypotheses", []):
        if h.get("margin") is None and "equivalen" in json.dumps(h).lower():
            problems.append(f"{h.get('id')}: equivalence without a registered margin")
    return problems


def load_template(root: Path) -> dict:
    return json.loads((root / "configs/v07/templates/study-template.json").read_text(encoding="utf-8"))
