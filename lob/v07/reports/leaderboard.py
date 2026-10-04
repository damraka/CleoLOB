"""Scientific leaderboard (workstream 87): a multi-column table that never collapses into one score.

Columns per model: development and selection objective, fresh-day objectives with intervals, AUC, support
coverage, parameters, fit compute, family errors and limitations. Rows are alphabetical, never ordered by a
composite rank: "best simulator" requires a registered criterion (the only one is the sealed G* rule).
"""
from __future__ import annotations

COLUMNS = ("development_objective", "selection_objective", "fresh_objective", "fresh_objective_ci", "auc",
           "support_coverage", "parameters", "fit_seconds", "family_errors", "limitations")


def leaderboard(models: dict[str, dict]) -> dict:
    rows = [{"model": name, **{c: m.get(c) for c in COLUMNS}} for name, m in sorted(models.items())]
    return {"columns": ["model", *COLUMNS], "rows": rows, "ordering": "alphabetical (no composite score)",
            "registered_selection_rule": "G* = argmin selection objective (sealed M6 rule); nothing else ranks models"}
