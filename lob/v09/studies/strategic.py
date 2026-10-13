"""strategic-response-v09 (H11): M50 self-impact rankings (endogenous vs exogenous strategic ecology). SYNTHETIC (E1/E2); see ``synthetic_studies``."""
from __future__ import annotations

from .synthetic_studies import run_for


def run(args) -> dict:
    return run_for(args, ["H11"], "strategic-response-v09")
