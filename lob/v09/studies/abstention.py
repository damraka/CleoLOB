"""abstention-study-v09 (H20): M84 contextual fallback ladder in shifted worlds (H18 and H19 are in the decisions hub). SYNTHETIC (E1/E2); see ``synthetic_studies``."""
from __future__ import annotations

from .synthetic_studies import run_for


def run(args) -> dict:
    return run_for(args, ["H20"], "abstention-study-v09")
