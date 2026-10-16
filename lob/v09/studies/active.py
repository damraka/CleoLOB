"""active-calibration-v09 (H13): M62 active calibration (cross-entropy acquisition vs random search). SYNTHETIC (E1/E2); see ``synthetic_studies``."""
from __future__ import annotations

from .synthetic_studies import run_for


def run(args) -> dict:
    return run_for(args, ["H13"], "active-calibration-v09")
