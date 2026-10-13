"""dro-study-v09 (H21): M79 distributionally robust (min-max regret) selection over ambiguity worlds. SYNTHETIC (E1/E2); see ``synthetic_studies``."""
from __future__ import annotations

from .synthetic_studies import run_for


def run(args) -> dict:
    return run_for(args, ["H21"], "dro-study-v09")
