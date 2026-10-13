"""mpc-study-v09 (H24): M80 adaptive vs non-adaptive MPC v2 after regime drift (H23 is decided on bounded replay, conclusion C1). SYNTHETIC (E1/E2); see ``synthetic_studies``."""
from __future__ import annotations

from .synthetic_studies import run_for


def run(args) -> dict:
    return run_for(args, ["H24"], "mpc-study-v09")
