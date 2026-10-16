"""rare-event-study-v09 (H28): M85 rare-event-aware selection with importance sampling. SYNTHETIC (E1/E2); see ``synthetic_studies``."""
from __future__ import annotations

from .synthetic_studies import run_for


def run(args) -> dict:
    return run_for(args, ["H28"], "rare-event-study-v09")
