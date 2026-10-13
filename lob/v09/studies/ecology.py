"""ecology-study-v09 (M50; H12): strategic vs static ecology response realism (re-sealed from hybrid-twin-v09)."""
from __future__ import annotations

from .wrappers import reseal


def run(args) -> dict:
    return reseal(args, analysis="ecology-study-v09", key="hybrid", hypothesis="H12",
                  fields=("h12", "h12_odd", "h12_even", "err", "bursts", "status"))
