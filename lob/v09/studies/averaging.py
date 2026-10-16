"""model-averaging-v09 (M17/M37; H5): dynamic model averaging (re-sealed from neural-point-process-v09)."""
from __future__ import annotations

from .wrappers import reseal


def run(args) -> dict:
    return reseal(args, analysis="model-averaging-v09", key="tpp", hypothesis="H5",
                  fields=("h5", "h5_odd", "h5_even", "dma_nll", "nll", "selected_single", "events"))
