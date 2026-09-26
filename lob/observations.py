"""Current simulator state and frozen train-only observation standardization."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from .capabilities import Capability, SIMULATOR_CONTRACT


OBSERVATION_CONTRACT = SIMULATOR_CONTRACT.restrict({
    Capability.AGGREGATE_DEPTH, Capability.TOP_OF_BOOK, Capability.SPREAD,
    Capability.BOOK_IMBALANCE, Capability.ORDER_IDENTITY, Capability.SIMULATED_FILLS})


FEATURES = tuple(f"{side}_{level}_{field}" for side in ("bid", "ask")
                 for level in range(5) for field in ("distance_ticks", "quantity")) + (
    "microprice_distance_ticks", "imbalance", "remaining_fraction", "time_fraction",
    "reserved_fraction", "available_fraction", "resting_fraction", "cancel_pending_fraction",
    "own_price_distance_ticks", "oldest_order_age_fraction", "risk_halted", "opposite_liquidity")


@dataclass(frozen=True)
class ObservationNormalization:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    clip: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", tuple(self.mean))
        object.__setattr__(self, "scale", tuple(self.scale))
        if (len(self.mean) != len(FEATURES) or len(self.scale) != len(FEATURES)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or min(self.scale) <= 0 or not np.isfinite(self.clip) or self.clip <= 0):
            raise ValueError("normalization requires 32 finite means, positive scales and clip")

    def apply(self, raw: np.ndarray) -> np.ndarray:
        if raw.shape != (len(FEATURES),) or not np.isfinite(raw).all():
            raise ValueError("nonfinite or incompatible current-state observation")
        return np.clip((raw - np.asarray(self.mean)) / np.asarray(self.scale),
                       -self.clip, self.clip).astype(np.float32)

    def as_dict(self) -> dict:
        return asdict(self)


def normalization(value: ObservationNormalization | Mapping[str, Any] | None) -> ObservationNormalization | None:
    return value if value is None or isinstance(value, ObservationNormalization) else ObservationNormalization(**value)


def fit_normalization(observations: list[np.ndarray]) -> dict:
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(FEATURES) or not np.isfinite(values).all():
        raise ValueError("at least two finite observations with the registered feature schema required")
    # One raw unit floors constant/near-constant features; no arbitrary market
    # depth or price scale is inserted. Natural ratios retain meaningful units.
    fitted = ObservationNormalization(tuple(values.mean(axis=0)), tuple(np.maximum(values.std(axis=0), 1.0)))
    return {"normalization": fitted.as_dict(), "features": FEATURES, "sample_count": len(values),
            "observations_sha256": hashlib.sha256(json.dumps(values.tolist(), separators=(",", ":"),
                                                              allow_nan=False).encode()).hexdigest(),
            "method": "training-only population mean/std; scale floor one raw unit; clip at +/-10; frozen during evaluation"}
