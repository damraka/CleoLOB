"""Bounded finite-factor design; never materialize a large Cartesian product."""
from __future__ import annotations

import math
import random
from typing import Any

from ..config import ResearchConfig, _set_dotted, canonical_json, config_hash

BUDGETS = {"tiny": 4, "small": 16, "medium": 64, "large": 256, "exhaustive": 4096}


def design_experiment(config: ResearchConfig, factors: dict[str, list[Any]], *,
                      budget: str = "tiny", seed: int = 0) -> dict[str, Any]:
    if budget not in BUDGETS:
        raise ValueError(f"budget must be one of {', '.join(BUDGETS)}")
    if not isinstance(factors, dict) or not factors or len(factors) > 32:
        raise ValueError("factors must contain 1 to 32 dotted configuration fields")
    keys = sorted(factors)
    for key in keys:
        values = factors[key]
        if not isinstance(values, list) or not values or len(values) > 1000:
            raise ValueError("each factor needs 1 to 1000 explicit values")
        if len({canonical_json(v) for v in values}) != len(values):
            raise ValueError(f"duplicate factor levels: {key}")
        if any(other.startswith(key + ".") for other in keys if key != other):
            raise ValueError("factors cannot overlap parent and child configuration fields")
    total = math.prod(len(factors[key]) for key in keys)
    if budget == "exhaustive" and total > BUDGETS[budget]:
        raise ValueError(f"exhaustive design has {total} configurations; cap is {BUDGETS[budget]}")
    count = min(total, BUDGETS[budget])
    rng = random.Random(seed)
    indices: list[int]
    if count == total:
        indices = list(range(total))
        method = "full_factorial"
    else:
        # Floyd's algorithm: uniform sampling without replacement, O(count) memory,
        # even when the theoretical lattice exceeds sys.maxsize.
        chosen: set[int] = set()
        for j in range(total - count, total):
            value = rng.randrange(j + 1)
            chosen.add(j if value in chosen else value)
        indices = sorted(chosen)
        method = "random_without_replacement"
    designs = []
    episodes = 0
    estimated_events = 0
    for index in indices:
        value = index
        overlay = {}
        data = config.model_dump(mode="json")
        for key in reversed(keys):
            options = factors[key]
            value, digit = divmod(value, len(options))
            overlay[key] = options[digit]
            _set_dotted(data, key, options[digit])
        candidate = ResearchConfig.model_validate(data)
        designs.append({"index": index, "overrides": overlay, "config_sha256": config_hash(candidate)})
        episodes += candidate.episode_count
        estimated_events += candidate.estimated_events
    return {"theoretical_configurations": total, "selected_configurations": count,
            "coverage_percent": 100 * count / total, "method": method, "budget": budget, "seed": seed,
            "episodes_planned": episodes, "estimated_events": estimated_events,
            "estimated_runtime_seconds": None, "runtime_note": "NOT ESTIMATED — measure an episode on this machine first.",
            "status": "DESIGN ONLY — NOT RUN", "designs": designs}
