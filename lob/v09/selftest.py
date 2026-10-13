"""Pure task functions used by the runner self-checks, the smoke and the worker-count determinism tests."""
from __future__ import annotations

import numpy as np


def draw(seed: int, n: int = 64) -> dict:
    """A seeded computation whose result depends only on its arguments."""
    x = np.random.default_rng(seed).standard_normal(n)
    return {"seed": seed, "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "max": float(x.max())}


def allocate(seed: int, mb: int) -> dict:
    """Hold ``mb`` MiB briefly (resource-contract tests)."""
    block = np.ones(int(mb) * 131_072, dtype=np.float64)
    return {"seed": seed, "sum": float(block.sum())}
