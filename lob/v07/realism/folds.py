"""Blocked, purged and embargoed temporal folds (workstream 59).

Market observations are serially dependent, so random K-fold splits leak information. ``purged_folds``
cuts ``n`` time-ordered windows into ``k`` contiguous blocks; each block is a test fold, and training uses
every other window except ``purge`` windows before the test block and ``embargo`` windows after it.
Ordinary random cross-validation is not used anywhere in v0.7.
"""
from __future__ import annotations

import numpy as np


def purged_folds(n: int, k: int, *, purge: int = 10, embargo: int = 10) -> list[tuple[np.ndarray, np.ndarray]]:
    if k < 2 or n < 2 * k:
        raise ValueError("need k >= 2 and at least two windows per fold")
    edges = np.linspace(0, n, k + 1).astype(int)
    folds = []
    for a, b in zip(edges[:-1], edges[1:]):
        test = np.arange(a, b)
        keep = np.ones(n, bool)
        keep[max(0, a - purge):min(n, b + embargo)] = False
        folds.append((np.flatnonzero(keep), test))
    return folds
