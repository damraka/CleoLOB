"""Deterministic random namespaces (M8).

Seeds are derived by SHA-256 from (namespace, study, purpose, index) under a registered root seed, so they are
identical across machines, Python versions and worker counts, and disjoint between namespaces by construction:
the namespace name is part of the hashed key. Guarded namespaces must pass the sealed design name, which is also
hashed, so a confirmatory study cannot reuse an exploratory seed stream.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .namespaces import GUARDED, NAMESPACES

ROOT_SEED = 1_000_000_001


def seed(namespace: str, study: str, purpose: str = "", index: int = 0, *, design: str | None = None,
         root: int = ROOT_SEED) -> int:
    if namespace not in NAMESPACES:
        raise ValueError(f"unknown namespace {namespace}")
    if namespace in GUARDED and not design:
        raise ValueError(f"seeds in guarded namespace {namespace} need the sealed design name")
    key = f"cleolob-v1|{root}|{namespace}|{design or ''}|{study}|{purpose}|{int(index)}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") & ((1 << 63) - 1)


def seeds(namespace: str, study: str, count: int, purpose: str = "", *, design: str | None = None) -> list[int]:
    return [seed(namespace, study, purpose, i, design=design) for i in range(int(count))]


def rng(namespace: str, study: str, purpose: str = "", index: int = 0, *, design: str | None = None):
    return np.random.default_rng(seed(namespace, study, purpose, index, design=design))


__all__ = ["seed", "seeds", "rng", "ROOT_SEED"]
