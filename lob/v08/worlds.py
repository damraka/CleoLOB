"""Plausible market worlds and registered world sets.

A ``World`` is a generator, a book model, a latency profile and labels
(generator family, posterior sample id, regime id, scenario label). A
``WorldSet`` is an ordered, weighted collection with provenance; robust
optimization, uncertainty decomposition and transfer studies draw their worlds
only from registered sets (never from ad hoc choices after outcomes are seen).

Scenario labels: ``CALIBRATED`` (point fit), ``POSTERIOR_SAMPLE``,
``MODEL_AVERAGE``, ``SYNTHETIC_TRUTH`` (a known data-generating world in a
recovery or misspecification benchmark), ``STRESS`` (a structured stress), and
``HOLDOUT`` (sealed synthetic holdout).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json

import numpy as np

from ..config import canonical_json
from .execution.simulator import LatencyProfile
from .generators.base import MarketGenerator
from .market.sim import BookModel

LABELS = ("CALIBRATED", "POSTERIOR_SAMPLE", "MODEL_AVERAGE", "SYNTHETIC_TRUTH", "STRESS", "HOLDOUT", "EXPLORATORY")


@dataclass
class World:
    world_id: str
    generator: MarketGenerator
    book: BookModel
    latency: LatencyProfile | None = None
    posterior_sample_id: int | None = None
    regime_id: str | None = None
    label: str = "CALIBRATED"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"unknown world label {self.label!r}")

    @property
    def family(self) -> str:
        return self.generator.name

    def to_dict(self) -> dict:
        return {"world_id": self.world_id, "generator": self.generator.to_dict(), "book": self.book.to_dict(),
                "latency": self.latency.to_dict() if self.latency else None,
                "posterior_sample_id": self.posterior_sample_id, "regime_id": self.regime_id, "label": self.label,
                "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> World:
        return cls(d["world_id"], MarketGenerator.from_dict(d["generator"]), BookModel.from_dict(d["book"]),
                   LatencyProfile.from_dict(d["latency"]) if d.get("latency") else None, d.get("posterior_sample_id"),
                   d.get("regime_id"), d.get("label", "CALIBRATED"), d.get("meta", {}))

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()

    def variant(self, world_id: str, **changes) -> World:
        book_changes = changes.pop("book_changes", None)
        out = replace(self, world_id=world_id, **changes)
        if book_changes:
            out = replace(out, book=out.book.replace(**book_changes))
            out.generator = _with_book(out.generator, out.book)
        return out


def _with_book(generator: MarketGenerator, book: BookModel) -> MarketGenerator:
    d = generator.to_dict()
    d["book"] = book.to_dict()
    return MarketGenerator.from_dict(d)


@dataclass
class WorldSet:
    name: str
    worlds: list[World]
    weights: list[float] | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [w.world_id for w in self.worlds]
        if len(set(ids)) != len(ids):
            raise ValueError("world ids must be unique")
        if self.weights is None:
            self.weights = [1.0 / len(self.worlds)] * len(self.worlds)
        w = np.asarray(self.weights, float)
        if len(w) != len(self.worlds) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be nonnegative, one per world")
        self.weights = (w / w.sum()).tolist()

    def __len__(self) -> int:
        return len(self.worlds)

    def __iter__(self):
        return iter(self.worlds)

    def to_dict(self) -> dict:
        return {"name": self.name, "weights": self.weights, "provenance": self.provenance,
                "worlds": [w.to_dict() for w in self.worlds]}

    @classmethod
    def from_dict(cls, d: dict) -> WorldSet:
        return cls(d["name"], [World.from_dict(w) for w in d["worlds"]], d["weights"], d.get("provenance", {}))

    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()

    def families(self) -> list[str]:
        return sorted({w.family for w in self.worlds})


def from_generator(generator: MarketGenerator, *, world_id: str = "nominal", label: str = "CALIBRATED",
                   latency: LatencyProfile | None = None, **meta) -> World:
    return World(world_id, generator, generator.book, latency, label=label, meta=meta)


def from_posterior(generator: MarketGenerator, particles: list[dict], weights: list[float], *, n: int, seed: int,
                   prefix: str) -> WorldSet:
    """``n`` worlds drawn (systematic resampling, seeded) from a weighted posterior over flattened parameters."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    rng = np.random.default_rng(seed)
    positions = (rng.random() + np.arange(n)) / n
    idx = np.minimum(np.searchsorted(np.cumsum(w), positions), len(w) - 1)
    worlds = []
    for k, i in enumerate(idx):
        g = generator.with_params(particles[int(i)])
        worlds.append(World(f"{prefix}-p{k:03d}", g, g.book, posterior_sample_id=int(i), label="POSTERIOR_SAMPLE",
                            meta={"theta": particles[int(i)]}))
    return WorldSet(prefix, worlds, None, {"source": "posterior", "particles": len(particles), "draws": n,
                                           "seed": seed})


def save(world_set: WorldSet, path) -> None:
    from pathlib import Path
    Path(path).write_text(json.dumps(world_set.to_dict(), sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def load(path) -> WorldSet:
    from pathlib import Path
    return WorldSet.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
