"""Stable typed public SDK of CleoLOB v1 (M140): the objects users and studies exchange.

Semver contract (``docs/v1-api-reference.md``): every name in ``__all__`` and every field below is stable for the
1.x series; additions are allowed, removals or semantic changes need 2.0. The objects are thin, validated,
serializable wrappers over the engine/data/evidence modules, so a user can work without knowing the internal layout:

* ``Dataset`` - one registered dataset (stream, date, role, capability, raw hashes) and its canonical events;
* ``MarketEvent`` / ``OrderBook`` - a canonical event and a reconstructed L2 book view;
* ``Generator`` - protocol every generative market model implements (``fit`` / ``sample`` / ``loglik``);
* ``MarketTwin`` - a fidelity-tagged simulation environment (``F0``-``F6``) with its claim permissions;
* ``Policy`` and ``ExecutionMandate`` - an execution strategy and the order it must execute;
* ``Scenario`` - a named market condition (twin + parameters + seed block);
* ``CalibrationResult``, ``StudyDesign``, ``Hypothesis``, ``EvidenceArtifact`` - the research objects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np

from .data.schema import EVENT_DTYPE, Capability, EventKind

FIDELITIES = ("F0", "F1", "F2", "F3", "F4", "F5", "F6")
FIDELITY_MEANING = {
    "F0": "observational evidence (no intervention)",
    "F1": "exact-observation historical replay (no agent)",
    "F2": "bounded counterfactual historical replay (interval over a registered assumption grid)",
    "F3": "hybrid historical twin (historical backbone + modelled response)",
    "F4": "calibrated classical generative market",
    "F5": "neural generative market",
    "F6": "strategic multi-agent market",
}
TERMINAL = ("ESTABLISHED", "FAILED", "NOT_ESTABLISHED", "INCONCLUSIVE", "NOT_AVAILABLE", "NOT_EVALUABLE")


class SDKError(ValueError):
    pass


@dataclass(frozen=True)
class MarketEvent:
    ts_exchange: int
    kind: str
    side: int
    price: int
    size: float
    order_id: int = -1
    seq: int = -1

    @classmethod
    def from_row(cls, row) -> MarketEvent:
        return cls(int(row["ts_exchange"]), EventKind(int(row["kind"])).name, int(row["side"]), int(row["price"]),
                   float(row["size"]), int(row["order_id"]), int(row["seq"]))


@dataclass(frozen=True)
class Dataset:
    id: str
    stream: str
    date: str
    role: str
    capability: Capability
    path: Path | None = None              # directory with book.parquet / trades.parquet / meta.json
    raw_sha256: dict = field(default_factory=dict)

    def events(self, kind: str = "book") -> Iterator[np.ndarray]:
        from .data import ingest
        if self.path is None:
            raise SDKError(f"{self.id} has no local events")
        yield from ingest.iter_events(Path(self.path) / f"{kind}.parquet")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["path"] = None if self.path is None else Path(self.path).as_posix()
        return d


class OrderBook:
    """Reconstructed L2 book (dense) with a read-only view API."""

    def __init__(self) -> None:
        from .data.reconstruction.l2 import DenseBook
        self._book = DenseBook()

    @classmethod
    def replay(cls, events) -> OrderBook:
        from .data.reconstruction import l2
        b = cls()
        for _ts, _seq, snap, m in l2.messages(events):
            l2.apply(b._book, m, snap)
        return b

    def best(self) -> tuple[int | None, int | None]:
        return self._book.best(1), self._book.best(-1)

    def depth(self, levels: int = 10) -> dict:
        bp, bq = self._book.levels(1, levels)
        ap, aq = self._book.levels(-1, levels)
        return {"bids": list(zip(bp.tolist(), bq.tolist())), "asks": list(zip(ap.tolist(), aq.tolist()))}


@runtime_checkable
class Generator(Protocol):
    """Every generative market model: fitted on frames, samples frames, scores held-out data."""
    name: str

    def fit(self, frame) -> Generator: ...

    def sample(self, n_bins: int, seed: int): ...

    def loglik(self, frame) -> np.ndarray: ...


@dataclass(frozen=True)
class ExecutionMandate:
    side: int                              # +1 buy, -1 sell
    quantity: int                          # lots
    horizon_s: float
    decision_interval_s: float = 5.0
    completion: str = "hard"               # hard: remainder sent at the horizon; soft: may stay unfilled
    name: str = "mandate"

    def __post_init__(self) -> None:
        if self.side not in (1, -1) or self.quantity <= 0 or self.horizon_s <= 0 or self.decision_interval_s <= 0:
            raise SDKError("invalid mandate")
        if self.completion not in ("hard", "soft"):
            raise SDKError("completion must be hard or soft")


@runtime_checkable
class Policy(Protocol):
    name: str

    def reset(self, mandate: ExecutionMandate) -> None: ...

    def act(self, observation: dict) -> list[dict]: ...


@dataclass(frozen=True)
class MarketTwin:
    name: str
    fidelity: str
    capabilities: dict
    description: str = ""

    def __post_init__(self) -> None:
        if self.fidelity not in FIDELITIES:
            raise SDKError(f"unknown fidelity {self.fidelity}")

    def permits(self, claim: str) -> bool:
        from .twins.permissions import permitted
        return permitted(self.fidelity, claim, self.capabilities)


@dataclass(frozen=True)
class Scenario:
    name: str
    twin: str
    parameters: dict = field(default_factory=dict)
    seeds: tuple[int, ...] = ()


@dataclass(frozen=True)
class CalibrationResult:
    model: str
    parameters: dict
    objective: float
    train: tuple[str, ...]
    test: tuple[str, ...]
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.train) & set(self.test):
            raise SDKError("calibration train and test datasets overlap")


@dataclass(frozen=True)
class Hypothesis:
    id: str
    title: str
    endpoint: str
    direction: str                         # "greater" | "less" | "noninferior" | "equivalent"
    threshold: float
    unit: str
    capability: tuple[str, ...]
    validation_level: str
    status: str = "CANDIDATE"

    def __post_init__(self) -> None:
        if self.direction not in ("greater", "less", "noninferior", "equivalent"):
            raise SDKError("invalid direction")


@dataclass(frozen=True)
class StudyDesign:
    name: str
    study: str
    namespace: str
    hypotheses: tuple[str, ...]
    datasets: tuple[str, ...]
    analysis: str
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hypotheses"], d["datasets"] = list(self.hypotheses), list(self.datasets)
        return d


@dataclass(frozen=True)
class EvidenceArtifact:
    path: Path
    status: str
    analysis: str
    result_sha256: str

    @classmethod
    def open(cls, path: str | Path) -> EvidenceArtifact:
        from .evidence import runs as R
        p = Path(path)
        v = R.verify_run(p)
        if not v["valid"]:
            raise SDKError(f"{p}: {v['issues']}")
        st = R.read_json(p / "status.json")
        return cls(p, st["status"], st["analysis"], R.read_json(p / "checksums.json")["result.json"])

    def result(self) -> Any:
        return json.loads((self.path / "result.json").read_text(encoding="utf-8"))


def events_array(rows: list[MarketEvent]) -> np.ndarray:
    out = np.zeros(len(rows), dtype=EVENT_DTYPE)
    for i, r in enumerate(rows):
        out[i] = (r.ts_exchange, np.iinfo(np.int64).min, r.seq, EventKind[r.kind], r.side, r.price, r.size,
                  r.order_id, -1)
    return out


__all__ = ["FIDELITIES", "FIDELITY_MEANING", "TERMINAL", "SDKError", "MarketEvent", "Dataset", "OrderBook",
           "Generator", "ExecutionMandate", "Policy", "MarketTwin", "Scenario", "CalibrationResult", "Hypothesis",
           "StudyDesign", "EvidenceArtifact", "events_array"]
