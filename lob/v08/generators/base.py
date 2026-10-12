"""Generator plugin interface, registry and serialization (v0.8 M8, brief section 9).

A v0.8 market generator decides how many events of each of the six marks occur
in the next 100 ms bin, given the book features and its own state. Where those
events land is decided by the shared ``BookModel``. Every generator exposes:

* ``fit(frame)`` -> fitted copy;
* ``initial_state(simulate)``, ``expected(state, x)``, ``draw(state, x, rng)``,
  ``update(state, realized, x, own=None)`` (the bin-level process);
* ``loglik(frame)`` -> one-step-ahead predictive log-likelihood per bin, filtering
  on the realized counts and states (``score`` summarizes it);
* ``simulate(n_bins, seed)`` / ``sample`` and ``posterior_predictive``;
* ``to_dict`` / ``from_dict`` (JSON round trip, registry-dispatched);
* ``capabilities`` and ``parameter_space()`` (the bounded box SBI may search).

Custom generators register with ``@register`` or through the
``cleolob.generators`` entry-point group; the CLI never needs editing.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np

from ..market.frame import EventFrame, MARKS
from ..market.sim import BookModel, simulate as _simulate

REGISTRY: dict[str, type] = {}
ENTRY_POINT_GROUP = "cleolob.generators"
K = len(MARKS)
CAPABILITIES = ("count_process", "state_dependent", "self_exciting", "latent_regimes", "queue_reactive", "mixture",
                "agent_ecology", "endogenous_response", "closed_form_loglik", "filtered_loglik")


def register(cls):
    if not getattr(cls, "name", None) or cls.name in REGISTRY:
        raise ValueError(f"generator name missing or duplicate: {getattr(cls, 'name', None)!r}")
    unknown = set(cls.capabilities) - set(CAPABILITIES)
    if unknown:
        raise ValueError(f"{cls.name}: unknown capabilities {sorted(unknown)}")
    REGISTRY[cls.name] = cls
    return cls


def load_plugins() -> list[str]:
    """Register third-party generators declared under the ``cleolob.generators`` entry-point group."""
    from importlib.metadata import entry_points
    loaded = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        cls = ep.load()
        if cls.name not in REGISTRY:
            register(cls)
            loaded.append(cls.name)
    return loaded


def get(name: str) -> type:
    if name not in REGISTRY:
        raise KeyError(f"unknown generator {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


_LOG_FACTORIAL = np.asarray([math.lgamma(k + 1.0) for k in range(4097)])


def poisson_logpmf(y, mu) -> np.ndarray:
    y = np.asarray(y)
    mu = np.maximum(np.asarray(mu, float), 1e-9)
    return y * np.log(mu) - mu - _LOG_FACTORIAL[np.minimum(y.astype(np.int64), 4096)]


class MarketGenerator:
    name = "base"
    version = "1"
    capabilities: frozenset = frozenset({"count_process"})

    def __init__(self, params: dict | None = None, book: BookModel | None = None) -> None:
        self.params = deepcopy(params) if params is not None else self.default_params()
        self.book = book or BookModel()
        self.fit_info: dict = {}

    # ------------------------------------------------------------------ process (override)
    @classmethod
    def default_params(cls) -> dict:
        return {"mu": [0.5, 0.5, 2.0, 2.0, 1.9, 1.9]}

    def initial_state(self, simulate: bool = False) -> Any:
        return None

    def expected(self, state: Any, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def draw(self, state: Any, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.poisson(np.clip(self.expected(state, x), 0.0, 60.0))

    def update(self, state: Any, realized: np.ndarray, x: np.ndarray, own: np.ndarray | None = None) -> Any:
        return state

    def fit(self, frame: EventFrame) -> MarketGenerator:
        raise NotImplementedError

    # ------------------------------------------------------------------ scoring
    def loglik(self, frame: EventFrame) -> np.ndarray:
        """One-step-ahead predictive log-likelihood per valid bin (filtering on realized history)."""
        state = self.initial_state(simulate=False)
        out = np.full(len(frame), np.nan)
        for i in range(len(frame)):
            x = frame.state[i]
            y = frame.counts[i]
            if frame.valid[i]:
                out[i] = float(np.sum(poisson_logpmf(y, self.expected(state, x))))
            state = self.update(state, y if frame.valid[i] else np.zeros(K, int), x)
        return out

    def score(self, frame: EventFrame) -> dict:
        ll = self.loglik(frame)
        ok = np.isfinite(ll)
        return {"loglik": float(ll[ok].sum()), "loglik_per_bin": float(ll[ok].mean()) if ok.any() else None,
                "bins": int(ok.sum()), "n_params": self.n_params()}

    def n_params(self) -> int:
        def count(v) -> int:
            if isinstance(v, dict):
                return sum(count(x) for x in v.values())
            if isinstance(v, (list, tuple)):
                return sum(count(x) for x in v)
            return 1 if isinstance(v, (int, float)) and not isinstance(v, bool) else 0
        return count(self.params)

    # ------------------------------------------------------------------ simulation
    def simulate(self, n_bins: int, seed: int, *, warmup_bins: int = 300, book: BookModel | None = None) -> EventFrame:
        return _simulate(self, book or self.book, n_bins, seed, warmup_bins=warmup_bins)

    sample = simulate

    def posterior_predictive(self, thetas: list[dict], n_bins: int, seeds: list[int]) -> list[EventFrame]:
        return [self.with_params(theta).simulate(n_bins, seed) for theta, seed in zip(thetas, seeds)]

    # ------------------------------------------------------------------ parameters
    def parameter_space(self) -> dict[str, tuple[float, float]]:
        """Bounded box of scalar parameters SBI may search (flattened names)."""
        return {}

    def get_flat(self) -> dict[str, float]:
        out = {}
        for key in self.parameter_space():
            target: Any = self.params
            for part in key.split("."):
                target = target[int(part)] if isinstance(target, list) else target[part]
            out[key] = float(target)
        return out

    def with_params(self, theta: dict) -> MarketGenerator:
        """Copy with flattened parameters replaced (``mu.0``, ``book.refill_mean``, ...)."""
        new = deepcopy(self)
        book_changes = {}
        for key, value in theta.items():
            if key.startswith("book."):
                book_changes[key[5:]] = value
                continue
            parts = key.split(".")
            target: Any = new.params
            for part in parts[:-1]:
                target = target[int(part)] if isinstance(target, list) else target[part]
            last = parts[-1]
            if isinstance(target, list):
                target[int(last)] = float(value)
            else:
                target[last] = float(value)
        if book_changes:
            new.book = new.book.replace(**book_changes)
        new._refresh()
        return new

    def _refresh(self) -> None:
        """Recompute cached arrays after parameter changes."""

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict:
        return {"generator": self.name, "version": self.version, "capabilities": sorted(self.capabilities),
                "params": json_ready(self.params), "book": self.book.to_dict(), "fit_info": json_ready(self.fit_info)}

    @staticmethod
    def from_dict(d: dict) -> MarketGenerator:
        cls = get(d["generator"])
        if str(d.get("version")) != str(cls.version):
            raise ValueError(f"{cls.name}: serialized version {d.get('version')} != {cls.version}")
        return cls._restore(d)

    @classmethod
    def _restore(cls, d: dict) -> MarketGenerator:
        g = cls(d["params"], BookModel.from_dict(d["book"]))
        g.fit_info = d.get("fit_info", {})
        g._refresh()
        return g

    def describe(self) -> dict:
        return {"generator": self.name, "version": self.version, "capabilities": sorted(self.capabilities),
                "n_params": self.n_params()}


def json_ready(value):
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value
