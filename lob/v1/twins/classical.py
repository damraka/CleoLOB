"""F4 classical generative market twin with the corrected v1 book calibration (M65).

``fit(caches)`` builds the v0.9 training frame (registered windows of the training days), fits the corrected,
state-dependent book model (``generators.classical.book.fit_book_model`` on absolute-price-aligned events) and every
registered classical family (Poisson, Hawkes, queue-reactive, regime switching, state dependent, mixture, static
ecology) on that frame. ``fit(caches, book="v09")`` reproduces the v0.9 defective book for comparisons (H13, H30).

Execution episodes run the unchanged v0.8 execution simulator; ``corrected_market()`` swaps the market driver it
constructs (``lob.v08.execution.simulator.MarketSim``) for ``MarketSimV1`` for the duration of the block, so a
``BookModelV1`` world places improving additions with its state-dependent table. ``MarketSimV1`` falls back to the
v0.8 behaviour for v0.8 book models, so the swap never changes a v0.9-book simulation.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib

import numpy as np

from ...v09.generators import classical as G9
from ..generators.classical import book as B

FAMILIES = list(G9.FAMILIES) + list(G9.STATIC_ECOLOGY)


def v1_book(caches: list[dict], frame) -> B.BookModelV1:
    from ...v09.data.daycache import HORIZON_S, WARMUP_S, WINDOW_STARTS_S
    evs = []
    for c in caches:
        for s in WINDOW_STARTS_S:
            a, b = int((s - WARMUP_S) * 10), int((s + HORIZON_S) * 10)
            evs.append(B.infer_book_events(c["bid_px"][a:b], c["ask_px"][a:b], c["depth"][a:b], c["volume"][a:b],
                                           c["valid"][a:b]))
    ev = {k: np.concatenate([e[k] for e in evs]) for k in evs[0] if isinstance(evs[0][k], np.ndarray)}
    ev["label"] = evs[0]["label"]
    base = G9.book_model(frame)
    hazard = B.improvement_table_hazard(frame.bid_px, frame.ask_px, frame.counts, frame.valid)
    return B.fit_book_model(ev, market_sizes=frame.samples["market_size"],
                            refill_depths=frame.depth[frame.valid][:, :, 1:].ravel(), mid_ticks=base.mid_ticks,
                            table=hazard)


def fit(caches: list[dict], *, book: str = "v1", families=None) -> dict:
    frame = G9.training_frame(caches)
    bm = v1_book(caches, frame) if book == "v1" else G9.book_model(frame)
    gens = {n: G9.fit_family(n, frame, bm) for n in (families or FAMILIES)}
    return {"frame": frame, "book": bm, "generators": gens, "book_kind": book}


@contextmanager
def corrected_market():
    sim = importlib.import_module("lob.v08.execution.simulator")
    saved = sim.MarketSim
    sim.MarketSim = B.MarketSimV1
    try:
        yield
    finally:
        sim.MarketSim = saved


def simulate(generator, n_bins: int, seed: int, *, warmup_bins: int = 600):
    if isinstance(generator.book, B.BookModelV1):
        return B.simulate_v1(generator, generator.book, n_bins, seed, warmup_bins=warmup_bins)
    from ...v08.market.sim import simulate as sim08
    return sim08(generator, generator.book, n_bins, seed, warmup_bins=warmup_bins)


def synthetic_episode(generator, policy, *, qty: int, seed: int, side: int = 1) -> dict:
    from ...v09 import twin
    with corrected_market():
        return twin.synthetic_episode(generator, policy, qty=qty, seed=seed, side=side)


__all__ = ["fit", "v1_book", "simulate", "synthetic_episode", "corrected_market", "FAMILIES"]
