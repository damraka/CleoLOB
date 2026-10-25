"""Memory-mapped and streaming replay (M44).

``stream_windows`` iterates registered execution windows of memory-mapped day frames (``data.columnar``) and yields
each window as numpy views - the day is never loaded; resident memory is bounded by the windows touched.
``stream_events`` replays a canonical event Parquet file row group by row group through the L2 reconstruction,
yielding the touch at chosen sampling times, again with bounded memory. ``to_event_frame`` converts a window into
the v0.8 ``EventFrame`` used by the generator and replay code (copying only that window).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..data import columnar, ingest
from ..data.reconstruction import l2

WINDOW_STARTS_S = tuple(h * 3600 + 600 for h in range(24))
WARMUP_S, HORIZON_S = 30, 300


def window_bins(start_s: int, *, warmup_s: int = WARMUP_S, horizon_s: int = HORIZON_S, bin_ms: int = 100):
    return (start_s - warmup_s) * 1000 // bin_ms, (start_s + horizon_s) * 1000 // bin_ms


def stream_windows(path: Path, starts=WINDOW_STARTS_S) -> Iterator[tuple[int, dict]]:
    f = columnar.open_frames(path)
    for s in starts:
        a, b = window_bins(s)
        yield s, f.window(a, b)


def to_event_frame(w: dict, *, tick: float):
    from ...v08.market.frame import EventFrame, features
    n = len(w["valid"])
    bid, ask = np.asarray(w["bid_px"]), np.asarray(w["ask_px"])
    depth = np.asarray(w["depth"], float)
    mid = np.asarray(w["mid"], float)
    state = np.stack([features(int(bid[i]), int(ask[i]), depth[i, 0], depth[i, 1], [mid[max(i - 10, 0)], mid[i]])
                      for i in range(n)]) if n else np.zeros((0, 6))
    return EventFrame(np.asarray(w["counts"], np.int64), state, bid.copy(), ask.copy(), depth.copy(),
                      np.asarray(w["volume"], float).copy(), mid.copy(), np.asarray(w["valid"], bool).copy(),
                      samples={}, meta={"source": "historical", "tick": tick})


def stream_events(path: Path, sample_every_ns: int) -> Iterator[tuple[int, int | None, int | None]]:
    """(sample time, best bid, best ask) every ``sample_every_ns`` of exchange time, streaming row groups."""
    book = l2.DenseBook()
    next_t = None
    for ts, _seq, snap, m in l2.messages(ingest.iter_events(Path(path))):
        if next_t is None:
            next_t = ts - ts % sample_every_ns + sample_every_ns
        while ts >= next_t:
            yield next_t, book.best(1), book.best(-1)
            next_t += sample_every_ns
        l2.apply(book, m, snap)


__all__ = ["stream_windows", "to_event_frame", "stream_events", "window_bins", "WINDOW_STARTS_S"]
