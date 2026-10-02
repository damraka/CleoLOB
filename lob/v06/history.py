"""Historical tapes for v0.6: ledger-recorded access, cached extraction, development lot scale.

Every call goes through ``lob.v06.data.acquire`` so each read is recorded in the
v0.6 ledger with its role-permitted use. Tapes are cached under the exact source
hashes in the ignored ``data/v06/cache`` directory. Quantities are converted to
lots with the development depth scale (the v0.5 rule) for every dataset,
including the cross-instrument BTC holdout (a deliberate transfer test).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..experiments.registry import PROJECT_ROOT
from . import protocol as pr
from .data import acquire
from .observables import measure
from .tape import BookTape, cached_tape

TICKS = {"ETH-PERPETUAL": 0.05, "BTC-PERPETUAL": 0.5}
BLOCK_S = 600.0


def load(dataset_id: str, *, use: str, analysis: str, purpose: str, root: Path = PROJECT_ROOT,
         scale: float | None = None) -> tuple[BookTape, dict]:
    """Native tape (``scale`` None) or lot tape, plus metadata (file names, validity, events)."""
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    declaration = pr.dataset_declaration(protocol, dataset_id)
    tick = TICKS[declaration["instrument"]]
    files = acquire(dataset_id, use=use, analysis=analysis, purpose=purpose, root=root)
    tape = cached_tape(files, tick=tick, root=root)
    meta = {"dataset_id": dataset_id, "role": declaration["role"], "instrument": declaration["instrument"],
            "tick": tick, "files": {k: v.name for k, v in files.items()}, "samples": int(len(tape.t)),
            "valid_fraction": float(tape.valid.mean()), "events": int(len(tape.trade_t)),
            "duration_s": float(tape.duration)}
    return (tape if scale is None else tape.rescaled(scale)), meta


def block_raws(tape: BookTape, block_s: float = BLOCK_S, *, min_valid: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Raw measurements of every complete block; blocks below ``min_valid`` validity are skipped and listed."""
    raws, blocks = [], []
    for k, block in enumerate(tape.blocks(block_s)):
        validity = float(block.valid.mean())
        entry = {"index": k, "start_s": float(block.t[0]) if len(block.t) else None, "valid_fraction": validity}
        if validity < min_valid:
            blocks.append({**entry, "used": False})
            continue
        raws.append(measure(block))
        blocks.append({**entry, "used": True})
    return raws, blocks


def valid_fraction_summary(blocks: list[dict]) -> dict:
    values = np.asarray([b["valid_fraction"] for b in blocks]) if blocks else np.asarray([])
    return {"blocks": len(blocks), "used": int(sum(b["used"] for b in blocks)),
            "mean_valid_fraction": float(values.mean()) if len(values) else None,
            "min_valid_fraction": float(values.min()) if len(values) else None}
