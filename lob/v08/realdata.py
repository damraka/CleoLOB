"""Ledger-gated access to historical tapes and conversion to EventFrames.

Access to every real dataset is recorded in the v0.8 ledger in stages
(``downloaded`` -> ``opened`` -> ``parsed`` -> ``evaluated``) with the dataset role
and the reading design. Consumed v0.5-v0.7 days are read from local files only.

**Acquisition rule for fresh holdouts** (registered): before any request for a
holdout file, an availability probe fetches the response headers of an
*already-consumed* control file from the same provider. If the probe fails, an
``attempt`` with outcome NOT_AVAILABLE is ledgered, no holdout byte is ever
requested and the holdout stays fresh. Only after a successful probe is the
``downloaded`` stage ledgered and the holdout requested.

Tapes come from the frozen v0.7 builder (``lob.v07.data.tape.cached_build``); a
tape is lot-scaled by the registered rule (lot = median touch size / 20 on that
dataset) and cut into registered windows that become ``EventFrame`` objects.
Windows with less than 95% valid bins are excluded and listed, never dropped
silently.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ..data_download import download_tardis_sample
from ..experiments.registry import sha256_file
from ..v07.data.access import SEARCH_DIRS, file_name
from ..v07.data.schema import venue_spec
from ..v07.data.tape import cached_build
from ..v07.generators.events import extract
from . import protocol as pr
from .market.frame import EventFrame, VISIBLE

DATA_DIR = "data/v08/tardis"
CACHE_DIR = "data/v08/cache"
CONTROL_URL = "https://datasets.tardis.dev/v1/deribit/trades/2021/01/01/ETH-PERPETUAL.csv.gz"
TOUCH_LOTS = 20.0
MIN_VALID = 0.95
EXTRACTOR_VERSION = "v08-frames-1"
_TYPES = {"incremental_book_L2": "l2", "trades": "trades"}


class DataNotAvailable(RuntimeError):
    """A registered dataset could not be obtained; dependent results are NOT_AVAILABLE."""


def probe(url: str = CONTROL_URL, timeout: float = 20.0) -> dict:
    """Headers-only availability probe on an already-consumed control file (never a holdout)."""
    from urllib.request import Request, urlopen
    request = Request(url, method="GET", headers={"Range": "bytes=0-0", "User-Agent": "CleoLOB/0.8",
                                                  "Accept-Encoding": "identity"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return {"ok": response.status in (200, 206), "status": int(response.status)}
    except Exception as exc:   # any network failure means NOT_AVAILABLE
        return {"ok": False, "status": getattr(exc, "code", None), "error": f"{type(exc).__name__}: {exc}"}


def dataset_files(declaration: dict, root: Path) -> dict[str, Path]:
    files = {}
    for data_type in declaration["data_types"]:
        name = file_name(declaration["venue"], data_type, declaration["date"], declaration["instrument"])
        found = [root / d / name for d in (*SEARCH_DIRS, DATA_DIR) if (root / d / name).is_file()]
        files[_TYPES[data_type]] = found[0] if found else root / DATA_DIR / name
    return files


def _record(root: Path, protocol: dict, d: dict, design: str, use: str, stage: str, reason: str, **payload) -> None:
    pr.append_event(root / pr.LEDGER_PATH, "access", dataset=d["id"], role=d["role"], design=design, reason=reason,
                    protocol=protocol, root=root, payload={"use": use, "stage": stage, **payload})


def acquire(dataset_id: str, *, use: str, design: str, purpose: str, root: Path, prober=probe,
            downloader=download_tardis_sample) -> dict[str, Path]:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    d = pr.dataset_declaration(protocol, dataset_id)
    if use not in pr.ROLE_USES[d["role"]] or use == "download":
        raise pr.ProtocolError(f"{dataset_id}: use {use!r} is not permitted for role {d['role']}")
    files = dataset_files(d, root)
    missing = sorted(k for k, p in files.items() if not p.is_file())
    holdout = d["role"] in pr.HOLDOUT_ROLES
    if holdout:
        state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), protocol)
        sealed = state.designs.get(design)
        if sealed is None or dataset_id not in sealed.get("reads", []):
            raise pr.ProtocolError(f"{dataset_id}: holdout access needs a sealed design that reads it")
        if missing:
            check = prober()
            if not check["ok"]:
                pr.append_event(root / pr.LEDGER_PATH, "attempt", design=design, dataset=dataset_id, role=d["role"],
                                reason="availability probe on a consumed control file failed; no holdout request made",
                                protocol=protocol, root=root,
                                payload={"outcome": "NOT_AVAILABLE", "directory": None,
                                         "no_directory_reason": "nothing was downloaded",
                                         "note": "dependent hypotheses are NOT_AVAILABLE; the holdout stays fresh",
                                         "probe": check})
                raise DataNotAvailable(f"{dataset_id} NOT_AVAILABLE: provider probe failed ({check})")
    elif missing:
        raise DataNotAvailable(f"{dataset_id}: consumed dataset files are not present locally: {missing}")
    _record(root, protocol, d, design, "download" if holdout else use, "downloaded",
            f"{'download start' if missing else 'local files used'}: {purpose}", files=missing)
    for kind in missing:
        inverse = {v: k for k, v in _TYPES.items()}
        try:
            downloader(d["venue"], d["instrument"], d["date"], inverse[kind], root / DATA_DIR,
                       max_bytes=2 * 1024**3, max_expanded_bytes=32 * 1024**3)
        except (OSError, ValueError) as exc:
            pr.append_event(root / pr.LEDGER_PATH, "attempt", design=design, dataset=dataset_id, role=d["role"],
                            reason="download failed; no substitution", protocol=protocol, root=root,
                            payload={"outcome": "NOT_AVAILABLE", "directory": DATA_DIR,
                                     "note": "dependent hypotheses are NOT_AVAILABLE", "error": str(exc)})
            raise DataNotAvailable(f"{dataset_id} NOT_AVAILABLE: {exc}") from exc
    files = dataset_files(d, root)
    hashes = {p.name: sha256_file(p) for p in files.values()}
    _record(root, protocol, d, design, use, "opened", f"source bytes hashed: {purpose}", source_sha256=hashes)
    return files


def load_frames(dataset_id: str, *, use: str, design: str, purpose: str, root: Path, windows: list[tuple[float, float]],
                prober=probe) -> tuple[list[EventFrame], dict]:
    """Registered windows of a dataset as EventFrames (cached), with exclusions and scale metadata."""
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    d = pr.dataset_declaration(protocol, dataset_id)
    files = acquire(dataset_id, use=use, design=design, purpose=purpose, root=root, prober=prober)
    spec = venue_spec(d["venue"], d["instrument"])
    key = hashlib.sha256("|".join([sha256_file(files["l2"]), sha256_file(files["trades"]), EXTRACTOR_VERSION,
                                   json.dumps(windows)]).encode()).hexdigest()[:24]
    cache = root / CACHE_DIR / f"{key}.npz"
    if cache.is_file():
        stored = np.load(cache, allow_pickle=False)
        meta = json.loads(str(stored["meta"]))
        frames = [EventFrame.from_arrays({k[len(f"w{i}__"):]: stored[k] for k in stored.files
                                          if k.startswith(f"w{i}__")}, meta["frames"][i])
                  for i in range(len(meta["frames"]))]
    else:
        tape, quality = cached_build(files, tick=float(spec.tick), root=root)
        valid = tape.valid
        touch = float(np.median(np.r_[tape.bq[valid, 0], tape.aq[valid, 0]]))
        lot = touch / TOUCH_LOTS
        scaled = tape.rescaled(lot)
        frames, excluded = [], []
        for start, seconds in windows:
            block = scaled.block(scaled.t[0] + start, scaled.t[0] + start + seconds)
            if len(block.t) < 50 or block.valid.mean() < MIN_VALID:
                excluded.append({"start_s": start, "seconds": seconds, "valid_fraction":
                                 float(block.valid.mean()) if len(block.t) else 0.0})
                continue
            f = frame_from_tape(block)
            f.meta.update({"dataset": dataset_id, "window_start_s": start, "lot_native": lot})
            frames.append(f)
        meta = {"dataset": dataset_id, "lot_native": lot, "lot_rule": f"median touch size / {TOUCH_LOTS}",
                "tick": float(spec.tick), "quality": quality["status"], "excluded_windows": excluded,
                "frames": [f.meta for f in frames], "extractor": EXTRACTOR_VERSION}
        cache.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"w{i}__{k}": v for i, f in enumerate(frames) for k, v in f.to_arrays().items()}
        tmp = cache.with_suffix(".part.npz")
        np.savez_compressed(tmp, meta=json.dumps(meta), **arrays)
        tmp.replace(cache)
    _record(root, protocol, d, design, use, "parsed", f"frames built: {purpose}", windows=len(frames),
            quality=meta["quality"])
    meta = {k: v for k, v in meta.items() if k != "frames"}
    return frames, meta


def mark_evaluated(dataset_id: str, *, use: str, design: str, run: str, root: Path) -> None:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    d = pr.dataset_declaration(protocol, dataset_id)
    _record(root, protocol, d, design, use, "evaluated", f"results computed in {run}", run=run)


def frame_from_tape(tape) -> EventFrame:
    """EventFrame from a lot-scaled v0.6 ``BookTape`` block (aggregate L2 + grouped prints)."""
    b = extract(tape)
    n = len(b.counts)
    tick = tape.tick
    valid = b.valid.copy()
    bid_px = np.zeros(n, dtype=np.int64)
    ask_px = np.zeros(n, dtype=np.int64)
    depth = np.zeros((n, 2, VISIBLE))
    for i in range(n):
        if not tape.valid[i]:
            continue
        bid_px[i] = int(round(tape.bp[i, 0] / tick))
        ask_px[i] = int(round(tape.ap[i, 0] / tick))
        for side, prices, sizes, sign in ((0, tape.bp[i], tape.bq[i], 1), (1, tape.ap[i], tape.aq[i], -1)):
            touch = bid_px[i] if side == 0 else ask_px[i]
            for p, q in zip(prices, sizes):
                if not math.isfinite(p):
                    continue
                k = (touch - int(round(p / tick))) * sign
                if 0 <= k < VISIBLE:
                    depth[i, side, k] = q
    mid_ticks = np.where(tape.valid, (tape.bp[:, 0] + tape.ap[:, 0]) / (2 * tick), np.nan)
    mid = _ffill(mid_ticks[: n + 1])
    ret10 = np.zeros(n)
    ret10[10:] = np.abs(mid[10:n] - mid[: n - 10])
    state = np.column_stack([b.state[:, 0], b.state[:, 1], b.state[:, 2], tape.bq[:n, 0], tape.aq[:n, 0], ret10])
    edges = np.searchsorted(tape.trade_t, tape.t[: n + 1], side="right")
    volume = np.zeros((n, 2))
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        q, s = tape.trade_q[lo:hi], tape.trade_s[lo:hi]
        volume[i, 0], volume[i, 1] = float(q[s > 0].sum()), float(q[s < 0].sum())
    samples = {"market_size": np.r_[b.sizes["market_buy"], b.sizes["market_sell"]]}
    adds = np.vstack([b.offsets[k] for k in b.offsets if k.startswith("add") and len(b.offsets[k])] or [np.zeros((0, 2))])
    cancels = np.vstack([b.offsets[k] for k in b.offsets if k.startswith("cancel") and len(b.offsets[k])]
                        or [np.zeros((0, 2))])
    samples.update({"add_level": adds[:, 0], "add_size": adds[:, 1], "cancel_level": np.maximum(cancels[:, 0], 0),
                    "cancel_frac": cancels[:, 1],
                    "trade_t": tape.trade_t - tape.t[0], "trade_px": tape.trade_p / tick, "trade_q": tape.trade_q,
                    "trade_s": tape.trade_s})
    mid_valid = np.isfinite(mid_ticks[:n]).all() if n else True
    return EventFrame(b.counts.astype(np.int64), np.nan_to_num(state), bid_px, ask_px, depth, volume, mid, valid,
                      samples=samples, meta={"source": "historical", "tick": tick, "tick_bps": float(
                          1e4 / np.nanmedian(mid_ticks)) if np.isfinite(mid_ticks).any() else None,
                          "mid_complete": bool(mid_valid)})


def _ffill(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, float).copy()
    last = np.nan
    for i in range(len(out)):
        if np.isfinite(out[i]):
            last = out[i]
        else:
            out[i] = last
    first = np.nonzero(np.isfinite(out))[0]
    if len(first):
        out[: first[0]] = out[first[0]]
    return np.nan_to_num(out)


def registered_windows(count: int = 24, seconds: float = 600.0, offset: float = 900.0) -> list[tuple[float, float]]:
    """Hourly windows: start at ``h * 3600 + offset`` for ``h < count``."""
    return [(h * 3600.0 + offset, seconds) for h in range(count)]
