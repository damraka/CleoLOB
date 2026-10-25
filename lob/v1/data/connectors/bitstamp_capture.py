"""Prospective, read-only capture of Bitstamp's public order-level (MBO) market data (M14, M15, M17).

Genuine order-level messages: ``live_orders_<pair>`` carries order ids with ``order_created``, ``order_changed``
and ``order_deleted`` events, ``live_trades_<pair>`` the executions (with maker/taker order ids), and REST
``order_book?group=2`` censuses list every resting order with its id. That makes queue position observable in
principle, unlike public L2, which is why v1 uses this source for the MBO hypotheses (H15).

Fresh by construction: segments are recorded after v0.9 closed and are never parsed by the capture. Each segment is
a write-once directory holding the exact message text with local receive clocks, a manifest and its SHA-256. A
disconnect, reconnect request or error ends the segment with an explicit ``GAP`` status; the supervisor waits and
starts a new segment, so a gap is never bridged silently. Interpretation belongs to the reconstruction layer
(``lob.v1.data.reconstruction.mbo``) and happens only for ledgered, registered uses.

Terms: Bitstamp documents public market-data use; commercial use needs a separate licence. Captures stay under the
ignored ``data/v1`` directory and are never redistributed. Requires the optional ``websockets`` package.

No order is ever sent: the only outbound messages are channel subscriptions and HTTPS GETs of the public census.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time
from urllib.request import Request, urlopen

WS_URL = "wss://ws.bitstamp.net"
REST_URL = "https://www.bitstamp.net/api/v2/order_book/{pair}/?group=2"
CHANNELS = ("live_orders_{pair}", "live_trades_{pair}", "order_book_{pair}")
MAX_REST_BYTES = 64 * 1024 * 1024
SCHEMA = "cleolob-v1-bitstamp-capture-1"
USER_AGENT = "CleoLOB-v1-research-capture (+https://github.com/damraka/CleoLOB)"
TERMS = {"source": "https://www.bitstamp.net/api/", "checked": "2026-10-25",
         "use": "non-commercial research; raw capture never redistributed"}


def _record(stream, kind: str, **fields) -> None:
    fields.update(kind=kind, recv_ns=time.time_ns(), mono_ns=time.perf_counter_ns())
    stream.write(json.dumps(fields, separators=(",", ":"), ensure_ascii=True) + "\n")


def _rest_snapshot(pair: str) -> str:
    request = Request(REST_URL.format(pair=pair), headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed public https endpoint
        body = response.read(MAX_REST_BYTES + 1)
    if len(body) > MAX_REST_BYTES:
        raise ValueError("REST census exceeds size bound")
    return body.decode("utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def segment_name(t_ns: int) -> str:
    return datetime.fromtimestamp(t_ns / 1e9, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


async def capture_segment(pair: str, seconds: float, snapshot_every: float, out: Path, *, connect=None) -> dict:
    """Record one segment into the write-once directory ``out``; returns its manifest."""
    if connect is None:
        import websockets
        connect = websockets.connect
    out.mkdir(parents=True, exist_ok=False)
    raw_path = out / "capture.jsonl.gz"
    status = {"status": "COMPLETE", "reason": None}
    started = time.time_ns()
    messages = 0
    with gzip.open(raw_path, "wt", encoding="utf-8", newline="\n") as stream:
        _record(stream, "capture_start", pair=pair, seconds=seconds, ws_url=WS_URL,
                channels=[c.format(pair=pair) for c in CHANNELS])
        try:
            async with connect(WS_URL, max_size=16 * 1024 * 1024, ping_interval=20) as ws:
                for channel in CHANNELS:
                    await ws.send(json.dumps({"event": "bts:subscribe", "data": {"channel": channel.format(pair=pair)}}))
                confirmed: set[str] = set()
                deadline = time.monotonic() + seconds
                next_snapshot = None
                loop = asyncio.get_running_loop()
                while time.monotonic() < deadline:
                    if (len(confirmed) == len(CHANNELS) and next_snapshot is not None
                            and time.monotonic() >= next_snapshot):
                        body = await loop.run_in_executor(None, _rest_snapshot, pair)
                        _record(stream, "rest_snapshot", body=body)
                        next_snapshot += snapshot_every
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    _record(stream, "ws", raw=message)
                    messages += 1
                    try:
                        parsed = json.loads(message)
                    except ValueError:
                        continue
                    event = parsed.get("event")
                    if event == "bts:subscription_succeeded":
                        confirmed.add(parsed.get("channel"))
                        if len(confirmed) == len(CHANNELS) and next_snapshot is None:
                            await asyncio.sleep(2.0)          # census after every stream is live
                            next_snapshot = time.monotonic()
                    elif event == "bts:request_reconnect":
                        status = {"status": "GAP", "reason": "venue requested reconnect"}
                        break
                if status["status"] == "COMPLETE" and len(confirmed) == len(CHANNELS):
                    body = await loop.run_in_executor(None, _rest_snapshot, pair)
                    _record(stream, "rest_snapshot", body=body)
                    tail = time.monotonic() + 3.0
                    while time.monotonic() < tail:
                        try:
                            _record(stream, "ws", raw=await asyncio.wait_for(ws.recv(), timeout=0.5))
                            messages += 1
                        except asyncio.TimeoutError:
                            continue
                elif len(confirmed) != len(CHANNELS):
                    status = {"status": "INCOMPLETE", "reason": f"subscriptions confirmed: {sorted(confirmed)}"}
        except Exception as exc:  # noqa: BLE001 - recorded as a gap, never repaired
            status = {"status": "GAP", "reason": f"{type(exc).__name__}: {exc}"}
        _record(stream, "capture_end", **status)
    manifest = {"schema": SCHEMA, "pair": pair, "requested_seconds": seconds, "snapshot_every_seconds": snapshot_every,
                "started_utc_ns": started, "ended_utc_ns": time.time_ns(), "file": raw_path.name,
                "sha256": _sha256(raw_path), "bytes": raw_path.stat().st_size, "messages": messages, **status,
                "python": platform.python_version(), "terms": TERMS,
                "label": "RAW prospective capture; unparsed; fresh until a ledgered access"}
    (out / "capture-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


async def supervise(pair: str, base: Path, *, segment_seconds: float, snapshot_every: float, total_seconds: float,
                    stop_file: Path | None = None, backoff_seconds: float = 15.0, connect=None) -> list[dict]:
    """Consecutive segments until ``total_seconds`` elapse or ``stop_file`` exists; gaps start a new segment."""
    base.mkdir(parents=True, exist_ok=True)
    end = time.monotonic() + total_seconds
    manifests = []
    index_path = base / "segments.jsonl"
    while time.monotonic() < end and not (stop_file and stop_file.exists()):
        name = segment_name(time.time_ns())
        target = base / name
        if target.exists():
            await asyncio.sleep(1.0)
            continue
        length = min(segment_seconds, max(end - time.monotonic(), 10.0))
        manifest = await capture_segment(pair, length, snapshot_every, target, connect=connect)
        manifests.append(manifest)
        with index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"segment": name, "status": manifest["status"], "sha256": manifest["sha256"],
                                "bytes": manifest["bytes"], "started_utc_ns": manifest["started_utc_ns"],
                                "ended_utc_ns": manifest["ended_utc_ns"], "reason": manifest["reason"]},
                               sort_keys=True) + "\n")
        if manifest["status"] != "COMPLETE":
            await asyncio.sleep(backoff_seconds)
    return manifests


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="prospective read-only Bitstamp MBO capture (segmented)")
    p.add_argument("--pair", default="btcusd")
    p.add_argument("--out", type=Path, default=Path("data/v1/capture/bitstamp"))
    p.add_argument("--segment-seconds", type=float, default=3600.0)
    p.add_argument("--snapshot-every", type=float, default=600.0)
    p.add_argument("--total-seconds", type=float, required=True)
    p.add_argument("--stop-file", type=Path, default=None)
    a = p.parse_args(argv)
    if not 10 <= a.segment_seconds <= 7200 or not 30 <= a.snapshot_every <= 3600 or a.total_seconds < 10:
        p.error("segment in [10, 7200] s, census interval in [30, 3600] s, total >= 10 s")
    manifests = asyncio.run(supervise(a.pair, a.out / a.pair, segment_seconds=a.segment_seconds,
                                      snapshot_every=a.snapshot_every, total_seconds=a.total_seconds,
                                      stop_file=a.stop_file))
    print(json.dumps({"segments": len(manifests),
                      "complete": sum(m["status"] == "COMPLETE" for m in manifests)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
