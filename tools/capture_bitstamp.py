"""Record a bounded capture of Bitstamp's public order-level market data.

The capture keeps every websocket message's exact text with local receive clocks,
plus REST ``order_book?group=2`` censuses (individual orders with IDs) at the start,
every ``--snapshot-every`` seconds and at the end. Nothing is parsed, repaired or
filtered here; interpretation belongs to ``lob.mbo_sources``. Reconnect requests,
disconnects and errors are recorded as explicit gap markers and end the capture,
so a gap can never be silently bridged.

Bitstamp documents public market-data use; commercial use requires a separate
licence. Captures stay under the ignored ``data/v05`` directory and are not
redistributed. Requires the optional ``websockets`` package.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from urllib.request import Request, urlopen

WS_URL = "wss://ws.bitstamp.net"
REST_URL = "https://www.bitstamp.net/api/v2/order_book/{pair}/?group=2"
CHANNELS = ("live_orders_{pair}", "live_trades_{pair}", "order_book_{pair}")
MAX_REST_BYTES = 64 * 1024 * 1024


def _record(stream, kind: str, **fields) -> None:
    fields.update(kind=kind, recv_ns=time.time_ns(), mono_ns=time.perf_counter_ns())
    stream.write(json.dumps(fields, separators=(",", ":"), ensure_ascii=True) + "\n")


def _rest_snapshot(pair: str) -> str:
    request = Request(REST_URL.format(pair=pair), headers={"User-Agent": "CleoLOB-research-capture/0.5"})
    with urlopen(request, timeout=30) as response:
        body = response.read(MAX_REST_BYTES + 1)
    if len(body) > MAX_REST_BYTES:
        raise ValueError("REST census exceeds size bound")
    return body.decode("utf-8")


async def capture(pair: str, seconds: float, snapshot_every: float, out: Path) -> dict:
    import websockets

    out.mkdir(parents=True, exist_ok=False)
    raw_path = out / "capture.jsonl.gz"
    status = {"status": "COMPLETE", "reason": None}
    started = time.time_ns()
    with gzip.open(raw_path, "wt", encoding="utf-8", newline="\n") as stream:
        _record(stream, "capture_start", pair=pair, seconds=seconds, ws_url=WS_URL,
                channels=[c.format(pair=pair) for c in CHANNELS])
        try:
            async with websockets.connect(WS_URL, max_size=16 * 1024 * 1024, ping_interval=20) as ws:
                for channel in CHANNELS:
                    await ws.send(json.dumps({"event": "bts:subscribe",
                                              "data": {"channel": channel.format(pair=pair)}}))
                confirmed: set[str] = set()
                deadline = time.monotonic() + seconds
                next_snapshot = None
                loop = asyncio.get_running_loop()
                while time.monotonic() < deadline:
                    if len(confirmed) == len(CHANNELS) and next_snapshot is not None and time.monotonic() >= next_snapshot:
                        body = await loop.run_in_executor(None, _rest_snapshot, pair)
                        _record(stream, "rest_snapshot", body=body)
                        next_snapshot += snapshot_every
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    _record(stream, "ws", raw=message)
                    try:
                        parsed = json.loads(message)
                    except ValueError:
                        continue
                    event = parsed.get("event")
                    if event == "bts:subscription_succeeded":
                        confirmed.add(parsed.get("channel"))
                        if len(confirmed) == len(CHANNELS) and next_snapshot is None:
                            # Census only after all streams are live, so buffered
                            # events can be aligned by exchange microtimestamp.
                            await asyncio.sleep(2.0)
                            next_snapshot = time.monotonic()
                    elif event == "bts:request_reconnect":
                        status = {"status": "GAP", "reason": "venue requested reconnect"}
                        break
                if status["status"] == "COMPLETE" and len(confirmed) == len(CHANNELS):
                    body = await loop.run_in_executor(None, _rest_snapshot, pair)
                    _record(stream, "rest_snapshot", body=body)
                    # Keep streaming briefly so events up to the end census are covered.
                    tail = time.monotonic() + 3.0
                    while time.monotonic() < tail:
                        try:
                            _record(stream, "ws", raw=await asyncio.wait_for(ws.recv(), timeout=0.5))
                        except asyncio.TimeoutError:
                            continue
                elif len(confirmed) != len(CHANNELS):
                    status = {"status": "INCOMPLETE", "reason": f"subscriptions confirmed: {sorted(confirmed)}"}
        except Exception as exc:  # recorded, never repaired
            status = {"status": "GAP", "reason": f"{type(exc).__name__}: {exc}"}
        _record(stream, "capture_end", **status)
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    manifest = {"schema": "cleolob-bitstamp-capture-1", "pair": pair, "requested_seconds": seconds,
                "snapshot_every_seconds": snapshot_every, "started_utc_ns": started,
                "ended_utc_ns": time.time_ns(), "file": raw_path.name, "sha256": digest,
                "bytes": raw_path.stat().st_size, **status,
                "python": platform.python_version(), "websockets": _version("websockets"),
                "terms": {"source": "https://www.bitstamp.net/api/", "checked": "2026-09-26",
                          "use": "non-commercial research; raw capture not redistributed"}}
    (out / "capture-manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


def _version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="btcusd")
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--snapshot-every", type=float, default=300.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 10 <= args.seconds <= 7200 or not 30 <= args.snapshot_every <= 3600:
        parser.error("seconds must be in [10, 7200] and snapshot interval in [30, 3600]")
    manifest = asyncio.run(capture(args.pair, args.seconds, args.snapshot_every, args.out))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    sys.exit(0 if manifest["status"] == "COMPLETE" else 2)


if __name__ == "__main__":
    main()
