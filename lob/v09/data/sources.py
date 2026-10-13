"""Registered data acquisition with redundancy, integrity and ledgered staged access (M10, M12, §29 of the brief).

``acquire`` downloads every registered source of one dataset (streaming to disk while hashing), verifies Binance
archives against the published SHA-256 ``.CHECKSUM`` file, and appends a ``downloaded`` access entry carrying the
raw source hashes to the ledger. Local copies are re-verified against their recorded hashes and never re-downloaded.
On any failure (HTTP error, network error, checksum mismatch) it records a retained ``attempt`` with outcome
``NOT_AVAILABLE`` and raises ``DataNotAvailable``; ``resolve`` then moves to the next registered backup slot of the
same pool. Nothing else may substitute, and consumed data are never re-labelled fresh.

Raw files live under ``data/v09/raw`` (gitignored; never committed or bundled).
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from ...preregistration import ProtocolError
from .. import protocol as pr

RAW_DIR = "data/v09/raw"
USER_AGENT = "CleoLOB-v0.9-research (+https://github.com/damraka/CleoLOB)"
CHUNK = 1 << 20


class DataNotAvailable(RuntimeError):
    """A registered source could not be obtained (recorded in the ledger as a NOT_AVAILABLE attempt)."""


def local_path(root: Path, declaration: dict, name: str) -> Path:
    url = declaration["sources"][name]["url"]
    return Path(root) / RAW_DIR / declaration["venue"] / declaration["instrument"] / declaration["date"] / \
        f"{name}{''.join(Path(url).suffixes[-2:]) if url.endswith('.csv.gz') else Path(url).suffix}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _get(url: str, timeout: float) -> urllib.request.addinfourl:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - registered https URLs only


def probe(url: str, timeout: float = 20.0) -> dict:
    """Header-only availability probe (reads no market data)."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return {"url": url, "status": response.status, "bytes": int(response.headers.get("Content-Length") or 0)}
    except urllib.error.HTTPError as exc:
        return {"url": url, "status": exc.code, "bytes": 0}
    except (urllib.error.URLError, OSError) as exc:
        return {"url": url, "status": None, "error": type(exc).__name__, "bytes": 0}


def _download(url: str, target: Path, *, timeout: float = 60.0, retries: int = 3) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    last: Exception | None = None
    for attempt in range(retries):
        try:
            digest, size = hashlib.sha256(), 0
            with _get(url, timeout) as response, partial.open("wb") as handle:
                for block in iter(lambda: response.read(CHUNK), b""):
                    handle.write(block)
                    digest.update(block)
                    size += len(block)
            os.replace(partial, target)
            return digest.hexdigest(), size
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            if 400 <= exc.code < 500:
                raise DataNotAvailable(f"HTTP {exc.code} for {url}") from None
            last = exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            partial.unlink(missing_ok=True)
            last = exc
        time.sleep(2.0 * (attempt + 1))
    raise DataNotAvailable(f"download failed after {retries} attempts: {type(last).__name__}: {last}")


def _checksum(url: str, timeout: float = 30.0) -> str:
    with _get(url, timeout) as response:
        text = response.read().decode("ascii", "replace")
    token = text.strip().split()[0] if text.strip() else ""
    if len(token) != 64:
        raise DataNotAvailable(f"malformed checksum file {url}")
    return token.lower()


def _record_attempt(root: Path, protocol: dict, d: dict, design: str, reason: str) -> None:
    pr.append_event(root / pr.LEDGER_PATH, "attempt", dataset=d["id"], role=d["role"], design=design,
                    reason=f"registered source failed: {reason}", protocol=protocol, root=root,
                    payload={"outcome": "NOT_AVAILABLE", "directory": None, "note": reason,
                             "no_directory_reason": "data acquisition failed before any run directory existed",
                             "kind": "acquisition"})


def recorded_hashes(state: pr.LedgerState, dataset_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for access in state.accesses.get(dataset_id, []):
        out.update(access.get("source_sha256", {}))
    return out


def acquire(dataset_id: str, *, use: str, design: str, purpose: str, root: Path,
            downloader=_download, checksum=_checksum) -> dict[str, Path]:
    """Download (or re-verify) every registered source of ``dataset_id``; returns name -> local path."""
    root = Path(root)
    protocol, state = pr.state_of(root)
    d = pr.dataset_declaration(protocol, dataset_id)
    if use not in pr.ROLE_USES[d["role"]]:
        raise ProtocolError(f"{dataset_id}: use {use!r} is not permitted for role {d['role']}")
    for a in state.attempts:
        if a.get("dataset") == dataset_id and a.get("outcome") in ("NOT_AVAILABLE", "QUALITY_REJECTED"):
            raise DataNotAvailable(f"{dataset_id} has a ledgered {a['outcome']} attempt; use the next backup slot")
    known = recorded_hashes(state, dataset_id)
    paths, hashes, sizes = {}, {}, {}
    try:
        for name, source in sorted(d["sources"].items()):
            path = local_path(root, d, name)
            if name in known and path.is_file():
                if sha256_file(path) != known[name]:
                    raise ProtocolError(f"{dataset_id}/{name}: local bytes differ from the ledgered hash")
                paths[name] = path
                continue
            digest, size = downloader(source["url"], path)
            if "checksum_url" in source:
                published = checksum(source["checksum_url"])
                if published != digest:
                    path.unlink(missing_ok=True)
                    raise DataNotAvailable(f"{name}: SHA-256 differs from the published checksum")
            if name in known and known[name] != digest:
                raise ProtocolError(f"{dataset_id}/{name}: re-downloaded bytes differ from the ledgered hash")
            paths[name], hashes[name], sizes[name] = path, digest, size
    except DataNotAvailable as exc:
        _record_attempt(root, protocol, d, design, str(exc))
        raise
    if hashes:
        pr.append_event(root / pr.LEDGER_PATH, "access", dataset=dataset_id, role=d["role"], design=design,
                        reason=purpose, protocol=protocol, root=root,
                        payload={"use": use, "stage": "downloaded", "source_sha256": hashes, "bytes": sizes,
                                 "checksum_verified": sorted(n for n in hashes if "checksum_url" in d["sources"][n])})
    return paths


def mark(dataset_id: str, *, stage: str, use: str, design: str, root: Path, reason: str, **payload) -> dict:
    """Ledger a later access stage (opened, parsed, evaluated) for an already downloaded dataset."""
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    d = pr.dataset_declaration(protocol, dataset_id)
    return pr.append_event(root / pr.LEDGER_PATH, "access", dataset=dataset_id, role=d["role"], design=design,
                           reason=reason, protocol=protocol, root=root, payload={"use": use, "stage": stage, **payload})


def quality_rejected(dataset_id: str, *, design: str, root: Path, report: dict) -> None:
    """Retain a QUALITY_REJECTED outcome; the slot is then replaced by the next registered backup."""
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    d = pr.dataset_declaration(protocol, dataset_id)
    pr.append_event(root / pr.LEDGER_PATH, "attempt", dataset=dataset_id, role=d["role"], design=design,
                    reason="quality engine rejected the dataset", protocol=protocol, root=root,
                    payload={"outcome": "NOT_AVAILABLE", "directory": None, "note": "QUALITY_REJECTED",
                             "no_directory_reason": "rejected before any study run", "quality": report,
                             "kind": "quality"})


def resolve(protocol: dict, state: pr.LedgerState, pool: str, stream: str, n: int) -> dict:
    """The first ``n`` usable slots of a stream: primaries in order, failed ones replaced by ordered backups."""
    venue, instrument = stream.split(":")
    rows = sorted((d for d in pr.datasets_by(protocol, pool=pool)
                   if d["venue"] == venue and d["instrument"] == instrument), key=lambda d: d["slot"])
    failed = {a["dataset"] for a in state.attempts if a.get("outcome") == "NOT_AVAILABLE" and a.get("dataset")}
    chosen, replaced = [], []
    for d in rows:
        if len(chosen) >= n:
            break
        if d["id"] in failed:
            replaced.append(d["id"])
            continue
        chosen.append(d["id"])
    return {"pool": pool, "stream": stream, "requested": n, "datasets": chosen, "failed": replaced,
            "status": "AVAILABLE" if len(chosen) == n else "NOT_AVAILABLE",
            "note": None if len(chosen) == n else f"registered slots exhausted ({len(chosen)}/{n})"}


def open_bytes(path: Path) -> io.BufferedReader:
    return Path(path).open("rb")
