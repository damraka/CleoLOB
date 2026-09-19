"""Bounded downloads of Tardis's public first-of-month sample files.

Files remain compressed. A download is accepted only after transport size,
optional provider MD5, and complete gzip/CRC checks. Existing files are reused
only with matching provenance and integrity; no data file is overwritten.
"""
from __future__ import annotations

from datetime import date as calendar_date, datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zlib


_HOST = "datasets.tardis.dev"
_KINDS = frozenset({"incremental_book_L2", "book_snapshot_5", "trades"})
_CHUNK = 1024 * 1024
_HEADERS = ("Content-Length", "Content-Type", "x-md5", "ETag", "Last-Modified", "Date")


def _positive_limit(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _trusted_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != _HOST or parsed.port not in (None, 443)
            or parsed.username is not None or parsed.password is not None or parsed.fragment
            or not parsed.path.startswith("/v1/")):
        raise ValueError("download and redirects must use the public HTTPS Tardis dataset endpoint")


class _TardisRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Validate before urllib follows the redirect, not merely after downloading.
        _trusted_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fingerprints(path: Path, max_bytes: int) -> tuple[int, str, str]:
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(min(_CHUNK, max_bytes - size + 1)):
            size += len(block)
            if size > max_bytes:
                raise ValueError(f"compressed file exceeds max_bytes={max_bytes}")
            sha.update(block)
            md5.update(block)
    return size, sha.hexdigest(), md5.hexdigest()


def _verify_gzip(path: Path, max_expanded_bytes: int) -> int:
    expanded = 0
    try:
        with gzip.open(path, "rb") as stream:
            while block := stream.read(min(_CHUNK, max_expanded_bytes - expanded + 1)):
                expanded += len(block)
                if expanded > max_expanded_bytes:
                    raise ValueError(f"gzip data exceeds max_expanded_bytes={max_expanded_bytes}")
    except (gzip.BadGzipFile, EOFError, zlib.error) as exc:
        raise ValueError(f"gzip integrity/CRC validation failed: {exc}") from exc
    if expanded == 0:
        raise ValueError("sample gzip contains no data")
    return expanded


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate provenance key: {key}")
        result[key] = value
    return result


def _reuse(path: Path, sidecar: Path, identity: dict[str, str], max_bytes: int,
           max_expanded_bytes: int) -> Path:
    if path.is_symlink() or sidecar.is_symlink() or not path.is_file() or not sidecar.is_file():
        raise FileExistsError("existing sample requires a regular data file and provenance sidecar")
    if sidecar.stat().st_size > 64 * 1024:
        raise ValueError("existing provenance sidecar exceeds 64 KiB")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != 1
            or any(metadata.get(key) != value for key, value in identity.items())):
        raise ValueError("existing provenance does not match the requested sample")
    _trusted_url(metadata.get("final_url", ""))
    size, sha, md5 = _fingerprints(path, max_bytes)
    if any(metadata.get(key) != value for key, value in {"bytes": size, "sha256": sha, "md5": md5}.items()):
        raise ValueError("existing sample integrity does not match its provenance")
    expanded = _verify_gzip(path, max_expanded_bytes)
    if metadata.get("expanded_bytes") != expanded:
        raise ValueError("existing expanded size does not match its provenance")
    return path


def download_tardis_sample(exchange: str, symbol: str, date: str, data_type: str,
                          out: str | Path, max_bytes: int = 64 * 1024**2, *,
                          max_expanded_bytes: int = 2 * 1024**3) -> Path:
    """Download one public sample and save ``<file>.provenance.json`` beside it.

    ``date`` is an ISO calendar date on the first day of a month. Exchange and
    symbol are provider identifiers, not arbitrary paths. No authentication is
    sent. Redirects are restricted to the same public dataset host. HTTP timeout
    is 30 seconds per socket operation; compressed and expanded size are bounded.

    The output directory is created if necessary. Data and sidecar are each
    published atomically without replacing existing files. An interrupted
    publication may leave an incomplete pair, which a later call rejects rather
    than overwriting. Only temporary files created by this call are removed.
    """
    _positive_limit(max_bytes, "max_bytes")
    _positive_limit(max_expanded_bytes, "max_expanded_bytes")
    if not isinstance(exchange, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", exchange):
        raise ValueError("exchange must be a lowercase provider identifier (letters, digits, hyphens)")
    if not isinstance(symbol, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", symbol):
        raise ValueError("symbol must be a provider identifier (letters, digits, dots, underscores, hyphens)")
    if not isinstance(data_type, str) or data_type not in _KINDS:
        raise ValueError(f"data_type must be one of {', '.join(sorted(_KINDS))}")
    if not isinstance(date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError("date must be YYYY-MM-DD")
    parsed_date = calendar_date.fromisoformat(date)
    if parsed_date.day != 1:
        raise ValueError("only first-of-month public samples are supported")
    url = f"https://{_HOST}/v1/{exchange}/{data_type}/{date.replace('-', '/')}/{symbol}.csv.gz"
    _trusted_url(url)
    directory = Path(out).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{exchange}_{data_type}_{date}_{symbol}.csv.gz"
    sidecar = path.with_name(path.name + ".provenance.json")
    identity = {"provider": "Tardis.dev", "exchange": exchange, "symbol": symbol,
                "date": date, "kind": data_type, "url": url}
    if path.exists() or path.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        return _reuse(path, sidecar, identity, max_bytes, max_expanded_bytes)

    temporary: list[Path] = []
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=directory, prefix=".cleo-tardis-",
                                         suffix=".part", delete=False) as output:
            data_temp = Path(output.name)
            temporary.append(data_temp)
            opener = build_opener(_TardisRedirectHandler())
            request = Request(url, headers={"Accept-Encoding": "identity", "User-Agent": "CleoLOB/0.2"})
            with opener.open(request, timeout=30) as response:
                _trusted_url(response.geturl())
                if response.getcode() != 200:
                    raise ValueError(f"sample GET requires HTTP 200, received {response.getcode()}")
                selected_headers = {name: response.headers[name] for name in _HEADERS
                                    if response.headers.get(name) is not None}
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    if not re.fullmatch(r"[0-9]+", content_length):
                        raise ValueError("invalid Content-Length header")
                    if int(content_length) > max_bytes:
                        raise ValueError(f"Content-Length exceeds max_bytes={max_bytes}")
                size = 0
                sha = hashlib.sha256()
                md5 = hashlib.md5(usedforsecurity=False)
                while block := response.read(min(_CHUNK, max_bytes - size + 1)):
                    size += len(block)
                    if size > max_bytes:
                        raise ValueError(f"download exceeds max_bytes={max_bytes}")
                    output.write(block)
                    sha.update(block)
                    md5.update(block)
                if content_length is not None and size != int(content_length):
                    raise ValueError("download length differs from Content-Length (truncated response)")
                provider_md5 = response.headers.get("x-md5")
                if provider_md5 is not None:
                    expected = provider_md5.strip().strip('"').lower()
                    if not re.fullmatch(r"[0-9a-f]{32}", expected) or expected != md5.hexdigest():
                        raise ValueError("provider x-md5 checksum does not match downloaded data")
                final_url = response.geturl()
            output.flush()
            os.fsync(output.fileno())

        expanded = _verify_gzip(data_temp, max_expanded_bytes)
        metadata = {"schema_version": 1, **identity, "final_url": final_url,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "response_headers": selected_headers, "bytes": size, "expanded_bytes": expanded,
                    "sha256": sha.hexdigest(), "md5": md5.hexdigest()}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=".cleo-tardis-", suffix=".part", delete=False) as output:
            provenance_temp = Path(output.name)
            temporary.append(provenance_temp)
            json.dump(metadata, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        # Hard-link publication is atomic and, unlike replace()/POSIX rename(),
        # refuses to overwrite another process's file in a concurrent download.
        os.link(data_temp, path)
        os.link(provenance_temp, sidecar)
        return path
    finally:
        for item in temporary:
            item.unlink(missing_ok=True)
