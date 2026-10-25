"""SHA-256 content-addressed raw store (M17) with streaming, chunked downloads (M15).

Raw files are stored once under ``data/v1/raw/sha256/<aa>/<digest><suffix>`` and referenced by digest; a per-URL
index (``data/v1/raw/index.jsonl``) maps every acquisition to its digest, size and time. A download streams into a
``.partial`` file while hashing, enforces a byte ceiling, verifies an optional published checksum and is renamed
atomically only when complete, so an interrupted download never leaves a file that looks valid. A URL whose content
changes between acquisitions is a recorded anomaly (``ContentChanged``), never an overwrite.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

USER_AGENT = "CleoLOB-v1-research (+https://github.com/damraka/CleoLOB)"
CHUNK = 1 << 20


class NotAvailable(RuntimeError):
    """The provider does not serve this object (HTTP 404/403/410) - a data status, not a crash."""


class ContentChanged(RuntimeError):
    pass


class ChecksumMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class Stored:
    sha256: str
    path: Path
    bytes: int
    url: str


def _suffix(url: str) -> str:
    name = url.rsplit("/", 1)[-1]
    for ext in (".csv.gz", ".data.zip", ".jsonl.gz", ".zip", ".gz", ".csv", ".json"):
        if name.endswith(ext):
            return ext
    return ""


class RawStore:
    def __init__(self, base: Path) -> None:
        self.base = Path(base)
        (self.base / "sha256").mkdir(parents=True, exist_ok=True)
        self.index_path = self.base / "index.jsonl"

    def path_of(self, digest: str, suffix: str = "") -> Path:
        return self.base / "sha256" / digest[:2] / f"{digest}{suffix}"

    def index(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if self.index_path.exists():
            for line in self.index_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    out[rec["url"]] = rec
        return out

    def lookup(self, url: str) -> Stored | None:
        rec = self.index().get(url)
        if rec is None:
            return None
        p = self.path_of(rec["sha256"], rec.get("suffix", ""))
        return Stored(rec["sha256"], p, rec["bytes"], url) if p.exists() else None

    def put_file(self, src: Path, url: str) -> Stored:
        """Ingest an existing local file (e.g. a capture segment) under ``url`` (a ``capture://`` name)."""
        h = hashlib.sha256()
        size = 0
        with Path(src).open("rb") as f:
            for chunk in iter(lambda: f.read(CHUNK), b""):
                h.update(chunk)
                size += len(chunk)
        return self._commit(Path(src), h.hexdigest(), size, url, copy=True)

    def _commit(self, tmp: Path, digest: str, size: int, url: str, *, copy: bool = False) -> Stored:
        prev = self.index().get(url)
        if prev is not None and prev["sha256"] != digest:
            if not copy:
                tmp.unlink(missing_ok=True)
            raise ContentChanged(f"{url}: content changed ({prev['sha256'][:12]} -> {digest[:12]})")
        suffix = _suffix(url)
        target = self.path_of(digest, suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not copy:
                tmp.unlink(missing_ok=True)
        elif copy:
            shutil.copyfile(tmp, target)
        else:
            os.replace(tmp, target)
        if prev is None:
            with self.index_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"url": url, "sha256": digest, "bytes": size, "suffix": suffix,
                                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, sort_keys=True) + "\n")
        return Stored(digest, target, size, url)

    def fetch(self, url: str, *, max_bytes: int = 4 << 30, checksum: str | None = None, timeout: float = 60.0,
              retries: int = 3, opener=None) -> Stored:
        """Stream ``url`` into the store (or return the stored copy); verifies ``checksum`` (sha256 hex) if given."""
        have = self.lookup(url)
        if have is not None:
            if checksum and have.sha256 != checksum:
                raise ChecksumMismatch(f"{url}: stored {have.sha256[:12]} != published {checksum[:12]}")
            return have
        opener = opener or (lambda u: urllib.request.urlopen(  # noqa: S310 - registered https URLs only
            urllib.request.Request(u, headers={"User-Agent": USER_AGENT}), timeout=timeout))
        tmp = self.base / "partial" / f"{hashlib.sha256(url.encode()).hexdigest()}.partial"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        last: Exception | None = None
        for attempt in range(retries):
            h, size = hashlib.sha256(), 0
            try:
                with opener(url) as response, tmp.open("wb") as out:
                    for chunk in iter(lambda: response.read(CHUNK), b""):
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError(f"{url}: exceeds the byte ceiling {max_bytes}")
                        h.update(chunk)
                        out.write(chunk)
                digest = h.hexdigest()
                if checksum and digest != checksum:
                    tmp.unlink(missing_ok=True)
                    raise ChecksumMismatch(f"{url}: downloaded {digest[:12]} != published {checksum[:12]}")
                return self._commit(tmp, digest, size, url)
            except urllib.error.HTTPError as exc:
                tmp.unlink(missing_ok=True)
                if exc.code in (403, 404, 410):
                    raise NotAvailable(f"HTTP {exc.code} for {url}") from None
                last = exc
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                tmp.unlink(missing_ok=True)
                last = exc
            time.sleep(min(2.0 ** attempt, 10.0))
        raise NotAvailable(f"{url}: failed after {retries} attempts ({type(last).__name__}: {last})")

    def verify(self) -> dict:
        """Re-hash every stored object against its name (offline integrity check)."""
        bad, n = [], 0
        for p in (self.base / "sha256").rglob("*"):
            if not p.is_file():
                continue
            n += 1
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(CHUNK), b""):
                    h.update(chunk)
            if not p.name.startswith(h.hexdigest()):
                bad.append(p.name)
        return {"objects": n, "corrupt": bad, "valid": not bad}


def published_sha256(url: str, opener=None, timeout: float = 30.0) -> str:
    """Binance-style ``<sha256>  <file>`` checksum file."""
    opener = opener or (lambda u: urllib.request.urlopen(  # noqa: S310
        urllib.request.Request(u, headers={"User-Agent": USER_AGENT}), timeout=timeout))
    try:
        with opener(url) as response:
            text = response.read(4096).decode("ascii", "replace").strip()
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404, 410):
            raise NotAvailable(f"HTTP {exc.code} for {url}") from None
        raise
    digest = text.split()[0] if text else ""
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ChecksumMismatch(f"malformed checksum file {url}")
    return digest.lower()


def probe(url: str, timeout: float = 20.0) -> dict:
    """HEAD request: reachability and size only (proves nothing about content quality)."""
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            return {"url": url, "status": r.status, "bytes": int(r.headers.get("Content-Length") or 0)}
    except urllib.error.HTTPError as exc:
        return {"url": url, "status": exc.code, "bytes": 0}
    except (urllib.error.URLError, OSError) as exc:
        return {"url": url, "status": None, "error": type(exc).__name__, "bytes": 0}


__all__ = ["RawStore", "Stored", "NotAvailable", "ContentChanged", "ChecksumMismatch", "published_sha256", "probe",
           "USER_AGENT"]
