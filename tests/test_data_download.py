"""Public sample transport integrity, bounds, and reuse; no network needed."""
from __future__ import annotations

from email.message import Message
import gzip
import hashlib
from io import BytesIO
import json
from urllib.request import Request

import pytest

from lob import data_download as download


URL = "https://datasets.tardis.dev/v1/binance-futures/incremental_book_L2/2024/01/01/BTCUSDT.csv.gz"


class Response(BytesIO):
    def __init__(self, body, headers=None, *, status=200, url=URL):
        super().__init__(body)
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = str(value)
        self.status = status
        self.url = url

    def getcode(self):
        return self.status

    def geturl(self):
        return self.url


def stub(monkeypatch, body, headers=None, **kwargs):
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append((request, timeout))
            return Response(body, headers, **kwargs)

    monkeypatch.setattr(download, "build_opener", lambda handler: Opener())
    return requests


def fetch(tmp_path, **kwargs):
    return download.download_tardis_sample("binance-futures", "BTCUSDT", "2024-01-01",
                                           "incremental_book_L2", tmp_path, **kwargs)


def test_normal_get_has_verified_provenance_and_reuses_without_network(tmp_path, monkeypatch):
    original = b"exchange,symbol,timestamp\nbinance-futures,BTCUSDT,123\n"
    compressed = gzip.compress(original, mtime=0)
    md5 = hashlib.md5(compressed, usedforsecurity=False).hexdigest()
    requests = stub(monkeypatch, compressed, {"Content-Length": len(compressed), "x-md5": md5})
    path = fetch(tmp_path)
    assert path.read_bytes() == compressed
    metadata = json.loads(path.with_name(path.name + ".provenance.json").read_text())
    assert metadata["sha256"] == hashlib.sha256(compressed).hexdigest()
    assert metadata["md5"] == md5
    assert metadata["expanded_bytes"] == len(original)
    assert metadata["url"] == URL
    assert metadata["kind"] == "incremental_book_L2"
    assert requests[0][0].get_method() == "GET"
    assert requests[0][0].get_header("Authorization") is None
    assert requests[0][1] == 30
    assert fetch(tmp_path) == path
    assert len(requests) == 1
    assert not list(tmp_path.glob(".cleo-tardis-*"))


@pytest.mark.parametrize("kind", ["incremental_book_L2", "book_snapshot_5", "trades"])
def test_supported_public_types(tmp_path, monkeypatch, kind):
    url = URL.replace("incremental_book_L2", kind)
    requests = stub(monkeypatch, gzip.compress(b"header\nrow\n"), url=url)
    result = download.download_tardis_sample("binance-futures", "BTCUSDT", "2024-01-01", kind, tmp_path)
    assert result.name == f"binance-futures_{kind}_2024-01-01_BTCUSDT.csv.gz"
    assert requests[0][0].full_url == url


@pytest.mark.parametrize("argument,value", [
    ("exchange", "../secret"), ("exchange", "Binance"), ("symbol", "BTC/USDT"),
    ("symbol", ".."), ("symbol", "A?token=x"), ("data_type", "quotes"),
    ("date", "2024-01-02"), ("date", "2024-1-01"), ("date", "2024-13-01"),
])
def test_invalid_identity_rejected_before_network(tmp_path, monkeypatch, argument, value):
    requests = stub(monkeypatch, b"unused")
    args = {"exchange": "binance-futures", "symbol": "BTCUSDT", "date": "2024-01-01",
            "data_type": "incremental_book_L2", "out": tmp_path}
    args[argument] = value
    with pytest.raises(ValueError):
        download.download_tardis_sample(**args)
    assert not requests
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
@pytest.mark.parametrize("field", ["max_bytes", "max_expanded_bytes"])
def test_invalid_size_limits(tmp_path, value, field):
    with pytest.raises(ValueError, match="positive integer"):
        fetch(tmp_path, **{field: value})


@pytest.mark.parametrize("with_length", [False, True])
def test_compressed_size_limit_cleans_only_own_temporary_file(tmp_path, monkeypatch, with_length):
    retained = tmp_path / "user-file"
    retained.write_text("preserve")
    body = gzip.compress(b"a,b,c\n1,2,3\n")
    stub(monkeypatch, body, {"Content-Length": len(body)} if with_length else {})
    with pytest.raises(ValueError, match="max_bytes"):
        fetch(tmp_path, max_bytes=len(body) - 1)
    assert list(tmp_path.iterdir()) == [retained]


def test_expansion_limit_catches_compressible_payload(tmp_path, monkeypatch):
    stub(monkeypatch, gzip.compress(b"0" * 10000))
    with pytest.raises(ValueError, match="max_expanded_bytes"):
        fetch(tmp_path, max_expanded_bytes=100)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("case", ["truncated", "crc", "html", "empty", "provider_md5", "content_length"])
def test_corrupt_or_incomplete_download_rejected(tmp_path, monkeypatch, case):
    body = gzip.compress(b"column\nvalue\n", mtime=0)
    headers = {}
    if case == "truncated":
        body = body[:-5]
    elif case == "crc":
        body = body[:-8] + bytes([body[-8] ^ 1]) + body[-7:]
    elif case == "html":
        body = b"<html>service unavailable</html>"
    elif case == "empty":
        body = gzip.compress(b"")
    elif case == "provider_md5":
        headers["x-md5"] = "0" * 32
    elif case == "content_length":
        headers["Content-Length"] = len(body) + 1
    stub(monkeypatch, body, headers)
    with pytest.raises(ValueError):
        fetch(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("url", [
    "http://datasets.tardis.dev/v1/file", "https://example.org/v1/file",
    "https://datasets.tardis.dev.evil.example/v1/file", "https://user@datasets.tardis.dev/v1/file",
    "https://datasets.tardis.dev:444/v1/file", "https://datasets.tardis.dev/account",
])
def test_redirect_rejected_before_following(url):
    with pytest.raises(ValueError, match="public HTTPS"):
        download._TardisRedirectHandler().redirect_request(Request(URL), None, 302, "Found", {}, url)


def test_public_same_host_redirect_accepted():
    url = URL.replace("2024/01", "2024/02")
    request = download._TardisRedirectHandler().redirect_request(Request(URL), None, 302, "Found", {}, url)
    assert request.full_url == url


@pytest.mark.parametrize("case", ["data_tamper", "identity_tamper", "hash_tamper", "missing_data", "missing_sidecar"])
def test_invalid_existing_data_never_redownloaded_or_overwritten(tmp_path, monkeypatch, case):
    requests = stub(monkeypatch, gzip.compress(b"x\n1\n"))
    path = fetch(tmp_path)
    sidecar = path.with_name(path.name + ".provenance.json")
    if case == "data_tamper":
        path.write_bytes(gzip.compress(b"x\n2\n"))
    elif case in {"identity_tamper", "hash_tamper"}:
        metadata = json.loads(sidecar.read_text())
        metadata["symbol" if case == "identity_tamper" else "sha256"] = "tampered"
        sidecar.write_text(json.dumps(metadata))
    elif case == "missing_data":
        path.unlink()
    else:
        sidecar.unlink()
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises((ValueError, FileExistsError)):
        fetch(tmp_path)
    assert len(requests) == 1
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_reuse_obeys_tighter_resource_limits(tmp_path, monkeypatch):
    compressed = gzip.compress(b"x" * 1000)
    requests = stub(monkeypatch, compressed)
    fetch(tmp_path)
    with pytest.raises(ValueError, match="max_bytes"):
        fetch(tmp_path, max_bytes=len(compressed) - 1)
    with pytest.raises(ValueError, match="max_expanded_bytes"):
        fetch(tmp_path, max_expanded_bytes=999)
    assert len(requests) == 1
