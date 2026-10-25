"""v1 evidence runs and provenance graph (M5)."""
import json

import pytest

from lob.v1.evidence import provenance as P, runs as R
from lob.v1.protocol import document as D
from tests.v1_fixtures import minimal_protocol, ws  # noqa: F401


def _finished(ws, out, *, upstream=None, datasets=(), namespace="explore"):
    run = R.new_run(out)
    b = R.bind(ws=ws, namespace=namespace, design=None, datasets=list(datasets), upstream=upstream or {},
               root=ws.root)
    return R.finalize(run, analysis="unit-test", config={"k": 1}, result={"value": 1.5}, binding=b, root=ws.root)


def test_provenance_run_roundtrip_and_tamper(ws):
    D.freeze(ws, minimal_protocol(), reason="freeze")
    out = ws.root / "results/v1/explore/a"
    info = _finished(ws, out)
    assert info["status"] == "COMPLETED"
    assert R.verify_run(out, ws=ws)["valid"]
    with pytest.raises(R.RunError):
        R.new_run(out)                                   # write-once
    (out / "result.json").write_text(json.dumps({"value": 2.0}))
    assert not R.verify_run(out)["valid"]


def test_provenance_portability_rules(tmp_path):
    for name, bad in (("nan", {"v": float("nan")}), ("abs", {"path": "C:\\Users\\someone\\data"})):
        out = R.new_run(tmp_path / name)
        info = R.finalize(out, analysis="x", config={}, result=bad, binding={})
        assert info["status"] == "INVALID" and R.verify_run(out)["status"] == "INVALID"   # sealed, never lost
    with pytest.raises(R.RunError):
        R.finalize(R.new_run(tmp_path / "cfg"), analysis="x", config={"v": float("inf")}, result={}, binding={})
    import numpy as np
    out = R.new_run(tmp_path / "r")
    R.finalize(out, analysis="x", config={}, result={"url": "https://quote-saver.bycsi.com/a", "ratio": "1:2",
                                                     1: np.int64(3), "a": np.arange(2)}, binding={})
    assert R.verify_run(out)["valid"] and R.read_json(out / "result.json")["1"] == 3


def test_provenance_binding_requires_ledgered_access(ws):
    D.freeze(ws, minimal_protocol(), reason="freeze")
    with pytest.raises(R.RunError):
        R.bind(ws=ws, namespace="explore", design=None, datasets=["d"], upstream={}, root=ws.root)
    ws.append("declare_fresh", dataset="d", reason="slot", payload={"role": "explore", "pool": "explore"})
    ws.append("acquire", dataset="d", reason="dl", payload={"sha256": {"l2": "a" * 64}})
    D.record_access(ws, "d", namespace="explore", stage="parsed", use="pilot", reason="read")
    b = R.bind(ws=ws, namespace="explore", design=None, datasets=["d"], upstream={}, root=ws.root)
    assert b["datasets"]["d"]["source_sha256"] == {"l2": "a" * 64}


def test_provenance_graph_lineage_and_binding_mismatch(ws):
    D.freeze(ws, minimal_protocol(), reason="freeze")
    a = ws.root / "results/v1/explore/a"
    _finished(ws, a)
    b = ws.root / "results/v1/explore/b"
    _finished(ws, b, upstream={"a": a})
    g = P.build(ws.root / "results/v1", ws.root)
    assert g["valid"], g["issues"]
    lin = P.lineage(g, "run:results/v1/explore/b")
    assert "run:results/v1/explore/a" in lin and any(n.startswith("code:") for n in lin)
    # rewriting the upstream result (and its checksums) breaks the downstream binding
    (a / "result.json").write_text(json.dumps({"value": 9}))
    sums = R.read_json(a / "checksums.json")
    sums["result.json"] = R.sha256_file(a / "result.json")
    (a / "checksums.json").write_text(json.dumps(sums))
    g = P.build(ws.root / "results/v1", ws.root)
    assert not g["valid"] and "different result" in g["issues"][0]


def test_provenance_failed_runs_are_finalised_with_status(tmp_path):
    out = R.new_run(tmp_path / "failed")
    R.finalize(out, analysis="x", config={}, result={"error": "boom"}, binding={}, status="INVALID", reason="boom")
    v = R.verify_run(out)
    assert v["valid"] and v["status"] == "INVALID"
    with pytest.raises(R.RunError):
        R.finalize(out, analysis="x", config={}, result={}, binding={})


def test_provenance_code_digest_is_stable():
    assert R.code_digest() == R.code_digest() and len(R.code_digest()) == 64
