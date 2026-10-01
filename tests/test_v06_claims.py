"""v0.6 claim graph and claim audit."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lob.v06 import claims as cl
from lob.v06 import evidence
from lob.v06 import protocol as pr
from tests import test_v06_protocol
from tests.test_v06_protocol import DEV

root = test_v06_protocol.root   # shared fixture


def _run(root: Path, status: str = "ESTABLISHED") -> Path:
    out = evidence.new_run(root / "results/v06/h1")
    evidence.finalize(out, analysis="unit-h1", dataset_ids=[DEV], config={"a": 1},
                      result={"improvement": {"status": status, "difference": -0.4, "ci_low": -0.6, "ci_high": -0.2}},
                      root=root, check_source=False)
    return out


def _claims(root: Path, run: Path) -> Path:
    claim = cl.make_claim(claim_id="H1", hypothesis_id="H1", run=run, result_path="improvement", metric="objective diff",
                          threshold="upper < 0", status_path="improvement/status", family="F1_calibration",
                          family_size=3, alpha=0.05 / 3, evidence_level="retrospective", interpretation="test",
                          root=root, estimate_key="difference")
    path = root / "claims.json"
    cl.write_claims(path, [claim], pr.protocol_sha256(pr.load_protocol(root / pr.PROTOCOL_PATH)))
    return path


def test_claim_numbers_come_from_the_run(root: Path) -> None:
    run = _run(root)
    path = _claims(root, run)
    claim = json.loads(path.read_text(encoding="utf-8"))["claims"][0]
    assert claim["estimate"] == -0.4 and claim["ci_high"] == -0.2 and claim["status"] == "ESTABLISHED"
    assert claim["dataset_roles"] == {DEV: "development"} and claim["evidence"] == "results/v06/h1"
    for field in cl.FIELDS:
        assert field in claim
    with pytest.raises(ValueError):
        cl.make_claim(claim_id="x", hypothesis_id=None, run=run, result_path="improvement", metric="m", threshold="t",
                      status_path=None, family=None, family_size=None, alpha=None, evidence_level="hype",
                      interpretation="", root=root)
    with pytest.raises(ValueError, match="duplicate"):
        cl.write_claims(root / "c.json", [claim, claim], "x")


def test_audit_detects_doc_mismatch_and_tampering(root: Path) -> None:
    run = _run(root)
    path = _claims(root, run)
    doc = root / "report.md"
    doc.write_text("| H1 | ESTABLISHED | retrospective |\n", encoding="utf-8")
    assert cl.audit(path, [doc], root=root)["valid"]
    doc.write_text("| H1 | NOT_ESTABLISHED | x |\n| H9 | FAILED | y |\n", encoding="utf-8")
    issues = cl.audit(path, [doc], root=root)["issues"]
    assert any("H1 says NOT_ESTABLISHED" in i for i in issues) and any("H9 has no machine-readable" in i for i in issues)
    doc.write_text("| H1 | ESTABLISHED | x |\n", encoding="utf-8")
    (run / "result.json").write_text(json.dumps({"improvement": {"status": "FAILED"}}) + "\n", encoding="utf-8")
    assert any("does not verify" in i for i in cl.audit(path, [doc], root=root)["issues"])


def test_risky_sentences_require_negation() -> None:
    text = ("No profitability claim follows. The strategy is profitable in live trading. "
            "Exact FIFO is not established. We find causal effects of cancellations.")
    flagged = cl.risky_sentences(text)
    assert flagged == ["The strategy is profitable in live trading.", "We find causal effects of cancellations."]
    assert cl.risky_sentences("alpha = 0.05 per family; adjusted alpha 0.0125.") == []


def test_doc_table_parses_status_rows_only() -> None:
    table = cl.doc_table("| Claim | Status |\n|---|---|\n| H2 | **NOT_ESTABLISHED** | x |\n| note | words | y |\n")
    assert table == {"H2": "NOT_ESTABLISHED"}
