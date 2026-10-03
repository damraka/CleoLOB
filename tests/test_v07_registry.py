"""v0.7 result registry, hypothesis table, claim graph, claim audit and negative-result retention."""
from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from lob.v07.evidence import runs
from lob.v07.protocol import core as pr
from lob.v07.reports import registry as rg
from tests.test_v07_protocol import COPIED, ROOT


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    for name in COPIED + tuple(f"configs/v07/{p.name}" for p in (ROOT / "configs/v07").glob("*.json")):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, tmp_path / name)
    pr.initialize_ledger(tmp_path)
    for day, h1 in (("deribit-eth-perp-2020-11-01", "ESTABLISHED"), ("deribit-eth-perp-2020-12-01", "NOT_ESTABLISHED")):
        out = runs.new_run(tmp_path / f"results/v07/m17/{day}")
        runs.finalize(out, analysis=f"m17-holdout-{day}", dataset_ids=[], config={}, root=tmp_path, result={
            "hypotheses": {"H1": {"estimate": -0.3, "ci": [-0.5, -0.1], "status": h1},
                           "H10": {"status": "FAILED"}, "H11": {"status": "ESTABLISHED"}},
            "domain_gap": {"G3_conditional_ar": {"difference": -0.01, "ci_low": -0.02, "ci_high": 0.0,
                                                 "status_H2": "NOT_ESTABLISHED", "qualifiers": ["VACUOUS"]}},
            "support": {}})
    pr.append_event(tmp_path / pr.LEDGER_PATH, "attempt", design="x", reason="t", root=tmp_path,
                    protocol=pr.load_protocol(tmp_path / pr.PROTOCOL_PATH),
                    payload={"outcome": "ABORTED", "directory": "results/v07/x", "note": "host memory"})
    return tmp_path


def test_hypothesis_table_is_complete_and_conservative(root) -> None:
    table = rg.hypothesis_table(root)
    assert [r["hypothesis"] for r in table] == [f"H{i}" for i in range(1, 14)]
    status = {r["hypothesis"]: r["status"] for r in table}
    assert status["H1"] == "NOT_ESTABLISHED"          # conjunction over both fresh days
    assert status["H10"] == "FAILED" and status["H11"] == "ESTABLISHED"
    assert status["H2"] == "NOT_ESTABLISHED" and status["H4"] == "NOT_AVAILABLE" and status["H12"] == "NOT_AVAILABLE"
    assert "ASSUMPTION_DEPENDENT" in next(r for r in table if r["hypothesis"] == "H1")["limitations"]


def test_registry_retains_attempts_and_negative_results(root) -> None:
    registry = rg.build_registry(root)
    assert len(registry["runs"]) == 2 and all(r["valid"] for r in registry["runs"])
    negative = rg.negative_results(rg.hypothesis_table(root), registry)
    assert negative["attempts"][0]["outcome"] == "ABORTED" and "H10" in negative["hypotheses"]


def test_claim_graph_and_audit(root, tmp_path_factory) -> None:
    table = rg.hypothesis_table(root)
    docs = root / "docs"
    docs.mkdir()
    good = docs / "a.md"
    good.write_text("H1 is NOT_ESTABLISHED on fresh data [C-H1].\nH10 FAILED [C-H10].\n", encoding="utf-8")
    graph = rg.claim_graph(table, [good], root)
    node = next(n for n in graph["claims"] if n["hypothesis"] == "H1")
    assert node["sentences"] == ["docs/a.md:1"] and all(node["artifacts"].values())
    assert rg.audit(table, graph, [good], root)["valid"]
    bad = docs / "b.md"
    bad.write_text("H1 shows the posterior improves realism [C-H1].\n[C-H99]\n", encoding="utf-8")
    report = rg.audit(table, rg.claim_graph(table, [bad], root), [bad], root)
    assert not report["valid"] and len(report["issues"]) == 2
    terms = rg.term_audit([bad], root)
    assert any(t["term"] == "improves" for t in terms)
