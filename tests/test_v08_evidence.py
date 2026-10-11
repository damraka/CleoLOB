"""v0.8 evidence: write-once runs, corruption detection, lineage, DAG, claim audit, figures, bundles (M35)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lob.v08 import artifacts, claims, figures, reporting
from tests.v08_fixtures import v08_root, v08_template  # noqa: F401


def _run(root: Path, name: str, result: dict, inputs=()) -> Path:
    out = artifacts.new_run(root / "results/v08" / name)
    artifacts.finalize(out, analysis=name.replace("/", "_"), config={"k": 1}, result=result, root=root, inputs=inputs,
                       check_source=False)
    return out


def test_run_is_write_once_bound_and_verifiable(v08_root) -> None:
    run = _run(v08_root, "a", {"x": 1})
    assert artifacts.verify_run(run, root=v08_root)["valid"]
    with pytest.raises(FileExistsError):
        artifacts.new_run(run)
    binding = json.loads((run / "binding.json").read_text())
    assert binding["frozen"] and len(binding["protocol_sha256"]) == 64


@pytest.mark.parametrize("damage", ["result_bytes", "remove_file", "extra_file", "binding_result"])
def test_corrupted_artifacts_fail_verification(v08_root, damage) -> None:
    run = _run(v08_root, "b", {"x": 1})
    if damage == "result_bytes":
        (run / "result.json").write_text('{"x":2}\n', encoding="utf-8")
    elif damage == "remove_file":
        (run / "lineage.json").unlink()
    elif damage == "extra_file":
        (run / "sneaky.json").write_text("{}", encoding="utf-8")
    else:
        b = json.loads((run / "binding.json").read_text())
        b["result_sha256"] = "0" * 64
        (run / "binding.json").write_text(json.dumps(b), encoding="utf-8")
    assert not artifacts.verify_run(run, root=v08_root)["valid"]


def test_lineage_breaks_when_upstream_changes(v08_root) -> None:
    up = _run(v08_root, "up", {"x": 1})
    down = _run(v08_root, "down", {"y": 2}, inputs=["results/v08/up"])
    assert artifacts.verify_run(down, root=v08_root)["valid"]
    (up / "result.json").write_text('{"x":3}\n', encoding="utf-8")
    report = artifacts.verify_run(down, root=v08_root)
    assert not report["valid"] and any("upstream result changed" in i for i in report["issues"])


def test_dag_is_acyclic_with_edges(v08_root) -> None:
    _run(v08_root, "p", {"x": 1})
    _run(v08_root, "q", {"x": 2}, inputs=["results/v08/p"])
    dag = artifacts.dag(v08_root / "results/v08", root=v08_root)
    assert dag["acyclic"] and not dag["dangling_inputs"]
    assert {"from": "results/v08/p", "to": "results/v08/q"} == {k: dag["edges"][0][k] for k in ("from", "to")}


def test_absolute_inputs_outside_root_are_refused(v08_root, tmp_path) -> None:
    out = artifacts.new_run(v08_root / "results/v08/c")
    with pytest.raises(ValueError):
        artifacts.finalize(out, analysis="c", config={}, result={}, root=v08_root, inputs=[str(tmp_path.parent)],
                           check_source=False)


def test_source_change_during_run_refuses_sealing(v08_root) -> None:
    out = artifacts.new_run(v08_root / "results/v08/d")
    artifacts._STARTED[out.resolve()] = {"lob/fake.py": "0" * 64}
    with pytest.raises(ValueError):
        artifacts.finalize(out, analysis="d", config={}, result={}, root=v08_root)


def test_nested_isolated_roots_are_skipped(v08_root) -> None:
    _run(v08_root, "e", {"x": 1})
    nested = v08_root / "results/v08/dev/smoke/root"
    (nested / "configs/v08").mkdir(parents=True)
    (nested / "configs/v08/protocol.json").write_text("{}", encoding="utf-8")
    (nested / "results/v08/x").mkdir(parents=True)
    (nested / "results/v08/x/binding.json").write_text("{}", encoding="utf-8")
    base = v08_root / "results/v08"
    found = artifacts.runs_under(base)
    assert found and all("smoke" not in p.relative_to(base).parts for p in found)


def _final(root: Path, statuses: dict) -> Path:
    hyps = {h: {"status": s, "estimate": None, "ci_low": None, "ci_high": None, "threshold": None, "n": None,
                "alpha": None, "qualifiers": [], "reason": "", "run": None, "members": {}, "secondary_mandate": None}
            for h, s in statuses.items()}
    return _run(root, "final", {"hypotheses": hyps, "statuses": statuses, "inputs": {}})


def _docs(root: Path, report: str, paper: str, readme: str = "") -> None:
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs/v08-final-report.md").write_text(report, encoding="utf-8")
    (root / "docs/v08-paper.md").write_text(paper, encoding="utf-8")
    (root / "README.md").write_text(f"x\n<!-- v08:start -->\n{readme}\n<!-- v08:end -->\n", encoding="utf-8")


ALL = "\n".join(f"H{k} is NOT_ESTABLISHED in simulation [C08-H{k}]" for k in range(1, 22))


def test_claim_audit_passes_on_consistent_documents(v08_root) -> None:
    _final(v08_root, {f"H{k}": "NOT_ESTABLISHED" for k in range(1, 22)})
    _docs(v08_root, ALL, ALL, "v0.8 results are reported with statuses [C08-H1] NOT_ESTABLISHED")
    report = claims.audit(v08_root)
    assert report["issues"] == [i for i in report["issues"] if "does not resolve" in i]   # extra claims lack runs here
    assert not [i for i in report["issues"] if "states" in i or "unknown" in i or "high-risk" in i]


def test_claim_audit_fails_closed(v08_root) -> None:
    _final(v08_root, {f"H{k}": "NOT_ESTABLISHED" for k in range(1, 22)})
    bad = ALL + "\nH8 is ESTABLISHED [C08-H8]\nUnknown [C08-H99]\nThis policy is profitable.\n"
    _docs(v08_root, bad, ALL.replace("[C08-H21]", ""))
    issues = claims.audit(v08_root)["issues"]
    assert any("C08-H8 states" in i for i in issues)
    assert any("unknown claim marker C08-H99" in i for i in issues)
    assert any("high-risk term 'profitable'" in i for i in issues)
    assert any("no claim markers for ['H21']" in i for i in issues)


def test_figures_write_svg_and_data(tmp_path) -> None:
    prov = {"run": "results/v08/x", "pointer": "/a"}
    for made in (figures.bar(tmp_path, "b", ["a", "b"], [1.0, None], title="t", ylabel="y", provenance=prov,
                             errors=[(0.1, 0.2), None]),
                 figures.lines(tmp_path, "l", {"s": ([0, 1], [1.0, 2.0])}, title="t", xlabel="x", ylabel="y",
                               provenance=prov),
                 figures.scatter(tmp_path, "s", [1.0, 2.0], [2.0, 1.0], title="t", xlabel="x", ylabel="y",
                                 provenance=prov),
                 figures.heatmap(tmp_path, "h", [[1.0, None], [0.5, 0.2]], ["r1", "r2"], ["c1", "c2"], title="t",
                                 provenance=prov)):
        svg = (tmp_path / made["figure"]).read_text(encoding="utf-8")
        assert svg.startswith("<svg") and json.loads((tmp_path / made["data"]).read_text())["provenance"] == prov


def test_bundle_export_and_offline_verification(v08_root) -> None:
    _final(v08_root, {f"H{k}": "NOT_AVAILABLE" for k in range(1, 22)})
    out = reporting.export_bundle(v08_root, out="bundle", final="results/v08/final")
    assert out["included"] >= 1
    assert reporting.verify_bundle(v08_root / "bundle")["valid"]
    (v08_root / "bundle/runs/final/result.json").write_text("{}", encoding="utf-8")
    assert not reporting.verify_bundle(v08_root / "bundle")["valid"]
