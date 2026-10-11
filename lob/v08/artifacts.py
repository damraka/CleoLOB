"""Write-once v0.8 run directories, lineage and the experiment DAG (M35).

A run directory holds ``config.json``, ``result.json``, ``provenance.json``
(line-ending-normalized source hashes, runtime, git state), ``binding.json``
(protocol hash, ledger anchor, dataset identities, config/result hashes),
``lineage.json`` (upstream runs and the result hashes it consumed) and
``checksums.json`` (byte integrity of every file, including figures and tables).
Paths inside artifacts are repository-relative POSIX paths; absolute machine
paths are refused.

``verify_run`` checks bytes, binding and lineage (an upstream run whose result
changed breaks every downstream run); it is never independent replication.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ..artifacts import seal_artifacts, verify_artifacts
from ..config import canonical_json
from ..experiments.registry import PROJECT_ROOT
from ..preregistration import document_sha256
from ..v07.evidence.runs import provenance as v07_provenance, source_manifest
from . import protocol as pr

RESULTS_ROOT = "results/v08"
MEANING = ("byte integrity plus protocol/ledger/dataset/config/result/lineage binding; "
           "not independent scientific replication")
_STARTED: dict[Path, dict] = {}


def _rel(path: Path, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        raise ValueError(f"artifact path must lie inside the repository root: {path}") from None


def new_run(out: str | Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    _STARTED[out.resolve()] = source_manifest()
    return out


def write_json(out: Path, name: str, value: Any) -> Path:
    target = Path(out) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(json.loads(canonical_json(value))) + "\n")
    return target


def write_text(out: Path, name: str, text: str) -> Path:
    target = Path(out) / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return target


def result_sha256(run: Path) -> str:
    return document_sha256(json.loads((Path(run) / "result.json").read_text(encoding="utf-8")))


def finalize(out: Path, *, analysis: str, config: Any, result: Any, root: Path = PROJECT_ROOT,
             dataset_ids: Iterable[str] = (), inputs: Iterable[str | Path] = (), seeds: dict | None = None,
             extra: dict | None = None, check_source: bool = True) -> dict:
    """Bind, record lineage and seal. Refuses if the implementation changed while the run executed."""
    out = Path(out)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    prov = v07_provenance()
    started = _STARTED.pop(out.resolve(), None)
    if check_source and started is not None and prov["source_files"] != started:
        changed = sorted(k for k in set(started) | set(prov["source_files"])
                         if started.get(k) != prov["source_files"].get(k))
        raise ValueError(f"implementation changed during the run ({changed[:5]}); attempt retained unsealed")
    config = json.loads(canonical_json(config))
    result = json.loads(canonical_json(result))
    lineage = {"schema": "cleolob-v08-lineage-1", "analysis": analysis,
               "inputs": [{"run": _rel(root / i if not Path(i).is_absolute() else Path(i), root),
                           "result_sha256": result_sha256(root / i if not Path(i).is_absolute() else Path(i))}
                          for i in inputs]}
    write_json(out, "config.json", config)
    write_json(out, "result.json", result)
    write_json(out, "provenance.json", prov)
    write_json(out, "lineage.json", lineage)
    binding = pr.bind_evidence(protocol=protocol, ledger_path=root / pr.LEDGER_PATH, analysis=analysis,
                               dataset_ids=dataset_ids, config=config, result=result, provenance=prov,
                               extra={"seeds": seeds or {}, "lineage_sha256": document_sha256(lineage),
                                      **(extra or {})})
    write_json(out, "binding.json", binding)
    seal_artifacts(out)
    return binding


def read_run(out: str | Path) -> dict:
    out = Path(out)
    return {name: json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
            for name in ("config", "result", "binding", "provenance", "lineage")}


def verify_run(out: str | Path, *, root: Path = PROJECT_ROOT, check_source: bool = False) -> dict:
    out = Path(out)
    issues: list[str] = []
    analysis = None
    try:
        issues.extend(verify_artifacts(out)["issues"])
        stored = read_run(out)
        analysis = stored["binding"].get("analysis")
        protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
        issues.extend(pr.verify_binding(stored["binding"], protocol=protocol, ledger_path=root / pr.LEDGER_PATH,
                                        config=stored["config"], result=stored["result"]))
        recorded = stored["provenance"]
        if hashlib.sha256(canonical_json(recorded["source_files"]).encode()).hexdigest() != recorded["source_sha256"]:
            issues.append("provenance source manifest does not match its digest")
        if stored["binding"].get("source_sha256") != recorded["source_sha256"]:
            issues.append("binding and provenance disagree on the source digest")
        if stored["binding"]["extra"].get("lineage_sha256") != document_sha256(stored["lineage"]):
            issues.append("lineage changed since binding")
        for upstream in stored["lineage"]["inputs"]:
            path = root / upstream["run"]
            if not (path / "result.json").is_file():
                issues.append(f"upstream run missing: {upstream['run']}")
            elif result_sha256(path) != upstream["result_sha256"]:
                issues.append(f"upstream result changed: {upstream['run']}")
        if check_source and recorded["source_files"] != source_manifest(root):
            issues.append("current implementation differs from the run's recorded source")
        text = json.dumps(stored)
        for marker in (":\\\\Users", "/home/", "/Users/"):
            if marker in text:
                issues.append(f"absolute machine path in artifact metadata ({marker})")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(str(exc))
    return {"valid": not issues, "issues": issues, "analysis": analysis, "meaning": MEANING}


def runs_under(base: Path) -> list[Path]:
    return sorted(p.parent for p in Path(base).rglob("binding.json"))


def verify_tree(base: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    runs = runs_under(Path(base))
    reports = {p.relative_to(base).as_posix() or ".": verify_run(p, root=root) for p in runs}
    return {"valid": bool(reports) and all(r["valid"] for r in reports.values()), "runs": len(reports),
            "invalid": sorted(k for k, r in reports.items() if not r["valid"]), "reports": reports,
            "meaning": MEANING}


def dag(base: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    """Experiment DAG: nodes are sealed runs, edges are lineage inputs (upstream -> downstream)."""
    nodes, edges = [], []
    for run in runs_under(Path(base)):
        rel = _rel(run, root)
        lineage = json.loads((run / "lineage.json").read_text(encoding="utf-8"))
        binding = json.loads((run / "binding.json").read_text(encoding="utf-8"))
        nodes.append({"run": rel, "analysis": lineage["analysis"], "result_sha256": binding["result_sha256"],
                      "protocol_sha256": binding["protocol_sha256"], "datasets": sorted(binding["datasets"])})
        edges.extend({"from": i["run"], "to": rel, "result_sha256": i["result_sha256"]} for i in lineage["inputs"])
    known = {n["run"] for n in nodes}
    dangling = [e for e in edges if e["from"] not in known]
    return {"schema": "cleolob-v08-dag-1", "nodes": nodes, "edges": edges, "dangling_inputs": dangling,
            "acyclic": _acyclic(nodes, edges)}


def _acyclic(nodes: list[dict], edges: list[dict]) -> bool:
    graph: dict[str, list[str]] = {n["run"]: [] for n in nodes}
    for e in edges:
        graph.setdefault(e["from"], []).append(e["to"])
    state: dict[str, int] = {}

    def visit(v: str) -> bool:
        if state.get(v) == 1:
            return False
        if state.get(v) == 2:
            return True
        state[v] = 1
        ok = all(visit(w) for w in graph.get(v, []))
        state[v] = 2
        return ok

    return all(visit(v) for v in list(graph))
