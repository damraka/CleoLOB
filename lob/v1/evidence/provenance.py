"""Provenance graph over sources, datasets, designs, code and runs (M5).

Nodes: ``dataset:<id>`` (with source hashes from the ledger), ``design:<name>`` (sealed hash), ``code:<sha>`` (lob
source digest), ``commit:<sha>``, ``run:<path>`` (result hash, status). Edges: ``read`` (run -> dataset),
``governed_by`` (run -> design), ``built_from`` (run -> code, run -> commit), ``uses`` (run -> upstream run). The
graph is acyclic by construction; ``build`` verifies it and ``lineage`` returns every node a run depends on.
"""
from __future__ import annotations

from pathlib import Path

from .runs import read_json, runs_under


def build(base: Path, root: Path) -> dict:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for run in runs_under(base):
        rel = run.relative_to(root).as_posix() if run.is_absolute() else run.as_posix()
        b, p, s = read_json(run / "binding.json"), read_json(run / "provenance.json"), read_json(run / "status.json")
        sums = read_json(run / "checksums.json")
        rid = f"run:{rel}"
        nodes[rid] = {"kind": "run", "status": s["status"], "analysis": p.get("analysis"),
                      "result_sha256": sums.get("result.json"), "namespace": b.get("namespace")}
        code = f"code:{p['code_sha256']}"
        nodes.setdefault(code, {"kind": "code"})
        edges.append({"from": rid, "to": code, "kind": "built_from"})
        if p.get("git_commit"):
            commit = f"commit:{p['git_commit']}"
            nodes.setdefault(commit, {"kind": "commit", "dirty_at_some_run": False})
            nodes[commit]["dirty_at_some_run"] |= bool(p.get("git_dirty"))
            edges.append({"from": rid, "to": commit, "kind": "built_from"})
        if b.get("design"):
            did = f"design:{b['design']}"
            nodes.setdefault(did, {"kind": "design", "sha256": b.get("design_sha256")})
            edges.append({"from": rid, "to": did, "kind": "governed_by"})
        for d, info in b.get("datasets", {}).items():
            nid = f"dataset:{d}"
            nodes.setdefault(nid, {"kind": "dataset", "source_sha256": info.get("source_sha256", {})})
            edges.append({"from": rid, "to": nid, "kind": "read"})
        for key, up in b.get("upstream", {}).items():
            edges.append({"from": rid, "to": f"run:{up['run']}", "kind": "uses", "key": key,
                          "result_sha256": up["result_sha256"]})
    issues = []
    for e in edges:
        if e["kind"] == "uses":
            tgt = nodes.get(e["to"])
            if tgt is None:
                issues.append(f"{e['from']} uses missing run {e['to']}")
            elif tgt.get("result_sha256") != e["result_sha256"]:
                issues.append(f"{e['from']} bound to a different result of {e['to']}")
    if not _acyclic(nodes, edges):
        issues.append("provenance graph has a cycle")
    return {"nodes": nodes, "edges": edges, "issues": issues, "valid": not issues}


def _acyclic(nodes: dict, edges: list[dict]) -> bool:
    out: dict[str, list[str]] = {}
    for e in edges:
        out.setdefault(e["from"], []).append(e["to"])
    state: dict[str, int] = {}

    def visit(n: str) -> bool:
        if state.get(n) == 1:
            return False
        if state.get(n) == 2:
            return True
        state[n] = 1
        ok = all(visit(m) for m in out.get(n, []))
        state[n] = 2
        return ok

    return all(visit(n) for n in list(out))


def lineage(graph: dict, node: str) -> list[str]:
    out: dict[str, list[str]] = {}
    for e in graph["edges"]:
        out.setdefault(e["from"], []).append(e["to"])
    seen, stack = [], [node]
    while stack:
        n = stack.pop()
        for m in out.get(n, []):
            if m not in seen:
                seen.append(m)
                stack.append(m)
    return sorted(seen)


__all__ = ["build", "lineage"]
