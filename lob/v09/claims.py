"""claim-audit-v09: claim graph and fail-closed wording audit (M89).

Graph: claim -> hypothesis -> final-analysis pointer -> sealed run (result hash, protocol hash) -> lineage. Documents
(the README v0.9 section and every ``docs/v09-*.md``) cite claims with ``[C09-<id>]`` markers. The audit fails when:

* a marker does not resolve to a registered claim, or a claim's pointer does not resolve in the sealed final run;
* a line with a hypothesis marker states a status word that differs from the evidence status;
* the final report or the paper omits a marker for any of H1-H36;
* a high-risk term (profit, alpha, live trading, HFT, production, exact fills, queue positions, causal, ...) appears
  without a qualification on the same line.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from ..v08.claims import QUALIFIER, RISK, _pointer
from . import artifacts, protocol as pr

MARKER = re.compile(r"\[C09-([A-Za-z0-9-]+)\]")
STATUS = re.compile(r"\b(NOT_ESTABLISHED|ESTABLISHED|FAILED|INVALID|INCONCLUSIVE|NOT_AVAILABLE|NOT_EVALUABLE|"
                    r"NOT_PROMOTED)\b")
README_START, README_END = "<!-- v09:start -->", "<!-- v09:end -->"
REQUIRED_DOCS = ("docs/v09-final-report.md", "docs/v09-paper.md")
EXEMPT_DOCS = ("docs/v09-implementation-plan.md",)


def _final_run(root: Path, final: str | None, bundle: Path | None) -> Path | None:
    if bundle is not None:
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        rel = Path(manifest["final"]).relative_to("results/v09")
        return bundle / "runs" / rel
    if final:
        return root / final
    found = [r for r in artifacts.runs_under(root / "results/v09")
             if json.loads((r / "binding.json").read_text(encoding="utf-8"))["analysis"] == "final-analysis-v09"]
    return sorted(found, key=lambda p: p.stat().st_mtime)[-1] if found else None


def graph(root: Path, *, final: str | None = None, bundle: Path | None = None) -> dict:
    registry = json.loads(((bundle / "configs" / "claim-registry.json") if bundle else
                           root / pr.CONFIG_DIR / "claim-registry.json").read_text(encoding="utf-8"))
    run = _final_run(root, final, bundle)
    result = binding = lineage = None
    if run is not None and (run / "result.json").is_file():
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        binding = json.loads((run / "binding.json").read_text(encoding="utf-8"))
        lineage = json.loads((run / "lineage.json").read_text(encoding="utf-8"))
    nodes = []
    for cid, c in registry["claims"].items():
        node = {"claim": cid, "hypothesis": c.get("hypothesis"), "statement": c["statement"], "pointer": c["pointer"],
                "run": None, "value": None, "status": None, "resolved": False}
        if result is not None:
            value = _pointer(result, c["pointer"])
            node.update(run=run.relative_to(bundle if bundle else root).as_posix(), value=value,
                        status=value if isinstance(value, str) else None, resolved=value is not None,
                        result_sha256=binding["result_sha256"], protocol_sha256=binding["protocol_sha256"],
                        sources=[i["run"] for i in lineage["inputs"]])
        nodes.append(node)
    return {"schema": "cleolob-v09-claim-graph-1", "claims": nodes,
            "edges": [{"from": n["claim"], "to": n["hypothesis"] or n["pointer"]} for n in nodes]}


def documents(root: Path) -> dict[str, list[str]]:
    docs = {}
    readme = root / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        docs["README.md"] = (text.split(README_START, 1)[1].split(README_END, 1)[0].splitlines()
                             if README_START in text and README_END in text else [])
    for path in sorted((root / "docs").glob("v09-*.md")):
        docs[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8").splitlines()
    return docs


def audit(root: Path, *, final: str | None = None, bundle: Path | None = None) -> dict:
    root = Path(root)
    issues, warnings = [], []
    g = graph(root, final=final, bundle=bundle)
    claims = {n["claim"]: n for n in g["claims"]}
    for n in g["claims"]:
        if not n["resolved"]:
            issues.append(f"claim {n['claim']} does not resolve to sealed evidence ({n['pointer']})")
    docs = documents(root)
    if "README.md" in docs and not docs["README.md"]:
        issues.append("README.md has no v0.9 section delimited by the v09 markers")
    seen: dict[str, set] = {}
    for name, lines in docs.items():
        in_code = False
        for number, line in enumerate(lines, 1):
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            for cid in MARKER.findall(line):
                seen.setdefault(name, set()).add(cid)
                node = claims.get(cid)
                if node is None:
                    issues.append(f"{name}:{number}: unknown claim marker C09-{cid}")
                    continue
                stated = set(STATUS.findall(line))
                if node["hypothesis"] and node["status"] and stated and node["status"] not in stated:
                    issues.append(f"{name}:{number}: C09-{cid} states {sorted(stated)} but the evidence says "
                                  f"{node['status']}")
            if in_code or name in EXEMPT_DOCS:
                continue
            for term in RISK.findall(line):
                if not QUALIFIER.search(line):
                    issues.append(f"{name}:{number}: high-risk term {term!r} without a qualification")
    for required in REQUIRED_DOCS:
        if required not in docs:
            issues.append(f"{required} is missing")
            continue
        absent = [f"H{k}" for k in range(1, 37) if f"H{k}" not in seen.get(required, set())]
        if absent:
            issues.append(f"{required} has no claim markers for {absent}")
    unused = sorted(set(claims) - set().union(*seen.values())) if seen else sorted(claims)
    if unused:
        warnings.append(f"registered claims not cited in any document: {unused}")
    return {"valid": not issues, "issues": issues, "warnings": warnings, "claims": len(claims),
            "markers": {k: sorted(v) for k, v in seen.items()}, "graph": g}


def audit_command(args) -> dict:
    return audit(Path(args.root), final=args.final, bundle=args.bundle)
