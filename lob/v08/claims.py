"""Claim graph and fail-closed claim audit (M35, brief section 28).

Graph: claim -> hypothesis -> test -> result artifact (run, JSON pointer, value, result hash) -> source runs
(lineage) -> protocol binding (protocol hash of the run). Documents (README v0.8 section, every ``docs/v08-*.md``)
carry markers ``[C08-<claim id>]``. The audit fails when:

* a marker does not resolve to a registered claim, or a claim's evidence run is missing;
* a line with a hypothesis marker states a status that differs from the evidence status;
* the final report or the paper omits any of H1-H21;
* a high-risk term appears without a qualification on the same line (negation, SYNTHETIC/RETROSPECTIVE,
  bounded/hypothetical, model/simulation wording, an explicit status word, ...).
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from . import artifacts, protocol as pr

MARKER = re.compile(r"\[C08-([A-Za-z0-9-]+)\]")
STATUS = re.compile(r"\b(NOT_ESTABLISHED|ESTABLISHED|FAILED|INVALID|INCONCLUSIVE|NOT_AVAILABLE|EXPLORATORY|"
                    r"ASSUMPTION_DEPENDENT|NOT_EVALUABLE)\b")
RISK = re.compile(r"\b(profit\w*|alpha|production[- ]ready|production|live[- ]trading|hft|high[- ]frequency|guarantee\w*|"
                  r"prove[sn]?|outperform\w*|superior\w*|validated|exact (?:historical )?fills?|actual historical fills?|"
                  r"queue positions?|hidden liquidity|causal\w*|permanent impact|robust(?:ly|ness)?)\b", re.I)
QUALIFIER = re.compile(r"\b(not|no|never|without|cannot|can't|none|nothing|neither|nor|only|synthetic|simulat\w*|"
                       r"bounded|hypothetical|model\w*|registered|assumption\w*|exploratory|retrospective|stylized|"
                       r"local research|research|under|within|does not|is not|are not|NOT_\w+|ESTABLISHED|FAILED|"
                       r"INCONCLUSIVE|criterion|criteria|optimization|optimi[sz]\w*|regret|cvar|claim\w*|bound\w*|"
                       r"evaluated|tested|hypothes\w*|question|whether|if)\b", re.I)
README_START, README_END = "<!-- v08:start -->", "<!-- v08:end -->"


def _pointer(doc, pointer: str):
    cur = doc
    for part in [p for p in pointer.split("/") if p]:
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)]
        else:
            return None
    return cur


def _retained_attempts(ledger: Path) -> set[str]:
    """Run directories (relative to ``results/v08``) that the ledger records as INVALID/ABORTED/... attempts."""
    if not ledger.is_file():
        return set()
    out = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line) if line.strip() else {}
        directory = (entry.get("payload") or {}).get("directory") if entry.get("event") == "attempt" else None
        if directory:
            out.add(directory.removeprefix("results/v08/").strip("/"))
    return out


def _runs_by_analysis(base: Path, excluded: set[str] = frozenset()) -> dict[str, Path]:
    """Latest sealed run per analysis, never a run the ledger retains as a failed attempt."""
    found: dict[str, Path] = {}
    if not base.is_dir():
        return found
    for run in artifacts.runs_under(base):
        if "/dev/" in run.as_posix() + "/" or run.relative_to(base).as_posix() in excluded:
            continue
        binding = json.loads((run / "binding.json").read_text(encoding="utf-8"))
        prev = found.get(binding["analysis"])
        if prev is None or run.stat().st_mtime >= prev.stat().st_mtime:
            found[binding["analysis"]] = run
    return found


def graph(root: Path, *, final: str = "results/v08/final", bundle: Path | None = None) -> dict:
    base = bundle / "runs" if bundle else root / "results/v08"
    registry = json.loads(((bundle / "configs" / "claim-registry.json") if bundle else
                           root / pr.CONFIG_DIR / "claim-registry.json").read_text(encoding="utf-8"))
    ledger = (bundle / "configs" / Path(pr.LEDGER_PATH).name) if bundle else root / pr.LEDGER_PATH
    runs = _runs_by_analysis(base, _retained_attempts(ledger))
    final_dir = (bundle / "runs" / Path(final).name) if bundle else root / final
    if (final_dir / "result.json").is_file():
        runs["final_analysis"] = final_dir
    nodes = []
    for cid, c in registry["claims"].items():
        run = runs.get(c["source_analysis"])
        node = {"claim": cid, "hypothesis": c["hypothesis"], "test": c["test"], "statement": c["statement"],
                "pointer": c["pointer"], "run": None, "value": None, "status": None, "resolved": False}
        if run is not None:
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            binding = json.loads((run / "binding.json").read_text(encoding="utf-8"))
            lineage = json.loads((run / "lineage.json").read_text(encoding="utf-8"))
            value = _pointer(result, c["pointer"])
            node.update(run=run.relative_to(bundle if bundle else root).as_posix(), value=value,
                        status=value if isinstance(value, str) else None, resolved=value is not None,
                        result_sha256=binding["result_sha256"], protocol_sha256=binding["protocol_sha256"],
                        sources=[i["run"] for i in lineage["inputs"]])
        nodes.append(node)
    return {"schema": "cleolob-v08-claim-graph-1", "claims": nodes,
            "edges": [{"from": n["claim"], "to": n["hypothesis"] or n["test"]} for n in nodes]}


def _documents(root: Path) -> dict[str, list[str]]:
    docs = {}
    readme = root / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        if README_START in text and README_END in text:
            docs["README.md"] = text.split(README_START, 1)[1].split(README_END, 1)[0].splitlines()
        else:
            docs["README.md"] = []
    for path in sorted((root / "docs").glob("v08-*.md")):
        docs[path.relative_to(root).as_posix()] = path.read_text(encoding="utf-8").splitlines()
    return docs


def audit(root: Path, *, final: str = "results/v08/final", bundle: Path | None = None) -> dict:
    root = Path(root)
    issues, warnings = [], []
    g = graph(root, final=final, bundle=bundle)
    claims = {n["claim"]: n for n in g["claims"]}
    for n in g["claims"]:
        if not n["resolved"]:
            issues.append(f"claim {n['claim']} does not resolve to sealed evidence ({n['pointer']})")
    docs = _documents(root)
    if "README.md" in docs and not docs["README.md"]:
        issues.append("README.md has no v0.8 section delimited by the v08 markers")
    seen: dict[str, set] = {}
    in_code = False
    for name, lines in docs.items():
        for number, line in enumerate(lines, 1):
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            for cid in MARKER.findall(line):
                seen.setdefault(name, set()).add(cid)
                node = claims.get(cid)
                if node is None:
                    issues.append(f"{name}:{number}: unknown claim marker C08-{cid}")
                    continue
                stated = set(STATUS.findall(line))
                if node["hypothesis"] and node["status"] and stated and node["status"] not in stated:
                    issues.append(f"{name}:{number}: C08-{cid} states {sorted(stated)} but the evidence says "
                                  f"{node['status']}")
            if in_code or name == "docs/v08-implementation-plan.md":
                continue
            for term in RISK.findall(line):
                if not QUALIFIER.search(line):
                    issues.append(f"{name}:{number}: high-risk term {term!r} without a qualification")
    for required in ("docs/v08-final-report.md", "docs/v08-paper.md"):
        if required not in docs:
            issues.append(f"{required} is missing")
            continue
        absent = [f"H{k}" for k in range(1, 22) if f"H{k}" not in seen.get(required, set())]
        if absent:
            issues.append(f"{required} has no claim markers for {absent}")
    unused = sorted(set(claims) - set().union(*seen.values())) if seen else sorted(claims)
    if unused:
        warnings.append(f"registered claims not cited in any document: {unused}")
    return {"valid": not issues, "issues": issues, "warnings": warnings, "claims": len(claims),
            "markers": {k: sorted(v) for k, v in seen.items()}, "graph": g}
