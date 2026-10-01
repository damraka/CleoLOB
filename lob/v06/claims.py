"""M15: machine-readable claim graph and automated claim audit.

A claim links claim -> hypothesis -> result -> dataset(s) -> protocol -> config ->
evidence artifact -> status, with the estimate, interval, alpha and family size
taken directly from sealed result files. ``audit`` checks that

* every claim's evidence run verifies (bytes + binding) and its numbers are
  re-derivable from the run's ``result.json`` at the recorded JSON path;
* the documentation claim table (rows ``| Hn | STATUS | ...``) matches the
  machine-readable statuses exactly;
* documentation sentences containing high-risk terms (profitability, alpha, HFT,
  exact FIFO/fills, causal real-market effects) also contain a negation or
  limitation marker.

It does not verify natural language generally; it flags mismatches for a human.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from ..experiments.registry import PROJECT_ROOT
from . import protocol as pr
from .evidence import verify_run

SCHEMA = "cleolob-v06-claims-1"
FIELDS = ("claim_id", "study_id", "hypothesis_id", "dataset_ids", "dataset_roles", "protocol_sha256",
          "config_sha256", "result_sha256", "evidence", "result_path", "metric", "threshold", "estimate", "ci_low",
          "ci_high", "alpha", "family", "family_size", "status", "evidence_level", "interpretation")
RISK_TERMS = re.compile(r"\b(profitab\w*|alpha\b(?![-_ ]?(level|=|0\.))|HFT|high[- ]frequency trading|exact FIFO|"
                        r"exact (historical )?(passive )?fills?|causal|live trading|production trading)", re.I)
NEGATION = re.compile(r"\b(no|not|never|cannot|without|nor|neither|none|withheld|unsupported|excluded?|"
                      r"limitation|does not|do not|is not|are not)\b", re.I)
EVIDENCE_LEVELS = ("confirmatory", "retrospective", "descriptive", "exploratory", "pilot")


def _get(document: dict, path: str):
    value = document
    for part in path.split("/"):
        if part == "":
            continue
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def make_claim(*, claim_id: str, hypothesis_id: str | None, run: Path, result_path: str, metric: str,
               threshold: str, status_path: str | None, family: str | None, family_size: int | None,
               alpha: float | None, evidence_level: str, interpretation: str, root: Path = PROJECT_ROOT,
               estimate_key: str = "estimate", status: str | None = None) -> dict:
    """Build one claim from a sealed run; numbers are read from the run, never typed by hand."""
    if evidence_level not in EVIDENCE_LEVELS:
        raise ValueError(f"unknown evidence level {evidence_level}")
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    binding = json.loads((run / "binding.json").read_text(encoding="utf-8"))
    node = _get(result, result_path)
    value = node if not isinstance(node, dict) else node.get(estimate_key, node.get("mean", node.get("difference")))
    claim = {"claim_id": claim_id, "study_id": binding["analysis"], "hypothesis_id": hypothesis_id,
             "dataset_ids": sorted(binding["datasets"]),
             "dataset_roles": {k: v["role"] for k, v in binding["datasets"].items()},
             "protocol_sha256": binding["protocol_sha256"], "config_sha256": binding["config_sha256"],
             "result_sha256": binding["result_sha256"],
             "evidence": run.relative_to(root).as_posix() if run.is_relative_to(root) else str(run),
             "result_path": result_path, "metric": metric, "threshold": threshold,
             "estimate": value if isinstance(value, (int, float, str)) or value is None else None,
             "ci_low": node.get("ci_low") if isinstance(node, dict) else None,
             "ci_high": node.get("ci_high") if isinstance(node, dict) else None, "alpha": alpha, "family": family,
             "family_size": family_size,
             "status": status if status is not None else _get(result, status_path) if status_path else None,
             "evidence_level": evidence_level, "interpretation": interpretation, "status_path": status_path}
    missing = [f for f in FIELDS if f not in claim]
    if missing:
        raise ValueError(f"claim {claim_id} missing {missing}")
    return claim


def write_claims(path: Path, claims: list[dict], protocol_sha256: str) -> dict:
    ids = [c["claim_id"] for c in claims]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate claim ids")
    document = {"schema": SCHEMA, "protocol_sha256": protocol_sha256, "claims": claims}
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    return document


def doc_table(text: str) -> dict[str, str]:
    """Rows of a markdown claim table: '| <claim_id> | <STATUS> | ...'."""
    table = {}
    for line in text.splitlines():
        match = re.match(r"\|\s*`?([A-Za-z0-9_.:-]+)`?\s*\|\s*\**([A-Z_]+)\**\s*\|", line)
        if match and match.group(2) in set(pr.STATUSES) | {"EQUIVALENT_WITHIN_MARGIN", "FAILED_MARGIN",
                                                            "NOT_EVALUABLE"}:
            table[match.group(1)] = match.group(2)
    return table


def risky_sentences(text: str) -> list[str]:
    flagged = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        if RISK_TERMS.search(sentence) and not NEGATION.search(sentence):
            flagged.append(sentence.strip())
    return flagged


def audit(claims_path: Path, docs: list[Path], *, root: Path = PROJECT_ROOT, check_runs: bool = True) -> dict:
    document = json.loads(claims_path.read_text(encoding="utf-8"))
    issues: list[str] = []
    if document.get("schema") != SCHEMA:
        issues.append("unsupported claims schema")
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    if document.get("protocol_sha256") != pr.protocol_sha256(protocol):
        issues.append("claims were built under a different protocol hash")
    verified: dict[str, bool] = {}
    for claim in document.get("claims", []):
        run = root / claim["evidence"]
        if check_runs:
            if claim["evidence"] not in verified:
                verified[claim["evidence"]] = verify_run(run, root=root)["valid"]
            if not verified[claim["evidence"]]:
                issues.append(f"{claim['claim_id']}: evidence run does not verify")
                continue
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        binding = json.loads((run / "binding.json").read_text(encoding="utf-8"))
        if binding["result_sha256"] != claim["result_sha256"]:
            issues.append(f"{claim['claim_id']}: result hash differs from the bound result")
        if claim.get("status_path") and _get(result, claim["status_path"]) != claim["status"]:
            issues.append(f"{claim['claim_id']}: status differs from the result file")
        node = _get(result, claim["result_path"])
        for key in ("ci_low", "ci_high"):
            if isinstance(node, dict) and node.get(key) != claim[key]:
                issues.append(f"{claim['claim_id']}: {key} differs from the result file")
        if claim["status"] not in set(pr.STATUSES) | {"EQUIVALENT_WITHIN_MARGIN", "FAILED_MARGIN", "NOT_EVALUABLE"}:
            issues.append(f"{claim['claim_id']}: unregistered status {claim['status']}")
    statuses = {c["claim_id"]: c["status"] for c in document.get("claims", [])}
    flagged = {}
    for doc in docs:
        text = doc.read_text(encoding="utf-8")
        for claim_id, status in doc_table(text).items():
            if claim_id in statuses and statuses[claim_id] != status:
                issues.append(f"{doc.name}: {claim_id} says {status}, artifacts say {statuses[claim_id]}")
            elif claim_id not in statuses and re.fullmatch(r"H\d+", claim_id):
                issues.append(f"{doc.name}: {claim_id} has no machine-readable claim")
        risky = risky_sentences(text)
        if risky:
            flagged[doc.name] = risky
    return {"valid": not issues, "issues": issues, "claims": len(statuses), "runs_checked": len(verified),
            "risky_sentences_for_review": flagged,
            "meaning": "artifact/document consistency and run verification; not independent replication"}
