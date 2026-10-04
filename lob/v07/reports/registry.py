"""Result registry, hypothesis table, claim graph and claim audit (workstreams 63, 66, 68, 69).

``build_registry(root)`` scans every v0.7 run directory and every ledger attempt and returns one searchable
manifest: sealed runs (analysis, datasets, commit, dirty flag, source digest, verification status) and
unsealed attempts (FAILED / ABORTED / INVALID / NOT_AVAILABLE), so negative and failed evidence is never
filtered out. ``hypothesis_table`` reads each registered hypothesis from its sealed run by a fixed JSON
location and reports estimate, interval, margin, alpha, family, MDE, status and limitations (effect size
first). Missing runs make a hypothesis NOT_AVAILABLE (pending), never silently absent.

The claim graph links claim -> hypothesis -> protocol -> datasets -> run -> statistic -> artifact (result
hash) -> table/figure -> sentence (documentation lines carrying the claim marker ``[C-Hn]``). The audit
fails on unresolved markers, statuses that differ from the sealed result, and high-risk terms without a
qualification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from ...experiments.registry import PROJECT_ROOT
from ..evidence.runs import verify_run
from ..protocol import core as pr

SCHEMA_REGISTRY = "cleolob-v07-result-registry-1"
SCHEMA_GRAPH = "cleolob-v07-claim-graph-1"
POSTERIOR_LIMITATION = ("depends on the G0 SMC-ABC posterior, whose synthetic recovery is ASSUMPTION_DEPENDENT "
                        "(2 of 3 recovery cases below the registered 80% coverage)")

ETH = ("deribit-eth-perp-2020-11-01", "deribit-eth-perp-2020-12-01")
RICHER = ("G1_state_hawkes", "G2_regime_switching", "G3_conditional_ar", "G4_neural_temporal")
RUNS = {"holdout": "results/v07/m17", "posterior": "results/v07/m6/posterior", "execution": "results/v07/m15/execution",
        "transfer": "results/v07/m16/transfer"}


def resolve(root: Path, run: str) -> str:
    """The latest sealed variant of a run directory (``run``, ``run-2``, ``run-3``, ...); aborted or unsealed
    attempts are skipped (they stay in the ledger and on disk)."""
    candidates = [run] + [f"{run}-{k}" for k in range(2, 10)]
    sealed = [c for c in candidates if (root / c / "binding.json").is_file()]
    return sealed[-1] if sealed else run


def _load(root: Path, run: str) -> dict | None:
    path = root / resolve(root, run) / "result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def build_registry(root: Path = PROJECT_ROOT) -> dict:
    runs = []
    for binding in sorted((root / "results/v07").rglob("binding.json")):
        run = binding.parent
        b = json.loads(binding.read_text(encoding="utf-8"))
        prov = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        runs.append({"run": run.relative_to(root).as_posix(), "analysis": b["analysis"],
                     "datasets": {k: {"role": v["role"], "freshness": v["freshness"]} for k, v in b["datasets"].items()},
                     "ledger_anchor": b["ledger_anchor"], "result_sha256": b["result_sha256"],
                     "git_commit": prov.get("git_commit"), "git_dirty": prov.get("git_dirty"),
                     "source_sha256": prov.get("source_sha256"), "valid": verify_run(run, root=root)["valid"]})
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), pr.load_protocol(root / pr.PROTOCOL_PATH))
    attempts = [{"design": a["design"], "outcome": a["outcome"], "directory": a.get("directory"), "note": a.get("note"),
                 "ledger_index": a["index"]} for a in state.attempts]
    return {"schema": SCHEMA_REGISTRY, "runs": runs, "attempts": attempts, "ledger_head": state.head,
            "ledger_entries": state.count, "protocol_sha256": state.protocol_sha256,
            "consumed": sorted(d for d in state.consumed if not d.startswith("period:")),
            "fresh_remaining": sorted(d for d in state.fresh if d not in state.consumed)}


def _row(hid, question, dataset, model, status, *, estimate=None, ci=None, margin=None, alpha=None, family=None,
         mde=None, n=None, run=None, limitations="", members=None):
    return {"hypothesis": hid, "question": question, "dataset": dataset, "model": model, "n": n, "estimate": estimate,
            "interval": ci, "margin": margin, "alpha": alpha, "family": family, "power_or_mde": mde, "status": status,
            "run": run, "limitations": limitations, "members": members or []}


def _conjunction(statuses):
    if not statuses or any(s is None for s in statuses):
        return "NOT_AVAILABLE"
    if all(s == "ESTABLISHED" for s in statuses):
        return "ESTABLISHED"
    if any(s in ("INVALID",) for s in statuses):
        return "INVALID"
    return "FAILED" if any(s == "FAILED" for s in statuses) else "NOT_ESTABLISHED"


def _any(statuses):
    if not statuses or all(s in (None, "NOT_AVAILABLE") for s in statuses):
        return "NOT_AVAILABLE"
    return "ESTABLISHED" if "ESTABLISHED" in statuses else ("FAILED" if all(s == "FAILED" for s in statuses)
                                                           else "NOT_ESTABLISHED")


def hypothesis_table(root: Path = PROJECT_ROOT) -> list[dict]:
    hyp = json.loads((root / pr.CONFIG_DIR / "hypotheses.json").read_text(encoding="utf-8"))["hypotheses"]
    power = json.loads((root / pr.CONFIG_DIR / "power-design.json").read_text(encoding="utf-8"))["families"]
    fams = json.loads((root / pr.CONFIG_DIR / "statistical-families.json").read_text(encoding="utf-8"))["families"]

    def mde(h):
        f = power.get(hyp[h]["family"], {})
        return next((v for k, v in f.items() if k.startswith("mde") or k == "one_sided_resolution"), None)
    rows = []
    days = {d: _load(root, f"{RUNS['holdout']}/{d}") for d in (*ETH, "deribit-btc-perp-2020-11-01",
                                                               "bitmex-xbtusd-2020-11-01")}
    # H1, H10, H11 per fresh ETH day
    for h, key in (("H1", "H1"), ("H10", "H10"), ("H11", "H11")):
        members = []
        for d in ETH:
            r = days[d]
            entry = (r or {}).get("hypotheses", {}).get(key)
            members.append({"dataset": d, **(entry or {"status": None})})
        rows.append(_row(h, hyp[h]["question"], list(ETH), {"H1": "G0_post vs G0_point"}.get(h, "EA vs G0_point"),
                         _conjunction([m["status"] for m in members]), alpha=fams[hyp[h]["family"]]["adjusted_alpha"],
                         family=hyp[h]["family"], margin=hyp[h]["margin"], mde=mde(h), members=members,
                         run=[resolve(root, f"{RUNS['holdout']}/{d}") for d in ETH], limitations=POSTERIOR_LIMITATION))
    for h, kind, field in (("H2", "domain_gap", "status_H2"), ("H3", "support", "status_H3")):
        members = []
        for d in ETH:
            for g in RICHER:
                e = ((days[d] or {}).get(kind) or {}).get(g)
                members.append({"member": f"{g}@{d}", "estimate": (e or {}).get("difference"),
                                "interval": [(e or {}).get("ci_low"), (e or {}).get("ci_high")],
                                "status": (e or {}).get(field), "qualifiers": (e or {}).get("qualifiers", [])})
        rows.append(_row(h, hyp[h]["question"], list(ETH), "G1-G4 vs G0_point", _any([m["status"] for m in members]),
                         alpha=fams[hyp[h]["family"]]["adjusted_alpha"], family=hyp[h]["family"],
                         margin=hyp[h]["margin"], mde=mde(h), members=members,
                         run=[resolve(root, f"{RUNS['holdout']}/{d}") for d in ETH]))
    for h, d in (("H4", "deribit-btc-perp-2020-11-01"), ("H5", "bitmex-xbtusd-2020-11-01")):
        e = ((days[d] or {}).get("hypotheses") or {}).get(h)
        rows.append(_row(h, hyp[h]["question"], d, (e or {}).get("G_star", "G*"), (e or {}).get("status", "NOT_AVAILABLE")
                         if days[d] else "NOT_AVAILABLE", estimate=(e or {}).get("estimate"), ci=(e or {}).get("ci"),
                         alpha=fams["F4_cross_market"]["adjusted_alpha"], family="F4_cross_market", mde=mde(h),
                         run=resolve(root, f"{RUNS['holdout']}/{d}"), limitations=POSTERIOR_LIMITATION if (e or {}).get("G_star") ==
                         "G0_post" else ""))
    post = _load(root, RUNS["posterior"])
    rows.append(_row("H6", hyp["H6"]["question"], "development", "G0_post",
                     (post or {}).get("H6", {}).get("status", "NOT_AVAILABLE"), family="D1_posterior_shape",
                     run=resolve(root, RUNS["posterior"]), limitations=POSTERIOR_LIMITATION + "; descriptive",
                     members=[{"multimodal_runs": (post or {}).get("H6", {}).get("multimodal_runs")}]))
    ex = _load(root, RUNS["execution"])
    for h, key in (("H7", "H7"), ("H8", "H8"), ("H9", "H9")):
        e = (ex or {}).get(key, {})
        members = (list(e.get("agents", {}).items()) if h == "H7" else
                   [(k, {"edge": v["edge"], "mean": v["pooled_mean_bps"], "ci": v["ci"]}) for k, v in
                    e.get("edges", {}).items()] if h == "H8" else [])
        rows.append(_row(h, hyp[h]["question"], "simulation (development-calibrated worlds)", "plausible world set",
                         e.get("status", "NOT_AVAILABLE"), family=hyp[h]["family"],
                         alpha=fams.get(hyp[h]["family"], {}).get("adjusted_alpha"), margin=hyp[h]["margin"],
                         mde=mde(h), run=resolve(root, RUNS["execution"]), limitations=POSTERIOR_LIMITATION,
                         members=[{"member": k, **v} for k, v in members]))
    tr = _load(root, RUNS["transfer"])
    for h in ("H12", "H13"):
        e = (tr or {}).get(h, {})
        rows.append(_row(h, hyp[h]["question"], "deribit-eth-perp-2021-01-01", "learned/classical policies",
                         e.get("status", "NOT_AVAILABLE"), family=hyp[h]["family"],
                         alpha=fams.get(hyp[h]["family"], {}).get("adjusted_alpha"), mde=mde(h), run=resolve(root, RUNS["transfer"]),
                         limitations="bounded historical replay (no impact, no exact fills); " + POSTERIOR_LIMITATION,
                         members=[{"member": k, **v} for k, v in (e.get("members") or e.get("pairs") or {}).items()]))
    order = {f"H{i}": i for i in range(1, 14)}
    return sorted(rows, key=lambda r: order[r["hypothesis"]])


# ----------------------------------------------------------------------------- claim graph and audit

MARKER = re.compile(r"\[C-(H\d{1,2})\]")
RISKY = ("realistic", "validated", "robust", "generalizes", "identified", "equivalent", "stable", "superior",
         "improves", "predictive", "transfer", "causal", "accurate", "production", "hft", "profit", "alpha", "optimal")
NEGATIONS = ("not ", "no ", "never", "none", "nothing", "neither", "nor ", "cannot", "without", "withheld",
             "failed", "not_established", "not established", "not_available", "inconclusive", "vacuous", "worse")
QUALIFIERS = ("only", "within", "under", "assum", "simulat", "registered", "bounded", "model_dependent", "exploratory",
              "relative", "descriptive", "if ", "whether", "limit", "caveat", "?", "h1", "h2", "h3", "h4", "h5", "h6",
              "h7", "h8", "h9", "h10", "h11", "h12", "h13")
REVIEW_PATH = "configs/v07/claim-review.json"
CLASSES = ("SUPPORTED", "QUALIFIED", "NEGATED", "OVERCLAIM")


def claim_graph(table: list[dict], docs: list[Path], root: Path = PROJECT_ROOT) -> dict:
    protocol = pr.protocol_sha256(pr.load_protocol(root / pr.PROTOCOL_PATH))
    sentences: dict[str, list[str]] = {}
    for doc in docs:
        for n, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for h in MARKER.findall(line):
                sentences.setdefault(h, []).append(f"{doc.relative_to(root).as_posix()}:{n}")
    nodes = []
    for row in table:
        runs = row["run"] if isinstance(row["run"], list) else [row["run"]]
        artifacts = {r: (pr.file_sha256(root / r / "result.json") if (root / r / "result.json").is_file() else None)
                     for r in runs if r}
        nodes.append({"claim": f"C-{row['hypothesis']}", "hypothesis": row["hypothesis"], "protocol_sha256": protocol,
                      "datasets": row["dataset"], "runs": runs, "statistic": {k: row[k] for k in
                                                                          ("estimate", "interval", "status")},
                      "artifacts": artifacts, "tables": ["docs/v07-final-report.md#hypotheses"],
                      "sentences": sentences.get(row["hypothesis"], [])})
    return {"schema": SCHEMA_GRAPH, "claims": nodes, "protocol_sha256": protocol}


def _identifier_only(line: str, term: str) -> bool:
    """The term appears only in code spans, links, a heading/topic label or as a significance level (not a claim)."""
    if line.lstrip().startswith("#"):
        return True
    stripped = re.sub(r"`[^`]*`|\[[^\]]*\]\([^)]*\)|alpha\s*[\d(=/]|posterior[ -]predictive|<!--.*?-->", " ",
                      line.lower())
    return not re.search(rf"\b{term}", stripped)


LIST_ITEM = re.compile(r"^\s*(?:[-*]|\d+\.)\s")


def _context(lines: list[str], n: int) -> str:
    """Lower-cased sentence context of line ``n`` (1-based).

    For a list item: the item plus its list header (the nearest earlier non-item line, if it ends with ':').
    Otherwise: the sentences of the paragraph that overlap the line.
    """
    i = n - 1
    if lines[i].startswith("  ") and not LIST_ITEM.match(lines[i]):   # continuation of a list item
        k = i
        while k > 0 and lines[k].startswith("  ") and not LIST_ITEM.match(lines[k]):
            k -= 1
        if LIST_ITEM.match(lines[k]):
            return f"{_context(lines, k + 1)} {lines[i].lower()}"
    if LIST_ITEM.match(lines[i]):
        j = i - 1
        while j >= 0 and (LIST_ITEM.match(lines[j]) or not lines[j].strip() or lines[j].startswith("  ")):
            j -= 1
        header = lines[j] if j >= 0 and lines[j].rstrip().endswith(":") else ""
        return f"{header} {lines[i]}".lower()
    start = i
    while start > 0 and lines[start - 1].strip() and not LIST_ITEM.match(lines[start - 1]):
        start -= 1
    end = i
    while end + 1 < len(lines) and lines[end + 1].strip() and not LIST_ITEM.match(lines[end + 1]):
        end += 1
    offset = sum(len(x) + 1 for x in lines[start:i])
    span = (offset, offset + len(lines[i]))
    text = " ".join(lines[start:end + 1])
    out, position = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        begin = text.index(sentence, position)
        position = begin + len(sentence)
        if begin < span[1] and position > span[0]:
            out.append(sentence)
    return " ".join(out).lower()


def review_key(file: str, term: str, text: str) -> str:
    return hashlib.sha256(f"{file}|{term}|{text}".encode()).hexdigest()[:16]


def term_audit(docs: list[Path], root: Path = PROJECT_ROOT, *, reviews: dict | None = None) -> list[dict]:
    """Classify every risky-term occurrence as SUPPORTED, QUALIFIED, NEGATED or OVERCLAIM.

    Automatic rules: identifiers, code and topic labels -> SUPPORTED; a negation in the term's sentence (or a
    list item's header) -> NEGATED; a qualifier or a hypothesis reference there -> QUALIFIED.
    Anything else needs a recorded manual review (``configs/v07/claim-review.json``); without one it stays
    REVIEW, which the audit reports as an issue.
    """
    reviews = reviews if reviews is not None else load_reviews(root)
    findings = []
    for doc in docs:
        lines = doc.read_text(encoding="utf-8").splitlines()
        fenced = False
        for n, line in enumerate(lines, 1):
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            low = line.lower()
            context = _context(lines, n).replace("*", "")
            for term in RISKY:
                if not re.search(rf"\b{term}", low):
                    continue
                file = doc.relative_to(root).as_posix()
                text = line.strip()[:200]
                if fenced or _identifier_only(line, term):
                    label, reason = "SUPPORTED", "identifier or topic label, not a claim"
                elif any(q in context for q in NEGATIONS):
                    label, reason = "NEGATED", "negated in its sentence or list header"
                elif any(q in context for q in QUALIFIERS):
                    label, reason = "QUALIFIED", "qualified or tied to a registered hypothesis"
                elif (r := reviews.get(review_key(file, term, text))) is not None:
                    label, reason = r["classification"], "manual review: " + r["reason"]
                else:
                    label, reason = "REVIEW", "needs manual review"
                findings.append({"file": file, "line": n, "term": term, "classification": label, "reason": reason,
                                 "text": text})
    return findings


def load_reviews(root: Path = PROJECT_ROOT) -> dict:
    path = root / REVIEW_PATH
    if not path.is_file():
        return {}
    entries = json.loads(path.read_text(encoding="utf-8"))["reviews"]
    for e in entries:
        if e["classification"] not in CLASSES:
            raise ValueError(f"invalid review classification {e['classification']}")
    return {e["key"]: e for e in entries}


def audit(table: list[dict], graph: dict, docs: list[Path], root: Path = PROJECT_ROOT, *,
          overclaims: list[dict] | None = None) -> dict:
    issues = []
    statuses = {r["hypothesis"]: r["status"] for r in table}
    for doc in docs:
        for n, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for h in MARKER.findall(line):
                if h not in statuses:
                    issues.append(f"{doc.name}:{n} marker C-{h} has no registered hypothesis")
                elif statuses[h] not in line and statuses[h] != "NOT_AVAILABLE":
                    issues.append(f"{doc.name}:{n} C-{h} sentence does not state the sealed status {statuses[h]}")
    for node in graph["claims"]:
        for run, digest in node["artifacts"].items():
            if digest is not None and not verify_run(root / run, root=root)["valid"]:
                issues.append(f"{node['claim']}: evidence run {run} does not verify")
    for o in overclaims or []:
        if o["classification"] in {"OVERCLAIM", "REVIEW"}:
            issues.append(f"{o['classification']} {o['file']}:{o['line']} '{o['term']}'")
    return {"valid": not issues, "issues": issues, "claims": len(graph["claims"])}


def negative_results(table: list[dict], registry: dict) -> dict:
    negative = {"NOT_ESTABLISHED", "FAILED", "INVALID", "INCONCLUSIVE", "NOT_AVAILABLE", "ASSUMPTION_DEPENDENT",
                "MODEL_DEPENDENT"}
    return {"hypotheses": [r["hypothesis"] for r in table if r["status"] in negative],
            "attempts": [a for a in registry["attempts"] if a["outcome"] in {"FAILED", "ABORTED", "INVALID",
                                                                              "NOT_AVAILABLE"}]}
