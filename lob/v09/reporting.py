"""report-v09 / evidence-export-v09 (M89): report, cards, datasheets and the public evidence bundle.

``report`` regenerates from sealed evidence only, into a sealed run whose lineage lists every input:
the H1-H36 table (JSON and Markdown), the carried-conclusion ladder, the run registry (sealed runs plus every
ledger attempt), the experiment DAG, the claim graph, model cards (generators, survival, neural models, offline RL),
policy cards (registered execution policies), datasheets (every pool and stream) and SVG figures with data and
provenance.

``export`` writes the compact public bundle (``examples/studies/v09``): the v0.9 configs, designs and ledger, every
sealed v0.9 run's small JSON/Markdown files and figures, and verification metadata. Raw provider data, day caches
and checkpoints are never included. ``verify_bundle`` checks bundle bytes, every run's config/result/lineage hashes
against its binding and the ledger hash chain, offline.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from ..preregistration import document_sha256
from ..v08 import figures
from . import artifacts, claims, protocol as pr

BASE = "results/v09"
EXCLUDE_PARTS = ("dev", "smoke")
FMT = lambda x: "—" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))  # noqa: E731


def _load(path: Path) -> dict | None:
    p = path / "result.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def registry(root: Path) -> dict:
    base = root / BASE
    runs = []
    for run in artifacts.runs_under(base) if base.is_dir() else []:
        b = json.loads((run / "binding.json").read_text(encoding="utf-8"))
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        runs.append({"run": run.relative_to(root).as_posix(), "analysis": b["analysis"],
                     "namespace": cfg.get("namespace") if isinstance(cfg, dict) else None,
                     "status": artifacts.run_status(run), "result_sha256": b["result_sha256"],
                     "protocol_sha256": b["protocol_sha256"], "datasets": sorted(b["datasets"]),
                     "git_commit": b.get("git_commit"), "valid": artifacts.verify_run(run, root=root)["valid"]})
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), pr.load_protocol(root / pr.PROTOCOL_PATH))
    return {"schema": "cleolob-v09-registry-1", "runs": runs, "attempts": state.attempts,
            "ledger_entries": state.count, "ledger_head": state.head, "protocol_sha256": state.protocol_sha256,
            "amendments": state.amendments, "incidents": state.incidents}


def hypothesis_markdown(final: dict) -> str:
    lines = ["| H | Title | Promotion | Status | Estimate | 95% interval | Threshold | Units | Evidence |",
             "|---|---|---|---|---|---|---|---|---|"]
    for h, r in final["hypotheses"].items():
        iv = r.get("interval_95") or {}
        interval = "—" if not iv else f"[{FMT(iv.get('ci_low'))}, {FMT(iv.get('ci_high'))}]"
        units = r.get("units")
        if r.get("transfer"):
            t = r["transfer"]
            best = [f"{c}: {v.get('status')}" for c, v in (t.get("conclusions") or {}).items()]
            estimate = "; ".join(best) or "—"
            units = len(t.get("target_units") or [])
        else:
            estimate = FMT(r.get("estimate"))
        lines.append(f"| {h} | {r['title']} | {r['promotion']} | **{r['status']}** | {estimate} | {interval} | "
                     f"{FMT(r['threshold'])} {r['threshold_units']} | {FMT(units)} | {r.get('evidence_level') or '—'} |")
    return "\n".join(lines) + "\n"


def ladder_markdown(final: dict) -> str:
    if not final.get("ladder"):
        return "No execution conclusion was carried; the ladder is empty.\n"
    rungs = list(next(iter(final["ladder"].values())))
    lines = ["| Conclusion | " + " | ".join(rungs) + " |", "|---|" + "---|" * len(rungs)]
    for cid, r in final["ladder"].items():
        lines.append(f"| {cid} | " + " | ".join(str(r[k].get("status")) for k in rungs) + " |")
    return "\n".join(lines) + "\n"


MODEL_CARDS = {
    "classical generators (Poisson, Hawkes, queue-reactive, regime-switching, state-dependent)": {
        "intended_use": "stylized order-flow simulation for research comparisons; calibrated on explore days",
        "not_for": "trading decisions; price forecasting; claims about real fills",
        "evidence": ["generator-fit-v09", "path-realism-v09", "tail-realism-v09", "realism-benchmark-v09",
                     "realism-gate-v09", "identifiability-v5"]},
    "strategic ecology (agent-based, F6)": {
        "intended_use": "endogenous-response worlds for synthetic decision experiments (E1/E2)",
        "not_for": "statements about real traders' behaviour or identities",
        "evidence": ["hybrid-twin-v09", "strategic-response-v09", "adversarial-study-v09", "counterexample-study-v09"]},
    "neural temporal point processes / sequence models / latent dynamics (CPU research models)": {
        "intended_use": "comparison with classical event models under registered metrics (H2-H4)",
        "not_for": "production forecasting; any use outside the registered windows",
        "evidence": ["neural-point-process-v09", "sequence-generator-v09", "latent-dynamics-v09"]},
    "queue-survival models (KM, Cox, Weibull AFT, boosted hazard, neural hazard)": {
        "intended_use": "back-of-queue clearance probabilities of hypothetical resting orders (H7)",
        "not_for": "queue positions or exact fills (never observed on public L2)",
        "evidence": ["survival-study-v09", "queue-reconstruction-v09", "fill-study-v09"]},
    "offline RL learners (BC, CQL, IQL, TD3+BC, decision policy) with OPE gates": {
        "intended_use": "synthetic worlds with logged propensities only (E1)",
        "not_for": "historical deployment; OPE on real days is NOT_AVAILABLE (no propensities)",
        "evidence": ["offline-rl-v09", "ope-study-v09"]},
}


def model_cards(final: dict, reg: dict) -> str:
    sealed = {r["analysis"] for r in reg["runs"] if r["valid"]}
    out = ["# v0.9 model cards", "", "Generated from sealed evidence by report-v09. Every model is a research model "
           "with fixed seeds; none is assumed better than the classical baselines.", ""]
    for name, card in MODEL_CARDS.items():
        out += [f"## {name}", "", f"- Intended use: {card['intended_use']}.", f"- Not for: {card['not_for']}.",
                "- Evidence runs: " + ", ".join(f"`{a}`" + ("" if a in sealed else " (not run)")
                                                for a in card["evidence"]), ""]
    return "\n".join(out) + "\n"


def policy_cards(protocol: dict, final: dict) -> str:
    out = ["# v0.9 policy cards", "", "Registered execution policies. Bounded-replay costs are HYPOTHETICAL and "
           "assumption-dependent; no policy is claimed to be profitable or production-ready.", ""]
    carried = final.get("ladder", {})
    for group, names in protocol["policies"].items():
        if not isinstance(names, list):
            continue
        out += [f"## {group}", ""]
        for n in names:
            roles = [cid for cid, c in protocol["conclusions"]["set"].items() if n in (c["treatment"], c["control"])]
            tag = f" (conclusions: {', '.join(roles)}{'; carried' if any(r in carried for r in roles) else ''})" \
                if roles else ""
            out.append(f"- `{n}`{tag}")
        out.append("")
    out += ["## Fallback ladder", ""] + [f"- {k}: {v}" for k, v in protocol["policies"]["fallback_ladder"].items()]
    return "\n".join(out) + "\n"


def datasheets(protocol: dict, final: dict) -> str:
    avail = final.get("data_availability", {})
    out = ["# v0.9 datasheets", "", "Public archives used for non-commercial research under the providers' terms; raw "
           "data are never committed or bundled. Capabilities are those of the public feeds (no order ids, no queue "
           "positions, no trader identities).", ""]
    for pool, p in protocol["pools"].items():
        a = avail.get(pool, {})
        out += [f"## Pool `{pool}` ({p['role']})", "",
                f"- Period: {p['start']} to {p['end']}; primaries per stream: {p['count']}; backups: {p['backups']}",
                f"- Streams: {', '.join(p['streams'])}",
                f"- Declared / accessed / NOT_AVAILABLE attempts: {a.get('declared', '—')} / {a.get('accessed', '—')} / "
                f"{a.get('not_available', '—')}", ""]
    seen = set()
    for d in protocol["datasets"]:
        key = (d["venue"], d["instrument"])
        if key in seen:
            continue
        seen.add(key)
        caps = d["capabilities"]
        out += [f"### {d['venue']} {d['instrument']}", "", f"- Provider: {d['provider']}; types: "
                f"{', '.join(d['data_types'])}; tick {d['tick_size']}, lot {d['lot_size']}",
                f"- Capabilities: depth {caps.get('depth_levels')}, incremental L2 {caps.get('incremental_l2')}, "
                f"aggressor side {caps.get('aggressor_side')}, order ids {caps.get('order_ids')}, queue clearance "
                f"observable {caps.get('queue_clearance_observable')}", ""]
    return "\n".join(out) + "\n"


def _figs(final: dict, figdir: Path) -> list[dict]:
    made = []
    order = ["ESTABLISHED", "NOT_ESTABLISHED", "FAILED", "NOT_AVAILABLE", "NOT_EVALUABLE", "INCONCLUSIVE"]
    statuses = [r["status"] for r in final["hypotheses"].values()]
    made.append(figures.bar(figdir, "hypothesis_status_counts", order, [statuses.count(s) for s in order],
                            title="H1-H36 terminal status counts", ylabel="hypotheses",
                            provenance={"run": "final", "pointer": "/hypotheses/*/status"}))
    tested = {h: r for h, r in final["hypotheses"].items() if r.get("estimate") is not None and r.get("interval_95")}
    if tested:
        names = sorted(tested, key=lambda h: int(h[1:]))
        made.append(figures.bar(figdir, "confirmatory_effects_over_threshold", names,
                                [tested[h]["estimate"] / max(abs(tested[h]["threshold"]) or 1.0, 1e-12) for h in names],
                                title="Confirmatory estimate / practical threshold (1 = threshold)", ylabel="ratio",
                                provenance={"run": "final", "pointer": "/hypotheses/*/estimate"}))
    return made


def report(args) -> dict:
    root = Path(args.root).resolve()
    final_dir = root / args.final
    final = _load(final_dir)
    if final is None:
        raise ValueError(f"final analysis run not found: {args.final}")
    target = artifacts.new_run(root / args.out)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    reg = registry(root)
    g = claims.graph(root, final=args.final)
    dag = artifacts.dag(root / BASE, root=root)
    figs = _figs(final, target / "figures")
    files = {"hypotheses.md": hypothesis_markdown(final), "ladder.md": ladder_markdown(final),
             "model-cards.md": model_cards(final, reg), "policy-cards.md": policy_cards(protocol, final),
             "datasheets.md": datasheets(protocol, final)}
    for name, text in files.items():
        (target / name).write_text(text, encoding="utf-8", newline="\n")
    for name, value in (("registry.json", reg), ("claim-graph.json", g), ("dag.json", dag),
                        ("hypotheses.json", final["hypotheses"]), ("figures.json", figs)):
        (target / name).write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    inputs = [args.final] + [run for _, run in (args.input or [])]
    result = {"statuses": {h: r["status"] for h, r in final["hypotheses"].items()}, "counts": final["counts"],
              "figures": [f["figure"] for f in figs], "runs": len(reg["runs"]), "attempts": len(reg["attempts"]),
              "dag_acyclic": dag["acyclic"],
              "files": sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())}
    artifacts.finalize(target, analysis="report-v09", config={"final": args.final, "inputs": inputs}, result=result,
                       root=root, inputs={f"in{i}": r for i, r in enumerate(inputs)})
    return {"valid": True, "status": "COMPLETED", "run": target.relative_to(root).as_posix(), "figures": len(figs),
            "counts": final["counts"]}


def export(args) -> dict:
    root = Path(args.root).resolve()
    target = root / args.out
    if target.exists():
        raise ValueError(f"bundle directory exists (write-once): {args.out}")
    (target / "configs").mkdir(parents=True)
    for p in sorted((root / pr.CONFIG_DIR).rglob("*")):
        if p.is_file() and not p.name.endswith(".lock"):
            dest = target / "configs" / p.relative_to(root / pr.CONFIG_DIR)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    included, excluded = [], []
    for run in artifacts.runs_under(root / BASE):
        rel = run.relative_to(root / BASE)
        if any(part in EXCLUDE_PARTS for part in rel.parts) or artifacts.run_status(run) != "COMPLETED":
            excluded.append(rel.as_posix())
            continue
        dest = target / "runs" / rel
        dest.mkdir(parents=True)
        for f in sorted(run.iterdir()):
            if f.is_file() and f.suffix in (".json", ".md"):
                shutil.copyfile(f, dest / f.name)
        if (run / "figures").is_dir():
            shutil.copytree(run / "figures", dest / "figures")
        included.append(rel.as_posix())
    verification = {"protocol": pr.verify_protocol_files(root),
                    "runs": artifacts.verify_tree(root / BASE, root=root)}
    manifest = {"schema": "cleolob-v09-public-bundle-1", "included_runs": included, "excluded_runs": excluded,
                "final": args.final,
                "excluded_classes": ["raw provider data (data/)", "day caches", "checkpoints",
                                     "failed attempts (listed in the ledger, directories not copied)"],
                "verification_at_export": {"protocol_valid": verification["protocol"]["valid"],
                                           "runs_valid": verification["runs"].get("valid")},
                "meaning": "byte integrity and binding consistency of copied evidence; not independent replication"}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n", encoding="utf-8",
                                          newline="\n")
    (target / "README.md").write_text(
        "# CleoLOB v0.9 public evidence bundle\n\nCompact evidence of the v0.9 study runs: frozen protocol, ledger, "
        "designs, sealed run results, report, cards and datasheets. Raw provider data and day caches are never "
        "included. Verify offline with `cleo verify-v09 --bundle examples/studies/v09`.\n", encoding="utf-8",
        newline="\n")
    from .artifacts import seal_artifacts
    seal_artifacts(target)
    return {"valid": bool(verification["protocol"]["valid"]), "bundle": args.out, "included": len(included),
            "excluded": len(excluded)}


def verify_bundle(path: Path) -> dict:
    from .artifacts import verify_artifacts
    path = Path(path)
    issues = list(verify_artifacts(path)["issues"])
    protocol = json.loads((path / "configs/protocol.json").read_text(encoding="utf-8"))
    try:
        state = pr.replay_ledger(pr.read_ledger(path / "configs/consumption-ledger.jsonl"), protocol)
    except ValueError as exc:
        issues.append(f"ledger: {exc}")
        state = None
    runs = sorted(p.parent for p in (path / "runs").rglob("binding.json")) if (path / "runs").is_dir() else []
    for run in runs:
        b = json.loads((run / "binding.json").read_text(encoding="utf-8"))
        for name, key in (("config.json", "config_sha256"), ("result.json", "result_sha256")):
            if document_sha256(json.loads((run / name).read_text(encoding="utf-8"))) != b[key]:
                issues.append(f"{run.relative_to(path).as_posix()}: {name} does not match its binding")
        lineage = json.loads((run / "lineage.json").read_text(encoding="utf-8"))
        if document_sha256(lineage) != b["extra"].get("lineage_sha256"):
            issues.append(f"{run.relative_to(path).as_posix()}: lineage does not match its binding")
        if state is not None and b["ledger_anchor"]["index"] >= state.count:
            issues.append(f"{run.relative_to(path).as_posix()}: ledger anchor beyond the bundled ledger")
    return {"valid": not issues, "issues": issues, "runs": len(runs),
            "ledger_entries": state.count if state else None,
            "meaning": "bundle bytes, run bindings and ledger chain verified offline; not independent replication"}
