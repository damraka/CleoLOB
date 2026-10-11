"""Report generation, public evidence export and bundle verification (M35).

``build_report`` regenerates, from sealed evidence only: the hypothesis table (JSON and Markdown), the run
registry (sealed runs plus every ledger attempt), the experiment DAG, the claim graph, figures with their data and
provenance, and Markdown tables. The report is itself a sealed run whose lineage lists every input run.

``export_bundle`` writes the compact public bundle (``examples/studies/v08``): v0.8 configs, designs and ledger,
dataset metadata, every non-pilot sealed run's small JSON files, the report, and verification metadata. Raw
provider data, caches and checkpoints are never included. ``verify_bundle`` checks bundle bytes, every included
run's config/result/lineage hashes against its binding and the ledger hash chain - offline.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from . import artifacts, claims, figures, protocol as pr
from ..preregistration import document_sha256

EXCLUDE_PARTS = ("dev", "smoke")


def _load(path: Path) -> dict | None:
    p = path / "result.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def registry(root: Path) -> dict:
    base = root / "results/v08"
    runs = []
    for run in artifacts.runs_under(base) if base.is_dir() else []:
        b = json.loads((run / "binding.json").read_text(encoding="utf-8"))
        runs.append({"run": run.relative_to(root).as_posix(), "analysis": b["analysis"],
                     "pilot": any(part in EXCLUDE_PARTS for part in run.relative_to(base).parts),
                     "namespace": json.loads((run / "config.json").read_text(encoding="utf-8")).get("namespace"),
                     "result_sha256": b["result_sha256"], "protocol_sha256": b["protocol_sha256"],
                     "datasets": sorted(b["datasets"]), "git_commit": b.get("git_commit"),
                     "valid": artifacts.verify_run(run, root=root)["valid"]})
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), pr.load_protocol(root / pr.PROTOCOL_PATH))
    return {"schema": "cleolob-v08-registry-1", "runs": runs, "attempts": state.attempts,
            "ledger_entries": state.count, "ledger_head": state.head, "protocol_sha256": state.protocol_sha256,
            "amendments": state.amendments}


def hypothesis_markdown(final: dict) -> str:
    lines = ["| Hypothesis | Status | Estimate | Interval | Threshold | n | Qualifiers | Note |",
             "|---|---|---|---|---|---|---|---|"]
    fmt = lambda x: "—" if x is None else (f"{x:.4g}" if isinstance(x, (int, float)) else str(x))  # noqa: E731
    for h, r in final["hypotheses"].items():
        interval = "—" if r["ci_low"] is None else f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]"
        lines.append(f"| {h} | **{r['status']}** | {fmt(r['estimate'])} | {interval} | {fmt(r['threshold'])} | "
                     f"{fmt(r['n'])} | {', '.join(r['qualifiers'])} | {r['reason'] or ''} |")
    return "\n".join(lines) + "\n"


def _figs(root: Path, final: dict, figdir: Path) -> list[dict]:
    made, missing = [], []
    inputs = final.get("inputs", {})

    def res(key):
        return _load(root / inputs[key]) if key in inputs else None

    statuses = list(final["statuses"].values())
    order = ["ESTABLISHED", "NOT_ESTABLISHED", "FAILED", "INCONCLUSIVE", "NOT_AVAILABLE", "NOT_EVALUABLE",
             "ASSUMPTION_DEPENDENT", "EXPLORATORY", "INVALID"]
    made.append(figures.bar(figdir, "hypothesis_status_counts", order, [statuses.count(s) for s in order],
                            title="H1-H21 status counts", ylabel="hypotheses",
                            provenance={"run": "final", "pointer": "/statuses"}))
    d = res("drift")
    if d:
        made.append(figures.bar(figdir, "calibration_drift_rolling_vs_frozen", list(d["per_scenario"]),
                                [sum(x["h1_rel"] for x in v) / len(v) for v in d["per_scenario"].values()],
                                title="Rolling vs frozen calibration: relative loss reduction", ylabel="relative reduction",
                                provenance={"run": inputs["drift"], "pointer": "/per_scenario/*/h1_rel"}))
        series = {s: (list(range(1, len(v[0]["curve"]) + 1)),
                      [sum(x["curve"][k] for x in v) / len(v) - sum(x["baseline"] for x in v) / len(v)
                       for k in range(len(v[0]["curve"]))]) for s, v in d["per_scenario"].items()}
        made.append(figures.lines(figdir, "calibration_half_life_curves", series, title="Frozen-fit degradation by horizon",
                                  xlabel="horizon (windows)", ylabel="loss - in-sample loss",
                                  provenance={"run": inputs["drift"], "pointer": "/per_scenario/*/curve"}))
    else:
        missing.append("drift")
    c = res("calutil")
    if c:
        valid = [r for r in c["rows"] if r.get("status") == "VALID"]
        made.append(figures.scatter(figdir, "calibration_vs_regret", [r["execution_loss"] for r in valid],
                                    [r["regret"] for r in valid], title="Execution-aware calibration loss vs regret",
                                    xlabel="execution-aware loss", ylabel="regret (bps)",
                                    labels=[r["candidate"] for r in valid],
                                    provenance={"run": inputs["calutil"], "pointer": "/rows"}))
    m = res("mpc")
    if m:
        t = m["mandates"]["primary"]["table"]["policies"]
        names = sorted(t)
        made.append(figures.bar(figdir, "regret_by_policy_primary", names, [t[p]["regret_mean_bps"] for p in names],
                                title="Mean episode regret, primary mandate (held-out worlds)", ylabel="bps",
                                provenance={"run": inputs["mpc"], "pointer": "/mandates/primary/table/policies"}))
        if "large" in m["mandates"]:
            t2 = m["mandates"]["large"]["table"]["policies"]
            made.append(figures.bar(figdir, "cost_by_policy_secondary", names,
                                    [t2[p]["economic_cost_mean_bps"] for p in names],
                                    title="Economic cost, secondary 400-lot mandate (held-out worlds)", ylabel="bps",
                                    provenance={"run": inputs["mpc"], "pointer": "/mandates/large/table/policies"}))
    comp = res("completion")
    if comp:
        pts = comp["mandates"]["primary"]["pareto"]["points"]
        names = sorted(pts)
        made.append(figures.scatter(figdir, "completion_cost_pareto", [pts[p]["economic_cost_bps"] for p in names],
                                    [pts[p]["completion_probability"] for p in names], labels=names,
                                    title="Completion probability vs economic cost (completion mode none)",
                                    xlabel="economic cost (bps)", ylabel="P(actual completion)",
                                    provenance={"run": inputs["completion"], "pointer": "/mandates/primary/pareto/points"}))
    r = res("robust")
    if r:
        oos = r["mandates"]["primary"]["out_of_sample"]
        names = sorted(oos)
        made.append(figures.scatter(figdir, "cvar_frontier", [oos[k]["mean_cost"] for k in names],
                                    [oos[k]["cvar_cost"] for k in names], labels=names,
                                    title="Out-of-sample mean vs CVaR(0.8) by selection criterion",
                                    xlabel="mean cost (bps)", ylabel="CVaR cost (bps)",
                                    provenance={"run": inputs["robust"], "pointer": "/mandates/primary/out_of_sample"}))
    u = res("uncertainty")
    if u:
        pairs = u["mandates"]["primary"]["decomposition"]["pairs"]
        names = sorted(pairs)
        made.append(figures.bar(figdir, "epistemic_share_by_pair", names,
                                [pairs[k].get("epistemic_share") for k in names],
                                errors=[(pairs[k]["epistemic_share"] - pairs[k]["epistemic_share_ci"][0],
                                         pairs[k]["epistemic_share_ci"][1] - pairs[k]["epistemic_share"])
                                        if "epistemic_share_ci" in pairs[k] else None for k in names],
                                title="Epistemic share of paired cost variance (H5)", ylabel="share",
                                provenance={"run": inputs["uncertainty"], "pointer": "/mandates/primary/decomposition/pairs"}))
    a = res("abstention")
    if a:
        p = a["mandates"]["primary"]
        made.append(figures.bar(figdir, "abstention_benefit", ["without abstention", "with abstention"],
                                [p["without_abstention"]["incorrect_rate"], p["with_abstention"]["incorrect_rate"]],
                                title="Incorrect selection rate (regret > 0.5 bps)", ylabel="rate",
                                provenance={"run": inputs["abstention"], "pointer": "/mandates/primary"}))
    cf = res("counterfactual")
    if cf:
        pairs = cf["mandates"]["primary"]["pairs"]
        names = sorted(pairs)
        made.append(figures.bar(figdir, "counterfactual_pair_bounds", names,
                                [((pairs[k].get("lower") or 0) + (pairs[k].get("upper") or 0)) / 2 for k in names],
                                errors=[((pairs[k]["upper"] - pairs[k]["lower"]) / 2,) * 2 if pairs[k].get("lower")
                                        is not None else None for k in names],
                                title="Bounded paired cost differences on twin tapes (H14)", ylabel="bps",
                                provenance={"run": inputs["counterfactual"], "pointer": "/mandates/primary/pairs"}))
    for extra_key, maker in zip(FIGURE_INPUTS, (_transfer_fig, _latency_fig, _ood_fig, _posterior_fig,
                                                _averaging_fig)):
        e = res(extra_key)
        if e:
            made.extend(maker(figdir, e, inputs[extra_key]))
    return made


FIGURE_INPUTS = ("transfer", "latency", "ood", "posterior", "averaging")


def _transfer_fig(figdir, e, run):
    km = e["mandates"]["primary"]["kendall_matrix"]
    return [figures.heatmap(figdir, "transfer_kendall_tau", km["kendall_tau"], km["cells"], km["cells"],
                            title="Kendall tau of neutral bounded costs across retrospective cells",
                            provenance={"run": run, "pointer": "/mandates/primary/kendall_matrix"})]


def _latency_fig(figdir, e, run):
    summary = e["summary"]
    profiles = ["zero", "uniform_5ms", "uniform_50ms", "uniform_250ms", "stale_feed_1s"]
    series = {p: (list(range(len(profiles))), [summary[p][k]["economic_cost_mean_bps"] for k in profiles])
              for p in ("twap", "heuristic", "mpc", "pov") if p in summary}
    return [figures.lines(figdir, "latency_sensitivity", series, title="Economic cost by latency profile",
                          xlabel="profile index (zero, 5ms, 50ms, 250ms, stale 1s)", ylabel="bps",
                          provenance={"run": run, "pointer": "/summary"})]


def _ood_fig(figdir, e, run):
    det = e["detectors"]["mahalanobis"]
    names = sorted(det["shifts"])
    return [figures.bar(figdir, "ood_detection_rates", names, [det["shifts"][k]["detection_rate"] for k in names],
                        title="Mahalanobis OOD detection rate per registered shift (FPR target 5%)", ylabel="rate",
                        provenance={"run": run, "pointer": "/detectors/mahalanobis"})]


def _posterior_fig(figdir, e, run):
    fam = e["gate"]["families"]
    names = sorted(fam)
    return [figures.bar(figdir, "generator_gate_auc", names,
                        [fam[k]["criteria"]["discriminator_auc"]["value"] for k in names],
                        title="Discriminator AUC per family on the selection day (gate max 0.75)", ylabel="AUC",
                        provenance={"run": run, "pointer": "/gate/families"})]


def _averaging_fig(figdir, e, run):
    path = e["report"]["dynamic_path_every_600_bins"]
    names = sorted(e["report"]["stacking_weights"])
    series = {n: (list(range(len(path))), [row[i] for row in path]) for i, n in enumerate(names)}
    return [figures.lines(figdir, "model_weights_through_time", series, title="Dynamic model weights (selection day)",
                          xlabel="600-bin step", ylabel="weight", provenance={"run": run,
                                                                              "pointer": "/report/dynamic_path_every_600_bins"})]


def build_report(root: Path, *, out: str, final: str, extra_inputs: dict | None = None) -> dict:
    root = Path(root)
    fdir = root / final
    final_result = _load(fdir)
    if final_result is None:
        raise ValueError(f"final analysis run not found: {final}")
    target = artifacts.new_run(root / out)
    inputs = dict(final_result.get("inputs", {}))
    inputs.update(extra_inputs or {})
    final_result = {**final_result, "inputs": inputs}
    figs = _figs(root, final_result, target / "figures")
    reg = registry(root)
    g = claims.graph(root, final=final)
    dag = artifacts.dag(root / "results/v08", root=root)
    (target / "hypotheses.md").write_text(hypothesis_markdown(final_result), encoding="utf-8", newline="\n")
    for name, value in (("registry.json", reg), ("claim-graph.json", g), ("dag.json", dag),
                        ("hypotheses.json", final_result["hypotheses"]), ("figures.json", figs)):
        (target / name).write_text(json.dumps(value, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    used = sorted({final} | set(inputs.values()))
    result = {"hypotheses": final_result["statuses"], "figures": [f["figure"] for f in figs], "runs": len(reg["runs"]),
              "attempts": len(reg["attempts"]), "dag_acyclic": dag["acyclic"],
              "files": sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file())}
    artifacts.finalize(target, analysis="report", config={"final": final, "inputs": inputs}, result=result, root=root,
                       inputs=used)
    return {"status": "SEALED", "run": target.relative_to(root).as_posix(), "figures": len(figs),
            "hypotheses": final_result["statuses"]}


def export_bundle(root: Path, *, out: str, final: str) -> dict:
    root = Path(root)
    target = root / out
    if target.exists():
        raise ValueError(f"bundle directory exists (write-once): {out}")
    (target / "configs").mkdir(parents=True)
    for p in sorted((root / pr.CONFIG_DIR).rglob("*")):
        if p.is_file() and not p.name.endswith(".lock"):
            dest = target / "configs" / p.relative_to(root / pr.CONFIG_DIR)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    included, excluded = [], []
    for run in artifacts.runs_under(root / "results/v08"):
        rel = run.relative_to(root / "results/v08")
        if any(part in EXCLUDE_PARTS for part in rel.parts):
            excluded.append(rel.as_posix())
            continue
        dest = target / "runs" / rel
        dest.mkdir(parents=True)
        for name in ("config.json", "result.json", "binding.json", "lineage.json", "provenance.json", "checksums.json"):
            shutil.copyfile(run / name, dest / name)
        for sub in ("figures",):
            if (run / sub).is_dir():
                shutil.copytree(run / sub, dest / sub)
        for f in run.glob("*.md"):
            shutil.copyfile(f, dest / f.name)
        for f in run.glob("*.json"):
            if not (dest / f.name).exists() and f.name not in ("selection.json", "heldout.json"):
                shutil.copyfile(f, dest / f.name)
        included.append(rel.as_posix())
    verification = {"protocol": pr.verify_protocol_files(root), "runs": artifacts.verify_tree(root / "results/v08",
                                                                                               root=root)}
    verification["runs"].pop("reports", None)
    manifest = {"schema": "cleolob-v08-public-bundle-1", "included_runs": included,
                "excluded_runs": excluded, "excluded_classes": ["raw provider data (data/)", "tape and frame caches",
                                                                "checkpoints (*.ckpt)", "world-set files (regenerable)",
                                                                "development pilot runs (listed, not copied)"],
                "verification_at_export": verification,
                "meaning": "byte integrity and binding consistency of copied evidence; not independent replication"}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n", encoding="utf-8",
                                          newline="\n")
    (target / "README.md").write_text(
        "# CleoLOB v0.8 public evidence bundle\n\nCompact, reproducible evidence of the v0.8 study runs. Raw provider "
        "data are never included. Verify with `cleo verify-v08 --bundle examples/studies/v08`.\n", encoding="utf-8",
        newline="\n")
    from .artifacts import seal_artifacts
    seal_artifacts(target)
    return {"valid": verification["protocol"]["valid"] and verification["runs"]["valid"], "bundle": out,
            "included": len(included), "excluded": len(excluded)}


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
            "meaning": "bundle bytes, run bindings and ledger chain verified offline; v0.7 history pins require the "
                       "repository; not independent replication"}
