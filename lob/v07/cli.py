"""v0.7 command-line workflows, attached to ``cleo`` by ``lob.cli``.

Commands validate their inputs and fail with a machine-readable status instead of falling back silently.
Run directories live under ``results/v07`` and are write-once. Capability gaps are printed explicitly, e.g.
``exact_fifo_position: NOT_AVAILABLE (reason: aggregate_l2 ...)``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=True, default=str))


def _common(child, *, out: bool = True, workers: bool = False) -> None:
    child.add_argument("--root", type=Path, default=Path("."))
    if out:
        child.add_argument("--out", required=True, help="run directory relative to --root (write-once)")
    if workers:
        child.add_argument("--workers", type=int, default=8)


def add_commands(commands: argparse._SubParsersAction) -> None:
    c = commands.add_parser("protocol-v07", help="verify the frozen v0.7 protocol, ledger and holdout status")
    c.add_argument("operation", choices=("verify", "status"))
    c.add_argument("--root", type=Path, default=Path("."))
    c = commands.add_parser("dataset-registry-v07", help="dataset registry v2: roles, freshness, capabilities, access")
    _common(c, out=False)
    c = commands.add_parser("validate-market-data-v07", help="validate a Tardis L2/trades pair with stream flags")
    c.add_argument("--venue", required=True)
    c.add_argument("--instrument", required=True)
    c.add_argument("--l2", type=Path, required=True)
    c.add_argument("--trades", type=Path)
    c.add_argument("--date")
    c = commands.add_parser("replay-v07", help="deterministic replay: hash chain and optional checkpoint")
    c.add_argument("--venue", required=True)
    c.add_argument("--instrument", required=True)
    c.add_argument("--l2", type=Path, required=True)
    c.add_argument("--checkpoint-at", type=int)
    for name, text in (("queue-study-v07", "M3 queue uncertainty study"),
                       ("generator-study-v07", "M5 generator fit and scores"),
                       ("calibration-v4", "M7 execution-aware calibration and surrogate/active study"),
                       ("posterior-study-v07", "M6 recovery, posterior and generator selection"),
                       ("identifiability-v2", "M8 identifiability v2"),
                       ("realism-v2", "M10/M17 simulation bank, sealed holdout design and holdout evaluation"),
                       ("domain-gap-v07", "summarize stored discriminator outputs of a holdout run"),
                       ("regime-study-v07", "M11 regime/drift/half-life study"),
                       ("drift-study-v07", "M11 regime/drift/half-life study (alias)"),
                       ("impact-study-v07", "M13 impact/TCA, ecology and stress studies"),
                       ("execution-study-v07", "M15 execution study across the plausible world set"),
                       ("robust-policy-study-v07", "M14 policy training and simulated predictions"),
                       ("model-risk-v07", "M15 model-risk study (alias of execution-study-v07)"),
                       ("transfer-study-v07", "M12/M16 transfer matrix, transfer lock and final transfer")):
        c = commands.add_parser(name, help=text)
        c.add_argument("phase", nargs="?", default="run")
        _common(c, out=True, workers=True)
        c.add_argument("--generator-run", default="results/v07/m5/fit")
        c.add_argument("--posterior-run", default="results/v07/m6/posterior-2")
        c.add_argument("--ea-run", default="results/v07/m7/execution-aware")
        c.add_argument("--bank-run", default="results/v07/m10/bank")
        c.add_argument("--selection-run", default="results/v07/m6/select")
        c.add_argument("--policy-run", default="results/v07/m14/policies")
        c.add_argument("--prediction-run", default="results/v07/m16/predictions")
        c.add_argument("--execution-run", default="results/v07/m15/execution")
        c.add_argument("--dataset")
    c = commands.add_parser("benchmark-v07", help="frozen benchmark tasks; 'public' reruns the public replication subset")
    c.add_argument("group", choices=("all", "public"))
    c.add_argument("--root", type=Path, default=Path("."))
    c = commands.add_parser("final-analysis-v07", help="M22 synthesis over the sealed runs (no new data access)")
    _common(c)
    c = commands.add_parser("report-v07", help="regenerate registry, hypothesis table, claim graph, figures, explorer")
    _common(c)
    c = commands.add_parser("evidence-export-v07", help="export the public v0.7 evidence bundle")
    _common(c)
    c.add_argument("--report", default="results/v07/report")
    c = commands.add_parser("claim-audit-v07", help="claim graph and wording audit over the v0.7 documents")
    _common(c, out=False)
    c = commands.add_parser("verify-v07", help="verify v0.7 runs (or a public bundle with --bundle)")
    c.add_argument("--root", type=Path, default=Path("."))
    c.add_argument("--bundle", type=Path)


def _protocol(a):
    from .protocol.core import verify_protocol_files
    r = verify_protocol_files(a.root)
    _print(r)
    return 0 if r["valid"] else 1


def _registry(a):
    from .data.capability import REQUIREMENTS, check
    from .data.schema import venue_spec
    from .protocol import core as pr
    protocol = pr.load_protocol(a.root / pr.PROTOCOL_PATH)
    state = pr.replay_ledger(pr.read_ledger(a.root / pr.LEDGER_PATH), protocol)
    out = []
    for d in protocol["datasets"]:
        spec = venue_spec(d["venue"], d["instrument"])
        caps = {}
        for name in ("exact_fifo_position", "quantity_ahead", "order_identity", "hidden_order_quantity",
                     "observed_order_fill", "trade_prints", "aggregate_depth"):
            caps[name] = "AVAILABLE" if spec.contract.supports(name) else f"NOT_AVAILABLE (reason: {spec.capability_level} source)"
        sources = {}
        for access in state.accesses.get(d["id"], []):
            sources.update(access.get("source_sha256", {}))
        out.append({"id": d["id"], "source": d["provider"], "venue": d["venue"], "instrument": d["instrument"],
                    "date": d["date"], "license": "Tardis.dev free sample terms (not redistributed)" if
                    d["provider"].startswith("Tardis") else "self-recorded public API capture (not redistributed)",
                    "schema": "cleolob-v07-canonical-1", "granularity": "100 ms grid from incremental L2; trades"
                    if d["capability_level"] == "aggregate_l2" else "order-level events",
                    "role": d["role"], "freshness": state.freshness(d["id"]), "capabilities": caps,
                    "analyses": {k: check(protocol, d["id"], k)["status"] for k in REQUIREMENTS},
                    "checksums": sources, "access_stages": state.stages.get(d["id"], []),
                    "first_access_index": state.first_access.get(d["id"]),
                    "known_issues": d.get("prior_use")})
    _print({"datasets": out})
    return 0


def _validate(a):
    from .adapters.tardis import TardisAdapter
    from .data.validation import validate_stream
    adapter = TardisAdapter(a.venue, a.instrument, l2=a.l2, trades=a.trades)
    r = validate_stream(adapter.records(), adapter.spec, date=a.date)
    _print(r)
    return 0 if r["status"] != "FAIL" else 1


def _replay(a):
    from .adapters.tardis import TardisAdapter
    from .replay.deterministic import BookReplay
    adapter = TardisAdapter(a.venue, a.instrument, l2=a.l2, trades=None)
    if a.checkpoint_at:
        partial = BookReplay().run(adapter.records(), stop=a.checkpoint_at)
        chain = BookReplay.resume(partial.checkpoint(), adapter.records()).finish()
    else:
        chain = BookReplay().run(adapter.records()).finish()
    _print({"chain": chain, "checkpoint_at": a.checkpoint_at})
    return 0


def _study(a):
    root, name, phase = a.root, a.command, a.phase
    if name == "queue-study-v07":
        from .queue import study
        r = study.run(a.out, root=root)
    elif name == "generator-study-v07":
        from .generators import study
        r = study.fit(a.out, root=root, workers=a.workers)
    elif name == "calibration-v4":
        if phase == "surrogate":
            from .calibration import surrogate_study
            r = surrogate_study.run(a.out, root=root, workers=a.workers)
        else:
            from .calibration import execution_aware
            r = execution_aware.run(a.out, posterior_run=a.posterior_run, root=root, workers=a.workers)
    elif name == "posterior-study-v07":
        from .posterior import study
        r = {"recovery": lambda: study.recovery(a.out, root=root, workers=a.workers),
             "run": lambda: study.posterior(a.out, root=root, workers=a.workers),
             "select": lambda: study.select(a.out, posterior_run=a.posterior_run, generator_run=a.generator_run,
                                            root=root, workers=a.workers)}[phase]()
    elif name == "identifiability-v2":
        from .identifiability import study
        r = study.run(a.out, posterior_run=a.posterior_run, root=root, workers=a.workers)
    elif name == "realism-v2":
        from .realism import holdout
        r = {"bank": lambda: holdout.bank(a.out, generator_run=a.generator_run, posterior_run=a.posterior_run,
                                          ea_run=a.ea_run, root=root, workers=a.workers),
             "seal": lambda: holdout.seal(bank_run=a.bank_run, selection_run=a.selection_run,
                                          generator_run=a.generator_run, root=root),
             "evaluate": lambda: holdout.evaluate(a.dataset, a.out, root=root)}[phase]()
    elif name == "domain-gap-v07":
        run = root / a.out
        r = {p.name: json.loads(p.read_text(encoding="utf-8"))["auc"] for p in sorted(run.glob("discriminator-*.json"))}
    elif name in {"regime-study-v07", "drift-study-v07"}:
        from .drift import study
        r = study.run(a.out, bank_run=a.bank_run, root=root)
    elif name == "impact-study-v07":
        from .impact import study
        from .execution import ecology_study
        r = {"run": lambda: study.run_impact(a.out, generator_run=a.generator_run, posterior_run=a.posterior_run,
                                             root=root, workers=a.workers),
             "stress": lambda: study.run_stress(a.out, root=root, workers=a.workers),
             "ecology": lambda: ecology_study.run(a.out, root=root)}[phase]()
    elif name in {"execution-study-v07", "model-risk-v07"}:
        from .robustness import study
        r = study.run(a.out, generator_run=a.generator_run, posterior_run=a.posterior_run, root=root, workers=a.workers)
    elif name == "robust-policy-study-v07":
        from .policies import learned
        from .transfer import study as tstudy
        r = {"run": lambda: learned.train(a.out, posterior_run=a.posterior_run, root=root, workers=a.workers),
             "predict": lambda: tstudy.predict(a.out, policy_run=a.policy_run, root=root)}[phase]()
    elif name == "transfer-study-v07":
        from .transfer import matrix, study as tstudy
        r = {"matrix": lambda: matrix.run(a.out, root=root, workers=a.workers),
             "seal": lambda: tstudy.seal(policy_run=a.policy_run, prediction_run=a.prediction_run,
                                         execution_run=a.execution_run, root=root),
             "run": lambda: tstudy.evaluate(a.out, root=root)}[phase]()
    else:
        raise KeyError(name)
    _print({"command": name, "phase": phase, "keys": sorted(r)[:50] if isinstance(r, dict) else None})
    return 0


def _benchmark(a):
    from .benchmark import tasks
    if a.group == "public":
        results = tasks.run(tasks.PUBLIC)
        digest = tasks.public_digest(results)
        expected = a.root / "examples/studies/v07/public-subset-expected.json"
        stored = json.loads(expected.read_text(encoding="utf-8"))["digest"] if expected.is_file() else None
        _print({"digest": digest, "expected": stored, "reproduced": digest == stored})
        return 0 if digest == stored else 1
    _print(tasks.run())
    return 0


def _final(a):
    from .reports import final
    r = final.run(a.out, root=a.root)
    _print({"robust_selection": r["robust_selection"]["selected"], "realism_to_decision": r["realism_to_decision"]["verdict"],
            "certified": {k: v["final_status"] for k, v in r["decision_benchmark"]["pairs"].items()}})
    return 0


def _report(a):
    from .reports import build
    r = build.build(a.root / a.out, root=a.root)
    _print({"hypotheses": {x["hypothesis"]: x["status"] for x in r["table"]}, "figures": len(r["figures"])})
    return 0


def _export(a):
    from .reports import export
    report = a.root / a.report
    generated = {name: report / name for name in ("registry.json", "hypotheses.json", "claim-graph.json",
                                                   "negative-results.json", "tables.md", "explorer.html")
                 if (report / name).is_file()}
    if (report / "figures").is_dir():
        generated["figures"] = report / "figures"
    for name in ("docs/v07-requirement-coverage.md", "docs/v07-final-report.md",
                 "examples/studies/v07/public-subset-expected.json", "examples/studies/v07/README.md"):
        if (a.root / name).is_file():
            generated[Path(name).name] = a.root / name
    m = export.export(a.root / a.out, root=a.root, generated=generated)
    _print({"included": len(m["included"]), "excluded": len(m["excluded"]), "meaning": m["meaning"]})
    return 0


def _claims(a):
    from .reports import registry
    docs = sorted((a.root / "docs").glob("v07-*.md")) + [a.root / "README.md"]
    table = registry.hypothesis_table(a.root)
    graph = registry.claim_graph(table, docs, a.root)
    terms = registry.term_audit(docs, a.root)
    result = registry.audit(table, graph, docs, a.root)
    result["terms_needing_review"] = [t for t in terms if t["classification"] == "REVIEW"]
    _print(result)
    return 0 if result["valid"] else 1


def _verify(a):
    if a.bundle:
        from .reports.export import verify
        r = verify(a.bundle)
    else:
        from .evidence.runs import verify_tree
        r = verify_tree(a.root / "results/v07", root=a.root)
        r = {k: v for k, v in r.items() if k != "reports"}
    _print(r)
    return 0 if r["valid"] else 1


STUDIES = ("queue-study-v07", "generator-study-v07", "calibration-v4", "posterior-study-v07", "identifiability-v2",
           "realism-v2", "domain-gap-v07", "regime-study-v07", "drift-study-v07", "impact-study-v07",
           "execution-study-v07", "robust-policy-study-v07", "model-risk-v07", "transfer-study-v07")
HANDLERS: dict[str, Callable] = {"protocol-v07": _protocol, "dataset-registry-v07": _registry,
                                 "validate-market-data-v07": _validate, "replay-v07": _replay,
                                 "benchmark-v07": _benchmark, "final-analysis-v07": _final, "report-v07": _report, "evidence-export-v07": _export,
                                 "claim-audit-v07": _claims, "verify-v07": _verify, **{s: _study for s in STUDIES}}


def dispatch(args) -> int | None:
    """Run a v0.7 command; None when the command is not a v0.7 command."""
    handler = HANDLERS.get(args.command)
    return None if handler is None else handler(args)
