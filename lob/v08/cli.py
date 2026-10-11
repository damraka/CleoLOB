"""v0.8 command-line workflows, attached to ``cleo`` by ``lob.cli``.

Every command validates its inputs, prints one JSON document and fails with a
machine-readable status (exit 1) instead of falling back silently. Study runs
are write-once directories (default under ``results/v08``) bound to the v0.8
protocol and ledger. Confirmatory seed namespaces (``confirm``, ``holdout``)
require a sealed design that opened them (``protocol-v08 open``).
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable

BUDGETS = ("tiny", "small", "medium", "large", "exhaustive")
NAMESPACES = ("dev", "confirm", "holdout")

# command -> (study module, help, phases or None)
STUDIES: dict[str, tuple[str, str, tuple[str, ...] | None]] = {
    "regime-study-v08": ("regime", "M5 regime detection: method comparison (synthetic truth and retrospective days)", None),
    "change-point-study-v08": ("drift", "M6/M7 change-point and drift detection, calibration half-life", None),
    "adaptive-calibration-v08": ("adaptive", "M14 frozen vs adaptive calibration on drift, regime and model-switch benchmarks",
                                 ("drift", "regime", "switch", "real")),
    "inference-v08": ("inference", "M12/M13 SBI recovery, real-data posteriors, hierarchical pooling, world sets",
                      ("recovery", "posterior", "hierarchical", "worlds")),
    "identifiability-v3": ("identifiability", "M15 identifiability v3 (local, global, profile, posterior, structural)", None),
    "posterior-study-v08": ("posterior", "M16 posterior predictive checks and the generator rejection gate", None),
    "model-averaging-v08": ("averaging", "M17 model-class uncertainty and dynamic model averaging", ("synthetic", "real")),
    "uncertainty-study-v08": ("uncertainty", "M18 uncertainty decomposition over the plausible world design", None),
    "ood-study-v08": ("ood", "M19 OOD detection: false positives, detection of known shifts, decision degradation", None),
    "conformal-study-v08": ("conformal", "M20 conformal coverage (exchangeable and shifted)", None),
    "calibration-utility-v08": ("calibration_utility", "M21 decision-focused calibration and misspecification benchmark", None),
    "completion-study-v08": ("completion", "M22/M23 completion-constrained execution and Pareto fronts", None),
    "robust-execution-v08": ("robust", "M24 robust selection across plausible worlds (robustness benchmark)", None),
    "mpc-study-v08": ("mpc", "M25/M26 controlled execution benchmark: baselines, adaptive variants, MPC", None),
    "policy-selection-v08": ("selection", "M27 contextual / regime-aware selection and mixture of experts", None),
    "policy-abstention-v08": ("abstention", "M27 OOD-aware abstention and safe fallback benchmark", None),
    "adversarial-stress-v08": ("stress", "M28 structured stress worlds and bounded adversarial search", None),
    "counterfactual-bounds-v08": ("counterfactual", "M29 historical execution bounds (synthetic coverage, retrospective days)",
                                  ("synthetic", "real")),
    "latency-study-v08": ("latency", "M30 feed latency, stale-book execution and cancel races", None),
    "transfer-study-v08": ("transfer", "M31/M32 cross-time/instrument/venue transfer and portfolio infrastructure",
                           ("retrospective", "fresh", "portfolio")),
    "ranking-transfer-v08": ("ranking", "M32 synthetic predicted vs bounded historical rankings", ("retrospective", "fresh")),
    "benchmark-v08": ("benchmark", "local research-workload performance measurements (not HFT benchmarks)", None),
    "final-analysis-v08": ("final", "evaluate H1-H21 from sealed runs (and the registered holdout acquisition)", None),
}


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=True, default=str))


def _study_args(c: argparse.ArgumentParser, phases) -> None:
    if phases:
        c.add_argument("phase", choices=phases)
    c.add_argument("--root", type=Path, default=Path("."))
    c.add_argument("--out", required=True, help="write-once run directory relative to --root")
    c.add_argument("--budget", choices=BUDGETS, default="tiny")
    c.add_argument("--namespace", choices=NAMESPACES, default="dev")
    c.add_argument("--design", help="sealed design name (required for confirm/holdout namespaces)")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--input", action="append", default=[], metavar="KEY=RUN",
                   help="upstream run directories (relative to --root), e.g. worlds=results/v08/worlds")


def add_commands(commands: argparse._SubParsersAction) -> None:
    c = commands.add_parser("protocol-v08", help="freeze/verify the v0.8 protocol; seal designs; open namespaces")
    c.add_argument("operation", choices=("verify", "status", "freeze", "seal", "open", "probe"))
    c.add_argument("--root", type=Path, default=Path("."))
    c.add_argument("--design")
    c.add_argument("--file", type=Path, help="design JSON to seal")
    c.add_argument("--reads", action="append", default=[])
    c.add_argument("--namespaces", action="append", default=[])
    c.add_argument("--namespace", choices=("confirm", "holdout"))
    c.add_argument("--posthoc", action="store_true")
    c.add_argument("--reason", default="")
    c = commands.add_parser("dataset-registry-v08", help="dataset registry v3: roles, freshness, capability matrix")
    c.add_argument("--root", type=Path, default=Path("."))
    c = commands.add_parser("validate-market-data-v08", help="M3 market-data quality engine (no silent repairs)")
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--l2", type=Path, help="Tardis incremental_book_L2 CSV(.gz)")
    src.add_argument("--synthetic", choices=("poisson", "hawkes", "queue_reactive", "regime_switching"),
                     help="validate a generated tape instead of a file")
    c.add_argument("--trades", type=Path)
    c.add_argument("--tick", type=float)
    c.add_argument("--lot", type=float)
    c.add_argument("--max-rows", type=int, default=2_000_000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--bins", type=int, default=3000)
    for name, (_, text, phases) in STUDIES.items():
        _study_args(commands.add_parser(name, help=text), phases)
    for name, text in (("report-v08", "regenerate hypothesis table, claim graph, figures and tables from evidence"),
                       ("evidence-export-v08", "export the public v0.8 evidence bundle (no restricted raw data)")):
        c = commands.add_parser(name, help=text)
        c.add_argument("--root", type=Path, default=Path("."))
        c.add_argument("--out", required=True)
        c.add_argument("--final", default="results/v08/final", help="final-analysis run directory")
    c = commands.add_parser("claim-audit-v08", help="fail-closed claim graph and wording audit of README and v0.8 docs")
    c.add_argument("--root", type=Path, default=Path("."))
    c.add_argument("--final", default="results/v08/final")
    c.add_argument("--bundle", type=Path, help="audit against a public bundle instead of local runs")
    c = commands.add_parser("verify-v08", help="verify v0.8 runs, the protocol, the DAG, or a public bundle")
    c.add_argument("--root", type=Path, default=Path("."))
    c.add_argument("--bundle", type=Path)
    c.add_argument("--runs", default="results/v08")
    c = commands.add_parser("smoke-v08", help="offline generated-data v0.8 smoke (functionality evidence only)")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--workers", type=int, default=1)


# ============================================================================= handlers


def _protocol(a) -> int:
    from . import protocol as pr
    root = a.root
    if a.operation in ("verify", "status"):
        r = pr.verify_protocol_files(root)
        _print(r)
        return 0 if r["valid"] else 1
    if a.operation == "freeze":
        written = pr.initialize_ledger(root)
        r = pr.verify_protocol_files(root)
        _print({"entries_written": len(written), **r})
        return 0 if r["valid"] else 1
    if a.operation == "seal":
        if not a.design or not a.file:
            raise ValueError("seal needs --design and --file")
        design = json.loads(Path(a.file).read_text(encoding="utf-8"))
        entry = pr.seal_design(root, a.design, design, reads=a.reads, namespaces=a.namespaces, posthoc=a.posthoc,
                               reason=a.reason or f"seal design {a.design}")
        _print({"sealed": a.design, "index": entry["index"], "sha256": entry["sha256"]})
        return 0
    if a.operation == "open":
        if not a.design or not a.namespace:
            raise ValueError("open needs --design and --namespace")
        entry = pr.open_namespace(root, a.namespace, design=a.design, reason=a.reason or f"{a.design} opens {a.namespace}")
        _print({"opened": a.namespace, "design": a.design, "entry": entry.get("index", entry)})
        return 0
    if a.operation == "probe":
        from .realdata import probe
        _print(probe())
        return 0
    raise KeyError(a.operation)


def _registry(a) -> int:
    from .datasets import report
    r = report(a.root)
    _print(r)
    return 0 if not r["issues"] else 1


def _validate(a) -> int:
    from . import quality
    if a.synthetic:
        from .generators import get
        frame = get(a.synthetic)().simulate(a.bins, a.seed)
        table = quality.from_frame(frame, tick=a.tick or 1.0)
        r = quality.validate(table, tick=a.tick or 1.0, lot=a.lot)
        r["source"] = f"synthetic:{a.synthetic}:seed={a.seed}"
    else:
        if not a.l2.is_file() or (a.trades is not None and not a.trades.is_file()):
            raise ValueError("input file not found")
        table = quality.from_tardis(a.l2, a.trades, max_rows=a.max_rows)
        r = quality.validate(table, tick=a.tick, lot=a.lot)
        r["source"] = {"l2": a.l2.name, "trades": a.trades.name if a.trades else None, "max_rows": a.max_rows}
    _print(r)
    return 0 if r["status"] != "FAIL" else 1


def _study(a) -> int:
    module_name, _, _ = STUDIES[a.command]
    module = importlib.import_module(f"lob.v08.studies.{module_name}")
    inputs = {}
    for item in a.input:
        if "=" not in item:
            raise ValueError("--input must be KEY=RUN")
        key, value = item.split("=", 1)
        inputs[key] = value
    a.inputs = inputs
    summary = module.run(a)
    _print(summary)
    return 0 if summary.get("status") not in ("INVALID", "ERROR") else 1


def _report(a) -> int:
    from .reporting import build_report
    r = build_report(a.root, out=a.out, final=a.final)
    _print(r)
    return 0


def _export(a) -> int:
    from .reporting import export_bundle
    r = export_bundle(a.root, out=a.out, final=a.final)
    _print(r)
    return 0 if r.get("valid", True) else 1


def _claims(a) -> int:
    from .claims import audit
    r = audit(a.root, final=a.final, bundle=a.bundle)
    _print(r)
    return 0 if r["valid"] else 1


def _verify(a) -> int:
    if a.bundle:
        from .reporting import verify_bundle
        r = verify_bundle(a.bundle)
    else:
        from . import artifacts, protocol as pr
        runs = a.root / a.runs
        tree = artifacts.verify_tree(runs, root=a.root) if runs.is_dir() else {"valid": True, "runs": 0, "invalid": []}
        proto = pr.verify_protocol_files(a.root)
        dag = artifacts.dag(runs, root=a.root) if runs.is_dir() else {"acyclic": True, "dangling_inputs": []}
        r = {"valid": tree["valid"] and proto["valid"] and dag["acyclic"] and not dag["dangling_inputs"],
             "protocol": {k: proto[k] for k in ("valid", "issues", "protocol_sha256", "ledger_entries")},
             "runs": tree["runs"], "invalid_runs": tree["invalid"], "dag_acyclic": dag["acyclic"],
             "dangling_inputs": dag["dangling_inputs"], "meaning": artifacts.MEANING}
    _print(r)
    return 0 if r["valid"] else 1


def _smoke(a) -> int:
    from .smoke import run
    r = run(a.out, workers=a.workers)
    _print(r)
    return 0 if r["valid"] else 1


HANDLERS: dict[str, Callable] = {"protocol-v08": _protocol, "dataset-registry-v08": _registry,
                                 "validate-market-data-v08": _validate, "report-v08": _report,
                                 "evidence-export-v08": _export, "claim-audit-v08": _claims, "verify-v08": _verify,
                                 "smoke-v08": _smoke, **{name: _study for name in STUDIES}}


def dispatch(args) -> int | None:
    """Run a v0.8 command; None when the command is not a v0.8 command."""
    handler = HANDLERS.get(args.command)
    return None if handler is None else handler(args)
