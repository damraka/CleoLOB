"""v0.9 command-line workflows, attached to ``cleo`` by ``lob.cli``.

Every command validates its arguments deterministically, prints one JSON document and fails with a machine-readable
status (exit 1) instead of falling back silently. Study runs are write-once directories (under ``results/v09`` by
convention) bound to the v0.9 protocol, ledger and confirmatory registry. Guarded seed namespaces (``confirm``,
``holdout``, ``final``) require a sealed design that opened them.

Common study options: ``--root``, ``--out``, ``--design`` (a sealed design name), ``--namespace``, ``--seed``,
``--workers``, ``--budget``, ``--resume`` (resume a checkpoint under ``<out>.checkpoint``) and repeatable
``--input KEY=RUN`` bindings to upstream sealed runs.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import re
from typing import Any

BUDGETS = ("tiny", "small", "medium", "large")
NAMESPACES = ("dev", "explore", "pilot", "confirm", "holdout", "final")

# command -> (study module under lob.v09.studies, help, phases or None)
STUDIES: dict[str, tuple[str, str, tuple[str, ...] | None]] = {
    "validate-market-data-v09": ("data", "M13-M19 market-data quality: timestamps, gaps, duplicates, invariants, normalization, profile, drift",
        None),
    "event-reconstruction-v09": ("reconstruction", "M20/M21 event-sourced L2 reconstruction and replay consistency (reference vs dense)",
        None),
    "historical-replay-v09": ("replay", "M26-M32 pure, latency-aware, race and bounded replay with validity scores",
        None),
    "queue-reconstruction-v09": ("queue", "M22-M24 queue-reconstruction confidence, observation uncertainty, ambiguous orderings",
        None),
    "latent-liquidity-v09": ("liquidity", "M25/M63-M65 hidden-liquidity bounds, refresh proxies, sensitivity",
        None),
    "fill-study-v09": ("fills", "M66/M68 fill-probability calibration of generators (H6); H8 from the decisions hub",
        ("pilot", "confirm")),
    "survival-study-v09": ("survival", "M67 queue-survival models: KM, Cox, AFT, boosted, neural (H7)",
        ("pilot", "confirm")),
    "markout-study-v09": ("markout", "M69/M70 post-fill markout and adverse selection (H9 from bounded replay)",
        None),
    "toxicity-study-v09": ("toxicity", "M70 toxicity / flow-risk proxies (descriptive)",
        None),
    "regime-study-v09": ("regimes", "regime ontology and detection on real days (descriptive)",
        None),
    "change-point-study-v09": ("drift", "change-point and drift detection on real days (descriptive)",
        None),
    "realism-benchmark-v09": ("realism", "M51-M54 realism scorecard v3 of the full-book generators",
        None),
    "realism-gate-v09": ("gate", "M55 capability-specific realism gate from a sealed benchmark run",
        None),
    "path-realism-v09": ("path_realism", "M52 path-level realism (signatures, spectra, first passage)",
        None),
    "tail-realism-v09": ("tail_realism", "M54 tail realism (EVT, spread extremes, depth collapse)",
        None),
    "generator-fit-v09": ("generator_fit", "M44/M45 classical generator calibration on real days (parameters, fit)",
        None),
    "neural-point-process-v09": ("tpp", "M45 neural temporal point processes vs classical (H2) and averaging (H5)",
        ("pilot", "confirm")),
    "sequence-generator-v09": ("sequence", "M46 sequence models vs classical conditional event models (H3)",
        ("pilot", "confirm")),
    "latent-dynamics-v09": ("latent", "M47/M48 latent dynamics and path realism (H4)",
        ("pilot", "confirm")),
    "hybrid-twin-v09": ("hybrid", "M49 hybrid historical twin response realism (H1, H12)",
        ("pilot", "confirm")),
    "ecology-study-v09": ("ecology", "M50 strategic vs static ecology (H12, re-sealed from hybrid-twin-v09)",
        None),
    "strategic-response-v09": ("strategic", "M50 self-impact and endogenous response in synthetic worlds (H11)",
        None),
    "inference-v09": ("inference", "M56-M59 sequential posterior updating (H14)",
        ("pilot", "confirm")),
    "sbc-v09": ("sbc", "M56/M57 simulation-based calibration of the SBI engines",
        None),
    "identifiability-v5": ("identifiability", "M61 structural + practical identifiability v5",
        None),
    "posterior-study-v09": ("posterior", "M60 posterior predictive checks v3",
        None),
    "model-averaging-v09": ("averaging", "M17/M37 dynamic model averaging (H5, re-sealed from the TPP run)",
        None),
    "active-calibration-v09": ("active", "M62 active calibration / simulation acquisition (H13)",
        None),
    "continual-calibration-v09": ("continual", "continual calibration and forgetting (H16)",
        ("pilot", "confirm")),
    "meta-adaptation-v09": ("meta", "few-shot / hierarchical adaptation vs full refit (H15)",
        ("pilot", "confirm")),
    "conformal-study-v09": ("conformal", "sequential conformal coverage under drift (H17)",
        ("pilot", "confirm")),
    "impact-study-v09": ("impact", "M71-M73 impact kernels, decomposition, state dependence (H10)",
        ("pilot", "confirm")),
    "cross-impact-v09": ("cross_impact", "M74 cross-impact estimation (descriptive)",
        None),
    "completion-study-v09": ("completion", "chance-constrained completion (H22, re-sealed from the decisions hub)",
        None),
    "dro-study-v09": ("dro", "M79 distributionally robust schedules over ambiguity sets (H21)",
        None),
    "mpc-study-v09": ("mpc", "M80 MPC v2: tail cost on bounded replay (H23) and adaptive MPC after drift (H24)",
        None),
    "meta-policy-v09": ("meta_policy", "M83 meta-policy / mixture of experts (H25, re-sealed from the decisions hub)",
        None),
    "abstention-study-v09": ("abstention", "M84 abstention v3 (H18, H19 from the hub) and contextual fallback (H20)",
        None),
    "offline-rl-v09": ("offline_rl", "M81 offline RL (BC, CQL, IQL, TD3+BC, DT) with support gates (H26)",
        None),
    "ope-study-v09": ("ope", "M81 off-policy evaluation with capability gates",
        None),
    "safe-improvement-v09": ("safe_improvement", "M82 safe policy improvement and gating (H27)",
        None),
    "rare-event-study-v09": ("rare_event", "M85 rare-event simulation and EVT-aware optimization (H28)",
        None),
    "adversarial-study-v09": ("adversarial", "M85 bounded adversarial stress search (failure envelopes)",
        None),
    "counterexample-study-v09": ("counterexample", "M85 minimum plausible counterexamples to conclusions",
        None),
    "counterfactual-bounds-v09": ("bounds", "M28/M78 bounded counterfactual execution and partial identification",
        ("pilot", "confirm")),
    "transfer-study-v09": ("transfer", "M86 cross-time / instrument / venue transfer of carried conclusions (H32-H34)",
        ('time', 'instrument', 'venue')),
    "representation-transfer-v09": ("representation", "M38-M43 learned representations and transfer (H35)",
        ("pilot", "confirm")),
    "portfolio-study-v09": ("portfolio", "M87 basket execution and cross-impact-aware scheduling",
        None),
    "ranking-transfer-v09": ("ranking", "policy ranking transfer across fidelities (descriptive)",
        None),
    "multifidelity-agreement-v09": ("decisions", "M33-M37/M83/M84/M88 hybrid-twin decisions and multi-fidelity agreement hub (H8, H18, H19, H22, H25, H29-H31)",
        None),
    "uncertainty-decomposition-v09": ("uncertainty", "uncertainty decomposition across registered sources",
        None),
    "value-of-information-v09": ("voi", "M88 value of information (oracle experiments)",
        None),
    "benchmark-v09": ("benchmark", "local research-workload performance measurements (not HFT benchmarks)",
        None),
    "final-analysis-v09": ("final", "M89 mechanical resolution of every promoted hypothesis and the final vault",
        None),
}
SPECIAL = ("protocol-v09", "hypothesis-promotion-v09", "dataset-registry-v09", "capability-audit-v09",
           "report-v09", "evidence-export-v09", "claim-audit-v09", "verify-v09", "smoke-v09")
COMMANDS = tuple(STUDIES) + SPECIAL
_INPUT = re.compile(r"[A-Za-z0-9_.-]+=[^=]+")


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=True, default=str))


def _input(value: str) -> tuple[str, str]:
    if not _INPUT.fullmatch(value):
        raise argparse.ArgumentTypeError("--input must be KEY=RUN")
    key, run = value.split("=", 1)
    return key, run


def _positive(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _study_args(c: argparse.ArgumentParser, phases) -> None:
    if phases:
        c.add_argument("phase", choices=phases)
    c.add_argument("--root", type=Path, default=Path("."))
    c.add_argument("--out", required=True, help="write-once run directory relative to --root")
    c.add_argument("--design", default=None, help="sealed design name (required for guarded namespaces)")
    c.add_argument("--namespace", choices=NAMESPACES, default="dev")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--workers", type=_positive, default=None)
    c.add_argument("--budget", choices=BUDGETS, default="small")
    c.add_argument("--resume", action="store_true", help="resume the checkpoint next to --out")
    c.add_argument("--input", type=_input, action="append", default=[], metavar="KEY=RUN")
    c.add_argument("--dataset", action="append", default=[], help="registered dataset id (repeatable)")


def add_commands(commands) -> None:
    c = commands.add_parser("protocol-v09", help="v0.9 protocol, ledger v2, firewall, runtime fingerprint, SBOM, lock")
    c.add_argument("operation", choices=("verify", "status", "build", "init", "fingerprint", "sbom", "lock"))
    c.add_argument("--root", type=Path, default=Path("."))
    c.add_argument("--base-commit", default=None, help="base commit recorded by 'build'")
    c = commands.add_parser("hypothesis-promotion-v09", help="M2 promotion gate (run) and confirmatory freeze")
    c.add_argument("operation", choices=("run", "freeze", "feasibility"))
    c.add_argument("--root", type=Path, default=Path("."))
    c.add_argument("--out", default=None)
    c.add_argument("--run", default=None, help="promotion run to freeze")
    c.add_argument("--input", type=_input, action="append", default=[], metavar="KEY=RUN")
    c = commands.add_parser("dataset-registry-v09", help="M10-M12 dataset registry v4, roles, freshness, trust")
    c.add_argument("--root", type=Path, default=Path("."))
    c.add_argument("--pool", default=None)
    c = commands.add_parser("capability-audit-v09", help="M11 capability graph: which hypotheses each source can serve")
    c.add_argument("--root", type=Path, default=Path("."))
    for name, (_module, help_text, phases) in STUDIES.items():
        _study_args(commands.add_parser(name, help=help_text), phases)
    c = commands.add_parser("report-v09", help="figures, tables, model/policy cards, datasheets and the report")
    c.add_argument("--root", type=Path, default=Path("."))
    c.add_argument("--out", required=True)
    c.add_argument("--final", required=True, help="sealed final-analysis-v09 run")
    c.add_argument("--input", type=_input, action="append", default=[], metavar="KEY=RUN")
    c = commands.add_parser("evidence-export-v09", help="export the public v0.9 evidence bundle")
    c.add_argument("--root", type=Path, default=Path("."))
    c.add_argument("--out", required=True)
    c.add_argument("--final", required=True)
    c = commands.add_parser("claim-audit-v09", help="fail-closed claim graph and wording audit of README and v0.9 docs")
    c.add_argument("--root", type=Path, default=Path("."))
    c.add_argument("--final", default=None)
    c.add_argument("--bundle", type=Path, default=None)
    c = commands.add_parser("verify-v09", help="verify protocol, ledger, runs, lineage, checkpoints and the bundle")
    c.add_argument("--root", type=Path, default=Path("."))
    c.add_argument("--bundle", type=Path, default=None)
    c = commands.add_parser("smoke-v09", help="fast end-to-end functionality smoke in an isolated root")
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=_positive, default=1)


def dispatch(args) -> int | None:
    name = args.command
    if name not in COMMANDS:
        return None
    if name in STUDIES:
        module = importlib.import_module(f"lob.v09.studies.{STUDIES[name][0]}")
        result = module.run(args)
    else:
        result = getattr(importlib.import_module("lob.v09.commands"), name.replace("-", "_"))(args)
    _print(result)
    return 0 if result.get("valid", True) and result.get("status") not in ("INVALID", "ERROR") else 1
