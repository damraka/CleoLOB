"""Export a compact, public v0.5 evidence bundle from local run directories (M10).

Only permitted summaries are exported. Raw provider data, captures, per-order and
per-episode rows, historical tapes, detailed historical summaries, fitted
historical model parameters and checkpoints stay local; each excluded file is
listed with its original SHA-256, and each exported run keeps its binding (protocol
hash, ledger anchor, dataset identities, config/result hashes) and original
checksum manifest hash. A privacy scan rejects absolute personal paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lob.artifacts import seal_artifacts, verify_artifacts  # noqa: E402
from lob.config import canonical_json  # noqa: E402

PRIVATE = re.compile(r"([A-Za-z]:\\\\?Users\\\\?|/home/[^/\"]+/|/Users/[^/\"]+/)")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pick(data, keys):
    return {k: data[k] for k in keys if k in data}


def summarize_result(analysis: str, result: dict) -> dict:
    if analysis.startswith("m1-m2"):
        m1 = result["m1"]
        report = m1.get("lifecycle_report", {})
        validation = m1.get("validation", {})
        return {"post_hoc_exploratory": result.get("post_hoc_exploratory", False), "gate": m1["gate"],
                "attempts": result.get("attempts"), "unsupported_order_created_types": m1.get("unsupported_order_created_types"),
                "adapter_stats": validation.get("adapter_stats"),
                "deterministic_replay": validation.get("deterministic_replay"),
                "reference_comparison": _pick(validation.get("reference_comparison", {}),
                                              ("compared", "exact", "exact_fraction", "price_level_mismatch",
                                               "quantity_mismatch")),
                "secondary_tolerant_reference_agreement": m1.get("secondary_tolerant_reference_agreement"),
                "lifecycle_counts": {k: v for k, v in report.items() if k not in {"anomaly_examples", "snapshot_checks"}},
                "census_checks": [_pick(c, ("census_orders", "exact_order_matches", "missing_from_replay",
                                            "extra_in_replay", "adjacent_priority_concordant",
                                            "adjacent_priority_discordant")) for c in report.get("snapshot_checks", [])],
                "m2": result.get("m2") if isinstance(result.get("m2"), dict) and "status" in result["m2"] else
                {"summary": result.get("m2", {}).get("summary"),
                 "observed_order_validation": result.get("m2", {}).get("observed_order_validation")}}
    if analysis.startswith("m2-l2"):
        return _pick(result, ("dataset_id", "summary", "by_size_and_lifetime", "scope"))
    if analysis.startswith("m3-develop"):
        return _pick(result, ("label", "admitted_families", "excluded_families", "dispersion_index", "training_losses",
                              "candidates", "seconds_per_seed", "interpretation"))
    if analysis == "m3-select":
        return {"selected_family": result["selected"]["family"], "selection_losses": result["selection_losses"],
                "improving_extensions": result["improving_extensions"],
                "combined_members": (result.get("combined") or {}).get("members")}
    if analysis == "m3-evaluate":
        return {"selected_family": result["selected_family"], "alpha_per_holdout": result["alpha_per_holdout"],
                "holdouts": {role: {"dataset": h["dataset"], "retrospective": h["retrospective"], "losses": h["losses"],
                                    "family_errors": {m: {f: e["error"] for f, e in errs.items()}
                                                      for m, errs in h["family_errors"].items()},
                                    "generalization_gate": h["generalization_gate"], "improvement_H3a": h["improvement_H3a"]}
                             for role, h in result["holdouts"].items()}, "interpretation": result["interpretation"]}
    if analysis.startswith("m4-"):
        return {"dataset_id": result["dataset_id"], "label": result["label"], "events": result["events"],
                "selected_family": result["selected_family"],
                "H4": {name: c["H4_by_horizon"] for name, c in result["comparison"].items()},
                "persistence": {name: c["persistence_60s_over_1s"] for name, c in result["comparison"].items()},
                "historical_resilience": _pick(result["historical_resilience"], ("events", "recovered_share")),
                "interpretation": result["interpretation"]}
    if analysis == "m6-thresholds":
        return _pick(result, ("development_blocks", "development_counts"))
    if analysis.startswith("m6-"):
        return {"dataset_id": result["dataset_id"], "role": result["role"], "blocks": result["blocks"],
                "regime_counts": result["regime_counts"], "regime_robustness": result["regime_robustness"],
                "per_regime": {k: ({"blocks": v["blocks"], "status": v["status"],
                                    "m3": v.get("m3"), "m4_H4": v.get("m4_H4")}) for k, v in result["per_regime"].items()},
                "different_venue_layer": result["different_venue_layer"], "retrospective": result["retrospective"]}
    if analysis == "m8-transfer":
        return _pick(result, ("mapping", "family_size", "alpha", "comparisons", "classification_counts", "rankings",
                              "synthetic_ranking_original_regime", "invalid_rows", "rows", "interpretation"))
    if analysis == "m11-benchmarks":
        return result
    raise ValueError(f"no export rule for analysis {analysis!r}")


def export_run(run: Path, target: Path) -> dict:
    binding = json.loads((run / "binding.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    analysis = binding["analysis"]
    target.mkdir(parents=True, exist_ok=False)
    public = summarize_result(analysis, result)
    (target / "summary.json").write_text(canonical_json(public) + "\n", encoding="utf-8", newline="\n")
    shutil.copyfile(run / "binding.json", target / "binding.json")
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    (target / "provenance.json").write_text(canonical_json({k: provenance[k] for k in ("git_commit", "git_dirty",
                                                            "source_sha256", "runtime")}) + "\n",
                                            encoding="utf-8", newline="\n")
    exported = {"summary.json", "binding.json", "provenance.json"}
    excluded = {p.relative_to(run).as_posix(): _sha(p) for p in sorted(run.rglob("*")) if p.is_file()
                and p.name not in {"binding.json", "checksums.json"}}
    return {"analysis": analysis, "source_run": run.name, "original_checksums_sha256": _sha(run / "checksums.json"),
            "original_result_sha256": _sha(run / "result.json"), "exported": sorted(exported),
            "excluded_with_original_sha256": excluded}


def export_policy_study(run: Path, target: Path) -> dict:
    """Synthetic M7 study: plan, locks, summaries and journal hash; checkpoints excluded."""
    target.mkdir(parents=True, exist_ok=False)
    keep = ["preregistration.json", "registration_seal.json", "normalization.json", "normalization_seal.json",
            "evaluation_lock.json", "training_summary.json", "result.json", "manifest.json"]
    for name in keep:
        shutil.copyfile(run / name, target / name)
    excluded = {p.relative_to(run).as_posix(): _sha(p) for p in sorted(run.rglob("*"))
                if p.is_file() and p.name not in keep}
    return {"analysis": "m7-policy-study", "source_run": run.name, "exported": keep,
            "excluded_with_original_sha256": excluded}


def privacy_scan(root: Path) -> list[str]:
    hits = []
    for path in root.rglob("*"):
        if path.is_file() and PRIVATE.search(path.read_text(encoding="utf-8", errors="ignore")):
            hits.append(path.relative_to(root).as_posix())
    return hits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="v0.5 run directories with binding.json")
    parser.add_argument("--policy-study", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    for run in args.runs:
        entries.append(export_run(run, out / run.parent.name / run.name))
    if args.policy_study:
        entries.append(export_policy_study(args.policy_study, out / "m7" / args.policy_study.name))
    for name in ("configs/v05/protocol.json", "configs/v05/consumption-ledger.jsonl"):
        (out / "registration").mkdir(exist_ok=True)
        shutil.copyfile(ROOT / name, out / "registration" / Path(name).name)
    (out / "export.json").write_text(canonical_json({"schema": "cleolob-v05-public-evidence-1", "runs": entries,
                                                     "exclusion_policy": __doc__}) + "\n", encoding="utf-8", newline="\n")
    hits = privacy_scan(out)
    if hits:
        raise SystemExit(f"privacy scan found absolute personal paths in: {hits}")
    seal_artifacts(out)
    print(json.dumps(verify_artifacts(out), indent=2))


if __name__ == "__main__":
    main()
