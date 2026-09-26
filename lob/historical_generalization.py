"""Local-only attribution of preserved failures and one preregistered fresh period.

Fitted historical models and detailed provider-derived artifacts stay local.
This module never fits a model. The earlier failed holdouts remain consumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import generalization as g
from .calibration import (_frame_hash, evaluate_calibration, extract_features,
                          generate_features, load_calibration)
from .calibration_diagnostics import diagnose_period, report_markdown
from .artifacts import portable_provenance
from .data_download import download_tardis_sample
from .experiments.registry import sha256_file, source_manifest
from .multiperiod import _write


def _seal(root: Path) -> None:
    _write(root, "manifest.json", {"algorithm": "sha256", "files": {
        p.name: sha256_file(p) for p in sorted(root.iterdir()) if p.is_file()}})


def _section(frame, start, end):
    result = frame.loc[(frame.timestamp_us >= start) & (frame.timestamp_us <= end)].copy().reset_index(drop=True)
    if result.empty:
        raise ValueError("registered period is absent")
    result.loc[0, "log_return"] = float("nan")
    return result


def _diagnostic(root, name, reference, target, original_status, generated, config):
    result = diagnose_period(reference, target, interval_us=1_000_000,
                             original_status=original_status, generated=generated,
                             block_rows=config["block_rows"], repetitions=config["bootstrap_repetitions"],
                             seed=config["diagnostic_seed"])
    _write(root, f"diagnostics-{name}.json", result)
    (root / f"diagnostics-{name}.md").write_text(report_markdown(result), encoding="utf-8")
    return result


def run_v03_diagnostics(study: str | Path, out: str | Path) -> dict:
    """Reconstruct the already consumed synthetic v0.3 fixture without refitting."""
    study, root = Path(study), Path(out)
    verified = g.verify_study(study)
    if not verified["valid"]:
        raise ValueError(f"original v0.3 study failed verification: {verified['issues']}")
    config = json.loads((study / "plan.json").read_text())["config"]
    if config["kind"] != "synthetic_smoke":
        raise ValueError("v0.3 fixture attribution only accepts explicitly synthetic data")
    model = json.loads((study / "model.json").read_text())
    g._check_model(model)
    root.mkdir(parents=True, exist_ok=False)
    settings = {"block_rows": 32, "bootstrap_repetitions": 999, "diagnostic_seed": 95201}
    provenance = portable_provenance()
    _write(root, "provenance.json", provenance)
    _write(root, "plan.json", {"kind": "consumed_v03_synthetic_diagnostics", "settings": settings,
                               "original_artifacts": {name: sha256_file(study / name) for name in
                                   ("plan.json", "model.json", "internal.json", "external.json")},
                               "fit_permitted": False, "real_market_evidence": False})
    development = g.synthetic_fixture(config["development"], config["interval_us"])
    reference = _section(development, model["train_start_us"], model["train_end_us"])
    if _frame_hash(reference) != model["train_frame_sha256"]:
        raise ValueError("reconstructed training does not match frozen model")
    results = []
    for name in ("internal", "external"):
        original = json.loads((study / f"{name}.json").read_text())
        target = (_section(development, original["start_us"], original["end_us"])
                  if name == "internal" else g.synthetic_fixture(config["external"], config["interval_us"]))
        if _frame_hash(target) != original["frame_sha256"]:
            raise ValueError("reconstructed target differs from consumed evidence")
        generated = g.generate_observables(model, config["simulation_rows"], config[f"{name}_seeds"][0])
        _diagnostic(root, name, reference, target, original["status"], generated, settings)
        results.append({"period": name, "original_status": original["status"], "status_changed": False})
    result = {"schema_version": 1, "kind": "synthetic_retrospective", "periods": results,
              "real_market_generalization": "NOT_ESTABLISHED", "model_unchanged": True}
    if source_manifest() != provenance["source_files"]:
        raise ValueError("source changed during registered diagnosis")
    _write(root, "result.json", result)
    _seal(root)
    return result


def run_historical(config: dict, out: str | Path, *, data_root: str | Path = ".") -> dict:
    """Use only the immutable 2020 model; download June after the local plan is saved.

    Do not rerun a consumed June file while describing it as fresh. A first-access
    marker is persisted beside the raw file before extraction, including failures.
    """
    base, root = Path(data_root), Path(out)
    config = json.loads(json.dumps(config, allow_nan=False))
    if config.get("schema_version") != 1 or config.get("kind") != "frozen_2020_local_diagnostics":
        raise ValueError("unsupported historical diagnosis configuration")
    model_path, original_path = base / config["model_path"], base / config["original_result_path"]
    for path, key in ((model_path, "model_file_sha256"), (original_path, "original_result_sha256")):
        if sha256_file(path) != config[key]:
            raise ValueError(f"immutable original {key} mismatch")
    model, original = load_calibration(model_path), json.loads(original_path.read_text())
    root.mkdir(parents=True, exist_ok=False)
    provenance = portable_provenance()
    _write(root, "provenance.json", provenance)
    _write(root, "plan.json", {"config": config, "fit_permitted": False,
                               "frozen_model_sha256": model.model_sha256,
                               "gates": original["test"]["thresholds"],
                               "publication": "LOCAL_ONLY; provider permission required for detailed redistribution",
                               "source_sha256": {p.name: sha256_file(p) for p in (
                                   Path(__file__), Path(__file__).with_name("calibration.py"),
                                   Path(__file__).with_name("calibration_diagnostics.py"))}})
    source = base / config["development_path"]
    if sha256_file(source) != config["development_sha256"]:
        raise ValueError("development source changed")
    print(json.dumps({"stage": "extract_consumed_april"}), flush=True)
    development, metadata = extract_features(source)
    reference = _section(development, model.train_start_us, model.train_end_us)
    if _frame_hash(reference) != model.train_frame_sha256:
        raise ValueError("historical reference does not match immutable model")
    generated = g.observe(generate_features(model, config["simulation_rows"], seed=config["generation_seed"]),
                          model.sample_interval_us)
    scores = []
    for name in ("validation", "internal_test", "test"):
        previous = original[name]
        if name == "test":
            path = base / config["consumed_external_path"]
            if sha256_file(path) != config["consumed_external_sha256"]:
                raise ValueError("consumed external source changed")
            print(json.dumps({"stage": "extract_consumed_may"}), flush=True)
            frame, meta = extract_features(path)
        else:
            frame, meta = development, metadata
        target = _section(frame, previous["holdout_start_us"], previous["holdout_end_us"])
        if _frame_hash(target) != previous["holdout_frame_sha256"]:
            raise ValueError("consumed target differs from original registered evidence")
        score = evaluate_calibration(model, target, meta)
        if score["status"] != previous["status"]:
            raise ValueError("reproduction does not match original failed assessment")
        _write(root, f"score-{name}.json", score)
        _diagnostic(root, name, reference, target, previous["status"], generated, config)
        scores.append({"period": name, "inspection": "consumed", "status": score["status"],
                       "rows": len(target), "source_sha256": score["holdout_source_sha256"]})
    fresh = config.get("fresh_source")
    fresh_status = "NOT_AVAILABLE"
    if fresh is not None:
        # Metadata-only plan selected this date before any attempt to download.
        directory = base / fresh["directory"]
        filename = f"{fresh['exchange']}_{fresh['data_type']}_{fresh['date']}_{fresh['symbol']}.csv.gz"
        marker = directory / (filename + ".v04-consumed.json")
        existing = directory / filename
        if marker.exists() or existing.exists():
            raise ValueError("planned fresh file is already present or consumed; do not relabel it fresh")
        try:
            print(json.dumps({"stage": "download_preregistered_june", "date": fresh["date"]}), flush=True)
            path = download_tardis_sample(fresh["exchange"], fresh["symbol"], fresh["date"],
                                         fresh["data_type"], directory, max_bytes=128 * 1024**2)
            intake = {"independent_period_id": f"{fresh['exchange']}:{fresh['symbol']}:{fresh['date']}",
                      "source_sha256": sha256_file(path), "plan_sha256": sha256_file(root / "plan.json"),
                      "sidecar_sha256": sha256_file(Path(str(path) + ".provenance.json")),
                      "inspection": "consumed", "reason": "persisted before first feature extraction"}
            with marker.open("x", encoding="utf-8") as stream:
                json.dump(intake, stream, indent=2)
            _write(root, "fresh-intake.json", intake)
            print(json.dumps({"stage": "extract_preregistered_june"}), flush=True)
            frame, meta = extract_features(path)
            score = evaluate_calibration(model, frame, meta)
            _write(root, "score-fresh-june.json", score)
            _diagnostic(root, "fresh-june", reference, frame, score["status"], generated, config)
            scores.append({"period": fresh["date"], "inspection_before_run": "fresh",
                           "status": score["status"], "rows": len(frame),
                           "source_sha256": score["holdout_source_sha256"]})
            fresh_status = score["status"]
        except (ValueError, OSError) as exc:
            fresh_status = "INVALID" if marker.exists() else "NOT_AVAILABLE"
            _write(root, "fresh-failure.json", {"status": fresh_status, "error": str(exc),
                                                "substitution_permitted": False})
    if sha256_file(model_path) != config["model_file_sha256"]:
        raise ValueError("frozen model changed during diagnosis")
    if source_manifest() != provenance["source_files"]:
        raise ValueError("source changed during registered diagnosis")
    result = {"schema_version": 1, "kind": "local_historical_diagnostics", "periods": scores,
              "fresh_external_status": fresh_status, "real_market_generalization": "NOT_ESTABLISHED",
              "model_unchanged": True, "old_failures_preserved": True,
              "publication_status": "PERMISSION_REQUIRED_FOR_DETAILED_ARTIFACTS",
              "limitations": "At most one fresh period from one venue/instrument; causal attribution not established."}
    _write(root, "result.json", result)
    _seal(root)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--v03-study")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if bool(args.config) == bool(args.v03_study):
        parser.error("supply exactly one of --config or --v03-study")
    result = (run_v03_diagnostics(args.v03_study, args.out) if args.v03_study else
              run_historical(json.loads(Path(args.config).read_text()), args.out, data_root=args.data_root))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
