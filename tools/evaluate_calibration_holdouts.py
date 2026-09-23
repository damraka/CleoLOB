"""Evaluate frozen July fits on August, then unopened September observations.

This is the one-shot execution of SEARCH_AMENDMENT.json, with source, model and
download provenance checks. Existing results are never overwritten or rerun.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

from lob.zi_calibration import evaluate_model, load_model


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="Continue after an orchestration failure, reusing completed results")
    args = parser.parse_args()
    root = args.root.resolve()
    calibration = root / "examples/studies/core/calibration"
    output = args.out or calibration / "frozen-external-evaluation"
    output.mkdir(parents=True, exist_ok=args.resume)
    if (output / "summary.json").exists():
        raise ValueError("external evaluation is already complete")
    amendment_path = calibration / "SEARCH_AMENDMENT.json"
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    sources = [root / "lob/engine.py", root / "lob/zi_calibration.py", Path(__file__).resolve()]
    source_hashes = {str(path.relative_to(root)): sha256(path) for path in sources}
    model_paths = {cohort: calibration / f"zi-{cohort}-amended-fit/model.json"
                   for cohort in ("eth", "btc", "pooled")}
    models = {cohort: load_model(path) for cohort, path in model_paths.items()}
    model_hashes = {cohort: sha256(path) for cohort, path in model_paths.items()}
    symbols = {"eth": ["ETH-PERPETUAL"], "btc": ["BTC-PERPETUAL"],
               "pooled": ["BTC-PERPETUAL", "ETH-PERPETUAL"]}
    phases = [("validation", "2026-08-01", amendment["validation_simulation_seeds"]),
              ("holdout", "2026-09-01", amendment["holdout_simulation_seeds"])]
    all_seeds = [seed for _phase, _date, seeds in phases for seed in seeds]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("validation and holdout simulator seeds must be disjoint")
    data_provenance = {}
    for _phase, date, _seeds in phases:
        for symbol in ("BTC-PERPETUAL", "ETH-PERPETUAL"):
            path = root / "data/public" / f"deribit_book_snapshot_5_{date}_{symbol}.csv.gz"
            sidecar = Path(f"{path}.provenance.json")
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            actual = sha256(path)
            if actual != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise ValueError(f"download provenance mismatch: {path.name}")
            data_provenance[path.name] = {"sha256": actual, "bytes": path.stat().st_size,
                                          "sidecar_sha256": sha256(sidecar), "provenance": record}
    plan = {"schema_version": 1, "registered_at_utc": utc_now(),
            "amendment_sha256": sha256(amendment_path), "source_sha256": source_hashes,
            "model_file_sha256": model_hashes,
            "model_sha256": {name: model["model_sha256"] for name, model in models.items()},
            "training_status": {name: model["training_comparison"]["status"]
                                for name, model in models.items()},
            "data_provenance": data_provenance,
            "ordered_phases": [{"phase": phase, "date": date, "simulation_seeds": seeds}
                               for phase, date, seeds in phases],
            "seconds_per_seed": amendment["evaluation_simulation_seconds_per_seed"],
            "model_changes_permitted": False,
            "python_version": sys.version, "platform": platform.platform(),
            "interpretation": "A failed external gate remains failed; subsequent execution conclusions are conditional on the synthetic simulator."}
    if args.resume:
        prior = json.loads((output / "plan.json").read_text(encoding="utf-8"))
        for key in ("amendment_sha256", "model_file_sha256", "model_sha256",
                    "data_provenance", "ordered_phases", "seconds_per_seed"):
            if prior[key] != plan[key]:
                raise ValueError(f"resume changes frozen plan field {key}")
        for key in ("lob/engine.py", "lob/zi_calibration.py"):
            native_key = str(Path(key))
            if prior["source_sha256"][native_key] != source_hashes[native_key]:
                raise ValueError("resume changes frozen scientific code")
        runner_key = str(Path("tools/evaluate_calibration_holdouts.py"))
        write_new(output / "orchestration-resume.json", {
            "resumed_at_utc": utc_now(),
            "reason": "Reporting-only KeyError: comparison families expose passed, not status. Completed evaluation retained without rerun.",
            "original_runner_sha256": prior["source_sha256"][runner_key],
            "resumed_runner_sha256": source_hashes[runner_key],
            "reused_result_sha256": {path.name: sha256(path) for path in output.glob("validation-*.json")},
            "scientific_sources_unchanged": True})
    else:
        write_new(output / "plan.json", plan)
    results = []
    for phase, date, seeds in phases:
        for cohort, model in models.items():
            for source in sources:
                if sha256(source) != source_hashes[str(source.relative_to(root))]:
                    raise RuntimeError("evaluation source changed during frozen evaluation")
            for name, path in model_paths.items():
                if sha256(path) != model_hashes[name]:
                    raise RuntimeError("frozen model changed during external evaluation")
            result_path = output / f"{phase}-{cohort}.json"
            if args.resume and result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if (result["phase"] != phase or result["cohort"] != cohort
                        or result["model_sha256"] != model["model_sha256"]
                        or result["model_file_sha256"] != model_hashes[cohort]
                        or result["simulation_seeds"] != seeds or not result["model_unchanged"]):
                    raise ValueError("completed result does not match frozen plan")
            else:
                started = utc_now()
                print(json.dumps({"phase": phase, "cohort": cohort, "started_at_utc": started}), flush=True)
                paths = [root / "data/public" / f"deribit_book_snapshot_5_{date}_{symbol}.csv.gz"
                         for symbol in symbols[cohort]]
                result = evaluate_model(model, paths, seeds=seeds,
                                        simulation_seconds=amendment["evaluation_simulation_seconds_per_seed"])
                result.update({"phase": phase, "cohort": cohort, "started_at_utc": started,
                               "completed_at_utc": utc_now(), "model_file_sha256": model_hashes[cohort],
                               "source_sha256": source_hashes,
                               "download_sidecars_verified": True})
                write_new(result_path, result)
            for item in result["instruments"].values():
                metadata = item["source"]
                if metadata["sha256"] != data_provenance[Path(metadata["path"]).name]["sha256"]:
                    raise RuntimeError("input changed after download provenance verification")
            row = {"phase": phase, "cohort": cohort, "status": result["status"],
                   "source_rows": sum(item["source"]["rows"] for item in result["instruments"].values()),
                   "samples": result["target_summary"]["samples"],
                   "valid_samples": result["target_summary"]["valid_samples"],
                   "valid_fraction": result["target_summary"]["valid_fraction"],
                   "family_errors": {family: item["error"] for family, item in result["comparison"]["families"].items()},
                   "failed_families": [family for family, item in result["comparison"]["families"].items()
                                       if not item["passed"]],
                   "completed_at_utc": result["completed_at_utc"]}
            results.append(row)
            print(json.dumps(row), flush=True)
    unique_sources = {}
    for phase, _date, _seeds in phases:
        for cohort in ("eth", "btc"):
            record = json.loads((output / f"{phase}-{cohort}.json").read_text(encoding="utf-8"))
            unique_sources.update({item["source"]["sha256"]: item["source"]
                                   for item in record["instruments"].values()})
    summary = {"schema_version": 1, "completed_at_utc": utc_now(), "results": results,
               "overall_status": "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL",
               "unique_external_files": len(unique_sources),
               "unique_external_source_rows": sum(item["rows"] for item in unique_sources.values()),
               "unique_external_samples": sum(item["samples"] for item in unique_sources.values()),
               "unique_external_valid_samples": sum(item["valid_samples"] for item in unique_sources.values()),
               "models_unchanged": all(sha256(path) == model_hashes[name] for name, path in model_paths.items()),
               "sources_unchanged": all(sha256(path) == source_hashes[str(path.relative_to(root))] for path in sources),
               "limitations": next(iter(models.values()))["protocol"]["limitations"]}
    write_new(output / "summary.json", summary)
    checksums = {path.name: sha256(path) for path in sorted(output.glob("*.json"))}
    write_new(output / "checksums.json", {"algorithm": "SHA-256", "files": checksums})
    print(json.dumps({"summary": str(output / "summary.json"), "status": summary["overall_status"]}), flush=True)


if __name__ == "__main__":
    main()
