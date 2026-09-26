"""Registered chronological studies with physically separate, late-read holdouts.

No model selection or refitting occurs after ``selection-seal.json``. Dataset
inspection history is an auditable declaration, not a claim that a hash proves
human non-inspection. Synthetic fixtures are always labeled synthetic.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .calibration_diagnostics import diagnose_period, report_markdown
from .artifacts import portable_provenance
from .experiments.registry import sha256_file, source_manifest
from .generalization import (FAMILIES, GATE, _check_model, _digest, _integer,
                             evaluate_observable_model, fit_observable_model,
                             generate_observables, observe, synthetic_fixture)


ROLES = {"development": 0, "selection": 1, "internal": 2, "external": 3}
CORE = ("plan.json", "validation.json", "model.json", "selection.json")


def _write(root: Path, name: str, value: dict) -> None:
    with (root / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _hash(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def validate_plan(config: dict) -> None:
    """Validate descriptors without opening, statting or hashing any data file."""
    if config.get("schema_version") != 1 or config.get("kind") not in {"synthetic", "feature_csv"}:
        raise ValueError("unsupported multi-period schema/kind")
    for key, low, high in (("interval_us", 1, 86_400_000_000), ("embargo_us", 0, 86_400_000_000),
                           ("simulation_rows", 64, 100000), ("max_pool_rows", 32, 4096),
                           ("block_rows", 2, 10000), ("bootstrap_repetitions", 99, 10000)):
        _integer(config[key], key, low, high)
    for name in ("exchange", "symbol"):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError(f"{name} is required")
    periods = config["periods"]
    if not isinstance(periods, list) or len(periods) < 4:
        raise ValueError("at least development, selection, internal and external periods required")
    roles = [period["role"] for period in periods]
    if (any(role not in ROLES for role in roles) or roles.count("development") != 1
            or roles.count("internal") != 1 or not {"selection", "external"} <= set(roles)
            or [ROLES[r] for r in roles] != sorted(ROLES[r] for r in roles)):
        raise ValueError("ordered development/selection/internal/external roles required")
    known = config.get("consumed_periods", [])
    if not isinstance(known, list):
        raise ValueError("consumed_periods must be a registry list")
    seen_ids, seen_paths, seen_hashes = set(), set(), set()
    seeds = [config["fit_seed"], config["diagnostic_seed"]]
    for role in ("selection", "internal", "external"):
        values = config[f"{role}_seeds"]
        if not isinstance(values, list) or not 1 <= len(values) <= 16:
            raise ValueError("one to 16 simulation seeds required per phase")
        seeds.extend(values)
    previous_end = None
    for period in periods:
        if not re.fullmatch("[a-zA-Z0-9_-]{1,100}", period["id"]) or period["id"] in seen_ids:
            raise ValueError("period identifiers must be distinct and safe")
        seen_ids.add(period["id"])
        _integer(period["start_us"], "start_us", 0, 2**63 - 1)
        _integer(period["rows"], "rows", 64, 1_000_000)
        end = period["start_us"] + (period["rows"] - 1) * config["interval_us"]
        if end >= 2**63 or (previous_end is not None
                            and period["start_us"] <= previous_end + config["embargo_us"]):
            raise ValueError("periods must be strictly chronological with the registered embargo")
        previous_end = end
        if config["kind"] == "synthetic":
            if period["inspection"] != "synthetic":
                raise ValueError("synthetic periods cannot be fresh historical evidence")
            seeds.append(period["seed"])
            shift = period.get("shift", 1.0)
            if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not .1 <= shift <= 10:
                raise ValueError("synthetic shift outside supported bounds")
        else:
            if period["inspection"] not in {"consumed", "fresh"}:
                raise ValueError("historical inspection must declare consumed or fresh")
            if not period.get("independent_period_id") or not period.get("provenance"):
                raise ValueError("source identity and provenance declarations required")
            for name in ("sha256", "source_sha256"):
                if not _hash(period.get(name)):
                    raise ValueError(f"registered {name} required")
            if period["sha256"] in seen_hashes:
                raise ValueError("duplicate canonical content cannot be independent evidence")
            seen_hashes.add(period["sha256"])
            path = Path(period["path"])
            if (path.is_absolute() or ".." in path.parts or ":" in period["path"]
                    or "\\" in period["path"] or period["path"] in seen_paths):
                raise ValueError("distinct portable per-period source paths required")
            seen_paths.add(period["path"])
            if period["inspection"] == "fresh":
                if period["role"] != "external":
                    raise ValueError("fresh evidence is reserved for final external evaluation")
                for record in known:
                    if (record.get("independent_period_id") == period["independent_period_id"]
                            or record.get("source_sha256") == period["source_sha256"]):
                        raise ValueError("previously consumed source cannot be registered fresh")
                for other in periods:
                    if (other is not period and other["inspection"] == "fresh" and (
                            other["independent_period_id"] == period["independent_period_id"]
                            or other["source_sha256"] == period["source_sha256"])):
                        raise ValueError("fresh external periods must have distinct source identities")
                    if other is not period and other["inspection"] == "consumed" and (
                            other["independent_period_id"] == period["independent_period_id"]
                            or other["source_sha256"] == period["source_sha256"]):
                        raise ValueError("a consumed period cannot be relabeled fresh")
    for seed in seeds:
        _integer(seed, "seed", 0, 2**63 - 1)
    if len(seeds) != len(set(seeds)):
        raise ValueError("fit, diagnostic, data and phase seeds must be disjoint")


def _load_period(config: dict, period: dict, data_root: Path) -> pd.DataFrame:
    if config["kind"] == "synthetic":
        frame = synthetic_fixture(period, config["interval_us"])
    else:
        path = data_root / period["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(data_root.resolve()):
            raise ValueError("data path escapes declared root")
        if sha256_file(path) != period["sha256"]:
            raise ValueError(f"{period['id']} source hash differs from registration")
        frame = pd.read_csv(path)
        if sha256_file(path) != period["sha256"]:
            raise ValueError("source changed while reading")
    expected = period["start_us"] + np.arange(period["rows"], dtype=np.int64) * config["interval_us"]
    if len(frame) != period["rows"] or not np.array_equal(frame.timestamp_us.to_numpy(), expected):
        raise ValueError("period rows/timestamps differ from registration")
    observe(frame, config["interval_us"])
    return frame


def _seal(root: Path) -> None:
    seal = json.loads((root / "selection-seal.json").read_text())
    if set(seal["files"]) != set(CORE):
        raise ValueError("selection seal has wrong file set")
    for name, digest in seal["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("selection seal mismatch")


def run_study(config: dict, out: str | Path, *, data_root: str | Path = ".") -> dict:
    """Register first, select using development/selection only, seal, then open tests."""
    config = json.loads(json.dumps(config, allow_nan=False))
    validate_plan(config)
    root, data_root = Path(out), Path(data_root)
    root.mkdir(parents=True, exist_ok=False)
    plan = {"config": config, "gate": GATE, "candidates": list(FAMILIES),
            "selection": "minimum mean selection-period loss, lexical family tie-break; no final refit",
            "freshness": "declaration checked against consumed registry; hashes cannot prove non-inspection",
            "claim_scope": "observable calibration only; no fills, trading edge or exchange truth"}
    _write(root, "plan.json", plan)
    provenance = portable_provenance()
    _write(root, "provenance.json", {"registered_at_utc": datetime.now(timezone.utc).isoformat(),
                                     **provenance})
    development = _load_period(config, config["periods"][0], data_root)
    validation = [(p, _load_period(config, p, data_root)) for p in config["periods"]
                  if p["role"] == "selection"]
    candidates, models = [], {}
    for family in FAMILIES:
        records = []
        try:
            model = fit_observable_model(development, family=family, interval_us=config["interval_us"],
                                         seed=config["fit_seed"], max_pool_rows=config["max_pool_rows"])
            models[family] = model
            for period, frame in validation:
                score = evaluate_observable_model(model, frame, seeds=config["selection_seeds"],
                                                   simulation_rows=config["simulation_rows"])
                records.append({"period_id": period["id"], **score})
            candidates.append({"family": family, "eligible": True, "periods": records,
                               "mean_loss": float(np.mean([r["loss"] for r in records])), "error": None})
        except (ValueError, RuntimeError) as exc:
            candidates.append({"family": family, "eligible": False, "periods": records,
                               "mean_loss": None, "error": f"{type(exc).__name__}: {exc}"})
    _write(root, "validation.json", {"candidates": candidates})
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        _write(root, "failure.json", {"status": "FAILED", "reason": "all selection candidates failed"})
        raise ValueError("all candidates failed; diagnostics retained")
    selected = min(eligible, key=lambda c: (c["mean_loss"], c["family"]))
    model = models[selected["family"]]
    _write(root, "model.json", model)
    # Selection hash binds only development and selection values, never holdout bytes.
    _write(root, "selection.json", {"family": selected["family"], "mean_loss": selected["mean_loss"],
                                    "model_sha256": model["model_sha256"]})
    _write(root, "selection-seal.json", {"sealed_before_holdout_access": True,
                                         "files": {name: sha256_file(root / name) for name in CORE}})
    scores, consumption = [], []
    for period in config["periods"]:
        if period["role"] not in {"internal", "external"}:
            continue
        _seal(root)
        if source_manifest() != provenance["source_files"]:
            raise ValueError("implementation changed after registration; attempt retained")
        # Persist consumption intent before the first attempt, including failed loads.
        consumption.append({"id": period["id"], "role": period["role"],
                            "previous_inspection": period["inspection"],
                            "independent_period_id": period.get("independent_period_id"),
                            "source_sha256": period.get("source_sha256"), "inspection": "consumed"})
        _write(root, f"access-{period['id']}.json", consumption[-1])
        try:
            frame = _load_period(config, period, data_root)
            _seal(root)
            score = evaluate_observable_model(model, frame, seeds=config[f"{period['role']}_seeds"],
                                               simulation_rows=config["simulation_rows"])
            generated = generate_observables(model, config["simulation_rows"],
                                             config[f"{period['role']}_seeds"][0])
            diagnostics = diagnose_period(development, frame, interval_us=config["interval_us"],
                                          original_status=score["status"], generated=generated,
                                          block_rows=config["block_rows"],
                                          repetitions=config["bootstrap_repetitions"],
                                          seed=config["diagnostic_seed"])
            _write(root, f"diagnostics-{period['id']}.json", diagnostics)
            (root / f"diagnostics-{period['id']}.md").write_text(report_markdown(diagnostics))
            record = {"period_id": period["id"], "role": period["role"],
                      "inspection_before_run": period["inspection"], **score}
        except (ValueError, RuntimeError, OSError) as exc:
            record = {"period_id": period["id"], "role": period["role"], "status": "INVALID",
                      "inspection_before_run": period["inspection"], "error": f"{type(exc).__name__}: {exc}"}
        _write(root, f"score-{period['id']}.json", record)
        scores.append(record)
        _seal(root)
    if source_manifest() != provenance["source_files"]:
        raise ValueError("implementation changed after registration; attempt retained")
    failed = any(score["status"] in {"FAIL", "INVALID"} for score in scores)
    fresh_ids = {p.get("independent_period_id") for p in config["periods"]
                 if p["role"] == "external" and p["inspection"] == "fresh"}
    result = {"schema_version": 1, "kind": config["kind"], "plan_sha256": _digest(plan),
              "status": "FAILED" if failed else "INCONCLUSIVE", "periods": scores,
              "model_sha256": model["model_sha256"], "model_unchanged": True,
              "fresh_independent_external_periods": len(fresh_ids),
              "real_market_generalization": "NOT_ESTABLISHED",
              "reason": "Registered observable diagnostics are bounded evidence, not broad market validation."}
    _write(root, "consumption-registry.json", {"consumed_periods": consumption})
    _write(root, "result.json", result)
    _write(root, "manifest.json", {"algorithm": "sha256", "files": {
        p.name: sha256_file(p) for p in sorted(root.iterdir()) if p.is_file()}})
    return result


def verify_study(out: str | Path) -> dict:
    root = Path(out).resolve()
    issues = []
    try:
        manifest = json.loads((root / "manifest.json").read_text())
        expected = manifest["files"]
        actual = {p.name for p in root.iterdir() if p.is_file() and p.name != "manifest.json"}
        if manifest["algorithm"] != "sha256" or actual != set(expected):
            raise ValueError("artifact file set or algorithm changed")
        for name, digest in expected.items():
            path = root / name
            if (path.name != name or path.is_symlink() or not path.is_file()
                    or not path.resolve().is_relative_to(root) or sha256_file(path) != digest):
                raise ValueError(f"unsafe, missing or modified artifact: {name}")
        _seal(root)
        model = json.loads((root / "model.json").read_text())
        _check_model(model)
        plan = json.loads((root / "plan.json").read_text())
        validate_plan(plan["config"])
        result = json.loads((root / "result.json").read_text())
        required = {*CORE, "provenance.json", "selection-seal.json", "consumption-registry.json", "result.json"}
        for period in result["periods"]:
            name = period["period_id"]
            required.update({f"access-{name}.json", f"score-{name}.json"})
            if period["status"] != "INVALID":
                required.update({f"diagnostics-{name}.json", f"diagnostics-{name}.md"})
        if not required <= set(expected):
            raise ValueError("required scientific provenance or diagnostic artifact missing")
        if result["plan_sha256"] != _digest(plan) or result["model_sha256"] != model["model_sha256"]:
            raise ValueError("result does not match plan/model")
        expected_ids = {p["id"] for p in plan["config"]["periods"] if p["role"] in {"internal", "external"}}
        if {p["period_id"] for p in result["periods"]} != expected_ids:
            raise ValueError("result does not cover registered periods")
        for record in result["periods"]:
            saved = json.loads((root / f"score-{record['period_id']}.json").read_text())
            if saved != record or (record["status"] != "INVALID"
                                   and record["model_sha256"] != model["model_sha256"]):
                raise ValueError("score/result/model mismatch")
    except (ValueError, KeyError, OSError, TypeError) as exc:
        issues.append(str(exc))
    return {"valid": not issues, "issues": issues}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v04-multiperiod-smoke.json")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    result = verify_study(args.out) if args.verify else run_study(
        json.loads(Path(args.config).read_text()), args.out, data_root=args.data_root)
    print(json.dumps(result if args.verify else {k: v for k, v in result.items() if k != "periods"}))
    if args.verify and not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
