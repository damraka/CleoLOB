"""Verify and export permitted v0.4 summaries; never copy raw historical inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from lob.artifacts import seal_artifacts, verify_artifacts
from lob.evidence import verify_evidence
from lob.experiments.registry import sha256_file, write_json
from lob.multiperiod import verify_study
from lob.policy_study import verify as verify_policy


def _verify_flat_manifest(source: Path) -> dict:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest["files"]
    actual = {p.name for p in source.iterdir() if p.is_file() and p.name != "manifest.json"}
    issues = []
    if manifest.get("algorithm") != "sha256" or set(expected) != actual:
        issues.append("diagnostic file set changed")
    for name, digest in expected.items():
        target = source / name
        if (target.name != name or target.is_symlink() or not target.is_file()
                or not target.resolve().is_relative_to(source.resolve()) or sha256_file(target) != digest):
            issues.append("unsafe or modified diagnostic artifact")
    return {"valid": not issues, "issues": issues}


def export(args) -> dict:
    sources = {"policy": (args.policy, verify_policy), "multiperiod": (args.multiperiod, verify_study),
               "mbo": (args.mbo, verify_artifacts), "scaling": (args.scaling, verify_evidence),
               "calls": (args.calls, verify_artifacts), "smoke": (args.smoke, verify_artifacts)}
    diagnostics = getattr(args, "v03_diagnostics", None)
    if diagnostics is not None:
        sources["v03-diagnostics"] = (diagnostics, _verify_flat_manifest)
    for name, (source, verifier) in sources.items():
        verified = verifier(source)
        if not verified["valid"]:
            raise ValueError(f"source {name} failed verification: {verified['issues']}")
    # Verification is not redistribution permission. This exporter is expressly
    # limited to the synthetic/local-performance families named here.
    def read(source, filename):
        return json.loads((source / filename).read_text(encoding="utf-8"))

    if read(args.policy, "preregistration.json").get("dataset", {}).get("kind") != "synthetic":
        raise ValueError("compact export permits only synthetic policy studies")
    if read(args.multiperiod, "plan.json").get("config", {}).get("kind") != "synthetic":
        raise ValueError("historical multi-period models/derived data cannot enter this public export")
    if read(args.mbo, "result.json").get("evidence_kind") != "synthetic":
        raise ValueError("historical MBO export needs separate explicit redistribution permission")
    if any(row.get("kind") != "synthetic" for row in read(args.scaling, "datasets.json")):
        raise ValueError("scaling export permits only synthetic fixtures")
    if read(args.calls, "result.json").get("dataset", {}).get("kind") != "synthetic":
        raise ValueError("call benchmark must use synthetic data")
    if read(args.smoke, "result.json").get("dataset") != "generated synthetic fixtures; no historical date":
        raise ValueError("smoke export must use the bundled synthetic fixture")
    if diagnostics is not None and read(diagnostics, "plan.json").get("kind") != "consumed_v03_synthetic_diagnostics":
        raise ValueError("only synthetic v0.3 failure diagnostics may be exported")
    policy_files = ["preregistration.json", "registration_seal.json", "normalization.json",
                    "normalization_seal.json", "training_summary.json", "evaluation_lock.json",
                    "result.json", "manifest.json"]
    args.out.mkdir(parents=True, exist_ok=False)
    original_hashes = {}
    for name, (source, _) in sources.items():
        destination = args.out / name
        destination.mkdir()
        names = policy_files if name == "policy" else [p.name for p in source.iterdir() if p.is_file()]
        # Smoke data are synthetic but omit copied fixture/trace content to keep export compact.
        if name == "smoke":
            names = ["result.json", "provenance.json", "checksums.json"]
        original_hashes[name] = {}
        for filename in sorted(names):
            path = source / filename
            if path.stat().st_size > 3_000_000:
                raise ValueError(f"compact payload exceeds 3 MB: {name}/{filename}")
            shutil.copyfile(path, destination / filename)
            original_hashes[name][filename] = sha256_file(path)
    models = []
    for path in sorted((args.policy / "models").glob("*.json")):
        if path.name.endswith((".training.json", ".attempt.json", ".failure.json")):
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        models.append({"file": path.name, "sha256": sha256_file(path), "metadata": metadata})
    write_json(args.out / "policy/model-metadata.json", models)
    write_json(args.out / "export.json", {
        "schema": "cleolob-v04-compact-v1", "source_artifact_hashes": original_hashes,
        "evidence_kind": "synthetic mechanics, synthetic policy/calibration, local performance",
        "excluded": ["model ZIP checkpoints", "full episode/training traces", "source snapshots",
                     "raw historical data", "restricted derived historical data and fitted models"],
        "verification": "Verify this export using its top-level checksums. Nested seals describe original full runs and deliberately include excluded files.",
        "real_historical_mbo_validation": "NOT_AVAILABLE",
        "historical_policy_superiority": "NOT_ESTABLISHED"})
    seal_artifacts(args.out)
    return verify_artifacts(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "multiperiod", "mbo", "scaling", "calls", "smoke", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--v03-diagnostics", type=Path)
    print(json.dumps(export(parser.parse_args())))


if __name__ == "__main__":
    main()
