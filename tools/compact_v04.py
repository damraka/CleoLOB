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


def export(args) -> dict:
    sources = {"policy": (args.policy, verify_policy), "multiperiod": (args.multiperiod, verify_study),
               "mbo": (args.mbo, verify_artifacts), "scaling": (args.scaling, verify_evidence),
               "calls": (args.calls, verify_artifacts), "smoke": (args.smoke, verify_artifacts)}
    for name, (source, verifier) in sources.items():
        verified = verifier(source)
        if not verified["valid"]:
            raise ValueError(f"source {name} failed verification: {verified['issues']}")
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
    print(json.dumps(export(parser.parse_args())))


if __name__ == "__main__":
    main()
