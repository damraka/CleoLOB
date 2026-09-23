"""Verify full v0.3 runs and export bounded evidence without checkpoints or traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from lob.artifacts import seal_artifacts, verify_artifacts
from lob.experiments.registry import sha256_file, write_json
from lob.generalization import verify_study
from lob.policy_study import verify as verify_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for source, verifier in ((args.calibration, verify_study), (args.policy, verify_policy),
                             (args.benchmark, verify_artifacts), (args.smoke, verify_artifacts)):
        checked = verifier(source)
        if not checked["valid"]:
            raise ValueError(f"source integrity failed: {source.name}: {checked['issues']}")
    args.out.mkdir(parents=True, exist_ok=False)
    selections = {
        "calibration": (args.calibration, [p.name for p in args.calibration.iterdir() if p.is_file()]),
        "policy": (args.policy, ["preregistration.json", "registration_seal.json", "training_summary.json",
                                 "evaluation_lock.json", "result.json", "manifest.json"]),
        "benchmark": (args.benchmark, ["profile.json", "result.json", "checksums.json"]),
        "smoke": (args.smoke, ["result.json", "provenance.json", "checksums.json"]),
    }
    originals = {}
    for kind, (source, names) in selections.items():
        dest = args.out / kind
        dest.mkdir()
        originals[kind] = {}
        for name in sorted(names):
            path = source / name
            if path.stat().st_size > 2_000_000:
                raise ValueError(f"compact artifact exceeds 2 MB: {kind}/{name}")
            shutil.copyfile(path, dest / name)
            originals[kind][name] = sha256_file(path)
    # Learning curves are summarized per fit; original trace hashes stay linked.
    metadata = []
    for path in sorted((args.policy / "models").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "episodes" in record:
            record["episode_count"] = len(record.pop("episodes"))
        metadata.append({"artifact": path.name, "sha256": sha256_file(path), "metadata": record})
    write_json(args.out / "policy/model-metadata.json", metadata)
    write_json(args.out / "export.json", {"format": "compact research summaries", "source_artifact_hashes": originals,
        "not_included": ["model ZIP checkpoints", "per-episode traces", "source snapshots", "raw inputs", "CI logs"],
        "verification": "Top-level checksums verify this export. Nested manifests describe original full runs; their excluded files are intentional."})
    seal_artifacts(args.out)
    print(json.dumps(verify_artifacts(args.out)))


if __name__ == "__main__":
    main()
