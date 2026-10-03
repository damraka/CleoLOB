"""Compact public v0.7 evidence bundle, reproducibility capsule and reproduction tiers (workstreams 70, 71).

The bundle copies only redistributable material: v0.7 registration documents (protocol, ledger, registries,
hypotheses, families, budgets, taxonomy, claim and scenario registries, power, holdout and transfer designs),
the portable files of every sealed non-pilot run (config, result, binding, provenance), the generated result
registry, hypothesis table, claim graph, figure sources and figures, the requirement coverage table, and the
public replication subset's expected outputs. Files derived from restricted provider data (sketches, windows,
episode tables, discriminator scores, model checkpoints) are excluded and listed with their SHA-256.
A guard refuses absolute local paths and credential-like text. ``verify`` checks byte integrity (tier 1)
and run bindings inside the bundle; it is never independent scientific replication.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from ...artifacts import seal_artifacts, verify_artifacts
from ...experiments.registry import PROJECT_ROOT, sha256_file
from ...v06.export import check_text

PUBLIC_RUN_FILES = ("config.json", "result.json", "binding.json", "provenance.json")
REGISTRATION = tuple(f"configs/v07/{n}" for n in (
    "protocol.json", "consumption-ledger.jsonl", "dataset-registry.json", "hypotheses.json", "statistical-families.json",
    "compute-budget.json", "result-taxonomy.json", "claim-registry.json", "scenario-taxonomy.json", "power-design.json",
    "holdout-design.json", "transfer-design.json", "benchmark-tasks.json")) + ("configs/v07/templates/study-template.json",)
TIERS = {
    "TIER_1": "byte integrity: checksums of every bundle file (cleo verify-artifact / verify-v07)",
    "TIER_2": "artifact regeneration: tables, figures, registry and claim graph regenerate from the sealed run files "
              "(cleo report-v07)",
    "TIER_3": "public-data reproduction: the public replication subset reruns from code and synthetic fixtures only "
              "(cleo benchmark-v07 public) and must match the stored digest",
    "TIER_4": "full reproduction: requires the Tardis.dev sample files (provider terms; not redistributed) and the "
              "self-recorded Bitstamp captures; rerun every registered study from the protocol",
}
MEANING = "TIER 1 byte integrity plus binding consistency; not independent scientific replication"


def export(out: str | Path, *, root: Path = PROJECT_ROOT, generated: dict[str, Path] | None = None) -> dict:
    out = Path(out)
    if out.exists():
        raise FileExistsError("public bundle exists; export to a new directory")
    out.mkdir(parents=True)
    included, excluded = [], []
    for name in REGISTRATION:
        source = root / name
        if source.is_file():
            target = out / "registration" / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            check_text(name, source.read_text(encoding="utf-8"))
            shutil.copyfile(source, target)
            included.append(target.relative_to(out).as_posix())
    base = root / "results/v07"
    for binding in sorted(base.rglob("binding.json")):
        run = binding.parent
        relative = run.relative_to(base).as_posix()
        if relative.startswith("pilot/"):
            continue
        for path in sorted(p for p in run.rglob("*") if p.is_file()):
            name = path.relative_to(run).as_posix()
            if name in PUBLIC_RUN_FILES:
                check_text(f"{relative}/{name}", path.read_text(encoding="utf-8"))
                target = out / "runs" / relative / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                included.append(target.relative_to(out).as_posix())
            elif name != "checksums.json":
                excluded.append({"path": f"results/v07/{relative}/{name}", "sha256": sha256_file(path),
                                 "bytes": path.stat().st_size,
                                 "reason": "derived from restricted provider data, model checkpoint or large output"})
    for name, source in (generated or {}).items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
            for p in target.rglob("*"):
                if p.is_file() and p.suffix in {".json", ".csv", ".md", ".svg", ".html", ".txt"}:
                    check_text(p.name, p.read_text(encoding="utf-8"))
                    included.append(p.relative_to(out).as_posix())
        else:
            check_text(name, source.read_text(encoding="utf-8"))
            shutil.copyfile(source, target)
            included.append(name)
    manifest = {"schema": "cleolob-v07-capsule-1", "included": sorted(included), "excluded": excluded, "tiers": TIERS,
                "meaning": MEANING}
    (out / "export.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    seal_artifacts(out)
    return manifest


def verify(bundle: str | Path) -> dict:
    bundle = Path(bundle)
    issues = list(verify_artifacts(bundle)["issues"])
    for path in bundle.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".md", ".svg", ".html", ".txt"}:
            try:
                check_text(path.name, path.read_text(encoding="utf-8"))
            except ValueError as exc:
                issues.append(str(exc))
    for binding in bundle.glob("runs/**/binding.json"):
        b = json.loads(binding.read_text(encoding="utf-8"))
        result = json.loads((binding.parent / "result.json").read_text(encoding="utf-8"))
        from ...preregistration import document_sha256
        if document_sha256(result) != b["result_sha256"]:
            issues.append(f"{binding.parent.name}: result does not match its binding")
    return {"valid": not issues, "issues": issues, "meaning": MEANING}
