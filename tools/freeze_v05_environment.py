"""Freeze the final v0.5 execution environment after M6 and before M7 (writes once).

Binds the source commit and implementation hashes of every semantic layer the
policy study and transfer study depend on, the calibrated-regime specification
selected in M3, the dataset registry snapshot, the evaluation designs and the
statistical family. The freeze document hash is appended to the ledger as
``environment-freeze``; M7 registration verifies it and every listed hash.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lob import preregistration as pr  # noqa: E402
from lob.config import canonical_json  # noqa: E402
from lob.dataset_registry import build  # noqa: E402

LAYERS = {
    "simulator": ["lob/engine.py", "lob/sim_v2.py", "lob/simulators.py", "lob/historical_sim.py"],
    "capability": ["lob/capabilities.py", "lob/fill_bounds.py", "lob/observations.py"],
    "completion": ["lob/completion.py", "lob/mandate.py", "lob/settlement.py", "lob/risk.py", "lob/accounting.py"],
    "execution": ["lob/rl_env.py", "lob/runner.py", "lob/execution.py", "lob/controls.py", "lob/scenarios.py"],
    "calibration": ["lob/observables_v2.py", "lob/calibration_v2.py", "lob/calibration_v2_study.py"],
    "statistics": ["lob/stats.py", "lob/core_study.py", "lob/policy_study_v05.py", "lob/policy_transfer.py"],
    "protocol": ["lob/preregistration.py", "lob/v05_evidence.py", "lob/dataset_registry.py"],
}
DESIGNS = ["configs/v05/protocol.json", "configs/v05/policy-study.json", "configs/v05/m3-design.json",
           "configs/v05/m4-design.json", "configs/v05/m6-design.json", "configs/v05/m8-design.json"]


def main() -> None:
    target = ROOT / "configs/v05/environment-freeze.json"
    if target.exists():
        raise SystemExit("environment already frozen; the freeze is write-once")
    status = subprocess.check_output(["git", "status", "--porcelain", "--", "lob", "configs"], cwd=ROOT, text=True)
    if status.strip():
        raise SystemExit(f"commit implementation and configs before freezing:\n{status}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    selection = json.loads((ROOT / "results/v05/m3/select/result.json").read_text(encoding="utf-8"))
    registry = build(ROOT)
    freeze = {
        "schema": "cleolob-v05-environment-freeze-1", "source_commit": commit,
        "implementation_sha256": {name: pr.text_sha256(ROOT / name) for files in LAYERS.values() for name in files},
        "layers": LAYERS,
        "calibrated_regime": {"family": selection["selected"]["family"], "config": selection["selected"]["config"],
                              "extensions": selection["selected"]["extensions"],
                              "source": "results/v05/m3/select (sealed in ledger as m3-selection)",
                              "select_result_sha256": pr.file_sha256(ROOT / "results/v05/m3/select/result.json"),
                              "external_status": "M3 gate FAILED on every holdout; synthetic calibrated regime only"},
        "design_sha256": {name: pr.text_sha256(ROOT / name) for name in DESIGNS},
        "dataset_registry_sha256": pr.document_sha256(registry),
        "statistical_family": {"M7": 128, "correction": "bonferroni"},
        "semantics": {"completion": "within-horizon completion (E5) separate from settlement; residual valuation never a fill",
                      "capability": "aggregate L2 never yields identity, FIFO or hidden size; historical replay uses bounded fills only",
                      "simulator": "v0.4 engine; calibrated regime = M3 selected ExtendedSimulator spec"}}
    target.write_text(canonical_json(freeze) + "\n", encoding="utf-8", newline="\n")
    protocol = pr.load_protocol(ROOT / "configs/v05/protocol.json")
    entry = pr.append_event(ROOT / "configs/v05/consumption-ledger.jsonl", "seal_design", {
        "analysis": "environment-freeze", "design_sha256": pr.document_sha256(freeze), "reads": [],
        "source_commit": commit, "note": "Final execution environment frozen after M6 and before M7."}, protocol=protocol)
    print(json.dumps({"freeze": str(target.relative_to(ROOT)), "ledger_index": entry["index"], "commit": commit}, indent=2))


if __name__ == "__main__":
    main()
