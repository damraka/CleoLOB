"""Verify sealed v0.9 runs without reading outcomes: seal, status, design, namespace, source commit and inputs.

    python tools/v09_runcheck.py --root . RUN [RUN ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check(root: Path, rel: str) -> dict:
    from lob.v09 import artifacts
    d = root / rel
    v = artifacts.verify_run(d, root=root)
    binding = json.loads((d / "binding.json").read_text(encoding="utf-8"))
    lineage = json.loads((d / "lineage.json").read_text(encoding="utf-8"))
    prov = json.loads((d / "provenance.json").read_text(encoding="utf-8"))
    inputs = []
    for i in lineage.get("inputs", []):
        ok = (artifacts.verify_run(root / i["run"], root=root)["valid"]
              and artifacts.result_sha256(root / i["run"]) == i["result_sha256"])
        inputs.append({"key": i["key"], "run": i["run"], "bound": ok})
    from lob.v09 import protocol as pr
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    frozen = pr.state_of(root)[1].confirmatory_sha256
    row = {"run": rel, "valid": v["valid"], "status": artifacts.run_status(d), "analysis": lineage.get("analysis"),
           "design": config.get("design"), "namespace": binding.get("extra", {}).get("namespace"),
           "git_commit": prov.get("git_commit"), "git_dirty": prov.get("git_dirty"),
           "registry_bound": binding.get("confirmatory_sha256") == frozen, "inputs": inputs}
    import re
    name = config.get("design") or re.sub(r"-\d+$", "", Path(rel).name)
    state = pr.state_of(root)[1]
    if row["namespace"] in ("confirm", "final") and name in state.designs:
        from lob.v09 import confirm as CF
        protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
        doc, _ = CF.substitute(pr.sealed_design(root, name), protocol, state)
        guarded = lambda ids: {x for x in ids if _role(protocol, pr, x) in pr.GUARDED_ROLES}  # noqa: E731
        want = guarded(_strings(doc))
        used = guarded(binding.get("datasets") or [])
        row.update(design=name, design_datasets=len(want), bound_datasets=len(used), datasets_match=want == used)
    row["ok"] = (v["valid"] and row["status"] == "COMPLETED" and not row["git_dirty"] and all(i["bound"] for i in inputs)
                 and (row["namespace"] not in ("confirm", "final") or row["registry_bound"])
                 and row.get("datasets_match", True))
    return row


def _role(protocol, pr, x):
    try:
        return pr.dataset_declaration(protocol, x)["role"]
    except Exception:                                              # noqa: BLE001 - non-dataset strings
        return None


def _strings(x) -> list[str]:
    if isinstance(x, dict):
        return [s for v in x.values() for s in _strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in _strings(v)]
    return [x] if isinstance(x, str) else []


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    rows = [check(a.root.resolve(), r) for r in a.runs]
    print(json.dumps(rows, indent=1))
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
