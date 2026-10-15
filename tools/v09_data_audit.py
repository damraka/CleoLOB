"""Audit confirmatory data acquisition against the sealed designs and the frozen backup order (read-only).

For every sealed ``confirm-*`` design: the registered primary slots it requests, which were prepared, which failed
(ledger ``attempt`` NOT_AVAILABLE), every backup substitution and whether it follows the frozen registered order
(per pool/stream: failed primaries in slot order take the first registered backups that are neither failed nor
requested), the final per-stream counts, and the quality status recorded at access. Fails closed (exit 1) on any
substitution outside the registered order, any prepared dataset without a cache or an access record, or any read of
a confirm / confirm_late / final dataset that no sealed design resolves to.

    python tools/v09_data_audit.py --root . [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def audit(root: Path) -> dict:
    from lob.v09 import confirm as CF, protocol as pr
    from lob.v09.data import daycache
    protocol, state = pr.state_of(root)
    failed = {a["dataset"]: a for a in state.attempts if a.get("outcome") == "NOT_AVAILABLE" and a.get("dataset")}
    entries = [json.loads(line) for line in (root / pr.LEDGER_PATH).read_text(encoding="utf-8").splitlines() if line]
    access = {}
    for e in entries:
        if e["event"] == "access" and e.get("dataset"):
            access.setdefault(e["dataset"], []).append(e)
    issues: list[str] = []
    requested: dict[tuple, set] = {}
    resolved_all: set[str] = set()
    designs = {}
    acquisition: set[str] = set()
    for name in sorted(state.designs):
        if not name.startswith("confirm-"):
            continue
        raw = pr.sealed_design(root, name)
        if name == "confirm-data":                          # acquisition design: registered primaries + backups
            acquisition = set(_ids(raw, protocol, pr))
            continue
        doc, mapping = CF.substitute(raw, protocol, state)
        ids_raw, ids_doc = _ids(raw, protocol, pr), _ids(doc, protocol, pr)
        for x in ids_raw:
            d = pr.dataset_declaration(protocol, x)
            requested.setdefault((d["pool"], f"{d['venue']}:{d['instrument']}"), set()).add(x)
        resolved_all |= set(ids_doc)
        designs[name] = {"requested": len(ids_raw), "resolved": len(ids_doc),
                         "substitutions": {k: v for k, v in mapping.items()}}
    # independent recomputation of the registered backup order per pool/stream
    expected: dict[str, str | None] = {}
    rows = []
    for (pool, stream), ids in sorted(requested.items()):
        primaries, backups = CF.slots(protocol, pool, stream)
        if pool == "explore":
            continue
        req = [p for p in primaries if p in ids]
        extra = sorted(ids - set(primaries))
        if extra:
            issues.append(f"{pool}/{stream}: designs request non-primary slots {extra}")
        bad = [p for p in req if p in failed]
        free = [b for b in backups if b not in failed and b not in ids]
        subs = {}
        for p in bad:                                       # slot order (primaries are already in slot order)
            subs[p] = free.pop(0) if free else None
        expected.update(subs)
        final = [subs.get(p, p) for p in req if subs.get(p, p) is not None]
        prepared = [x for x in final if daycache.cache_path(root, x).is_file()]
        quality = {}
        for x in final:
            q = [e["payload"].get("quality_status") for e in access.get(x, []) if e["payload"].get("quality_status")]
            quality[q[-1] if q else "NO_ACCESS_RECORD"] = quality.get(q[-1] if q else "NO_ACCESS_RECORD", 0) + 1
        for x in final:
            if not daycache.cache_path(root, x).is_file():
                issues.append(f"{x}: resolved for a sealed design but no prepared cache")
            if x not in access:
                issues.append(f"{x}: resolved for a sealed design but no ledger access record")
        rows.append({"pool": pool, "stream": stream, "requested_primary": len(req),
                     "successful_primary": len([p for p in req if p not in failed]), "failed_primary": bad,
                     "failure_reasons": {p: failed[p].get("note") or failed[p].get("reason") for p in bad},
                     "substitutions": subs, "unfilled": [p for p, b in subs.items() if b is None],
                     "final_count": len(final), "prepared": len(prepared), "quality_status": quality})
    for name, d in designs.items():
        for p, b in d["substitutions"].items():
            if expected.get(p, "<absent>") != b:
                issues.append(f"{name}: substitution {p} -> {b} is outside the registered backup order "
                              f"(expected {expected.get(p)})")
            d["substitutions"][p] = {"backup": b, "preregistered_backup": b is None or
                                     b in CF.slots(protocol, *_pool_stream(protocol, pr, p))[1]}
            if b is not None and not d["substitutions"][p]["preregistered_backup"]:
                issues.append(f"{name}: {b} is not a registered backup of {p}'s stream")
    if resolved_all - acquisition:
        issues.append(f"resolved datasets outside the sealed confirm-data design: {sorted(resolved_all - acquisition)}")
    guarded = {"confirm", "confirm_late", "final"}
    stray = sorted(x for x, es in access.items() if pr.dataset_declaration(protocol, x)["pool"] in guarded
                   and x not in resolved_all and x not in failed)
    if stray:
        issues.append(f"reads of guarded datasets no sealed design resolves to: {stray}")
    return {"valid": not issues, "issues": issues, "streams": rows, "designs": designs,
            "failed_attempts": sorted(failed), "guarded_reads": sorted(x for x in access
                                                                        if pr.dataset_declaration(protocol, x)["pool"]
                                                                        in guarded)}


def _pool_stream(protocol, pr, x):
    d = pr.dataset_declaration(protocol, x)
    return d["pool"], f"{d['venue']}:{d['instrument']}"


def _ids(doc, protocol, pr) -> list[str]:
    out: list[str] = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and x not in out:
            try:
                d = pr.dataset_declaration(protocol, x)
            except Exception:                                      # noqa: BLE001 - non-dataset strings
                return
            if d["role"] in pr.GUARDED_ROLES:
                out.append(x)
    walk(doc)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args()
    out = audit(a.root.resolve())
    text = json.dumps(out, indent=1, sort_keys=True)
    if a.json:
        a.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if out["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
