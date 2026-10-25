"""RETROSPECTIVE dependence diagnostic of every promoted v0.9 hypothesis (Issue B of the v1.0 audit, M0).

Reads only the sealed v0.9 public bundle (``examples/studies/v09``) and the sealed v0.9 designs, recovers the
registered unit values of each promoted hypothesis with the unchanged v0.9 extraction code
(``lob.v09.studies.final.unit_values``) and recomputes the decision interval with calendar dates as the
independent unit (``lob.v1.statistics``). Nothing is written into the v0.9 bundle and no v0.9 status changes:
every output is labelled RETROSPECTIVE.

    python tools/v1_retro_dependence.py --out examples/studies/v1/retrospective/dependence.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from lob.v09.studies import final as F
from lob.v1 import statistics as S

LABEL = "RETROSPECTIVE diagnostic of sealed v0.9 evidence; not a preregistered test and not a v0.9 status change"
SEED = 1_000_036
ALPHA = 0.05
TRANSFER_SOURCE = {"H32": "confirm/confirm-bounds-late", "H33": "confirm/confirm-bounds", "H36": "final/vault-bounds"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _date(unit: str) -> str:
    return unit[-10:]


def _stream(unit: str) -> str:
    return unit[:-11]


def _bundle_run(bundle: Path, run: str) -> Path:
    return bundle / "runs" / Path(run).relative_to("results/v09")


def summarize(values: list[float], units: list[str], threshold: float, registered: dict | None, seed: int) -> dict:
    dates = [_date(u) for u in units]
    streams = [_stream(u) for u in units]
    by_stream: dict[str, list[float]] = defaultdict(list)
    for v, s in zip(values, streams):
        by_stream[s].append(v)
    ct = S.cluster_t(values, dates, threshold=threshold, alpha=ALPHA)
    out = {
        "threshold": threshold, "units": len(units), "distinct_dates": len(set(dates)),
        "streams": sorted(set(streams)), "mean": float(np.mean(values)),
        "registered_v09_interval": registered,
        "date_cluster_t": ct.to_dict(),
        "date_sign_flip": S.sign_flip(values, dates, threshold=threshold, seed=seed),
        "date_cluster_bootstrap": S.cluster_bootstrap(values, dates, alpha=ALPHA, seed=seed),
        "dependence": S.design_effect(values, dates),
        "per_stream_mean": {s: float(np.mean(v)) for s, v in sorted(by_stream.items())},
        "min_unit": float(np.min(values)),
    }
    if len(set(streams)) > 1 and len(set(dates)) > 1:
        out["date_x_stream_multiway_t"] = S.multiway_t(values, dates, streams, threshold=threshold).to_dict()
    out["cluster_t_lower_exceeds_threshold"] = bool(ct.ci_low > threshold)
    out["sign_flip_reaches_alpha"] = bool(out["date_sign_flip"]["p_one_sided"] <= ALPHA)
    return out


def analyse(bundle: Path, designs: Path) -> dict:
    final_path = bundle / "runs/final/final-analysis/result.json"
    final = json.loads(final_path.read_text())
    inputs = {str(final_path.relative_to(bundle)).replace("\\", "/"): _sha(final_path)}
    rows: dict[str, dict] = {}
    for i, (h, rec) in enumerate(sorted(final["hypotheses"].items(), key=lambda kv: int(kv[0][1:]))):
        if rec.get("promotion") != "PROMOTED":
            continue
        if rec.get("test") == "transfer":
            path = bundle / "runs" / TRANSFER_SOURCE[h] / "result.json"
            inputs[str(path.relative_to(bundle)).replace("\\", "/")] = _sha(path)
            source = json.loads(path.read_text())
            t = rec["transfer"]
            rows[h] = {"test": "transfer", "status_v09": rec["status"], "conclusions": {}}
            for j, c in enumerate(t["carried"]):
                units = t["target_units"]
                vals = [float(source["conclusions"][c]["units"][u]["lower"]) for u in units]
                iv = t["conclusions"][c]["interval_95"]
                rows[h]["conclusions"][c] = summarize(vals, units, float(t["conclusions"][c]["threshold"]),
                                                      [iv["ci_low"], iv["ci_high"]], SEED + 10 * i + j)
            continue
        run = _bundle_run(bundle, rec["run"])
        design_path = designs / f"{rec['design']}.json"
        inputs[str((run / "result.json").relative_to(bundle)).replace("\\", "/")] = _sha(run / "result.json")
        doc = json.loads(design_path.read_text())
        values = F.unit_values(h, json.loads((run / "result.json").read_text()), doc.get("design", doc))
        units = rec["units_used"]
        vals = [float(values[u]) for u in units]
        level = float(rec.get("level") or 0.0)
        test = rec["test"]
        # the registered decision quantity: superiority -> value vs threshold; noninferiority -> value + margin vs
        # 0; agreement -> value - level vs threshold
        if test == "noninferiority":
            shifted, thr = [v + float(rec["threshold"]) for v in vals], 0.0
        elif test == "agreement":
            shifted, thr = [v - level for v in vals], float(rec["threshold"])
        else:
            shifted, thr = vals, float(rec["threshold"])
        iv = rec["interval_95"]
        rows[h] = {"test": test, "status_v09": rec["status"], "decision_quantity": test,
                   **summarize(shifted, units, thr, [iv["ci_low"], iv["ci_high"]], SEED + 10 * i)}
    return {"label": LABEL, "alpha": ALPHA, "inputs": inputs, "hypotheses": rows,
            "method": {"independent_unit": "calendar date (all streams of a date move together)",
                       "primary": "cluster-robust mean, CR1 SE, Student t on G-1 df",
                       "randomization": "exact cluster sign-flip (floor 2^-G)",
                       "secondary": "percentile cluster bootstrap; two-way date x stream CGM where both vary"}}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bundle", type=Path, default=Path("examples/studies/v09"))
    p.add_argument("--designs", type=Path, default=Path("configs/v09/designs"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    result = analyse(a.bundle, a.designs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for h, r in result["hypotheses"].items():
        for name, row in (r["conclusions"].items() if "conclusions" in r else [("", r)]):
            ct = row["date_cluster_t"]
            print(f"{h:4} {name:3} units={row['units']:2} dates={row['distinct_dates']:2} mean={row['mean']:.4g} "
                  f"thr={row['threshold']:.4g} clusterT=[{ct['ci_low']:.4g},{ct['ci_high']:.4g}] "
                  f"p={ct['p_one_sided']:.3g} signflip_p={row['date_sign_flip']['p_one_sided']:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
