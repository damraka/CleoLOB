"""Engineering dry runs of the v1 studies on consumed v0.9 days (explore namespace; never evidence).

    python tools/v1_dryrun.py --study tpp --days 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lob.v1.research.studies import common as CM, decisions as DS, replication as RP, tpp  # noqa: E402

DAYS = ("2023-08-19", "2023-08-21", "2023-08-27", "2023-08-29", "2023-09-02", "2023-09-12", "2023-10-08",
        "2023-10-17")


def datasets(days: int) -> list[str]:
    return [f"bybit-linear-{s}-{d}" for d in DAYS[:days] for s in ("btcusdt", "ethusdt")]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--study", action="append", required=True)
    p.add_argument("--days", type=int, default=4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--tag", default="1")
    a = p.parse_args(argv)
    ds = datasets(a.days)
    out = {}
    for study in a.study:
        ctx = CM.Context(a.root, "explore")
        target = a.root / f"results/v1/dryrun/{study}-{a.tag}"
        try:
            if study == "tpp":
                r = tpp.run(ctx, target, held=ds, train=None, scope="pooled", seed=1, workers=a.workers,
                            hypotheses=("H1", "H2", "H21", "H22", "H26"))
            elif study == "fills":
                r = RP.fills(ctx, target, evaluate=ds, train=None, seed=1, workers=a.workers)
            elif study == "survival":
                r = RP.survival(ctx, target, evaluate=ds, train=None, seed=1, workers=a.workers)
            elif study == "continual":
                r = RP.continual(ctx, target, sequence=ds, seed=1, workers=a.workers)
            elif study == "conformal":
                r = RP.conformal(ctx, target, evaluate=ds, train=None, workers=a.workers)
            elif study == "decisions":
                r = DS.run(ctx, target, evaluate=ds, train=None, seed=1, workers=a.workers)
            elif study == "bounds":
                r = RP.bounds(ctx, target, evaluate=ds, conclusions=["C1", "C2", "C3"], workers=a.workers)
            else:
                raise SystemExit(f"unknown study {study}")
            out[study] = {"status": r["status"], "run": r["run"], "clusters": r["result"].get("clusters")}
        except Exception:  # noqa: BLE001 - dry runs report failures
            out[study] = {"status": "FAILED", "error": traceback.format_exc()[-3000:]}
        print(json.dumps({study: out[study]}, indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
