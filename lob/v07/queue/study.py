"""M3 queue-uncertainty study (development/retrospective; descriptive, no confirmatory hypothesis).

1. Learn the cancellation queue-position distribution on the Bitstamp development
   capture and check it on the Bitstamp validation capture (both consumed v0.5
   data; ASSUMPTION_DEPENDENT on tracked price-time priority).
2. On the ETH development day, place hypothetical passive child orders at the best
   bid at fixed times and compare filled quantity across the five queue models
   (pro-rata and learned cancellation positions). This measures how much queue
   assumptions alone move passive fills; it is not a fill estimate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...experiments.registry import PROJECT_ROOT, sha256_file
from ..adapters.bitstamp import BitstampAdapter
from ..data import access
from ..evidence.runs import finalize, new_run
from ..protocol import core as pr
from ..v06_compat import depth_scale
from . import learned
from .models import MODELS, all_models, level_events, uniform_cdf

CAPTURES = {"bitstamp-btcusd-mbo-dev": "data/v05/bitstamp/dev-900s/capture.jsonl.gz",
            "bitstamp-btcusd-mbo-validation": "data/v05/bitstamp/validation-1800s/capture.jsonl.gz"}
DESIGN = "m3-queue-uncertainty"
SEED = 70601
ORDERS = 400
LIFETIME_S = 60.0
QTY_LOTS = 14.0


def _ledger_mbo(dataset: str, path: Path, stage: str, root: Path) -> None:
    pr.append_event(root / pr.LEDGER_PATH, "access", dataset=dataset, role="mbo_retrospective", design=DESIGN,
                    reason="M3 cancellation queue-position estimation (retrospective, consumed capture)",
                    protocol=pr.load_protocol(root / pr.PROTOCOL_PATH), root=root,
                    payload={"use": "retrospective", "stage": stage,
                             "source_sha256": {path.relative_to(root / "data/v05/bitstamp").as_posix():
                                               sha256_file(path)}})


def run(out: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    out = new_run(root / out if not Path(out).is_absolute() else out)
    positions = {}
    for dataset, rel in CAPTURES.items():
        path = root / rel
        _ledger_mbo(dataset, path, "opened", root)
        extracted = learned.cancellation_positions(BitstampAdapter(path).records())
        _ledger_mbo(dataset, path, "parsed", root)
        positions[dataset] = extracted
    fit_u = positions["bitstamp-btcusd-mbo-dev"]["positions"]
    check_u = positions["bitstamp-btcusd-mbo-validation"]["positions"]
    cdf = learned.EmpiricalCDF(fit_u)
    ks_two_sample = float(np.max(np.abs(np.searchsorted(np.sort(fit_u), np.linspace(0, 1, 201), side="right") / len(fit_u)
                                        - np.searchsorted(np.sort(check_u), np.linspace(0, 1, 201), side="right")
                                        / len(check_u))))
    mbo = {"fit": learned.summary(fit_u, seed=SEED), "check": learned.summary(check_u, seed=SEED + 1),
           "skipped": {k: v["skipped"] for k, v in positions.items()}, "learned_cdf": cdf.to_dict(),
           "fit_vs_check_ks": ks_two_sample,
           "status": "ASSUMPTION_DEPENDENT",
           "assumption": "price-time priority with additions at the back and size increases or reprices re-queued; "
                         "the Bitstamp feed does not establish it (fifo_established=False)"}
    for dataset in CAPTURES:
        access.mark_evaluated(dataset, use="retrospective", design=DESIGN, run=str(out.relative_to(root).as_posix()),
                              root=root)
    # Queue-model sensitivity on the development day (lots; development depth scale from v0.6).
    tape, meta = access.load_tape("deribit-eth-perp-2020-04-01", use="develop", design=DESIGN,
                                  purpose="M3 queue-model sensitivity on development data", root=root)
    tape = tape.rescaled(depth_scale(tape))
    rng = np.random.default_rng(SEED)
    starts = np.sort(rng.uniform(tape.t[0] + 60, tape.t[-1] - LIFETIME_S - 1, ORDERS))
    rows = {f"{m}|{c}": [] for m in MODELS for c in ("uniform", "learned")}
    skipped = 0
    for start in starts:
        i = int(np.searchsorted(tape.t, start))
        if not tape.valid[i]:
            skipped += 1
            continue
        price = float(tape.bp[i, 0])
        join, events = level_events(tape, side=1, price=price, start=float(tape.t[i]), stop=float(tape.t[i]) + LIFETIME_S)
        for name, f in (("uniform", uniform_cdf), ("learned", cdf)):
            result = all_models(events, qty=QTY_LOTS, level_at_join=join, cdf=f)
            for model in MODELS:
                rows[f"{model}|{name}"].append(result[model]["filled"] / QTY_LOTS)
    sensitivity = {key: {"mean_fill_fraction": float(np.mean(v)), "complete_fraction": float(np.mean(np.asarray(v) >= 1 - 1e-9)),
                         "n": len(v)} for key, v in rows.items()}
    spread = sensitivity["optimistic|uniform"]["mean_fill_fraction"] - sensitivity["conservative|uniform"]["mean_fill_fraction"]
    result = {"mbo": mbo, "development_sensitivity": sensitivity, "orders": ORDERS, "skipped_invalid": skipped,
              "lifetime_s": LIFETIME_S, "qty_lots": QTY_LOTS, "bound_width_mean_fill_fraction": spread,
              "dataset_quality": meta["quality"]["status"], "label": "DEVELOPMENT / RETROSPECTIVE, descriptive"}
    access.mark_evaluated("deribit-eth-perp-2020-04-01", use="develop", design=DESIGN,
                          run=str(out.relative_to(root).as_posix()), root=root)
    finalize(out, analysis="m3-queue-uncertainty", dataset_ids=[*CAPTURES, "deribit-eth-perp-2020-04-01"],
             config={"orders": ORDERS, "lifetime_s": LIFETIME_S, "qty_lots": QTY_LOTS, "seed": SEED,
                     "bins": learned.BINS}, result=result, root=root, seeds={"orders": SEED})
    return result
