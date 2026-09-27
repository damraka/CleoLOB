"""Book resilience after depleting aggressive events (M4).

Depletion and replenishment are measured on the same tape as every other v0.5
observable. Aggregate data show net depth, so replenishment is net: new
liquidity cannot be separated from cancellations elsewhere on the side.
Recovery times are right-censored at ``CENSOR_S`` and reported with the
recovered share, never as uncensored means.
"""
from __future__ import annotations

import numpy as np

from .observables_v2 import SAMPLE_DT, Tape

CENSOR_S = 60.0
CURVE_S = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)


def depletion_events(tape: Tape, *, min_depletion: float = 0.5) -> list[dict]:
    """Events removing at least ``min_depletion`` of the opposite best-level depth."""
    valid = tape.valid
    index = np.searchsorted(tape.t, tape.trade_t, side="right") - 1
    sides = {1.0: (tape.aq[:, 0], tape.aq.sum(axis=1)), -1.0: (tape.bq[:, 0], tape.bq.sum(axis=1))}
    rows = []
    for k, size, sign, when in zip(index, tape.trade_q, tape.trade_s, tape.trade_t):
        if k < 0 or not valid[k]:
            continue
        l1, top5 = sides[1.0 if sign > 0 else -1.0]
        if l1[k] <= 0 or size < min_depletion * l1[k]:
            continue
        rows.append({"k": int(k), "time": float(when), "sign": float(sign), "size": float(size),
                     "pre_l1": float(l1[k]), "pre_top5": float(top5[k]), "l1": l1, "top5": top5})
    return rows


def recovery(tape: Tape, events: list[dict], *, level: float = 0.9) -> dict:
    valid = tape.valid
    horizon = int(CENSOR_S / SAMPLE_DT)
    times, censored = [], 0
    curves = {f"{h:g}s": [] for h in CURVE_S}
    for event in events:
        k = event["k"]
        stop = min(len(tape.t), k + 1 + horizon)
        window = event["l1"][k + 1:stop]
        hit = np.flatnonzero((window >= level * event["pre_l1"]) & valid[k + 1:stop])
        if len(hit):
            times.append(float(tape.t[k + 1 + hit[0]] - event["time"]))
        else:
            censored += 1
        for h in CURVE_S:
            j = k + int(round(h / SAMPLE_DT))
            if j < len(tape.t) and valid[j] and event["pre_top5"] > 0:
                curves[f"{h:g}s"].append(event["top5"][j] / event["pre_top5"])
    times_arr = np.asarray(times)
    return {"events": len(events), "recovered": len(times), "censored_at_60s": censored,
            "recovered_share": len(times) / len(events) if events else None,
            "recovery_time_quantiles_s": np.quantile(times_arr, [0.25, 0.5, 0.75]).tolist() if len(times) else None,
            "net_replenishment_curve_top5": {k: (float(np.mean(v)) if v else None) for k, v in curves.items()},
            "definition": f"first sample with opposite best-level depth >= {level:.0%} of pre-event depth; "
                          "censored at 60 s; replenishment is net opposite top-5 depth relative to pre-event"}
