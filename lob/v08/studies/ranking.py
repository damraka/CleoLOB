"""``ranking-transfer-v08``: synthetic predicted rankings versus bounded historical rankings (M32, H13).

The synthetic prediction is the held-out plausible-world mean economic cost of each historical-replay policy from
the sealed controlled execution benchmark (``--input mpc=...``). The historical target comes from bounded replays
(``--input transfer=...``, retrospective cells): pair verdicts under shared assumptions and neutral mean costs.
``retrospective`` results are EXPLORATORY (consumed days). ``fresh`` reports NOT_AVAILABLE when the fresh transfer
acquisition was blocked (``--input fresh=...``).
"""
from __future__ import annotations

from ..transfer import compare
from . import common as C


def run(args) -> dict:
    out = C.start(args)
    if args.phase == "fresh":
        fresh = C.read_result(C.input_run(args, "fresh"))
        result = {"status": fresh["status"], "acquisition": fresh["acquisition"],
                  "reason": "fresh holdouts unavailable (registered probe failed)" if fresh["status"] == "NOT_AVAILABLE"
                  else "data acquired: bounded replays required before H13 can be evaluated"}
        return C.finish(args, out, analysis="ranking_fresh", config={}, result=result,
                        inputs=[args.inputs["fresh"]])
    mpc = C.read_result(C.input_run(args, "mpc"))
    tr = C.read_result(C.input_run(args, "transfer"))
    out_cells = {}
    for mname in tr["mandates"]:
        table = mpc["mandates"].get(mname, {}).get("table", {}).get("policies")
        if not table:
            continue
        source = {p: v["economic_cost_mean_bps"] for p, v in table.items()}
        for cell, verdicts in tr["mandates"][mname]["pair_verdicts"].items():
            target = tr["mandates"][mname]["neutral_costs"][cell]
            names = [p for p in target if p in source]
            out_cells[f"{mname}|{cell}"] = compare(source, target, determinate=verdicts, names=names, k=3)
    result = {"retrospective": out_cells,
              "labels": {"evidence": "RETROSPECTIVE; EXPLORATORY (consumed days cannot support H13)",
                         "source": "plausible-world mean costs (confirm seeds)", "target": "bounded historical replays"}}
    return C.finish(args, out, analysis="ranking_retrospective", config={}, result=C.json_safe(result),
                    inputs=[args.inputs["mpc"], args.inputs["transfer"]])
