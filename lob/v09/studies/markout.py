"""markout-study-v09 (M69, M70; H9): post-fill markouts of hypothetical passive fills under bounded replay.

H9 is the registered execution conclusion C5 (toxicity-proxy-aware passive vs naive passive, small mandate, 5 s
markout of hypothetical passive fills, decided by both queue bounds) with completion non-inferiority. This command
re-seals C5 from a counterfactual-bounds-v09 run (``--input bounds=RUN``) together with the completion difference per
day, so H9 has its own lineage-bound record. Markouts are BOUNDED and HYPOTHETICAL; "toxicity" is a flow-risk proxy,
never an identification of informed traders.
"""
from __future__ import annotations

from . import common


def run(args) -> dict:
    upstream = common.read_input(args, "bounds")
    out = common.start(args)
    c5 = upstream["conclusions"]["C5"]
    result = {"hypothesis": "H9", "conclusion": "C5", "definition": c5["definition"], "units": c5["units"],
              "completion_units": c5.get("completion_units"),
              "pilot": {"H9": c5["pilot"]} if upstream.get("phase") == "pilot" else None,
              "completion_margin": 0.02, "upstream": common.inputs(args)["bounds"],
              "label": "BOUNDED, HYPOTHETICAL; toxicity is a flow-risk proxy"}
    return common.finish(args, out, analysis="markout-study-v09", config={"input": "bounds"}, result=result)
