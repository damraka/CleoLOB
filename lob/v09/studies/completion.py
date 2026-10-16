"""completion-study-v09 (H22): re-sealed from the upstream run given as ``--input decisions=RUN`` (lineage-bound, no recomputation)."""
from __future__ import annotations

from . import common


def run(args) -> dict:
    upstream = common.read_input(args, "decisions")
    out = common.start(args)
    units = {k: {f: v.get(f) for f in ("h22", "h22_odd", "h22_even", "h22_cost_diff", "windows") if f in v} for k, v in upstream["units"].items()}
    pilot = (upstream.get("pilot") or {}).get("H22")
    return common.finish(args, out, analysis="completion-study-v09", config={"input": "decisions", "hypothesis": "H22"},
                         result={"hypothesis": "H22", "units": units, "pilot": {"H22": pilot} if pilot else None,
                                 "upstream": common.inputs(args)["decisions"]})
