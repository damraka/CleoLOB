"""safe-improvement-v09 (H27): re-sealed from the upstream run given as ``--input rl=RUN`` (lineage-bound, no recomputation)."""
from __future__ import annotations

from . import common


def run(args) -> dict:
    upstream = common.read_input(args, "rl")
    out = common.start(args)
    units = {k: {f: v.get(f) for f in ("h27", "h27_odd", "h27_even", "harmful", "deployed_ungated", "deployed_gated") if f in v} for k, v in upstream["worlds"].items()}
    pilot = (upstream.get("pilot") or {}).get("H27")
    return common.finish(args, out, analysis="safe-improvement-v09", config={"input": "rl", "hypothesis": "H27"},
                         result={"hypothesis": "H27", "units": units, "pilot": {"H27": pilot} if pilot else None,
                                 "upstream": common.inputs(args)["rl"]})
