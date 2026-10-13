"""meta-policy-v09 (H25): re-sealed from the upstream run given as ``--input decisions=RUN`` (lineage-bound, no recomputation)."""
from __future__ import annotations

from . import common


def run(args) -> dict:
    upstream = common.read_input(args, "decisions")
    out = common.start(args)
    units = {k: {f: v.get(f) for f in ("h25", "h25_odd", "h25_even", "best_global", "meta_regret", "windows") if f in v} for k, v in upstream["units"].items()}
    pilot = (upstream.get("pilot") or {}).get("H25")
    return common.finish(args, out, analysis="meta-policy-v09", config={"input": "decisions", "hypothesis": "H25"},
                         result={"hypothesis": "H25", "units": units, "pilot": {"H25": pilot} if pilot else None,
                                 "upstream": common.inputs(args)["decisions"]})
