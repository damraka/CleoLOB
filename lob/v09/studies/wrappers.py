"""Thin lineage-linked studies that re-seal one hypothesis' part of an upstream run.

``ecology-study-v09`` (H12) reads ``--input hybrid=RUN`` (a hybrid-twin-v09 run) and ``model-averaging-v09`` (H5)
reads ``--input tpp=RUN`` (a neural-point-process-v09 run). The upstream run is verified and bound as lineage; nothing
is recomputed, so the numbers are identical to the upstream run by construction.
"""
from __future__ import annotations

from . import common


def reseal(args, *, analysis: str, key: str, hypothesis: str, fields: tuple[str, ...]) -> dict:
    upstream = common.read_input(args, key)
    out = common.start(args)
    units = {k: {f: v.get(f) for f in fields if f in v} for k, v in upstream["units"].items()}
    pilot = upstream.get("pilot") or {}
    result = {"phase": upstream["phase"], "hypothesis": hypothesis, "units": units,
              "pilot": {hypothesis: pilot[hypothesis]} if hypothesis in pilot else None,
              "upstream": common.inputs(args)[key]}
    return common.finish(args, out, analysis=analysis, config={"input": key, "hypothesis": hypothesis},
                         result=result)
