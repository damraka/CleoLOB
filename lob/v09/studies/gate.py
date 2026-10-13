"""realism-gate-v09 (M55): capability-specific realism decisions and the global gate from a sealed benchmark run.

Reads ``--input benchmark=RUN`` (realism-benchmark-v09). Per instrument and generator it reports the capability
decisions (ACCEPTABLE / PARTIALLY_ACCEPTABLE / REJECTED) and per study class the global gate (ACCEPTABLE_MODEL_SET /
PARTIAL_MODEL_SET / NO_ACCEPTABLE_MODEL). The overall gate per study class is the worst over instruments (fail closed).
Model cards (M56 of the reporting layer) are generated from this run.
"""
from __future__ import annotations

from ..realism import benchmark as RB
from . import common

ORDER = ("NO_ACCEPTABLE_MODEL", "PARTIAL_MODEL_SET", "ACCEPTABLE_MODEL_SET")


def run(args) -> dict:
    bench = common.read_input(args, "benchmark")
    out = common.start(args)
    per_stream = {s: {"decisions": {g: v["decisions"] for g, v in r["generators"].items()}, "global": r["global"]}
                  for s, r in bench["streams"].items()}
    overall = {cls: min((r["global"][cls] for r in bench["streams"].values()), key=ORDER.index)
               for cls in RB.STUDY_CLASSES}
    usable = {s: {cls: sorted(g for g, d in r["decisions"].items()
                              if all(d[c] in ("ACCEPTABLE", "PARTIALLY_ACCEPTABLE") for c in RB.STUDY_CLASSES[cls]))
                  for cls in RB.STUDY_CLASSES} for s, r in per_stream.items()}
    result = {"per_stream": per_stream, "global": overall, "usable_generators": usable,
              "rule": "fail closed: a generator REJECTED for a capability never supports a claim that needs it"}
    return common.finish(args, out, analysis="realism-gate-v09", config={"input": "benchmark"}, result=result)
