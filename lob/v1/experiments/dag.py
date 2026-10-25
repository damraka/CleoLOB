"""Experiment DAG planner, runner and resume (M150).

A study plan is a DAG of named steps; each step names a command function, its arguments, the steps whose run
directories it consumes, and its output run directory. ``plan`` validates the DAG (unique names, known
dependencies, acyclic) and returns a deterministic topological order; ``status`` reports which steps are complete
(their run directory verifies), blocked (a dependency is missing or not COMPLETED) or ready; ``execute`` runs ready
steps in order, one at a time (the resource contract: one heavy study family at a time), skipping complete steps,
so re-running a plan resumes it. A step's failure stops the plan and is reported; it is never skipped silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from pathlib import Path

from ..evidence import runs as R


@dataclass(frozen=True)
class Step:
    name: str
    fn: str                              # "module:function", called as fn(out=Path, inputs={dep: Path}, **args)
    out: str                             # run directory relative to the root
    depends: tuple[str, ...] = ()
    args: dict = field(default_factory=dict)


class PlanError(ValueError):
    pass


def plan(steps: list[Step]) -> list[Step]:
    names = [s.name for s in steps]
    if len(set(names)) != len(names):
        raise PlanError("duplicate step names")
    by = {s.name: s for s in steps}
    for s in steps:
        for d in s.depends:
            if d not in by:
                raise PlanError(f"{s.name} depends on unknown step {d}")
    order, state = [], {}

    def visit(n: str) -> None:
        if state.get(n) == 1:
            raise PlanError(f"cycle through {n}")
        if state.get(n) == 2:
            return
        state[n] = 1
        for d in sorted(by[n].depends):
            visit(d)
        state[n] = 2
        order.append(by[n])

    for n in sorted(names):
        visit(n)
    return order


def step_status(root: Path, s: Step) -> str:
    out = root / s.out
    if not (out / "checksums.json").exists():
        return "MISSING"
    v = R.verify_run(out)
    if not v["valid"]:
        return "CORRUPT"
    return v["status"]


def status(root: Path, steps: list[Step]) -> dict:
    order = plan(steps)
    st = {s.name: step_status(root, s) for s in order}
    out = {}
    for s in order:
        if st[s.name] == "COMPLETED":
            out[s.name] = "COMPLETE"
        elif any(st[d] != "COMPLETED" for d in s.depends):
            out[s.name] = "BLOCKED"
        elif st[s.name] == "MISSING":
            out[s.name] = "READY"
        else:
            out[s.name] = st[s.name]            # INVALID / CORRUPT / RESOURCE_EXHAUSTED: needs a new attempt
    return {"order": [s.name for s in order], "steps": out}


def execute(root: Path, steps: list[Step], *, stop_after: int | None = None) -> dict:
    ran = []
    for s in plan(steps):
        if stop_after is not None and len(ran) >= stop_after:
            break
        cur = step_status(root, s)
        if cur == "COMPLETED":
            continue
        if cur != "MISSING":
            return {"status": "STOPPED", "step": s.name, "reason": f"existing run is {cur}; register a new attempt",
                    "ran": ran}
        by = {x.name: x for x in steps}
        if any(step_status(root, by[d]) != "COMPLETED" for d in s.depends):
            return {"status": "BLOCKED", "step": s.name, "ran": ran}
        mod, _, fname = s.fn.partition(":")
        fn = getattr(importlib.import_module(mod), fname)
        inputs = {d: root / by[d].out for d in s.depends}
        fn(out=root / s.out, inputs=inputs, **s.args)
        ran.append(s.name)
        if step_status(root, s) != "COMPLETED":
            return {"status": "FAILED", "step": s.name, "ran": ran}
    return {"status": "COMPLETED", "ran": ran}


__all__ = ["Step", "PlanError", "plan", "status", "execute", "step_status"]
