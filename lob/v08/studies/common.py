"""Shared plumbing for v0.8 studies: run directories, seeds, designs, worlds, episode tasks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .. import artifacts, protocol as pr
from ..execution import mpc  # noqa: F401  (registers the MPC policies)
from ..execution import policies as P
from ..execution.simulator import Fees, LatencyProfile, Mandate, run_episode
from ..generators.base import MarketGenerator
from ..market.book import BUY
from ..planning import budget as get_budget
from ..worlds import World, WorldSet


def root_of(args) -> Path:
    return Path(args.root)


def start(args) -> Path:
    out = root_of(args) / args.out
    return artifacts.new_run(out)


def design(args, defaults: dict) -> dict:
    """Study sizes: the sealed design for confirm/holdout namespaces, else the budget defaults."""
    if args.namespace in ("confirm", "holdout"):
        if not args.design:
            raise pr.ProtocolError(f"namespace {args.namespace} requires --design")
        sealed = pr.sealed_design(root_of(args), args.design)
        pr.namespace_guard(args.namespace, root=root_of(args), design=args.design)
        return sealed
    b = get_budget(args.budget)
    d = dict(defaults.get(args.budget) or defaults.get("small") or {})
    d.setdefault("budget", b.name)
    return d


def seeds(args, study: str, count: int) -> list[int]:
    return pr.seeds(args.namespace, study, count, root=root_of(args), design=args.design)


def finish(args, out: Path, *, analysis: str, config: dict, result: dict, datasets=(), inputs=(), seed_info=None
           ) -> dict:
    config = {**config, "namespace": args.namespace, "design": args.design, "budget": args.budget}
    artifacts.finalize(out, analysis=analysis, config=config, result=result, root=root_of(args),
                       dataset_ids=datasets, inputs=list(inputs), seeds=seed_info or {})
    return {"status": "SEALED", "analysis": analysis, "run": out.relative_to(root_of(args)).as_posix(),
            "namespace": args.namespace}


def input_run(args, key: str, default: str | None = None) -> Path:
    rel = args.inputs.get(key, default)
    if rel is None:
        raise ValueError(f"missing --input {key}=<run directory>")
    path = root_of(args) / rel
    if not (path / "binding.json").is_file():
        raise ValueError(f"input {key} is not a sealed v0.8 run: {rel}")
    verdict = artifacts.verify_run(path, root=root_of(args))
    if not verdict["valid"]:
        raise ValueError(f"input {key} fails verification: {verdict['issues'][:3]}")
    return path


def read_result(path: Path) -> dict:
    return json.loads((path / "result.json").read_text(encoding="utf-8"))


def mandate(root: Path, **overrides) -> Mandate:
    m = pr.load_protocol(root / pr.PROTOCOL_PATH)["mandate"]
    fees = Fees(**m["fees"])
    base = Mandate(side=BUY if m["side"] == "buy" else -1, qty=int(m["qty_lots"]), horizon_s=m["horizon_s"],
                   decision_dt_s=m["decision_dt_s"], warmup_s=m["warmup_s"], settlement_s=m["settlement_s"],
                   completion=m["completion_primary"], urgency_fraction=m["urgency_fraction"],
                   penalty_bps=m["penalty_bps"], fees=fees)
    return base.replace(**overrides) if overrides else base


def mandate_dict(m: Mandate) -> dict:
    return m.to_dict()


def mandate_from(d: dict) -> Mandate:
    d = dict(d)
    d["fees"] = Fees(**d["fees"])
    return Mandate(**d)


def registered_policies(root: Path) -> dict[str, dict]:
    p = pr.load_protocol(root / pr.PROTOCOL_PATH)["policies"]
    out = {}
    for group in ("baselines", "adaptive", "mpc"):
        out.update(p[group])
    return out


def policy(name: str, params: dict | None = None) -> P.Policy:
    if name.startswith("param:"):
        u, pa, c = (float(x) for x in name.split(":")[1].split(","))
        return P.make("param_schedule", urgency=u, passive=pa, catchup=c)
    return P.make(name, **(params or {}))


def candidate_name(urgency: float, passive: float, catchup: float) -> str:
    return f"param:{urgency:g},{passive:g},{catchup:g}"


def episode_task(world: dict, policy_name: str, params: dict, mandate: dict, seed: int, latency: dict | None = None,
                 context: dict | None = None, extra: dict | None = None) -> dict:
    """Module-level, JSON-in/JSON-out episode (picklable for the process pool)."""
    w = World.from_dict(world)
    lat = LatencyProfile.from_dict(latency) if latency else None
    row = run_episode(w, policy(policy_name, params), mandate_from(mandate), seed, latency=lat, context=context)
    row["policy"] = policy_name
    if extra:
        row.update(extra)
    return row


def load_world_set(path: Path, name: str) -> WorldSet:
    return WorldSet.from_dict(json.loads((path / f"{name}.json").read_text(encoding="utf-8")))


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def generator_from(d: dict) -> MarketGenerator:
    return MarketGenerator.from_dict(d)


def evaluate(args, out: Path, worlds: list, policies: list, mandate_obj, seed_list: list[int], *, tag: str,
             latency: dict | None = None, contexts: dict | None = None, extras: dict | None = None) -> list[dict]:
    """Episodes for every (world, policy, seed) through the checkpointed deterministic runner.

    ``policies``: list of (name, params). Rows come back sorted by task id (scheduling-independent).
    """
    from ..runner import Task, run_tasks
    md = mandate_obj.to_dict()
    tasks = []
    for w in worlds:
        wd = w.to_dict()
        for name, params in policies:
            for s in seed_list:
                ctx = (contexts or {}).get(w.world_id)
                extra = {"world_label": w.label, **((extras or {}).get(w.world_id, {}))}
                tasks.append(Task(f"{tag}|{w.world_id}|{name}|{s:010d}", "lob.v08.studies.common:episode_task",
                                  {"world": wd, "policy_name": name, "params": params or {}, "mandate": md, "seed": s,
                                   "latency": latency, "context": ctx, "extra": extra}))
    ckpt = Path(str(out) + ".ckpt") / tag
    binding = {"study": getattr(args, "command", "?"), "out": args.out, "tag": tag, "namespace": args.namespace,
               "design": args.design}
    rows, info = run_tasks(tasks, workers=args.workers, checkpoint=ckpt, binding=binding)
    return rows


def by(rows: list[dict], *keys: str) -> dict:
    out: dict = {}
    for r in rows:
        out.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    return out


def world_means(rows: list[dict], value: str = "economic_cost_bps") -> dict:
    """(world_id, policy) -> mean over VALID seeds (None if any episode is INVALID: retained, never dropped)."""
    out = {}
    for (w, p), group in by(rows, "world_id", "policy").items():
        if any(r["status"] != "VALID" or r.get(value) is None for r in group):
            out[(w, p)] = None
        else:
            out[(w, p)] = float(np.mean([r[value] for r in group]))
    return out


def invalid_count(rows: list[dict]) -> int:
    return sum(r.get("status") != "VALID" for r in rows)
