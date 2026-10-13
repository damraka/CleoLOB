"""Shared plumbing for v0.9 studies: roots, namespaces, seeds, inputs, units, sealing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...preregistration import ProtocolError
from .. import artifacts, protocol as pr
from ..runner import ResourceContract

BUDGET_SCALE = {"tiny": 0.1, "small": 0.25, "medium": 0.5, "large": 1.0}
USE_OF_NAMESPACE = {"dev": "develop", "explore": "explore", "pilot": "explore", "confirm": "confirm",
                    "final": "final"}


def root_of(args) -> Path:
    return Path(getattr(args, "root", ".")).resolve()


def start(args) -> Path:
    root = root_of(args)
    out = Path(args.out)
    if out.is_absolute():
        raise ValueError("--out must be relative to --root")
    if getattr(args, "namespace", "dev") in pr.GUARDED_NAMESPACES and not getattr(args, "design", None):
        raise ProtocolError(f"namespace {args.namespace!r} requires --design (a sealed design that opened it)")
    if getattr(args, "design", None) and args.namespace in pr.GUARDED_NAMESPACES:
        pr.namespace_guard(args.namespace, root=root, design=args.design)
    return artifacts.new_run(root / out)


def design_document(args) -> dict | None:
    if not getattr(args, "design", None):
        return None
    return pr.sealed_design(root_of(args), args.design)


def seeds(args, study: str, count: int) -> list[int]:
    ns = args.namespace
    return pr.seeds(ns, f"{study}:{args.seed}", count, root=root_of(args) if ns in pr.GUARDED_NAMESPACES else None,
                    design=getattr(args, "design", None))


def contract(args, **overrides) -> ResourceContract:
    protocol = pr.load_protocol(root_of(args) / pr.PROTOCOL_PATH)
    ceil = protocol["resource_ceilings"]
    base = {"ram_ceiling_mb": float(ceil["ram_mb_per_study"]), "workers_default": int(ceil["workers_default"]),
            "workers_max": int(ceil["workers_max"])}
    return ResourceContract(**{**base, **overrides})


def inputs(args) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, run in getattr(args, "input", []) or []:
        if key in out:
            raise ValueError(f"--input {key} given twice")
        path = root_of(args) / run
        if not (path / "binding.json").is_file():
            raise ValueError(f"--input {key}={run} is not a sealed run")
        out[key] = run
    return out


def read_input(args, key: str) -> dict:
    runs = inputs(args)
    if key not in runs:
        raise ValueError(f"missing required --input {key}=RUN")
    report = artifacts.verify_run(root_of(args) / runs[key], root=root_of(args))
    if not report["valid"]:
        raise ValueError(f"--input {key} does not verify: {report['issues'][:3]}")
    return json.loads((root_of(args) / runs[key] / "result.json").read_text(encoding="utf-8"))


def finish(args, out: Path, *, analysis: str, config: dict, result: dict, datasets=(), seed_info=None,
           status: str = "COMPLETED", status_reason: str = "", resources: dict | None = None,
           summary: Any = None, metrics: Any = None) -> dict:
    binding = artifacts.finalize(out, analysis=analysis, config=config, result=json_safe(result), root=root_of(args),
                                 dataset_ids=datasets, inputs=inputs(args), seeds=seed_info,
                                 namespace=getattr(args, "namespace", "dev"), status=status,
                                 status_reason=status_reason, resources=resources, summary=json_safe(summary)
                                 if summary is not None else None, metrics=json_safe(metrics)
                                 if metrics is not None else None)
    return {"valid": True, "status": status, "run": out.relative_to(root_of(args)).as_posix(),
            "analysis": analysis, "result_sha256": binding["result_sha256"]}


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if np.isfinite(x) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def units(args, pool: str, streams: list[str], n: int) -> dict:
    """Registered slot resolution (primaries, failures replaced by ordered backups)."""
    from ..data import store
    return store.resolve_units(root_of(args), pool, streams, n)


def use_for(args) -> str:
    use = USE_OF_NAMESPACE.get(args.namespace)
    if use is None:
        raise ProtocolError(f"namespace {args.namespace} does not read real data")
    return use
