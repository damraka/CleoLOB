"""Shared machinery of the v1 studies: ledgered loading, pool/stream selection, date clusters and run output.

A ``Context`` binds a study run to the workspace, the run's namespace and (for guarded namespaces) its sealed
design. ``ctx.load(dataset)`` is the only way a study obtains data: it resolves the dataset's registered role,
checks that the role may be read in the namespace the dataset is being read for (``calibrate`` data in
``calibrate``, ``confirm`` data in ``confirm`` ...), records the ledgered access and returns the memory-mapped day
cache. Every dataset a run reads is remembered with its namespace and enters the run binding.

Unit values are keyed by dataset id; ``cluster_values`` averages the streams of each calendar date into one cluster
value (the registered independent unit) and ``decide`` applies the registered decision rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np

from ...data import acquire as A, daycache as DC, store as S
from ...evidence import runs as R
from ...protocol import document as D, seeds as SD
from ... import statistics as ST

ROLE_NAMESPACE = {"explore": "explore", "calibrate": "calibrate", "confirm": "confirm", "replicate": "replicate",
                  "vault": "vault", "development": "dev", "consumed": "explore"}


def registry(root: Path) -> dict:
    return json.loads((Path(root) / "configs/v1/dataset-registry.json").read_text(encoding="utf-8"))


def candidates(root: Path) -> dict[str, dict]:
    doc = json.loads((Path(root) / "configs/v1/hypotheses-candidates.json").read_text(encoding="utf-8"))
    return {h["id"]: h for h in doc["hypotheses"]}


def date_of(dataset: str) -> str:
    return dataset[-10:]


def stream_of(dataset: str) -> str:
    parts = dataset.split("-")
    return f"{parts[0]}-{parts[1]}:{parts[2].upper()}"


@dataclass
class Context:
    root: Path
    namespace: str
    design: str | None = None
    ws: D.Workspace | None = None
    read: dict = field(default_factory=dict)            # dataset -> namespace
    started: float = field(default_factory=time.time)
    _reg: dict | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.ws = self.ws or D.Workspace(self.root)

    @property
    def reg(self) -> dict:
        if self._reg is None:
            self._reg = {d["id"]: d for d in registry(self.root)["datasets"]}
        return self._reg

    def role(self, dataset: str) -> str:
        if dataset in self.reg:
            return self.reg[dataset]["role"]
        if dataset in self.ws.state().consumed:
            return "consumed"
        raise KeyError(f"{dataset} is neither registered nor consumed")

    def spec(self, dataset: str) -> A.DatasetSpec:
        if dataset not in self.reg:            # a consumed (pre-v1) dataset: engineering dry runs only
            conn, seg, sym = dataset.split("-")[:3]
            return A.DatasetSpec(dataset, conn, seg, sym.upper(), date_of(dataset), "consumed", "consumed", 0,
                                 "primary")
        d = self.reg[dataset]
        return A.DatasetSpec(d["id"], d["connector"], d["segment"], d["symbol"], d["date"], d["role"], d["pool"],
                             d["slot"], d["slot_kind"])

    def prepare(self, datasets: list[str], *, use: str, workers: int = 2) -> dict[str, dict]:
        """Ledger every access first (in this process), then build the missing caches in parallel workers.

        Returns {dataset: {"status", "cache"}}; QUALITY_REJECTED datasets are reported, never silently used.
        """
        from concurrent.futures import ProcessPoolExecutor
        from ...data.connectors import base as CB
        raw = S.RawStore(self.root / "data/v1/raw")
        state = self.ws.state()
        jobs, out = [], {}
        for ds in datasets:
            role = self.role(ds)
            ns = ROLE_NAMESPACE[role]
            design = self.design if ns in ("confirm", "replicate", "vault") else None
            if ds not in state.acquired:
                out[ds] = {"status": "NOT_AVAILABLE", "cache": None}
                continue
            D.record_access(self.ws, ds, namespace=ns, stage="parsed", use=use, design=design,
                            reason=f"{use}: day cache of {ds}")
            self.read[ds] = ns
            path = DC.cache_path(self.root, ds)
            if path.exists():
                from ...data import columnar
                out[ds] = {"status": columnar.open_frames(path).meta["meta"]["quality"]["status"],
                           "cache": path.as_posix()}
                continue
            sp = self.spec(ds)
            files = {s.name: raw.lookup(s.url).path for s in CB.get(sp.connector).sources(sp.segment, sp.symbol,
                                                                                            sp.date)
                     if raw.lookup(s.url) is not None}
            jobs.append((ds, sp, files, dict(state.acquired[ds])))
        if jobs:
            with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
                futs = {ex.submit(DC.build, ds, sp.connector, sp.segment, sp.symbol, sp.date, files,
                                  root=self.root, raw_sha256=sha): ds for ds, sp, files, sha in jobs}
                for f, ds in futs.items():
                    r = f.result()
                    out[ds] = {"status": r["status"], "cache": r["cache"]}
        return out

    def load(self, dataset: str, *, use: str) -> dict:
        role = self.role(dataset)
        ns = ROLE_NAMESPACE[role]
        design = self.design if ns in ("confirm", "replicate", "vault") else None
        info = DC.prepare(self.ws, self.spec(dataset), S.RawStore(self.root / "data/v1/raw"), namespace=ns,
                          design=design, use=use, root=self.root)
        self.read[dataset] = ns
        if info["status"] != "AVAILABLE":
            raise QualityRejected(dataset, info)
        return DC.load(Path(info["cache"]))

    def seeds(self, study: str, n: int, purpose: str = "") -> list[int]:
        return SD.seeds(self.namespace, study, n, purpose,
                        design=self.design if self.namespace in ("confirm", "replicate", "vault") else None)


class QualityRejected(RuntimeError):
    def __init__(self, dataset: str, info: dict) -> None:
        super().__init__(f"{dataset} is {info['status']}")
        self.dataset, self.info = dataset, info


def pool_datasets(root: Path, pool: str, streams: list[str], *, acquired: set[str] | None = None,
                  max_dates: int | None = None, require_all_streams: bool = False) -> list[str]:
    """Registered slot order of a pool restricted to streams (and to acquired datasets when given)."""
    rows = [d for d in registry(root)["datasets"] if d["pool"] == pool and d["stream"] in streams]
    rows.sort(key=lambda d: (d["slot"], streams.index(d["stream"])))
    if acquired is not None:
        rows = [d for d in rows if d["id"] in acquired]
    by_date: dict[str, list[str]] = {}
    for d in rows:
        by_date.setdefault(d["date"], []).append(d["id"])
    dates = list(by_date)
    if require_all_streams:
        dates = [x for x in dates if len(by_date[x]) == len(streams)]
    if max_dates is not None:
        dates = dates[:max_dates]
    return [ds for x in dates for ds in by_date[x]]


def cluster_values(unit_values: dict[str, float | None]) -> dict[str, float]:
    """Calendar-date cluster means of the finite unit values (streams of one date averaged)."""
    by: dict[str, list[float]] = {}
    for ds, v in unit_values.items():
        if v is not None and np.isfinite(v):
            by.setdefault(date_of(ds), []).append(float(v))
    return {d: float(np.mean(v)) for d, v in sorted(by.items())}


def pilot_record(unit_values: dict[str, float | None], *, threshold: float, compute_hours: float,
                 capabilities_ok: bool = True) -> dict:
    """Cluster-level pilot summary used by the frozen promotion rule."""
    cl = cluster_values(unit_values)
    x = np.asarray(list(cl.values()), float)
    rec = {"clusters": int(len(x)), "units": int(sum(v is not None for v in unit_values.values())),
           "cluster_values": cl, "threshold": threshold, "compute_hours": compute_hours,
           "capabilities_ok": capabilities_ok}
    if len(x) >= 2:
        se = float(x.std(ddof=1) / np.sqrt(len(x)))
        rec.update({"mean": float(x.mean()), "sd_cluster_means": float(x.std(ddof=1)), "se": se,
                    "planning_effect": float(x.mean() - se)})
    else:
        rec.update({"mean": float(x.mean()) if len(x) else None, "sd_cluster_means": None, "se": None,
                    "planning_effect": None})
    return rec


def finalize(ctx: Context, out: Path, *, analysis: str, config: dict, result, upstream: dict | None = None,
             status: str = "COMPLETED", reason: str | None = None) -> dict:
    run = R.new_run(out)
    b = R.bind(ws=ctx.ws, namespace=ctx.namespace, design=ctx.design, datasets=dict(ctx.read),
               upstream=upstream or {}, root=ctx.root)
    info = R.finalize(run, analysis=analysis, config={**config, "namespace": ctx.namespace, "design": ctx.design},
                      result=result, binding=b, status=status, reason=reason, started=ctx.started, root=ctx.root)
    D.record_attempt(ctx.ws, study=analysis, run=Path(out).relative_to(ctx.root).as_posix()
                     if Path(out).is_absolute() else Path(out).as_posix(), outcome=status, namespace=ctx.namespace,
                     detail=reason or "")
    return info


def cluster_test(values: dict[str, float], *, threshold: float, alpha: float = 0.05, seed: int = 0) -> dict:
    keys = list(values)
    x = [values[k] for k in keys]
    ct = ST.cluster_t(x, keys, threshold=threshold, alpha=alpha)
    sf = ST.sign_flip(x, keys, threshold=threshold, seed=seed)
    return {"cluster_t": ct.to_dict(), "sign_flip": sf}


__all__ = ["Context", "QualityRejected", "pool_datasets", "cluster_values", "pilot_record", "finalize",
           "cluster_test", "registry", "candidates", "date_of", "stream_of", "ROLE_NAMESPACE"]
