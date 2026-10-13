"""Deterministic parallel task execution with checkpoint v3 and resource contracts (§40, §41 of the brief).

A study is a list of ``Task`` objects: unique ``task_id``, an importable ``module:function``, JSON keyword arguments
that carry every seed, plus the seed and namespace recorded separately for audit. Each task is a pure function of
its arguments; results are returned sorted by ``task_id`` whatever the worker count, so worker count is never
scientifically meaningful (tested with 1, 4 and N workers).

Checkpoint v3: ``manifest.json`` binds the run to the design hash, input digest, code digest and the exact task
set; ``tasks.jsonl`` holds one record per completed task with ``task_id``, ``seed``, ``namespace``,
``design_sha256``, ``input_digest``, ``code_digest``, ``task_spec_digest``, ``result_digest``, ``status`` and the
result. Resuming validates the manifest (any change refuses the resume), skips completed tasks, rejects duplicates
and corrupted records, discards (and reports) a truncated final line, and never touches sealed evidence.

Resource contracts: ``ResourceContract`` registers the RAM ceiling, worker default/maximum, task estimate and
checkpoint interval. Workers above the registered maximum (or the ``CLEOLOB_MAX_WORKERS`` environment cap) are
refused. When the parent's observed resident memory plus the workers' reported peaks exceed the ceiling, the run
stops cleanly after the in-flight tasks, keeps its checkpoint and reports ``RESOURCE_EXHAUSTED``.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from ..config import canonical_json
from ..v08.runner import disable_power_throttling
from . import runtime

SCHEMA = "cleolob-v09-checkpoint-3"


class ResumeError(RuntimeError):
    """A checkpoint cannot be resumed safely (settings, task set or records changed)."""


class ResourceExhausted(RuntimeError):
    """The registered resource ceiling was exceeded; the checkpoint is kept for a registered resume."""


@dataclass(frozen=True)
class ResourceContract:
    ram_ceiling_mb: float = 12_000.0
    workers_default: int = 4
    workers_max: int = 8
    task_estimate: int | None = None
    disk_estimate_mb: float | None = None
    gpu: str = "none"
    checkpoint_every: int = 1

    def to_dict(self) -> dict:
        return {"ram_ceiling_mb": self.ram_ceiling_mb, "workers_default": self.workers_default,
                "workers_max": self.workers_max, "task_estimate": self.task_estimate,
                "disk_estimate_mb": self.disk_estimate_mb, "gpu": self.gpu,
                "checkpoint_every": self.checkpoint_every, "abort": "RESOURCE_EXHAUSTED after in-flight tasks"}

    @classmethod
    def from_dict(cls, d: dict) -> ResourceContract:
        return cls(**{k: d[k] for k in ("ram_ceiling_mb", "workers_default", "workers_max", "task_estimate",
                                        "disk_estimate_mb", "gpu", "checkpoint_every") if k in d})

    def workers(self, requested: int | None) -> int:
        cap = self.workers_max
        env = os.environ.get("CLEOLOB_MAX_WORKERS")
        if env:
            try:
                cap = min(cap, max(1, int(env)))
            except ValueError:
                raise ValueError("CLEOLOB_MAX_WORKERS must be an integer") from None
        n = self.workers_default if requested is None else int(requested)
        if n < 1:
            raise ValueError("workers must be >= 1")
        if n > cap:
            raise ValueError(f"{n} workers exceed the registered maximum {cap}")
        return n


@dataclass(frozen=True)
class Task:
    task_id: str
    fn: str
    kwargs: dict = field(default_factory=dict)
    seed: int | None = None
    namespace: str = "dev"

    def spec_digest(self) -> str:
        return hashlib.sha256(canonical_json({"id": self.task_id, "fn": self.fn, "kwargs": self.kwargs,
                                              "seed": self.seed, "namespace": self.namespace}).encode()).hexdigest()


def _resolve(fn: str) -> Callable:
    module, name = fn.split(":")
    return getattr(importlib.import_module(module), name)


def _execute(task: Task) -> tuple[str, Any, float | None]:
    value = json.loads(canonical_json(_resolve(task.fn)(**task.kwargs)))
    return task.task_id, value, runtime.peak_memory_mb()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def task_set_digest(tasks: list[Task]) -> str:
    return digest(sorted(t.spec_digest() for t in tasks))


def code_digest() -> str:
    from ..v07.evidence.runs import source_manifest
    return digest(source_manifest())


def _record(task: Task, value: Any, context: dict) -> str:
    return canonical_json({"task_id": task.task_id, "seed": task.seed, "namespace": task.namespace,
                           "design_sha256": context["design_sha256"], "input_digest": context["input_digest"],
                           "code_digest": context["code_digest"], "task_spec_digest": task.spec_digest(),
                           "result_digest": digest(value), "status": "COMPLETED", "result": value})


def read_checkpoint(directory: Path, tasks: dict[str, Task] | None = None) -> tuple[dict | None, dict[str, Any], list[str]]:
    manifest_path, results_path = directory / "manifest.json", directory / "tasks.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    done: dict[str, Any] = {}
    notes: list[str] = []
    if results_path.is_file():
        lines = results_path.read_text(encoding="utf-8").split("\n")
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if number >= len(lines) - 1:
                    notes.append(f"discarded truncated final checkpoint line {number}")
                    continue
                raise ResumeError(f"corrupt checkpoint line {number}") from None
            if digest(record["result"]) != record.get("result_digest"):
                raise ResumeError(f"checkpoint line {number} fails its result digest")
            if record["task_id"] in done:
                raise ResumeError(f"duplicate checkpoint record for {record['task_id']}")
            if tasks is not None:
                task = tasks.get(record["task_id"])
                if task is None:
                    raise ResumeError(f"checkpoint holds a record for unknown task {record['task_id']}")
                if record.get("task_spec_digest") != task.spec_digest():
                    raise ResumeError(f"task {record['task_id']} spec changed since it was checkpointed")
            if manifest is not None:
                for key in ("design_sha256", "input_digest", "code_digest"):
                    if record.get(key) != manifest.get(key):
                        raise ResumeError(f"checkpoint line {number}: {key} differs from the manifest")
            done[record["task_id"]] = record["result"]
    return manifest, done, notes


def run_tasks(tasks: list[Task], *, workers: int | None = None, checkpoint: Path | None = None,
              design: Any = None, inputs: Any = None, contract: ResourceContract | None = None,
              progress: Callable[[int, int], None] | None = None) -> tuple[list[Any], dict]:
    """Run tasks; returns (results sorted by task id, run info). Raises ResourceExhausted with the checkpoint kept."""
    contract = contract or ResourceContract()
    n_workers = contract.workers(workers)
    by_id = {t.task_id: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("task ids must be unique")
    context = {"design_sha256": digest(design if design is not None else {}),
               "input_digest": digest(inputs if inputs is not None else {}), "code_digest": code_digest()}
    info: dict = {"tasks": len(tasks), "workers": n_workers, "resumed": 0, "notes": [],
                  "contract": contract.to_dict(), "peak_worker_memory_mb": 0.0}
    done: dict[str, Any] = {}
    handle = None
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        checkpoint.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": SCHEMA, **context, "task_set_sha256": task_set_digest(tasks),
                    "task_count": len(tasks)}
        stored, done, notes = read_checkpoint(checkpoint, by_id)
        info["notes"].extend(notes)
        if stored is not None:
            if stored != manifest:
                changed = sorted(k for k in set(stored) | set(manifest) if stored.get(k) != manifest.get(k))
                raise ResumeError(f"checkpoint settings differ from this run ({changed}); refusing to resume")
            info["resumed"] = len(done)
            if notes:
                (checkpoint / "tasks.jsonl").write_text(
                    "".join(_record(by_id[k], v, context) + "\n" for k, v in done.items()),
                    encoding="utf-8", newline="\n")
        else:
            (checkpoint / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8",
                                                      newline="\n")
        handle = (checkpoint / "tasks.jsonl").open("a", encoding="utf-8", newline="\n")
    pending = [t for t in sorted(tasks, key=lambda t: t.task_id) if t.task_id not in done]
    results = dict(done)

    def store(task_id: str, value: Any, peak: float | None) -> None:
        results[task_id] = value
        if peak:
            info["peak_worker_memory_mb"] = max(info["peak_worker_memory_mb"], peak)
        if handle:
            handle.write(_record(by_id[task_id], value, context) + "\n")
            handle.flush()
        if progress:
            progress(len(results), len(tasks))

    def over_ceiling() -> bool:
        parent = runtime.memory_mb() or 0.0
        return parent + info["peak_worker_memory_mb"] * (n_workers if n_workers > 1 else 0) > contract.ram_ceiling_mb

    exhausted = False
    try:
        if n_workers <= 1 or len(pending) <= 1:
            for t in pending:
                store(*_execute(t))
                if over_ceiling():
                    exhausted = True
                    break
        else:
            with ProcessPoolExecutor(max_workers=n_workers, initializer=disable_power_throttling) as pool:
                queue = list(pending)
                running = {pool.submit(_execute, queue.pop(0)) for _ in range(min(n_workers, len(queue)))}
                while running:
                    finished, running = wait(running, return_when=FIRST_COMPLETED)
                    for future in finished:
                        store(*future.result())
                    if over_ceiling():
                        exhausted = True
                    while queue and not exhausted and len(running) < n_workers:
                        running.add(pool.submit(_execute, queue.pop(0)))
    finally:
        if handle:
            handle.close()
    info["completed"] = len(results)
    if exhausted and len(results) < len(tasks):
        raise ResourceExhausted(f"RESOURCE_EXHAUSTED: {len(results)}/{len(tasks)} tasks completed; the checkpoint is "
                                f"kept for a registered resume (ceiling {contract.ram_ceiling_mb} MB)")
    return [results[i] for i in sorted(by_id)], info
