"""Deterministic parallel task execution with checkpoint/resume (M34).

A study is a list of ``Task`` objects (unique ``task_id``, an importable
``module:function`` and JSON keyword arguments that carry every seed). Results
never depend on scheduling: each task is a pure function of its arguments, and
outputs are returned sorted by ``task_id`` whatever the worker count.

Checkpointing: ``manifest.json`` binds the run to its configuration, source and
protocol hashes and to the exact task set; ``results.jsonl`` holds one hashed
line per completed task. Resuming validates the manifest (any changed setting
refuses the resume), skips completed tasks, rejects duplicates, and discards
(and reports) a truncated last line left by an interruption.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable

from ..config import canonical_json

SCHEMA = "cleolob-v08-checkpoint-1"


class ResumeError(RuntimeError):
    """A checkpoint cannot be resumed safely (settings, task set or records changed)."""


@dataclass(frozen=True)
class Task:
    task_id: str
    fn: str
    kwargs: dict = field(default_factory=dict)

    def digest(self) -> str:
        return hashlib.sha256(canonical_json({"id": self.task_id, "fn": self.fn, "kwargs": self.kwargs}).encode()
                              ).hexdigest()


def _resolve(fn: str) -> Callable:
    module, name = fn.split(":")
    return getattr(importlib.import_module(module), name)


def _execute(task: Task) -> tuple[str, Any]:
    return task.task_id, json.loads(canonical_json(_resolve(task.fn)(**task.kwargs)))


def _line(task_id: str, result: Any) -> str:
    body = canonical_json(result)
    return canonical_json({"task_id": task_id, "result": result,
                           "result_sha256": hashlib.sha256(body.encode()).hexdigest()})


def task_set_digest(tasks: list[Task]) -> str:
    return hashlib.sha256(canonical_json(sorted(t.digest() for t in tasks)).encode()).hexdigest()


def read_checkpoint(directory: Path) -> tuple[dict | None, dict[str, Any], list[str]]:
    manifest_path, results_path = directory / "manifest.json", directory / "results.jsonl"
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
            body = canonical_json(record["result"])
            if hashlib.sha256(body.encode()).hexdigest() != record["result_sha256"]:
                raise ResumeError(f"checkpoint line {number} fails its hash")
            if record["task_id"] in done:
                raise ResumeError(f"duplicate checkpoint record for {record['task_id']}")
            done[record["task_id"]] = record["result"]
    return manifest, done, notes


def run_tasks(tasks: list[Task], *, workers: int = 1, checkpoint: Path | None = None, binding: dict | None = None,
              progress: Callable[[int, int], None] | None = None) -> tuple[list[Any], dict]:
    """Run tasks; returns (results sorted by task id, run info)."""
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("task ids must be unique")
    info: dict = {"tasks": len(tasks), "workers": workers, "resumed": 0, "notes": []}
    done: dict[str, Any] = {}
    handle = None
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        checkpoint.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": SCHEMA, "binding": binding or {}, "task_set_sha256": task_set_digest(tasks),
                    "task_count": len(tasks)}
        stored, done, notes = read_checkpoint(checkpoint)
        info["notes"].extend(notes)
        if stored is not None:
            if stored != manifest:
                changed = sorted(k for k in set(stored) | set(manifest) if stored.get(k) != manifest.get(k))
                raise ResumeError(f"checkpoint settings differ from this run ({changed}); refusing to resume")
            unknown = set(done) - set(ids)
            if unknown:
                raise ResumeError(f"checkpoint holds results for unknown tasks {sorted(unknown)[:5]}")
            info["resumed"] = len(done)
            if notes:   # rewrite without the truncated line so appends stay well formed
                (checkpoint / "results.jsonl").write_text("".join(_line(k, v) + "\n" for k, v in done.items()),
                                                          encoding="utf-8", newline="\n")
        else:
            (checkpoint / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8", newline="\n")
        handle = (checkpoint / "results.jsonl").open("a", encoding="utf-8", newline="\n")
    pending = [t for t in tasks if t.task_id not in done]
    results = dict(done)
    try:
        if workers <= 1 or len(pending) <= 1:
            for t in pending:
                task_id, value = _execute(t)
                results[task_id] = value
                if handle:
                    handle.write(_line(task_id, value) + "\n")
                    handle.flush()
                if progress:
                    progress(len(results), len(tasks))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_execute, t) for t in pending]
                for future in as_completed(futures):
                    task_id, value = future.result()
                    results[task_id] = value
                    if handle:
                        handle.write(_line(task_id, value) + "\n")
                        handle.flush()
                    if progress:
                        progress(len(results), len(tasks))
    finally:
        if handle:
            handle.close()
    ordered = [results[i] for i in sorted(ids)]
    info["completed"] = len(results)
    return ordered, info
