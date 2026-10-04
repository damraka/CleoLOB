"""Deterministic parallel execution with checkpoints and resume (workstream 54).

``run(tasks, fn, workers=..., checkpoint=dir)`` evaluates ``fn(task)`` for every
task; results are returned in task order regardless of completion order, so a
single-process run and a parallel run are identical whenever ``fn`` is a pure
function of its task (every v0.7 simulation task carries its own seed). With a
checkpoint directory each finished result is written atomically as
``<index>.json`` (canonical JSON); a rerun skips finished indices, so an
interrupted study resumes without recomputation. Resource use is bounded by the
worker count and chunk size.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from ...config import canonical_json


def task_key(task: Any) -> str:
    return hashlib.sha256(canonical_json(task).encode()).hexdigest()[:16]


def _store(directory: Path, index: int, task: Any, result: Any) -> None:
    target = directory / f"{index:06d}.json"
    temporary = target.with_suffix(".part")
    temporary.write_text(canonical_json({"task_key": task_key(task), "result": result}) + "\n", encoding="utf-8",
                         newline="\n")
    temporary.replace(target)


def _load(directory: Path, index: int, task: Any):
    path = directory / f"{index:06d}.json"
    if not path.is_file():
        return None
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored["task_key"] != task_key(task):
        raise ValueError(f"checkpoint {path.name} belongs to a different task; refusing to mix runs")
    return stored


def run(tasks: list, fn: Callable, *, workers: int = 1, checkpoint: str | Path | None = None) -> dict:
    directory = Path(checkpoint) if checkpoint else None
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
    results: list = [None] * len(tasks)
    pending = []
    resumed = 0
    for i, task in enumerate(tasks):
        stored = _load(directory, i, task) if directory else None
        if stored is not None:
            results[i] = stored["result"]
            resumed += 1
        else:
            pending.append(i)
    if workers <= 1:
        for i in pending:
            results[i] = json.loads(canonical_json(fn(tasks[i])))
            if directory:
                _store(directory, i, tasks[i], results[i])
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fn, tasks[i]): i for i in pending}
            for future in as_completed(futures):
                i = futures[future]
                results[i] = json.loads(canonical_json(future.result()))
                if directory:
                    _store(directory, i, tasks[i], results[i])
    return {"results": results, "computed": len(pending), "resumed": resumed, "workers": workers,
            "digest": hashlib.sha256(canonical_json(results).encode()).hexdigest()}
