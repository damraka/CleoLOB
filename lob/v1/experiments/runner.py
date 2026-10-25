"""Multi-worker deterministic checkpoint/resume runner (M48) with resource guards.

* A ``Task`` names a registered function (``"module:function"``) and JSON arguments; its id is the SHA-256 of both,
  so the same task always has the same id and a changed argument is a different task.
* ``run`` writes ``manifest.json`` (task-set digest, code digest, profile) into the checkpoint directory on first
  start and refuses to resume a checkpoint whose task set differs. Each finished task is written atomically as
  ``done/<id>.json``; a resume executes only the missing tasks. Results are returned ordered by task id, so they do
  not depend on worker count or completion order (tested).
* A lock file (``O_EXCL``, holding the pid) prevents two runners writing the same checkpoint concurrently; a stale
  lock from a dead process is reported, never silently stolen, unless ``steal_stale=True``.
* Before each submission the runner checks free physical memory against ``min_free_gb``; below it, in-flight tasks
  finish, completed results stay checkpointed and the run ends ``RESOURCE_EXHAUSTED`` (resume later).
* Task exceptions are recorded per task (``failed/<id>.json``) and the run ends ``INVALID``; nothing is retried
  silently.
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
import hashlib
import importlib
import json
import os
from pathlib import Path
import traceback

PROFILES = {"smoke": {"workers": 1, "min_free_gb": 0.5}, "small": {"workers": 2, "min_free_gb": 4.0},
            "medium": {"workers": 2, "min_free_gb": 8.0}, "full": {"workers": 2, "min_free_gb": 8.0},
            "replication": {"workers": 2, "min_free_gb": 8.0}}


def free_gb() -> float:
    from ...v09.batch import free_gb as _free
    return _free()


@dataclass(frozen=True)
class Task:
    fn: str
    args: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        body = json.dumps({"fn": self.fn, "args": self.args}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()[:24]


def _call(fn: str, args: dict):
    mod, _, name = fn.partition(":")
    return getattr(importlib.import_module(mod), name)(**args)


def _execute(task: Task) -> tuple[str, bool, object]:
    try:
        return task.id, True, _call(task.fn, task.args)
    except Exception:  # noqa: BLE001 - recorded per task
        return task.id, False, traceback.format_exc()


def task_set_digest(tasks: list[Task]) -> str:
    return hashlib.sha256("\n".join(sorted(t.id for t in tasks)).encode()).hexdigest()


def _write_atomic(path: Path, value) -> None:
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _pid_alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            import ctypes
            h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not h:
                return False
            code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            ctypes.windll.kernel32.CloseHandle(h)
            return code.value == 259           # STILL_ACTIVE
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class RunnerLocked(RuntimeError):
    pass


def _lock(ckpt: Path, steal_stale: bool) -> Path:
    lock = ckpt / "runner.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pid = int(lock.read_text() or 0)
        if _pid_alive(pid):
            raise RunnerLocked(f"{ckpt} is being written by live process {pid}") from None
        if not steal_stale:
            raise RunnerLocked(f"{ckpt} has a stale lock from dead process {pid}; resume with steal_stale") from None
        lock.unlink()
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return lock


def run(tasks: list[Task], checkpoint: Path, *, profile: str = "smoke", workers: int | None = None,
        min_free_gb: float | None = None, steal_stale: bool = False, inline: bool = False) -> dict:
    prof = PROFILES[profile]
    workers = workers or prof["workers"]
    min_free = prof["min_free_gb"] if min_free_gb is None else min_free_gb
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("duplicate tasks")
    ckpt = Path(checkpoint)
    (ckpt / "done").mkdir(parents=True, exist_ok=True)
    (ckpt / "failed").mkdir(exist_ok=True)
    lock = _lock(ckpt, steal_stale)
    try:
        digest = task_set_digest(tasks)
        man = ckpt / "manifest.json"
        if man.exists():
            m = json.loads(man.read_text())
            if m["task_set"] != digest:
                raise ValueError("checkpoint belongs to a different task set; use a new checkpoint directory")
            resumed = True
        else:
            from ..evidence.runs import code_digest
            _write_atomic(man, {"task_set": digest, "tasks": len(tasks), "profile": profile,
                                "code_sha256": code_digest()})
            resumed = False
        done = {p.stem for p in (ckpt / "done").glob("*.json")}
        todo = [t for t in tasks if t.id not in done]
        status, failed = "COMPLETED", []

        def record(tid, ok, value):
            if ok:
                _write_atomic(ckpt / "done" / f"{tid}.json", {"id": tid, "value": value})
            else:
                _write_atomic(ckpt / "failed" / f"{tid}.json", {"id": tid, "error": value})
                failed.append(tid)

        if inline or workers <= 1:
            for t in todo:
                if free_gb() < min_free:
                    status = "RESOURCE_EXHAUSTED"
                    break
                record(*_execute(t))
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                pending, queue = set(), list(todo)
                while queue or pending:
                    while queue and len(pending) < workers:
                        if free_gb() < min_free:
                            status = "RESOURCE_EXHAUSTED"
                            queue.clear()
                            break
                        pending.add(ex.submit(_execute, queue.pop(0)))
                    if not pending:
                        break
                    finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in finished:
                        record(*f.result())
        if failed and status == "COMPLETED":
            status = "INVALID"
        results = {}
        for t in sorted(tasks, key=lambda x: x.id):
            p = ckpt / "done" / f"{t.id}.json"
            if p.exists():
                results[t.id] = json.loads(p.read_text())["value"]
        return {"status": status, "resumed": resumed, "completed": len(results), "tasks": len(tasks),
                "failed": sorted(failed), "results": results, "workers": workers, "profile": profile}
    finally:
        lock.unlink(missing_ok=True)


__all__ = ["Task", "run", "PROFILES", "task_set_digest", "RunnerLocked", "free_gb"]
