"""Compute accounting (workstream 56) and environment capture additions (workstream 72).

``Meter`` records, for one study: wall-clock, process CPU time, peak resident
memory of the main process where the OS exposes it (Windows API via ctypes,
``resource`` on POSIX; otherwise NOT_AVAILABLE), OS, Python, CPU model and
logical CPU count, plus counters the study sets (simulations, simulated seconds,
episodes, training steps, seeds). GPU: NOT_AVAILABLE on this machine (no CUDA
device); the GPU stack is never required. Worker-process CPU is not included
in process CPU time and is reported as wall-clock x workers upper bound.
Runtime benchmarks are local software timings, never exchange or HFT latency.
"""
from __future__ import annotations

import os
import platform
import sys
import time


def peak_memory_mb() -> float | None:
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return counters.PeakWorkingSetSize / 2**20
            return None
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 1024
    except (OSError, AttributeError, ImportError):
        return None


def gpu_status() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            return {"available": True, "device": torch.cuda.get_device_name(0)}
    except ImportError:
        pass
    return {"available": False, "status": "NOT_AVAILABLE", "reason": "no CUDA device; GPU stack optional"}


def environment() -> dict:
    keys = ("CLEOLOB_WORKERS", "PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")
    return {"os": platform.platform(), "python": platform.python_version(), "machine": platform.machine(),
            "logical_cpus": os.cpu_count(), "gpu": gpu_status(),
            "environment_variables": {k: os.environ.get(k) for k in keys},
            "line_ending_policy": "source hashes normalize CRLF to LF; evidence files are written with LF",
            "deterministic_flags": {"numpy": "explicit Generator per stochastic component; no global state",
                                    "torch": "torch.use_deterministic_algorithms(True) for policy training; "
                                             "single thread per worker"}}


class Meter:
    def __init__(self, label: str, workers: int = 1) -> None:
        self.label, self.workers = label, workers
        self.counters: dict[str, float] = {}

    def __enter__(self) -> Meter:
        self._wall, self._cpu = time.perf_counter(), time.process_time()
        return self

    def add(self, name: str, value: float = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def __exit__(self, *exc) -> None:
        self.wall_s = time.perf_counter() - self._wall
        self.cpu_s = time.process_time() - self._cpu

    def record(self) -> dict:
        return {"label": self.label, "wall_s": self.wall_s, "main_process_cpu_s": self.cpu_s,
                "workers": self.workers, "cpu_upper_bound_s": self.wall_s * self.workers,
                "peak_memory_mb_main": peak_memory_mb(), "counters": self.counters, **environment()}
