"""Runtime fingerprinting, dependency provenance (SBOM) and the locked environment (M7, M8, M9, M42).

* ``fingerprint`` records the interpreter, platform, CPU, memory, the versions of every numerical dependency, the
  optional accelerator (CUDA device and versions when PyTorch sees one; CPU otherwise) and deterministic-setting
  limitations. It never contains absolute machine paths or user names.
* ``sbom`` lists every installed distribution (name, version, declared license, requirement strings) in a
  CycloneDX-like JSON document with a content digest.
* ``lock`` writes/validates ``configs/v09/environment.lock``: exact versions of the distributions CleoLOB imports.
  ``check_lock`` reports drift; a drifted environment is recorded in run provenance, never silently ignored.
* ``memory_mb`` / ``peak_memory_mb`` read the process resident set without psutil (Windows API or /proc).
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import sys

from ..config import canonical_json

LOCK_PATH = "configs/v09/environment.lock"
CORE_DISTRIBUTIONS = ("numpy", "pandas", "gymnasium", "pydantic", "PyYAML")
OPTIONAL_DISTRIBUTIONS = ("torch", "stable-baselines3", "scipy", "hypothesis", "pytest", "ruff", "build", "twine",
                          "pyarrow")


def _version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def accelerator() -> dict:
    """Optional GPU information; repository correctness never depends on it."""
    info: dict = {"device": "cpu", "torch": _version("torch")}
    if info["torch"] is None:
        info["note"] = "PyTorch not installed; neural models unavailable (optional extra 'neural')"
        return info
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_build"] = torch.version.cuda
        if info["cuda_available"]:
            info.update({"device": "cuda", "gpu": torch.cuda.get_device_name(0),
                         "cudnn": torch.backends.cudnn.version()})
        info["deterministic_limitations"] = (
            "CPU kernels are deterministic for the registered models; CUDA kernels may be nondeterministic even with "
            "seeds, so registered confirmatory neural runs use the CPU device")
    except Exception as exc:  # noqa: BLE001 - optional accelerator probing must never fail a run
        info["error"] = type(exc).__name__
    return info


def _total_memory_mb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.ullTotalPhys / 2**20
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, AttributeError):
        pass
    return None


def _process_memory() -> tuple[float | None, float | None]:
    """(current, peak) resident memory of this process in MiB."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            psapi = ctypes.WinDLL("psapi")
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / 2**20, counters.PeakWorkingSetSize / 2**20
            return None, None
        status = Path("/proc/self/status")
        if status.is_file():
            values = {}
            for line in status.read_text().splitlines():
                key, _, rest = line.partition(":")
                if key in ("VmRSS", "VmHWM"):
                    values[key] = float(rest.split()[0]) / 1024
            return values.get("VmRSS"), values.get("VmHWM")
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return None, peak / (2**20 if sys.platform == "darwin" else 1024)
    except (OSError, ValueError, AttributeError, ImportError):
        return None, None


def memory_mb() -> float | None:
    return _process_memory()[0]


def peak_memory_mb() -> float | None:
    return _process_memory()[1]


def fingerprint() -> dict:
    total = _total_memory_mb()
    return {"schema": "cleolob-v09-runtime-1", "python": platform.python_version(),
            "implementation": platform.python_implementation(), "system": platform.system(),
            "release": platform.release(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
            "total_memory_gb": round(total / 1024, 1) if total else None,
            "packages": {n: _version(n) for n in CORE_DISTRIBUTIONS + OPTIONAL_DISTRIBUTIONS},
            "accelerator": accelerator(), "lock": check_lock(Path(__file__).resolve().parents[2])}


def _requirements(dist: md.Distribution) -> list[str]:
    return sorted(dist.requires or [])


def sbom() -> dict:
    """CycloneDX-like software bill of materials of the running environment."""
    components = []
    for dist in sorted(md.distributions(), key=lambda d: (d.metadata["Name"] or "").lower()):
        name = dist.metadata["Name"]
        if not name:
            continue
        components.append({"type": "library", "name": name, "version": dist.version,
                           "license": dist.metadata.get("License-Expression") or dist.metadata.get("License") or None,
                           "requires": _requirements(dist)})
    for c in components:
        if c["license"] and len(c["license"]) > 120:
            c["license"] = c["license"][:117] + "..."
    body = {"bomFormat": "CycloneDX-like", "specVersion": "cleolob-v09-sbom-1",
            "metadata": {"component": {"name": "cleolob", "version": _version("cleolob")},
                         "python": platform.python_version()}, "components": components}
    return {**body, "digest": hashlib.sha256(canonical_json(body).encode()).hexdigest()}


def lock_document() -> dict:
    pins = {n: _version(n) for n in CORE_DISTRIBUTIONS + OPTIONAL_DISTRIBUTIONS}
    return {"schema": "cleolob-v09-environment-lock-1", "python": ".".join(platform.python_version_tuple()[:2]),
            "pins": {k: v for k, v in sorted(pins.items()) if v is not None},
            "note": "exact versions of the research environment; the container path installs these pins"}


def write_lock(root: Path) -> dict:
    document = lock_document()
    (root / LOCK_PATH).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    return document


def check_lock(root: Path) -> dict:
    path = Path(root) / LOCK_PATH
    if not path.is_file():
        return {"status": "NO_LOCK"}
    locked = json.loads(path.read_text(encoding="utf-8"))
    drift = {n: {"locked": v, "installed": _version(n)} for n, v in locked["pins"].items() if _version(n) != v}
    return {"status": "MATCH" if not drift else "DRIFT", "drift": drift,
            "lock_sha256": hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}


def requirements_text(document: dict) -> str:
    """pip requirement lines for the container path."""
    return "".join(f"{name}=={version}\n" for name, version in document["pins"].items()
                   if name not in ("build", "twine", "ruff", "pytest", "hypothesis"))


def torch_threads(default: int = 2) -> int:
    """Cap PyTorch intra-op threads per worker (``CLEOLOB_TORCH_THREADS``) so parallel workers do not oversubscribe."""
    try:
        import torch
    except ImportError:
        return 0
    n = int(os.environ.get("CLEOLOB_TORCH_THREADS", default))
    torch.set_num_threads(max(1, n))
    return n
