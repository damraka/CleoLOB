"""Runtime helpers: thread caps per worker, environment report for ``cleo doctor``."""
from __future__ import annotations

import importlib.util
import os
import platform


def cap_threads(default: int = 2) -> int:
    """Cap PyTorch and BLAS threads per worker (``CLEOLOB_TORCH_THREADS``) so parallel workers never oversubscribe."""
    n = max(1, int(os.environ.get("CLEOLOB_TORCH_THREADS", default)))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass
    return n


def optional(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def doctor() -> dict:
    from . import VERSION
    from .engines import accelerated, gpu
    from .data import firewall
    out = {"cleolob": VERSION, "python": platform.python_version(), "platform": platform.platform(terse=True),
           "optional": {m: optional(m) for m in ("numba", "torch", "pyarrow", "fastapi", "uvicorn", "websockets",
                                                 "scipy", "plotly", "gymnasium")},
           "accelerated_kernel": "numba" if accelerated.HAVE_NUMBA else "interpreted",
           "gpu": gpu.available(), "consumed_through": firewall.CONSUMED_THROUGH}
    missing = [m for m in ("pyarrow",) if not out["optional"][m]]
    out["status"] = "OK" if not missing else "DEGRADED"
    out["missing_for_core"] = missing
    return out


__all__ = ["cap_threads", "optional", "doctor"]
