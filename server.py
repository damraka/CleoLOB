"""CLEO — LOB Execution Lab server.

FastAPI backend that serves the custom web UI and runs simulations as background
jobs with progress polling.

    python server.py            # -> http://127.0.0.1:8000 (opens browser)
    python server.py --port 8080 --no-browser
"""
from __future__ import annotations

import argparse
import threading
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from lob.runner import run_pair

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="CLEO — LOB Execution Lab", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class RunParams(BaseModel):
    qty: int = Field(10_000, ge=500, le=200_000)
    horizon: float = Field(60.0, ge=10, le=300)
    dt: float = Field(0.5, ge=0.1, le=2.0)
    seed: int = Field(42, ge=0, le=1_000_000)
    latency_ms: float = Field(10.0, ge=0, le=100)
    resilience: float = Field(0.8, ge=0, le=3)
    market_rate: float = Field(6.0, ge=0.5, le=30)
    risk_aversion: float = Field(1e-6, ge=1e-9, le=1e-2)
    model_path: str = "models/ppo_lob"


JOBS: Dict[str, Dict[str, Any]] = {}
LOCK = threading.Lock()


def _worker(job_id: str, params: Dict[str, Any]) -> None:
    def progress(phase: str, frac: float) -> None:
        with LOCK:
            JOBS[job_id].update(phase=phase, pct=float(frac))

    try:
        payload = run_pair(params, progress)
        with LOCK:
            JOBS[job_id].update(status="done", pct=1.0, result=payload)
    except Exception as exc:  # surface errors to the UI instead of dying silently
        with LOCK:
            JOBS[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}")


@app.post("/api/run")
def api_run(params: RunParams) -> Dict[str, str]:
    job_id = uuid.uuid4().hex[:12]
    with LOCK:
        JOBS[job_id] = {"status": "running", "phase": "queued", "pct": 0.0}
    threading.Thread(target=_worker, args=(job_id, params.model_dump()),
                     daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def api_job(job_id: str) -> Dict[str, Any]:
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return {k: job[k] for k in ("status", "phase", "pct") if k in job} | \
               ({"error": job["error"]} if "error" in job else {})


@app.get("/api/result/{job_id}")
def api_result(job_id: str) -> Dict[str, Any]:
    with LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job.get("status") != "done":
        raise HTTPException(409, "job not finished")
    return job["result"]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    print(f"\n  CLEO — LOB Execution Lab\n  {url}\n")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
