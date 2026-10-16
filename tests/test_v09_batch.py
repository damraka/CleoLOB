"""Fail-closed batch execution: a child's nonzero exit propagates, blocks dependents and is ledgered."""
from __future__ import annotations

import subprocess
import sys

from lob.v09 import artifacts, batch, protocol as pr

from .v09_fixtures import v09_root, v09_template  # noqa: F401

SEAL = ("import sys; from pathlib import Path; from lob.v09 import artifacts; r=Path(sys.argv[1]); "
        "o=artifacts.new_run(r/sys.argv[2]); artifacts.finalize(o, analysis='batch-test', config={}, result={'ok': 1}, "
        "root=r, check_source=False)")


def _seal_step(name, root, out, needs=()):
    return {"name": name, "argv": [sys.executable, "-c", SEAL, str(root), out], "out": out, "needs": list(needs),
            "design": "explore-batch-test"}


def test_nonzero_exit_fails_blocks_dependents_and_is_ledgered(v09_root, tmp_path) -> None:
    root = v09_root
    before = pr.state_of(root)[1].count
    steps = [
        {"name": "fails", "argv": [sys.executable, "-c", "import sys; sys.exit(3)"], "out": "results/v09/explore/f",
         "design": "explore-batch-test"},
        _seal_step("dependent", root, "results/v09/explore/dep", needs=["fails"]),
        _seal_step("transitive", root, "results/v09/explore/tr", needs=["dependent"]),
        _seal_step("independent", root, "results/v09/explore/ok"),
    ]
    out = batch.run_batch(root, steps, logdir=tmp_path / "logs", min_free_gb=0.0, echo=lambda m: None)
    status = {r["name"]: r["status"] for r in out["steps"]}
    assert not out["valid"]
    assert status == {"fails": "FAILED", "dependent": "SKIPPED", "transitive": "SKIPPED", "independent": "SUCCEEDED"}
    assert not (root / "results/v09/explore/dep").exists() and not (root / "results/v09/explore/tr").exists()
    _, state = pr.state_of(root)
    assert state.count == before + 1
    attempt = state.attempts[-1]
    assert attempt["outcome"] == "INVALID" and attempt["exit_code"] == 3 and attempt["design"] == "explore-batch-test"
    assert attempt["directory"] is None and attempt["no_directory_reason"]
    assert artifacts.verify_run(root / "results/v09/explore/ok", root=root)["valid"]


def test_zero_exit_without_a_sealed_run_is_a_failure(v09_root, tmp_path) -> None:
    steps = [{"name": "silent", "argv": [sys.executable, "-c", "pass"], "out": "results/v09/explore/none",
              "design": "explore-batch-test"},
             _seal_step("after", v09_root, "results/v09/explore/after", needs=["silent"])]
    out = batch.run_batch(v09_root, steps, logdir=tmp_path / "logs", min_free_gb=0.0, echo=lambda m: None)
    assert [r["status"] for r in out["steps"]] == ["FAILED", "SKIPPED"]
    assert pr.state_of(v09_root)[1].attempts[-1]["exit_code"] == -1


def test_sealed_runs_are_never_rerun_or_overwritten(v09_root, tmp_path) -> None:
    step = _seal_step("once", v09_root, "results/v09/explore/once")
    assert batch.run_batch(v09_root, [step], logdir=tmp_path, min_free_gb=0.0, echo=lambda m: None)["valid"]
    digest = (v09_root / "results/v09/explore/once/result.json").read_bytes()
    again = batch.run_batch(v09_root, [step], logdir=tmp_path, min_free_gb=0.0, echo=lambda m: None)
    assert again["steps"][0]["status"] == "ALREADY_SEALED"
    assert (v09_root / "results/v09/explore/once/result.json").read_bytes() == digest


def test_batch_cli_exit_code_propagates(v09_root, tmp_path) -> None:
    import json
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([{"name": "x", "argv": [sys.executable, "-c", "import sys; sys.exit(7)"],
                                 "out": "results/v09/explore/x", "design": "explore-batch-test"}]), encoding="utf-8")
    rc = subprocess.call([sys.executable, "-m", "lob.v09.batch", str(spec), "--root", str(v09_root),
                          "--logdir", str(tmp_path / "logs"), "--min-free-gb", "0"])
    assert rc == 1


def test_pipeline_data_rounds_use_a_fresh_directory_after_a_sealed_round(v09_root) -> None:
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("v09_pipeline", Path(__file__).resolve().parents[1] / "tools" /
                                                  "v09_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = "results/v09/confirm"
    assert mod.fresh(v09_root, base, "data-prep") == f"{base}/data-prep"
    o = artifacts.new_run(v09_root / base / "data-prep")
    artifacts.finalize(o, analysis="t", config={}, result={}, root=v09_root, check_source=False)
    assert mod.target(v09_root, base, "data-prep") is None          # a sealed study step is skipped ...
    assert mod.fresh(v09_root, base, "data-prep") == f"{base}/data-prep-2"   # ... a new data round is not
