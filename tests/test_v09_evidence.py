"""v0.9 run schema v3, lineage DAG, checkpoint v3, worker-count determinism and resource contracts."""
from __future__ import annotations

import json
import os

import pytest

from lob.v09 import artifacts, runner, runtime
from lob.v09.runner import ResourceContract, ResourceExhausted, ResumeError, Task
from tests.v09_fixtures import v09_root, v09_template  # noqa: F401


def _tasks(n: int = 12) -> list[Task]:
    return [Task(f"t{i:03d}", "lob.v09.selftest:draw", {"seed": 1000 + i}, seed=1000 + i, namespace="dev")
            for i in range(n)]


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_results_are_worker_count_independent(workers) -> None:
    contract = ResourceContract(workers_max=8)
    base, _ = runner.run_tasks(_tasks(), workers=1, contract=contract)
    got, info = runner.run_tasks(_tasks(), workers=workers, contract=contract)
    assert got == base and info["completed"] == 12


def test_checkpoint_records_carry_v3_fields_and_resume(tmp_path) -> None:
    ckpt = tmp_path / "ckpt"
    tasks = _tasks(6)
    runner.run_tasks(tasks[:6], workers=1, checkpoint=ckpt, design={"d": 1}, inputs={"i": 1})
    records = [json.loads(line) for line in (ckpt / "tasks.jsonl").read_text().splitlines()]
    for key in ("task_id", "seed", "namespace", "design_sha256", "input_digest", "code_digest", "task_spec_digest",
                "result_digest", "status"):
        assert key in records[0]
    # truncate the last record: resume discards it, recomputes and reproduces identical results
    text = (ckpt / "tasks.jsonl").read_text()
    (ckpt / "tasks.jsonl").write_text(text[: len(text) - 25])
    again, info = runner.run_tasks(tasks, workers=2, checkpoint=ckpt, design={"d": 1}, inputs={"i": 1})
    fresh, _ = runner.run_tasks(tasks, workers=1)
    assert again == fresh and info["resumed"] == 5 and info["notes"]


def test_resume_refuses_changed_design_inputs_or_tasks(tmp_path) -> None:
    ckpt = tmp_path / "ckpt"
    runner.run_tasks(_tasks(3), checkpoint=ckpt, design={"d": 1})
    with pytest.raises(ResumeError):
        runner.run_tasks(_tasks(3), checkpoint=ckpt, design={"d": 2})
    with pytest.raises(ResumeError):
        runner.run_tasks(_tasks(4), checkpoint=ckpt, design={"d": 1})


def test_corrupt_or_duplicate_checkpoint_records_are_refused(tmp_path) -> None:
    ckpt = tmp_path / "ckpt"
    runner.run_tasks(_tasks(3), checkpoint=ckpt)
    lines = (ckpt / "tasks.jsonl").read_text().splitlines()
    record = json.loads(lines[0])
    record["result"]["mean"] = 99.0
    (ckpt / "tasks.jsonl").write_text("\n".join([json.dumps(record)] + lines[1:]) + "\n")
    with pytest.raises(ResumeError, match="digest"):
        runner.run_tasks(_tasks(3), checkpoint=ckpt)
    (ckpt / "tasks.jsonl").write_text("\n".join([lines[0], lines[0]] + lines[1:]) + "\n")
    with pytest.raises(ResumeError, match="duplicate"):
        runner.run_tasks(_tasks(3), checkpoint=ckpt)


def test_worker_caps_and_resource_exhaustion(tmp_path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="registered maximum"):
        ResourceContract(workers_max=2).workers(3)
    monkeypatch.setenv("CLEOLOB_MAX_WORKERS", "1")
    with pytest.raises(ValueError, match="registered maximum"):
        ResourceContract(workers_max=8).workers(2)
    monkeypatch.delenv("CLEOLOB_MAX_WORKERS")
    tiny = ResourceContract(ram_ceiling_mb=1.0, workers_max=2)
    tasks = [Task(f"a{i}", "lob.v09.selftest:allocate", {"seed": i, "mb": 1}, seed=i) for i in range(4)]
    with pytest.raises(ResourceExhausted, match="RESOURCE_EXHAUSTED"):
        runner.run_tasks(tasks, workers=1, checkpoint=tmp_path / "c", contract=tiny)
    assert (tmp_path / "c" / "tasks.jsonl").read_text().strip(), "the checkpoint is kept after a resource abort"


def test_runtime_fingerprint_is_portable() -> None:
    fp = runtime.fingerprint()
    text = json.dumps(fp)
    assert fp["python"] and fp["logical_cpus"] == os.cpu_count()
    for marker in (":\\\\", "/home/", "/Users/"):
        assert marker not in text
    bom = runtime.sbom()
    assert bom["components"] and len(bom["digest"]) == 64


def test_run_schema_v3_seals_and_verifies(v09_root) -> None:
    out = artifacts.new_run(v09_root / "results/v09/explore/a")
    binding = artifacts.finalize(out, analysis="test-a", config={"k": 1}, result={"v": 2}, root=v09_root,
                                 check_source=False, summary={"s": 1}, resources={"contract": {"ram": 1}})
    assert binding["confirmatory_sha256"] is None
    for name in ("config", "result", "summary", "provenance", "lineage", "resources", "status", "binding",
                 "checksums"):
        assert (out / f"{name}.json").is_file(), name
    assert artifacts.verify_run(out, root=v09_root)["valid"]
    down = artifacts.new_run(v09_root / "results/v09/explore/b")
    artifacts.finalize(down, analysis="test-b", config={}, result={"w": 3}, root=v09_root, check_source=False,
                       inputs={"upstream": "results/v09/explore/a"})
    graph = artifacts.dag(v09_root / "results/v09", root=v09_root)
    assert graph["acyclic"] and not graph["dangling_inputs"] and graph["edges"][0]["key"] == "upstream"
    # tampering upstream invalidates the downstream run (lineage) and the upstream (bytes)
    path = out / "result.json"
    path.write_text(path.read_text().replace("2", "3"))
    assert not artifacts.verify_run(out, root=v09_root)["valid"]
    report = artifacts.verify_run(down, root=v09_root)
    assert not report["valid"] and any("upstream result changed" in i for i in report["issues"])


def test_run_status_and_portability_are_enforced(v09_root) -> None:
    out = artifacts.new_run(v09_root / "results/v09/explore/c")
    with pytest.raises(ValueError, match="run status"):
        artifacts.finalize(out, analysis="x", config={}, result={}, root=v09_root, status="GREAT", check_source=False)
    out2 = artifacts.new_run(v09_root / "results/v09/explore/d")
    with pytest.raises(ValueError, match="absolute machine path"):
        artifacts.finalize(out2, analysis="x", config={"p": "C:\\Users\\someone\\x"}, result={}, root=v09_root,
                           check_source=False)
    out3 = artifacts.new_run(v09_root / "results/v09/explore/e")
    artifacts.finalize(out3, analysis="x", config={}, result={"n": 0}, root=v09_root, status="NOT_AVAILABLE",
                       status_reason="registered source failed", check_source=False)
    assert artifacts.run_status(out3) == "NOT_AVAILABLE" and artifacts.verify_run(out3, root=v09_root)["valid"]
    with pytest.raises(FileExistsError):
        artifacts.new_run(v09_root / "results/v09/explore/e")
