"""v1 experiment orchestration: checkpointed runner (M48) and DAG planner (M150)."""
import json
import os

import pytest

from lob.v1.experiments import dag as D, runner as RN


def tasks(n=6):
    return [RN.Task("tests.v1_tasks:square", {"x": i}) for i in range(n)]


def test_runner_results_independent_of_workers_and_resumable(tmp_path):
    a = RN.run(tasks(), tmp_path / "a", inline=True)
    b = RN.run(tasks(), tmp_path / "b", workers=2, profile="small", min_free_gb=0.0)
    assert a["status"] == b["status"] == "COMPLETED" and a["results"] == b["results"]
    assert sorted(a["results"].values()) == [i * i for i in range(6)]
    # resume: delete half the checkpoint, rerun, identical results
    done = sorted((tmp_path / "a" / "done").glob("*.json"))
    for p in done[:3]:
        p.unlink()
    c = RN.run(tasks(), tmp_path / "a", inline=True)
    assert c["resumed"] and c["results"] == a["results"]
    with pytest.raises(ValueError):
        RN.run(tasks(5), tmp_path / "a", inline=True)        # a different task set never reuses a checkpoint


def test_runner_task_ids_are_content_hashes():
    assert RN.Task("m:f", {"x": 1}).id == RN.Task("m:f", {"x": 1}).id != RN.Task("m:f", {"x": 2}).id
    with pytest.raises(ValueError):
        RN.run([RN.Task("m:f", {})] * 2, "unused")


def test_runner_failures_are_recorded_not_retried(tmp_path):
    ts = tasks(3) + [RN.Task("tests.v1_tasks:fail", {"x": 1})]
    r = RN.run(ts, tmp_path / "f", inline=True)
    assert r["status"] == "INVALID" and len(r["failed"]) == 1 and r["completed"] == 3
    assert "boom 1" in json.loads((tmp_path / "f" / "failed" / f"{r['failed'][0]}.json").read_text())["error"]


def test_runner_resource_guard_and_lock(tmp_path):
    r = RN.run(tasks(), tmp_path / "g", inline=True, min_free_gb=1e9)
    assert r["status"] == "RESOURCE_EXHAUSTED" and r["completed"] == 0
    (tmp_path / "h").mkdir()
    (tmp_path / "h" / "runner.lock").write_text(str(os.getpid()))
    with pytest.raises(RN.RunnerLocked):
        RN.run(tasks(), tmp_path / "h", inline=True)
    (tmp_path / "h" / "runner.lock").write_text("999999")       # dead pid
    with pytest.raises(RN.RunnerLocked):
        RN.run(tasks(), tmp_path / "h", inline=True)
    assert RN.run(tasks(), tmp_path / "h", inline=True, steal_stale=True)["status"] == "COMPLETED"


def steps():
    return [D.Step("a", "tests.v1_tasks:make_run", "r/a", args={"value": 1.0}),
            D.Step("b", "tests.v1_tasks:make_run", "r/b", depends=("a",), args={"value": 2.0}),
            D.Step("c", "tests.v1_tasks:make_run", "r/c", depends=("a", "b"), args={"value": 4.0})]


def test_dag_plan_order_and_validation():
    assert [s.name for s in D.plan(steps())] == ["a", "b", "c"]
    with pytest.raises(D.PlanError):
        D.plan(steps() + [D.Step("a", "x:y", "r/z")])
    with pytest.raises(D.PlanError):
        D.plan([D.Step("x", "m:f", "o", depends=("y",)), D.Step("y", "m:f", "o2", depends=("x",))])
    with pytest.raises(D.PlanError):
        D.plan([D.Step("x", "m:f", "o", depends=("missing",))])


def test_dag_execute_resume_and_failure(tmp_path):
    s = steps()
    assert D.status(tmp_path, s)["steps"] == {"a": "READY", "b": "BLOCKED", "c": "BLOCKED"}
    assert D.execute(tmp_path, s, stop_after=1)["ran"] == ["a"]
    r = D.execute(tmp_path, s)
    assert r["status"] == "COMPLETED" and r["ran"] == ["b", "c"]
    assert json.loads((tmp_path / "r/c/result.json").read_text())["total"] == 4 + 1 + (2 + 1)
    assert D.execute(tmp_path, s)["ran"] == []                   # complete plans are not rerun
    bad = [D.Step("x", "tests.v1_tasks:make_invalid", "r/x"), D.Step("y", "tests.v1_tasks:make_run", "r/y",
                                                                    depends=("x",))]
    out = D.execute(tmp_path, bad)
    assert out["status"] == "FAILED" and out["step"] == "x"
    assert D.status(tmp_path, bad)["steps"] == {"x": "INVALID", "y": "BLOCKED"}
