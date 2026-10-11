"""v0.8 statistics (M33), planning and checkpointed parallel runner (M34)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from lob.v08 import planning, statistics as st
from lob.v08.runner import ResumeError, Task, read_checkpoint, run_tasks


def test_paired_bootstrap_and_superiority_vs_equivalence_are_distinct() -> None:
    rng = np.random.default_rng(0)
    better = rng.normal(-2.0, 1.0, 200)
    iv = st.paired_bootstrap(better, seed=1)
    assert iv.high < 0 and st.superiority(iv, threshold=0.5) == "ESTABLISHED"
    noisy = rng.normal(0.0, 5.0, 20)
    iv2 = st.paired_bootstrap(noisy, seed=1)
    assert st.superiority(iv2) == "NOT_ESTABLISHED"
    # not significant is not equivalent: a wide interval cannot be EQUIVALENT within a tight margin
    assert st.equivalence(noisy, 0.5)["status"] in ("NOT_ESTABLISHED", "NOT_EQUIVALENT")
    tight = rng.normal(0.0, 0.1, 400)
    assert st.equivalence(tight, 0.2)["status"] == "EQUIVALENT_WITHIN_MARGIN"
    assert st.superiority(st.paired_bootstrap(rng.normal(2.0, 1.0, 200), seed=2)) == "FAILED"


def test_noninferiority() -> None:
    rng = np.random.default_rng(1)
    assert st.noninferiority(rng.normal(0.1, 0.5, 300), 0.5)["status"] == "NONINFERIOR_WITHIN_MARGIN"
    assert st.noninferiority(rng.normal(2.0, 0.5, 300), 0.5)["status"] == "INFERIOR_BEYOND_MARGIN"
    assert st.noninferiority([1.0], 0.5)["status"] == "NOT_EVALUABLE"


def test_multiplicity_corrections() -> None:
    p = [0.001, 0.01, 0.02, 0.04, 0.3]
    assert st.bonferroni(p, 0.05) == [True, True, False, False, False]     # 0.01 <= 0.05 / 5 (boundary)
    assert st.holm(p, 0.05) == [True, True, False, False, False]
    assert st.benjamini_hochberg(p, 0.05) == [True, True, True, True, False]
    assert st.adjust(p, "none", 0.05) == [True, True, True, True, False]


def test_sign_test_and_wilson() -> None:
    r = st.sign_test([1, 1, 1, 1, 1, 1, 1, 1, -1, 0])
    assert r["n"] == 9 and r["p_value"] == pytest.approx(2 * (1 + 9) / 512)
    lo, hi = st.wilson(50, 100)
    assert lo < 0.5 < hi and st.wilson(0, 0) == (0.0, 1.0)


def test_power_mde_and_required_n_are_consistent() -> None:
    n = st.required_n(2.0, 0.5, alpha=0.05, power=0.8)
    assert st.power_of(0.5, 2.0, n) >= 0.8 - 1e-9
    assert st.mde(2.0, n) <= 0.5 + 1e-9
    w = st.required_n_for_width(2.0, 1.0)
    assert 2 * st.zq(0.975) * 2.0 / math.sqrt(w) <= 1.0 + 1e-9


def test_sequential_boundaries_are_decreasing_for_obrien_fleming() -> None:
    b = st.sequential_boundaries(4, 0.05, n_paths=40_000, seed=0)
    assert b[0] > b[-1] > 1.8 and len(b) == 4
    pocock = st.sequential_boundaries(4, 0.05, kind="pocock", n_paths=40_000, seed=0)
    assert max(pocock) - min(pocock) < 0.6


def test_rank_statistics() -> None:
    assert st.kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert st.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert st.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert st.ranks([3.0, 1.0, 1.0]).tolist() == [3.0, 1.5, 1.5]
    costs = np.column_stack([np.zeros(50), np.ones(50)])
    ru = st.rank_uncertainty(costs, ["a", "b"], n_boot=100)
    assert ru["a"]["p_best"] == 1.0
    assert st.regret(np.asarray([[1.0, 3.0]])).tolist() == [[0.0, 2.0]]


def test_cvar_and_hierarchical_bootstrap() -> None:
    assert st.cvar([1, 2, 3, 4], 0.75) == pytest.approx(4.0)
    assert st.cvar([1, 2, 3, 4], 0.5) == pytest.approx(3.5)
    groups = {f"w{i}": np.random.default_rng(i).normal(i, 1, 30) for i in range(5)}
    iv = st.hierarchical_bootstrap(groups, n_boot=300)
    assert iv.low < 2.0 < iv.high
    assert 0.0 <= st.bayes_prob_below(np.random.default_rng(3).normal(-1, 1, 100)) <= 1.0


def _square(x: int, seed: int) -> dict:
    return {"x": x, "y": x * x, "noise": float(np.random.default_rng(seed).random())}


def _tasks(n=12):
    return [Task(f"t{i:03d}", "tests.test_v08_statistics:_square", {"x": i, "seed": 100 + i}) for i in range(n)]


def test_serial_and_parallel_runs_are_identical() -> None:
    serial, _ = run_tasks(_tasks(), workers=1)
    parallel, _ = run_tasks(_tasks(), workers=3)
    assert json.dumps(serial, sort_keys=True) == json.dumps(parallel, sort_keys=True)
    assert [r["x"] for r in serial] == list(range(12))


def test_checkpoint_resume_skips_done_and_refuses_changed_settings(tmp_path) -> None:
    ck = tmp_path / "ck"
    first, info = run_tasks(_tasks(5), checkpoint=ck, binding={"config": "a"})
    assert info["resumed"] == 0
    again, info = run_tasks(_tasks(5), checkpoint=ck, binding={"config": "a"})
    assert info["resumed"] == 5 and again == first
    with pytest.raises(ResumeError):
        run_tasks(_tasks(5), checkpoint=ck, binding={"config": "b"})
    with pytest.raises(ResumeError):
        run_tasks(_tasks(6), checkpoint=ck, binding={"config": "a"})


def test_partial_checkpoint_recovers_and_truncated_line_is_discarded(tmp_path) -> None:
    ck = tmp_path / "ck"
    run_tasks(_tasks(4), checkpoint=ck, binding={})
    path = ck / "results.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:2]) + "\n" + lines[2][:15], encoding="utf-8")    # interrupted mid-write
    out, info = run_tasks(_tasks(4), checkpoint=ck, binding={})
    assert info["resumed"] == 2 and any("truncated" in n for n in info["notes"]) and len(out) == 4
    assert len(read_checkpoint(ck)[1]) == 4


def test_corrupted_or_duplicated_checkpoint_is_refused(tmp_path) -> None:
    ck = tmp_path / "ck"
    run_tasks(_tasks(3), checkpoint=ck, binding={})
    path = ck / "results.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines + [lines[0]]) + "\n", encoding="utf-8")
    with pytest.raises(ResumeError):
        read_checkpoint(ck)
    record = json.loads(lines[0])
    record["result"]["y"] = 999
    path.write_text("\n".join([json.dumps(record)] + lines[1:]) + "\n", encoding="utf-8")
    with pytest.raises(ResumeError):
        read_checkpoint(ck)


def test_duplicate_task_ids_are_refused() -> None:
    with pytest.raises(ValueError):
        run_tasks([Task("a", "x:y"), Task("a", "x:y")])


def test_planning_budgets_estimates_and_seed_plans(tmp_path) -> None:
    assert planning.budget("tiny").seeds < planning.budget("exhaustive").seeds
    with pytest.raises(KeyError):
        planning.budget("enormous")
    rate = {"seconds_per_bin": 1e-4, "events_per_bin": 5.0}
    est = planning.estimate(episodes=1000, workers=4, rate=rate)
    assert est["events"] == 1000 * planning.BINS_PER_EPISODE * 5 and est["wall_seconds"] < est["cpu_seconds"]
    plan = planning.seed_plan(sd=3.0, alpha=0.05, power=0.8, effect=1.0, ci_width=2.0)
    assert plan["seeds"] >= plan["mde_driven"]["seeds"] and plan["seeds"] >= plan["ci_driven"]["seeds"]
    assert plan["achieved_mde"] <= 1.0 + 1e-9
    assert planning.within_budget(est, planning.budget("tiny"))["within_budget"] is False
    path = planning.write_plan(tmp_path / "plan", {"estimate": est, "seeds": plan})
    assert json.loads(path.read_text())["seeds"]["seeds"] == plan["seeds"]
