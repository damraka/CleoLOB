"""v0.8 decision layer: robust criteria (M24), completion and Pareto (M22, M23), selection/abstention (M27),
stress worlds (M28), transfer metrics (M32), decision-focused calibration (M21) and portfolio routing (M31)."""
from __future__ import annotations

import numpy as np
import pytest

from lob.v08 import decision_calibration as dc, portfolio as pf, robust, selection, stress, transfer
from lob.v08.execution import policies as P
from lob.v08.execution.completion import completion_summary, constrained_choice, objectives, pareto_front
from lob.v08.execution.simulator import Fees, Mandate, run_episode
from lob.v08.generators import EndogenousGenerator, PoissonGenerator
from lob.v08.ood import Reference
from lob.v08.market.frame import summaries
from lob.v08.worlds import World, WorldSet, from_posterior, load, save
from tests.v08_fixtures import tiny_world

COST = np.asarray([[1.0, 1.0, 9.0],      # a: cheap except one bad world
                   [3.0, 3.0, 3.0],      # b: flat
                   [0.5, 6.0, 6.0]])     # c: best only in world 0
NAMES = ["a", "b", "c"]


def test_robust_criteria_on_a_known_matrix() -> None:
    out = robust.select_all(NAMES, COST, np.ones(3), nominal=0, level=2 / 3)
    c = {k: v["choice"] for k, v in out["criteria"].items()}
    assert c["nominal"] == "c" and c["posterior_expected"] == "b" or c["posterior_expected"] == "a"
    assert c["worst_case"] == "b" and c["minimax_regret"] == "b" and not out["criteria_agree"]
    assert out["criteria"]["posterior_expected"]["values"]["a"] == pytest.approx(11 / 3)


def test_kl_dro_lies_between_mean_and_max() -> None:
    w = np.ones(3) / 3
    for row in COST:
        v = robust.kl_dro_value(row, w, 0.1)
        assert row.mean() - 1e-9 <= v <= row.max() + 1e-9
    assert robust.kl_dro_value(COST[0], w, 0.0) == pytest.approx(COST[0].mean())
    assert robust.kl_dro_value(COST[0], w, 10.0) == pytest.approx(9.0, abs=0.05)


def test_completion_constraint_removes_incomplete_candidates() -> None:
    comp = np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.5, 1.0, 1.0]])
    out = robust.select_all(NAMES, COST, np.ones(3), completion=comp, completion_threshold=0.95)
    assert "c" not in out["feasible"]
    none = robust.select_all(NAMES, COST, np.ones(3), completion=comp * 0, completion_threshold=0.95)
    assert all(v["status"] == "NO_FEASIBLE_CANDIDATE" for v in none["criteria"].values())


def _rows(world, name, mode, seeds=(1, 2, 3)):
    return [run_episode(world, P.make(name), Mandate(qty=20, completion=mode), s) for s in seeds]


def test_completion_summary_never_rewards_incomplete_low_cost() -> None:
    w = tiny_world(market=0.05)
    passive = [run_episode(w, _Passive(), Mandate(qty=20, completion="none"), s) for s in (1, 2, 3)]
    twap = _rows(w, "twap", "none")
    summaries_ = {"passive": completion_summary(passive), "twap": completion_summary(twap)}
    assert summaries_["passive"]["completion_probability"] < 0.5
    choice = constrained_choice(summaries_, threshold=0.3)       # Wilson lower bound of 3/3 is about 0.44
    assert choice["choice"] == "twap" and "passive" in choice["infeasible"]
    assert summaries_["passive"]["economic_cost_given_incomplete_bps"] is not None


class _Passive(P.Policy):
    name = "passive_test"

    def decide(self, obs):
        return [] if obs.resting or obs.context.get("pending_limit") else [("limit", obs.remaining, 0, 0)]


def test_pareto_front_keeps_tradeoffs() -> None:
    pts = {"cheap_incomplete": {"economic_cost_bps": 1.0, "completion_probability": 0.2},
           "dear_complete": {"economic_cost_bps": 3.0, "completion_probability": 1.0},
           "dominated": {"economic_cost_bps": 4.0, "completion_probability": 0.5}}
    front = pareto_front(pts, ["economic_cost_bps", "completion_probability"])
    assert set(front["front"]) == {"cheap_incomplete", "dear_complete"} and front["dominated"] == ["dominated"]
    obj = objectives(_rows(tiny_world(), "twap", "none"))
    assert {"economic_cost_bps", "completion_probability", "cvar90_cost_bps"} <= set(obj)


def _cases():
    out = []
    for i in range(12):
        regime = i % 2
        costs = {"x": 1.0 + 2 * regime, "y": 3.0 - 2 * regime, "safe": 2.5}
        out.append(selection.Case(f"c{i}", regime, [1.0 - regime, float(regime)], ood_score=float(i == 11) * 10,
                                  costs=costs))
    return out


def test_selectors_and_abstention() -> None:
    cases = _cases()
    names = ["x", "y", "safe"]
    glob = selection.Selector("global", names, safe_baseline="safe").fit(cases)
    reg = selection.Selector("regime", names, safe_baseline="safe").fit(cases)
    moe = selection.Selector("moe", names, safe_baseline="safe").fit(cases)
    eg, er, em = (selection.evaluate(s, cases, margin=0.5) for s in (glob, reg, moe))
    assert er["mean_regret"] == 0.0 < eg["mean_regret"] and em["mean_regret"] == 0.0
    ab = selection.evaluate(reg, cases, margin=0.5, abstention=selection.Abstention(5.0, 1e9))
    assert ab["abstention_rate"] == pytest.approx(1 / 12) and ab["fallback_rate"] >= ab["abstention_rate"]
    with pytest.raises(ValueError):
        selection.Selector("regime", names, safe_baseline="unregistered")
    with pytest.raises(ValueError):
        selection.Selector("magic", names, safe_baseline="safe")


@pytest.mark.parametrize("scenario", stress.SCENARIOS)
def test_every_stress_scenario_builds_a_runnable_labelled_world(scenario) -> None:
    w = stress.apply(tiny_world(), scenario)
    assert w.label == "STRESS" and scenario in w.world_id
    row = run_episode(w, P.make("twap"), Mandate(qty=10, horizon_s=20.0, warmup_s=5.0), 1)
    assert row["status"] == "VALID"


def test_adversarial_search_respects_plausibility_radius() -> None:
    base = tiny_world()
    ref = Reference(np.asarray([summaries(base.generator.simulate(400, k, book=base.book)) for k in range(8)]))
    res = stress.adversarial_search(base, lambda w: 1.0, reference=ref, radius=0.0, budget=3, seed=1, n_bins=300)
    assert res["evaluated"] == 0 and res["rejected_implausible"] == 3 and res["best"] is None
    res = stress.adversarial_search(base, lambda w: float(w.book.refill_mean), reference=ref, radius=1e9, budget=3,
                                    seed=1, n_bins=300)
    assert res["evaluated"] == 3 and res["best"]["objective"] == max(c["objective"] for c in res["candidates"])


def test_transfer_metrics() -> None:
    src = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    perfect = transfer.compare(src, dict(src), k=2)
    assert perfect["kendall_tau"] == pytest.approx(1.0) and perfect["pairwise"]["accuracy"] == 1.0
    assert perfect["top_k"]["retention"] == 1.0 and perfect["regret_transfer"]["target_regret"] == 0.0
    det = {"a|b": "B_LOWER", "a|c": "INCONCLUSIVE"}
    r = transfer.compare(src, {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}, determinate=det)
    assert r["kendall_tau"] == pytest.approx(-1.0) and r["pairwise"]["agree"] < r["pairwise"]["determinate_pairs"]
    assert transfer.binomial_greater(5, 5) == pytest.approx(1 / 32)
    stab = transfer.dominance_stability({"x": {"p": "A_LOWER", "q": "B_LOWER"}, "y": {"p": "A_LOWER", "q": "A_LOWER"}})
    assert stab["stable"] == ["p"] and stab["unstable"] == ["q"] and stab["stability"] == 0.5


def test_decision_calibration_pair_metrics_and_associations() -> None:
    truth = {"a": 1.0, "b": 2.0, "c": 3.0}
    good = dc.pair_metrics(truth, {"a": 1.1, "b": 2.1, "c": 3.1})
    bad = dc.pair_metrics(truth, {"a": 3.0, "b": 2.0, "c": 1.0})
    assert good["regret"] == 0.0 and good["kendall_tau"] == pytest.approx(1.0) and bad["regret"] == 2.0
    rows = [{"loss": float(i), "regret": float(i) * 0.1} for i in range(10)]
    assoc = dc.associations(rows, metrics=("loss",), n_boot=200)
    assert assoc["loss"]["spearman"] == pytest.approx(1.0) and assoc["loss"]["ci"][0] > 0.5
    assert dc.associations(rows[:3], metrics=("loss",))["loss"]["status"] == "NOT_EVALUABLE"


def test_routing_allocations_and_multi_venue_episode() -> None:
    venues = [pf.Venue("deep", tiny_world("d", refill=14.0), Fees(1.5, 0.0), 0.0),
              pf.Venue("thin", tiny_world("t", refill=4.0), Fees(0.5, 0.0), 0.05)]
    views = [(100, [10] * 10), (100, [2] * 10)]
    for r in pf.ROUTERS:
        alloc = pf.route(r, 15, views, venues, 1, 99.5)
        assert sum(alloc) == 15 and min(alloc) >= 0
    assert pf.route("single:1", 7, views, venues, 1, 99.5) == [0, 7]
    ep = pf.multi_venue_episode(venues, "sor", qty=40, horizon_s=20.0, warmup_s=5.0, seed=1)
    assert ep["completion"] and abs(sum(ep["venue_share"]) - 1) < 1e-9
    wb = World("b", EndogenousGenerator.wrap(PoissonGenerator()), PoissonGenerator().book)
    b = pf.basket_episode(tiny_world("a"), wb, qty_a=20, qty_b=20, cross_impact=1.0, horizon_s=20.0, warmup_s=5.0,
                          seed=1)
    assert b["completion"] and "SYNTHETIC" in b["label"]


def test_world_sets_round_trip_and_posterior_draws(tmp_path) -> None:
    g = PoissonGenerator()
    ws = from_posterior(g, [{"market": 0.5}, {"market": 2.0}], [0.0, 1.0], n=3, seed=1, prefix="p")
    assert all(w.generator.params["market"] == 2.0 for w in ws.worlds) and all(w.label == "POSTERIOR_SAMPLE" for w in ws)
    save(ws, tmp_path / "w.json")
    again = load(tmp_path / "w.json")
    assert again.digest() == ws.digest() and len(again) == 3
    with pytest.raises(ValueError):
        WorldSet("dup", [ws.worlds[0], ws.worlds[0]])
