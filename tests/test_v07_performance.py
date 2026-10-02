"""v0.7 performance tooling: deterministic parallel equivalence, checkpoint/resume, compute metering,
and differential tests of accelerated paths against reference implementations."""
from __future__ import annotations

import numpy as np
import pytest

from lob.v07.benchmark import compute, parallel


def _sim_task(task):
    from lob.engine import ExchangeSimulator, SimConfig
    seed, seconds = task
    sim = ExchangeSimulator(SimConfig(seed=seed, record_events=False))
    sim.step(seconds)
    return {"seed": seed, "mid": sim.book.mid(), "trades": len(sim.book.trades)}


def test_single_process_and_parallel_results_are_identical(tmp_path) -> None:
    tasks = [(s, 20.0) for s in range(6)]
    single = parallel.run(tasks, _sim_task, workers=1)
    multi = parallel.run(tasks, _sim_task, workers=3)
    assert single["digest"] == multi["digest"] and single["results"] == multi["results"]


def test_checkpoint_resume_and_mismatch_refusal(tmp_path) -> None:
    tasks = [(s, 5.0) for s in range(4)]
    first = parallel.run(tasks[:2], _sim_task, checkpoint=tmp_path)
    resumed = parallel.run(tasks, _sim_task, checkpoint=tmp_path)
    assert first["computed"] == 2 and resumed["resumed"] == 2 and resumed["computed"] == 2
    assert resumed["results"][:2] == first["results"]
    with pytest.raises(ValueError):
        parallel.run([(99, 5.0)] + tasks[1:], _sim_task, checkpoint=tmp_path)


def test_meter_records_environment() -> None:
    with compute.Meter("t", workers=2) as meter:
        meter.add("simulations", 3)
        sum(range(10000))
    record = meter.record()
    assert record["counters"]["simulations"] == 3 and record["wall_s"] >= 0 and record["logical_cpus"]
    assert record["gpu"]["available"] in (True, False)
    assert record["peak_memory_mb_main"] is None or record["peak_memory_mb_main"] > 1


def test_numpy_gru_matches_torch_reference() -> None:
    torch = pytest.importorskip("torch")
    from lob.v07.generators.counts import GRUCounts
    torch.manual_seed(0)
    gru = torch.nn.GRU(9, 8, batch_first=True)
    head = torch.nn.Linear(8, 6)
    sd = {k: v.detach().numpy().astype(float) for k, v in gru.state_dict().items()}
    model = GRUCounts(weights={"w_ih": sd["weight_ih_l0"], "w_hh": sd["weight_hh_l0"], "b_ih": sd["bias_ih_l0"],
                               "b_hh": sd["bias_hh_l0"], "w_out": head.weight.detach().numpy().astype(float),
                               "b_out": head.bias.detach().numpy().astype(float)},
                      hidden=8, guard={"state_min": np.full(3, -1e9), "state_max": np.full(3, 1e9),
                                       "rate_cap": np.full(6, 1e9)})
    rng = np.random.default_rng(1)
    realized = rng.integers(0, 4, (20, 6)).astype(float)
    features = rng.normal(size=(20, 3))
    h = model.initial()
    for r, f in zip(realized, features):
        h = model.update(h, r, f)
    x = torch.tensor(np.hstack([np.log1p(realized), features])[None], dtype=torch.float32)
    out, _ = gru(x)
    np.testing.assert_allclose(h, out[0, -1].detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(model.rates(h), np.minimum(np.exp(head(out[0, -1]).detach().numpy()), 1e9), rtol=1e-4)


def test_precomputed_support_matches_v06_reference() -> None:
    from lob.v06.domain_gap import Standardizer, support
    from lob.v07.realism.metrics import _Support
    rng = np.random.default_rng(2)
    history = rng.normal(0, 1, (80, 5))
    sims = [rng.normal(0.2, 1, (40, 5)) for _ in range(4)]
    scaler = Standardizer(np.vstack(sims))
    reference = 1 - support(history, sims, scaler)["out_of_support_fraction"]
    fast = _Support(history, sims, scaler).coverage(np.arange(4), np.arange(80))
    assert fast == pytest.approx(reference)
