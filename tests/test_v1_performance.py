"""v1 performance architecture: reference core, accelerated kernel, compatibility, columnar, mmap replay, batch MC,
GPU and CPU fallback, benchmark suite (M40-M47, M49)."""
import numpy as np
import pytest

from lob.v1.data import columnar
from lob.v1.engines import accelerated as A, batch as B, benchmark as BM, gpu as G, replay as RP
from lob.v1.engines.matching import BUY, MatchingEngine
from lob.v1.engines.reference import ReferenceEngine

PARAMS = A.KernelParams(levels=12, width=1024, improve_bp=(2500, 4500, 6500), refill_mean=6)


def _inputs(paths=6, events=1500, seed=3):
    return A.draw_inputs(paths, events, seed=seed, refill_mean=PARAMS.refill_mean)


def test_reference_core_is_typed_and_benchmarked():
    r = BM.matching_throughput(1_500)
    assert r["v2"]["commands_per_s"] > 0 and r["reference"]["commands"] == 1_500
    e, ref = MatchingEngine(), ReferenceEngine()
    for o in BM._commands(800, 5):
        e.submit(o)
        ref.submit(o)
    assert e.depth(BUY, 30) == ref.depth(BUY, 30)


@pytest.mark.skipif(not A.HAVE_NUMBA, reason="numba not installed (optional accel extra)")
def test_accelerated_numba_kernel_runs_and_flags_overflow():
    out = A.simulate(PARAMS, _inputs(), backend="numba")
    assert out["ok"].all() and (out["ask"] - out["bid"] >= 1).all()
    tiny = A.KernelParams(levels=12, width=40, refill_mean=6)
    inp = A.draw_inputs(4, 3000, seed=1, mark_probs=(0.45, 0.05, 0.2, 0.2, 0.05, 0.05), refill_mean=6)
    o2 = A.simulate(tiny, inp, backend="numba")
    assert not o2["ok"].all()                                       # the touch reached the window edge


@pytest.mark.skipif(not A.HAVE_NUMBA, reason="numba not installed")
def test_compat_numba_equals_interpreted_kernel_bitwise():
    inp = _inputs(paths=5, events=2000, seed=11)
    a = A.simulate(PARAMS, inp, backend="numba")
    b = A.simulate(PARAMS, inp, backend="python")
    assert G.equal(a, b)
    tiny = A.KernelParams(levels=12, width=40, refill_mean=6)
    inp2 = A.draw_inputs(4, 3000, seed=1, mark_probs=(0.45, 0.05, 0.2, 0.2, 0.05, 0.05), refill_mean=6)
    assert G.equal(A.simulate(tiny, inp2, backend="numba"), A.simulate(tiny, inp2, backend="python"))


def test_compat_improvement_table_changes_spreads():
    inp = _inputs(paths=8, events=3000, seed=2)
    no_imp = A.summary(A.simulate(A.KernelParams(levels=12, width=1024, refill_mean=6), inp, backend="python"))
    imp = A.summary(A.simulate(PARAMS, inp, backend="python"))
    assert imp["mean_spread"] < no_imp["mean_spread"]


torch = pytest.importorskip("torch") if G.available()["torch"] else None


@pytest.mark.skipif(not G.available()["torch"], reason="torch not installed")
def test_cpu_fallback_torch_cpu_equals_kernel_bitwise():
    inp = _inputs(paths=7, events=1200, seed=5)
    ref = A.simulate(PARAMS, inp, backend="python")
    assert G.equal(G.simulate_torch(PARAMS, inp, device="cpu"), ref)
    tiny = A.KernelParams(levels=12, width=40, refill_mean=6)
    inp2 = A.draw_inputs(4, 3000, seed=1, mark_probs=(0.45, 0.05, 0.2, 0.2, 0.05, 0.05), refill_mean=6)
    assert G.equal(G.simulate_torch(tiny, inp2, device="cpu"), A.simulate(tiny, inp2, backend="python"))


@pytest.mark.skipif(not G.available()["cuda"], reason="no CUDA device / CUDA build of torch")
def test_gpu_cuda_equals_kernel_bitwise():
    inp = _inputs(paths=64, events=1000, seed=9)
    ref = A.simulate(PARAMS, inp, backend="numba" if A.HAVE_NUMBA else "python")
    assert G.equal(G.simulate_torch(PARAMS, inp, device="cuda"), ref)


def test_batch_schedule_is_deterministic_and_order_independent():
    sc = {"tight": PARAMS, "loose": A.KernelParams(levels=12, width=1024, refill_mean=6)}
    plan = B.plan(sc, 10, batch_size=4)
    assert [b.id for b in plan] == [b.id for b in B.plan(sc, 10, batch_size=4)]
    backend = "numba" if A.HAVE_NUMBA else "python"
    r1 = B.run(plan, sc, 400, backend=backend)
    r2 = B.run(list(reversed(B.plan(sc, 10, batch_size=3))), sc, 400, backend=backend)
    assert r1 == r2                                        # batch size and order do not change any episode
    red = B.reduce(r1)
    assert red["tight"]["mean_spread"] < red["loose"]["mean_spread"]


def test_columnar_roundtrip_and_mmap_window(tmp_path):
    n = 5000
    rng = np.random.default_rng(0)
    frames = {"bid_px": np.arange(n, dtype=np.int64), "ask_px": np.arange(n, dtype=np.int64) + 1,
              "valid": rng.random(n) > 0.1, "depth": rng.random((n, 2, 10)).astype(np.float32),
              "counts": rng.integers(0, 5, (n, 6)).astype(np.int32), "volume": rng.random((n, 2)).astype(np.float32),
              "mid": rng.random(n + 1), "counters": {"messages": 3}}
    columnar.write_frames(frames, tmp_path / "f.arrow", meta={"day": "x"})
    f = columnar.open_frames(tmp_path / "f.arrow")
    for k, v in frames.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(f[k], v) and f[k].dtype == v.dtype
    assert f.extra["counters"] == {"messages": 3} and f.meta["meta"] == {"day": "x"}
    w = f.window(100, 200)
    assert len(w["valid"]) == 100 and len(w["mid"]) == 101 and np.array_equal(w["bid_px"], frames["bid_px"][100:200])


def test_mmap_stream_windows_and_event_frames(tmp_path):
    n = 864_000 // 24
    frames = {"bid_px": np.full(n, 1000, np.int64), "ask_px": np.full(n, 1001, np.int64), "valid": np.ones(n, bool),
              "depth": np.ones((n, 2, 10), np.float32), "counts": np.zeros((n, 6), np.int32),
              "volume": np.zeros((n, 2), np.float32), "mid": np.full(n + 1, 1000.5)}
    columnar.write_frames(frames, tmp_path / "f.arrow")
    got = list(RP.stream_windows(tmp_path / "f.arrow", starts=(600,)))
    s, w = got[0]
    a, b = RP.window_bins(600)
    assert s == 600 and len(w["valid"]) == b - a
    ef = RP.to_event_frame(w, tick=0.1)
    assert len(ef) == b - a and ef.state.shape == (b - a, 6)


def test_benchmark_suite_quick(tmp_path):
    rep = BM.run(tmp_path, quick=True)
    assert rep["matching"]["v2"]["commands_per_s"] > 0
    assert rep["columnar"]["file_bytes"] > 0 and rep["kernel"]["python"]["events_per_s"] > 0
    if A.HAVE_NUMBA:
        assert rep["kernel"]["python"]["equal_to_numba"] is True
    for k, v in rep["kernel"].items():
        if k.startswith("torch") and v.get("equal_to_cpu_kernel") is not None:
            assert v["equal_to_cpu_kernel"] is True
