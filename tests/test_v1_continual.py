"""v1 continual learning, meta-learning and representation transfer (M127-M129)."""
import numpy as np
import pytest

from lob.v1.continual import methods as CM

torch = pytest.importorskip("torch")


def _task(shift, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4)).astype(np.float32)
    y = ((X[:, 0] + shift * X[:, 1]) > 0).astype(np.int64)
    return X, y


def test_framework_reports_forgetting_and_transfer():
    from lob.v09.representations import sequence as S
    tasks = []
    for k, shift in enumerate((0.0, 2.0, -2.0)):
        rng = np.random.default_rng(k)
        X = rng.standard_normal((800, S.N_FEATURES if hasattr(S, "N_FEATURES") else 6)).astype(np.float32) \
            if False else None
        del X, rng
    # build sequence-model tasks in the v0.9 layout from synthetic frames
    rng = np.random.default_rng(1)
    for shift in (0.0, 1.5):
        n = 600
        X = rng.standard_normal((n, 8)).astype(np.float32)
        y = np.clip(np.round(X[:, 0] + shift * X[:, 1]), -1, 1).astype(np.int64) + 1
        wid = np.repeat(np.arange(n // 60), 60)
        idx = S.windows_of(X, wid, S.HISTORY)
        tasks.append((X, y, idx[wid[idx] % 2 == 0], idx[wid[idx] % 2 == 1]))
    out = CM.compare_continual(tasks, steps=40, methods=("naive", "replay"))
    assert set(out) == {"naive", "replay"} and 0 in out["naive"]["forgetting"] and 1 in out["naive"]["forward_nll"]


def test_meta_learning_reptile_adapts_faster_than_scratch():
    tasks = [_task(s, seed=k) for k, s in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0))]
    meta = CM.Reptile(4, 2, seed=0).meta_fit(tasks, iterations=150)
    res = CM.few_shot_vs_refit(meta, _task(1.5, seed=9), shots=16)
    assert res["adapted_nll"] < res["scratch_nll"]


def test_representation_transfer_and_planning():
    rng = np.random.default_rng(2)
    Zt = rng.standard_normal((2000, 2))
    W = rng.standard_normal((2, 10))
    X = Zt @ W + 0.05 * rng.standard_normal((2000, 10))
    rep = CM.Representation(10, latent=2, steps=400).fit(X[:1500])
    y = Zt[:, 0] * 2 + 0.1 * rng.standard_normal(2000)
    pr = CM.transfer_probe(rep, X[1500:], y[1500:], train_frac=0.1)
    assert pr["latent_probe_mse"] < 1.0
    Z = rep.encode(X[:500])
    A = rng.integers(0, 2, 500)
    Zn = Z + np.where(A[:, None] == 1, 0.5, -0.5)
    dyn = CM.fit_latent_dynamics(Z, A, Zn, 2)
    assert CM.plan(Z[0], dyn, lambda z: float(z.sum())) == 0              # action 0 lowers the latent sum
