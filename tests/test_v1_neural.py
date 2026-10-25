"""v1 neural generative models (M80-M89): real training, inference, persistence and held-out evaluation."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lob.v08.generators import PoissonGenerator  # noqa: E402
from lob.v08.market.sim import BookModel, simulate  # noqa: E402
from lob.v1.generators.classical import book as B  # noqa: E402
from lob.v1.generators.neural import counts as NC, dynamics as DY, flows as FL, response as RS, tpp as TP  # noqa: E402
from lob.v1.generators.neural.base import NeuralModel  # noqa: E402
from lob.v1.generators.point_process.hawkes import UnmarkedHawkes  # noqa: E402


@pytest.fixture(scope="module")
def hawkes_data():
    g = UnmarkedHawkes()
    g.mu, g.a, g.beta, g.p = 1.0, 0.5, 3.0, np.array([0.3, 0.1, 0.1, 0.3, 0.1, 0.1])
    t, m = g.simulate(1500, 4)
    h = len(t) // 2
    return [(t[:h], m[:h])], [(t[h:], m[h:])], len(t) / 1500


@pytest.fixture(scope="module")
def frames():
    base = BookModel(refill_mean=6.0, limit_size_mean=3.0, market_size_mean=2.0, mid_ticks=4000)
    g = PoissonGenerator({**PoissonGenerator.default_params(), "market": 1.2, "add": 4.0, "cancel": 3.0}, base)
    return [simulate(g, base, 900, seed=s, warmup_bins=100) for s in range(4)], base


def test_tpp_neural_fits_scores_and_samples(hawkes_data):
    train, test, rate = hawkes_data
    m = TP.RMTPP(steps=150, seed=1).fit(train)
    nll = float(m.nll(test).mean())
    assert np.isfinite(nll) and nll < float(UnmarkedHawkes().fit(train).nll(test).mean()) + 0.2
    ts, ms = m.sample(200.0, 3)
    assert abs(len(ts) / 200 - rate) / rate < 0.35 and set(ms) <= set(range(6))
    assert np.all(np.diff(ts) > 0)
    a, w, c = 1.0, -0.5, 0.2
    d = TP.invert_compensator(a, w, c, 0.7)
    assert abs((np.exp(a + w * d) - np.exp(a)) / w + c * d - 0.7) < 1e-6


def test_transformer_event_model_samples(hawkes_data):
    train, test, _ = hawkes_data
    m = TP.EventTransformer(steps=100, seed=2).fit(train)
    ts, _ = m.sample(50.0, 1)
    assert len(ts) > 10 and np.isfinite(m.nll(test).mean())


def test_autoregressive_counts_beat_poisson_and_persist(frames, tmp_path):
    fr, base = frames
    m = NC.AutoregressiveCounts(steps=150, history=16, seed=1).fit(fr[:3])
    nll = NC.mean_nll(m, fr[3:])
    pois = PoissonGenerator().fit(fr[0])
    p_nll = -np.nanmean(pois.loglik(fr[3]))
    assert np.isfinite(nll) and nll < p_nll + 0.5            # learned counts are competitive with the truth family
    m.save(tmp_path / "ar")
    m2 = NeuralModel.load(tmp_path / "ar")
    assert type(m2).__name__ == "AutoregressiveCounts" and abs(NC.mean_nll(m2, fr[3:]) - nll) < 1e-9


def test_recurrent_counts_drive_a_book_reproducibly(frames):
    fr, base = frames
    m = NC.RecurrentCounts(steps=100, history=8, seed=3).fit(fr[:2])
    v1 = B.BookModelV1(**base.to_dict(), improve_by_spread=((2, 0.6), (3, 0.6), (4, 0.6)))
    m.book = v1
    a = B.simulate_v1(m, v1, 200, 7, warmup_bins=20)
    b = B.simulate_v1(m, v1, 200, 7, warmup_bins=20)
    assert np.array_equal(a.counts, b.counts) and np.array_equal(a.bid_px, b.bid_px)


def test_state_space_trains_and_samples():
    rng = np.random.default_rng(0)
    z = np.cumsum(rng.standard_normal((900, 1)) * 0.1, axis=0)
    x = np.hstack([z + 0.1 * rng.standard_normal((900, 1)) for _ in range(5)]).astype(np.float32)
    w = [x[i:i + 300] for i in range(0, 900, 300)]
    m = DY.StateSpace(steps=120, seq=30, seed=1).fit(w[:2])
    assert np.isfinite(m.nll(w[2:])) and m.sample(50, 1).shape == (50, 5)


def test_latent_sde_recovers_mean_reversion():
    rng = np.random.default_rng(1)
    y = np.zeros((4000, 2), np.float32)
    for t in range(1, 4000):
        y[t] = y[t - 1] - 0.2 * y[t - 1] + 0.3 * rng.standard_normal(2)
    m = DY.NeuralSDE(steps=400, seed=1).fit([y[:3000]])
    assert m.nll([y[3000:]]) < DY.gaussian_ar1_nll([y[3000:]]) + 0.05
    path = m.sample(np.zeros(2, np.float32), 2000, 3)
    assert 0.2 < path.std(0).mean() < 0.8                     # stationary SD 0.3 / sqrt(1 - 0.64) = 0.5


def test_flow_and_diffusion_conditional_densities():
    rng = np.random.default_rng(2)
    c = rng.standard_normal((6000, 4)).astype(np.float32)
    x = np.column_stack([c[:, 0] + 0.3 * rng.standard_normal(6000),
                         np.abs(c[:, 1]) + 0.2 * rng.standard_exponential(6000)]).astype(np.float32)
    f = FL.ConditionalFlow(steps=400, seed=1).fit(x[:5000], c[:5000])
    g = FL.GaussianConditional().fit(x[:5000], c[:5000])
    assert f.nll(x[5000:], c[5000:]) < g.nll(x[5000:], c[5000:])          # the non-linear |c1| term needs the flow
    fm = FL.FlowMatching(steps=600, seed=1).fit(x[:5000], c[:5000])
    s = fm.sample(c[5000:], 1)
    assert s.shape == (1000, 2) and FL.energy_distance(s, x[5000:]) < FL.energy_distance(
        rng.standard_normal((1000, 2)) * 3, x[5000:])


def test_response_ensemble_and_envelope_failure_detection():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((2000, 7))
    Y = np.column_stack([X[:, :3].sum(1), X[:, 3] ** 2, np.sin(X[:, 4]), X[:, 5], X[:, 6], X[:, 0] * X[:, 1]])
    m = RS.NeuralResponse(steps=300, members=3, seed=1).fit(X[:1500], Y[:1500])
    err = RS.response_error(m.predict(X[1500:]), Y[1500:], Y[:1500].std(0))
    base = RS.response_error(np.tile(Y[:1500].mean(0), (500, 1)), Y[1500:], Y[:1500].std(0))
    assert err < 0.6 * base and m.uncertainty(X[1500:]).shape == (500, 6)
    ok = RS.envelope(train_inputs=X[:1500], eval_inputs=X[1500:], val_loss=1.0, test_loss=1.02,
                     closed_loop={"spread": 1.1}, training_stats={"spread": 1.0})
    assert ok["status"] == "PASS"
    bad = RS.envelope(train_inputs=X[:1500], eval_inputs=X[1500:] * 6, closed_loop={"spread": 156.0},
                      training_stats={"spread": 1.05})
    assert bad["status"] == "FAIL" and bad["checks"]["feedback_stability"]["status"] == "FAIL"
