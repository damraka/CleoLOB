"""Neural hybrid replay response model (M88) and validity/uncertainty/failure envelopes (M89).

``NeuralResponse``: a seed ensemble of MLPs predicting the six registered post-burst response targets of the v0.9
hybrid twin (signed mid move and hit-side depth recovery at +1, +5, +30 s; ``lob.v09.generators.hybrid``) from the
pre-burst state features. Its error is scored with the v0.9 ``response_error`` definition (squared error
standardized by the training targets' SD), so it is directly comparable with the v0.9 boosted ``HybridResponse``;
the ensemble spread is the predictive uncertainty.

``envelope`` (M89) states where a fitted model may be used. Checks, each PASS / WARN / FAIL:

* ``input_domain`` - share of evaluation inputs outside the per-feature [0.5%, 99.5%] training range and
  the 99th-percentile Mahalanobis distance ratio;
* ``generalization`` - held-out minus validation loss relative to the validation loss;
* ``ensemble_disagreement`` - mean ensemble SD relative to the target SD (when an ensemble exists);
* ``feedback_stability`` - for simulators: whether a closed-loop simulation's statistics stay within a registered
  factor of the training statistics (the recurrent count model's runaway spread fails this check).

The overall status is the worst check; FAIL means the model must not support any claim needing that capability.
"""
from __future__ import annotations

import numpy as np

from .base import NeuralModel, fingerprint, mlp, torch


class NeuralResponse(NeuralModel):
    name = "neural_response"

    def __init__(self, *, hidden: int = 64, members: int = 5, steps: int = 1500, batch: int = 256, lr: float = 2e-3,
                 seed: int = 0, device: str = "cpu", n_in: int = 6, n_out: int = 6) -> None:
        super().__init__(seed=seed, device=device, hidden=hidden, members=members, steps=steps, batch=batch, lr=lr,
                         n_in=n_in, n_out=n_out)

    def extra_state(self):
        return {k: getattr(self, k).tolist() for k in ("mx", "sx", "my", "sy")}

    def load_extra(self, e):
        for k in ("mx", "sx", "my", "sy"):
            setattr(self, k, np.asarray(e[k], np.float32))

    def build(self):
        T = torch()
        h = self.hp["hidden"]
        self.net = T.nn.ModuleList([mlp([self.hp["n_in"], h, h, self.hp["n_out"]])
                                    for _ in range(self.hp["members"])])

    def fit(self, X: np.ndarray, Y: np.ndarray):
        T = torch()
        self.hp["n_in"], self.hp["n_out"] = int(X.shape[1]), int(Y.shape[1])
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-6
        self.my, self.sy = Y.mean(0), Y.std(0) + 1e-6
        Xs = T.as_tensor(((X - self.mx) / self.sx).astype(np.float32))
        Ys = T.as_tensor(((Y - self.my) / self.sy).astype(np.float32))
        self.data_fingerprint = fingerprint(X, Y)
        self.build()
        rng = np.random.default_rng(self.seed)
        boots = [rng.integers(0, len(Xs), len(Xs)) for _ in range(self.hp["members"])]

        def loss(b):
            return sum(((m(Xs[bt[b % len(bt)]]) - Ys[bt[b % len(bt)]]) ** 2).mean()
                       for m, bt in zip(self.net, boots)) / len(self.net)

        self._train(loss, len(Xs), steps=self.hp["steps"], batch=self.hp["batch"], lr=self.hp["lr"])
        return self

    def predict(self, X: np.ndarray, *, members: bool = False):
        T = torch()
        Xs = T.as_tensor(((X - self.mx) / self.sx).astype(np.float32))
        with T.no_grad():
            P = np.stack([m(Xs).numpy() for m in self.net]) * self.sy + self.my
        return P if members else P.mean(0)

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X, members=True).std(0)


def response_error(pred: np.ndarray, Y: np.ndarray, sd_train: np.ndarray) -> float:
    """v0.9 definition: mean over bursts and targets of the squared error standardized by the training SD."""
    return float((((pred - Y) / np.maximum(sd_train, 1e-12)) ** 2).mean())


def _status(value, warn, fail, *, higher_is_worse=True):
    if value is None or not np.isfinite(value):
        return "WARN"
    if higher_is_worse:
        return "FAIL" if value >= fail else "WARN" if value >= warn else "PASS"
    return "FAIL" if value <= fail else "WARN" if value <= warn else "PASS"


def envelope(*, train_inputs: np.ndarray, eval_inputs: np.ndarray, val_loss: float | None = None,
             test_loss: float | None = None, ensemble_sd: np.ndarray | None = None, target_sd: np.ndarray | None = None,
             closed_loop: dict | None = None, training_stats: dict | None = None, factor: float = 3.0) -> dict:
    lo, hi = np.percentile(train_inputs, [0.5, 99.5], axis=0)
    outside = float(((eval_inputs < lo) | (eval_inputs > hi)).any(axis=1).mean())
    mu, cov = train_inputs.mean(0), np.cov(train_inputs.T) + 1e-9 * np.eye(train_inputs.shape[1])
    inv = np.linalg.inv(cov)

    def maha(x):
        d = x - mu
        return np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))
    ratio = float(np.percentile(maha(eval_inputs), 99) / max(np.percentile(maha(train_inputs), 99), 1e-12))
    checks = {"input_domain": {"outside_fraction": outside, "mahalanobis_p99_ratio": ratio,
                               "status": max(_status(outside, 0.05, 0.25), _status(ratio, 1.5, 3.0),
                                             key=["PASS", "WARN", "FAIL"].index)}}
    if val_loss is not None and test_loss is not None:
        gap = (test_loss - val_loss) / max(abs(val_loss), 1e-12)
        checks["generalization"] = {"relative_gap": gap, "status": _status(gap, 0.1, 0.5)}
    if ensemble_sd is not None and target_sd is not None:
        r = float(np.mean(ensemble_sd / np.maximum(target_sd, 1e-12)))
        checks["ensemble_disagreement"] = {"relative_sd": r, "status": _status(r, 0.25, 0.75)}
    if closed_loop is not None and training_stats is not None:
        worst, detail = 0.0, {}
        for k, v in closed_loop.items():
            ref = training_stats.get(k)
            if ref is None:
                continue
            ratio_k = max(abs(v), 1e-12) / max(abs(ref), 1e-12)
            ratio_k = max(ratio_k, 1 / ratio_k)
            detail[k] = {"closed_loop": v, "training": ref, "ratio": ratio_k}
            worst = max(worst, ratio_k)
        checks["feedback_stability"] = {"worst_ratio": worst, "detail": detail,
                                        "status": _status(worst, factor / 1.5, factor)}
    order = ["PASS", "WARN", "FAIL"]
    overall = max((c["status"] for c in checks.values()), key=order.index)
    return {"checks": checks, "status": overall,
            "rule": "FAIL in any check means the model supports no claim needing that capability"}


__all__ = ["NeuralResponse", "response_error", "envelope"]
