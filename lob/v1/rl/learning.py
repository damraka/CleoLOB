"""Offline learning (M121-M126): behaviour cloning, CQL, IQL, a Decision Transformer, OPE with support diagnostics,
and conservative policy-improvement gating.

* Data: logged episodes from the registered stochastic behaviour policy of v0.9 (``lob.v09.rl.Logged``), which
  records the propensity of every action it took. ``collect`` runs it in F4 worlds (corrected book) - the only
  place where propensities exist. Real public data carry no propensities, so ``ope`` refuses episodes without them
  (``PropensityMissing``): an invented behaviour policy never justifies an off-policy estimate.
* Learners: BC (multinomial logistic), CQL (fitted Q-iteration with the conservative log-sum-exp penalty), IQL
  (expectile value + advantage-weighted BC) - the v0.9 implementations reused unchanged - and ``DecisionTransformer``
  (new in v1): a causal Transformer over (return-to-go, state, previous action) tokens predicting the action, trained
  by cross-entropy, deployed with a registered high return-to-go target (Chen et al. 2021).
* OPE: per-decision IS, weighted PDIS and per-decision doubly robust with an FQE critic (v0.9), plus support
  diagnostics: state coverage, OOD action rate and the effective sample size of the importance weights.
* Gating: deploy a learned policy only if the bootstrap 95% lower bound of its DR improvement over the baseline
  exceeds the registered margin and support passes; otherwise fall back to the baseline.
"""
from __future__ import annotations

import numpy as np

from ...v09 import rl as R9
from ..generators.neural.base import NeuralModel, torch


class PropensityMissing(ValueError):
    pass


def collect(generator, *, episodes: int, qty: int, seed: int = 0) -> list[dict]:
    from ..twins.classical import corrected_market
    out = []
    with corrected_market():
        for k in range(episodes):
            out.append(R9.run_logged(generator, qty, seed + k, 1 if k % 2 == 0 else -1))
    return out


def learners(eps: list[dict]) -> dict:
    S, A, Pm, R, nxt, done, rtg = R9._stack(eps)
    bc = R9.fit_bc(S, A)
    return {"bc": bc, "cql": R9.fit_q(S, A, R, nxt, done, kind="cql"),
            "iql": R9.fit_q(S, A, R, nxt, done, kind="iql")}


class DecisionTransformer(NeuralModel):
    name = "decision_transformer"

    def __init__(self, *, context: int = 8, hidden: int = 32, heads: int = 2, layers: int = 2, steps: int = 600,
                 batch: int = 64, lr: float = 2e-3, target_quantile: float = 0.9, seed: int = 0,
                 device: str = "cpu") -> None:
        super().__init__(seed=seed, device=device, context=context, hidden=hidden, heads=heads, layers=layers,
                         steps=steps, batch=batch, lr=lr, target_quantile=target_quantile)
        self.target = 0.0

    def extra_state(self):
        return {"target": self.target, "mu": self.mu.tolist(), "sd": self.sd.tolist(), "rscale": self.rscale}

    def load_extra(self, e):
        self.target, self.mu, self.sd = e["target"], np.asarray(e["mu"]), np.asarray(e["sd"])
        self.rscale = e["rscale"]

    def build(self):
        T = torch()
        H, d = self.hp["hidden"], len(R9.STATE)

        class Net(T.nn.Module):
            def __init__(self, ctx, layers, heads):
                super().__init__()
                self.inp = T.nn.Linear(d + 1 + R9.N_ACTIONS + 1, H)
                self.pos = T.nn.Embedding(ctx, H)
                layer = T.nn.TransformerEncoderLayer(H, heads, 2 * H, dropout=0.0, batch_first=True)
                self.enc = T.nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
                self.out = T.nn.Linear(H, R9.N_ACTIONS)

            def forward(self, x):
                n = x.shape[1]
                h = self.inp(x) + self.pos(T.arange(n))[None]
                mask = T.triu(T.full((n, n), float("-inf")), 1)
                return self.out(self.enc(h, mask=mask))

        self.net = Net(self.hp["context"], self.hp["layers"], self.hp["heads"])

    def _tokens(self, states, prev_actions, rtg, steps):
        oh = np.zeros((len(prev_actions), R9.N_ACTIONS + 1), np.float32)
        oh[np.arange(len(prev_actions)), np.asarray(prev_actions) + 1] = 1.0      # index 0 = no previous action
        z = (np.asarray(states, float) - self.mu) / self.sd
        return np.column_stack([z, np.asarray(rtg)[:, None] / self.rscale, oh]).astype(np.float32)

    def fit(self, eps: list[dict]):
        T = torch()
        S = np.asarray([s for e in eps for s in e["s"]], float)
        self.mu, self.sd = S.mean(0), S.std(0) + 1e-6
        rtgs = [np.cumsum(np.asarray(e["r"])[::-1])[::-1] for e in eps]
        self.rscale = float(np.abs(np.concatenate(rtgs)).mean() + 1e-6)
        self.target = float(np.quantile([r[0] for r in rtgs], self.hp["target_quantile"]))
        C = self.hp["context"]
        X, Y = [], []
        for e, rtg in zip(eps, rtgs):
            prev = [-1] + list(e["a"][:-1])
            tok = self._tokens(e["s"], prev, rtg, None)
            for k in range(len(e["a"])):
                a = max(0, k - C + 1)
                seq = tok[a:k + 1]
                pad = np.zeros((C - len(seq), tok.shape[1]), np.float32)
                X.append(np.vstack([pad, seq]))
                Y.append(e["a"][k])
        X, Y = T.as_tensor(np.stack(X)), T.as_tensor(np.asarray(Y))
        self.build()
        ce = T.nn.CrossEntropyLoss()
        self._train(lambda b: ce(self.net(X[b])[:, -1], Y[b]), len(X), steps=self.hp["steps"],
                    batch=self.hp["batch"], lr=self.hp["lr"])
        return self

    def policy(self):
        """Score function s -> action logits with the registered return target (history-free deployment)."""
        T = torch()

        def fn(s):
            tok = self._tokens([s], [-1], [self.target], None)
            pad = np.zeros((self.hp["context"] - 1, tok.shape[1]), np.float32)
            with T.no_grad():
                return self.net(T.as_tensor(np.vstack([pad, tok])[None]))[0, -1].numpy()
        return fn


def ope(eps: list[dict], fn, fqe=None) -> dict:
    for e in eps:
        if "p" not in e or e["p"] is None or len(e["p"]) != len(e["a"]):
            raise PropensityMissing("off-policy evaluation needs the logged behaviour propensities of every action")
    res = R9.ope(eps, fn, fqe)
    w = []
    for e in eps:
        wt = 1.0
        for s, a, p in zip(e["s"], e["a"], e["p"]):
            wt *= R9.pi_probs(fn, s)[a] / max(p[a], 1e-6)
        w.append(wt)
    w = np.asarray(w)
    ess = float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))
    return {**res, "effective_sample_size": ess, "episodes": len(eps)}


def gate(candidate_dr: np.ndarray, baseline_returns: np.ndarray, *, margin: float = 0.0, support: float = 1.0,
         min_support: float = 0.9, ood_rate: float | None = None, max_ood: float = 0.1, ess: float | None = None,
         min_ess: float = 10.0, seed: int = 0) -> dict:
    """Conservative improvement: deploy iff LCB95(candidate - baseline) > margin and every support check passes
    (state coverage, OOD action rate and - when given - the trajectory importance-weight effective sample size)."""
    diff = np.asarray(candidate_dr, float) - float(np.mean(baseline_returns))
    lo = R9.lcb(diff, seed)
    ok_support = support >= min_support and (ood_rate is None or ood_rate <= max_ood) and         (ess is None or ess >= min_ess)
    deploy = bool(lo > margin and ok_support)
    return {"deploy": deploy, "lcb95_improvement": lo, "support_ok": ok_support,
            "decision": "deploy candidate" if deploy else "fall back to baseline"}


__all__ = ["collect", "learners", "DecisionTransformer", "ope", "gate", "PropensityMissing"]
