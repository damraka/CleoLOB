"""Offline RL, off-policy evaluation and safe policy improvement in synthetic worlds (M81, M82; H26, H27).

Historical public data carry no behaviour propensities, so IPS / SNIPS / DR are NOT_AVAILABLE on real data by
construction (``capability_report``). In calibrated synthetic worlds a registered stochastic behaviour policy logs
its action propensities, which makes OPE estimable there (E1 evidence only).

MDP (discrete actions at every 5 s decision): state = (time fraction left, inventory fraction left, spread, signed
imbalance, log touch depth, recent volatility, behind-schedule gap); actions = {0 wait, 1 rest the TWAP slice
passively, 2 market the TWAP slice, 3 market twice the slice}. Per-step reward = minus the bps cost of the fills
between this decision and the next (gross vs arrival, plus registered fees), with the terminal residual valued as in
the v0.8 report; returns are summed per episode (no discounting).

Learners (all CPU, seeded): ``BC`` (multinomial logistic), ``CQL`` (fitted Q-iteration with the conservative
log-sum-exp penalty), ``IQL`` (expectile value + advantage-weighted behaviour cloning), ``TD3BC`` (discrete-action
adaptation: Q-learning with a behaviour-cloning log-likelihood bonus) and ``DecisionPolicy`` (return-conditioned
classifier, a Decision-Transformer-style policy without attention; registered as the sequence-policy variant).
OPE: per-decision importance sampling (PDIS), weighted PDIS and per-decision doubly robust with an FQE critic.
Support: fraction of evaluation states within the behaviour data's state range and the OOD action rate (actions the
behaviour policy gave probability < 0.05). Safe improvement: deploy a candidate only if the bootstrap 95 % lower
bound of its DR-estimated improvement over the best classical baseline exceeds 0 (else fall back).
"""
from __future__ import annotations

import math

import numpy as np

from ..v08.execution.policies import Policy
from ..v08.execution.simulator import Observation, cancel_all, limit, market

N_ACTIONS = 4
STATE = ("time_left", "inventory_left", "spread", "imbalance_signed", "log_touch", "vol", "behind")


def state_of(obs: Observation) -> np.ndarray:
    mids = np.asarray(obs.recent_mids[-100:], float)
    vol = float(np.std(np.diff(mids))) if len(mids) > 3 else 0.0
    tau = obs.step / obs.steps
    left = obs.remaining / max(obs.target, 1)
    return np.asarray([1 - tau, left, float(obs.spread), obs.side * float(obs.context.get("imbalance5", 0.0)),
                       math.log1p(float(obs.touch_depth())), vol, tau - (1 - left)])


def act(obs: Observation, a: int) -> list:
    out = [cancel_all()] if obs.resting else []
    left = obs.remaining
    if left <= 0:
        return out
    slice_ = max(1, int(math.ceil(left / max(1, obs.steps - obs.step))))
    if obs.step == obs.steps - 1:
        return out + [market(left)]
    if a == 1:
        return out + [limit(min(slice_, left))]
    if a == 2:
        return out + [market(min(slice_, left))]
    if a == 3:
        return out + [market(min(2 * slice_, left))]
    return out


class Logged(Policy):
    """Stochastic behaviour policy (softmax over a registered heuristic score) that logs (s, a, propensity)."""
    name = "behaviour"
    defaults = {"temperature": 1.0, "seed": 0}

    def reset(self, ctx):
        super().reset(ctx)
        self.rng = np.random.default_rng(np.random.SeedSequence([9, 81, int(self.params["seed"])]))
        self.log: list[tuple[np.ndarray, int, np.ndarray, float]] = []

    def probs(self, s: np.ndarray) -> np.ndarray:
        behind, spread = s[6], s[2]
        score = np.asarray([-0.5 - 3 * behind, 0.5 - 2 * behind - 0.2 * (spread - 1), 3 * behind, 4 * behind - 0.5])
        z = score / self.params["temperature"]
        e = np.exp(z - z.max())
        return 0.7 * e / e.sum() + 0.3 / N_ACTIONS          # epsilon floor keeps every action supported

    def decide(self, obs: Observation) -> list:
        s = state_of(obs)
        p = self.probs(s)
        a = int(self.rng.choice(N_ACTIONS, p=p))
        self.log.append((s, a, p, obs.t))
        return act(obs, a)


class Greedy(Policy):
    """Deterministic policy from a learned action-value or action-probability function ``fn(s) -> scores``."""
    name = "learned"
    defaults = {}

    def __init__(self, fn, label: str) -> None:
        super().__init__()
        self.fn, self.name, self.label = fn, label, label

    def decide(self, obs: Observation) -> list:
        return act(obs, int(np.argmax(self.fn(state_of(obs)))))


def run_logged(generator, qty: int, seed: int, side: int) -> dict:
    """One behaviour episode: per-step (s, a, p, r) with rewards from the fills between decisions."""
    from ..v08.execution.simulator import Fees, Mandate
    from ..v08.execution.simulator import Episode
    from ..v08.worlds import World
    pol = Logged(seed=seed)
    mandate = Mandate(side=side, qty=int(qty), horizon_s=300.0, decision_dt_s=5.0, warmup_s=30.0, completion="hard",
                      fees=Fees(taker_bps=5.5, maker_bps=2.0))
    ep = Episode(World("w", generator, generator.book), pol, mandate, seed, context={"seed": seed})
    rep = ep.run()
    arrival = rep["arrival_mid_ticks"]
    times = [x[3] for x in pol.log] + [math.inf]
    rewards = np.zeros(len(pol.log))
    for f in ep.fills:
        t, price, q, maker = f[0], f[1], f[2], f[3]
        k = max(0, int(np.searchsorted(times, t, side="right")) - 1)
        if k < len(rewards):
            cost = side * (price - arrival) * q / (arrival * qty) * 1e4 + q / qty * (2.0 if maker else 5.5)
            rewards[k] -= cost
    if len(rewards):
        resid = rep.get("hypothetical_residual_cost_bps") or 0.0
        rewards[-1] -= resid + rep.get("completion_penalty_bps", 0.0)
    return {"s": [x[0].tolist() for x in pol.log], "a": [x[1] for x in pol.log],
            "p": [x[2].tolist() for x in pol.log], "r": rewards.tolist(), "cost": rep.get("optimization_cost_bps")}


# ----------------------------------------------------------------------------- learners


def _stack(eps):
    S = np.asarray([s for e in eps for s in e["s"]], float)
    A = np.asarray([a for e in eps for a in e["a"]], int)
    Pm = np.asarray([p for e in eps for p in e["p"]], float)
    R = np.asarray([r for e in eps for r in e["r"]], float)
    done = np.asarray([k == len(e["a"]) - 1 for e in eps for k in range(len(e["a"]))])
    nxt = np.vstack([np.r_[np.asarray(e["s"][1:], float).reshape(-1, len(STATE)), np.zeros((1, len(STATE)))]
                     for e in eps])
    rtg = np.asarray([sum(e["r"][k:]) for e in eps for k in range(len(e["r"]))], float)
    return S, A, Pm, R, nxt, done, rtg


class _Std:
    def __init__(self, S):
        self.mu, self.sd = S.mean(0), np.where(S.std(0) > 0, S.std(0), 1.0)

    def __call__(self, S):
        return (np.asarray(S, float) - self.mu) / self.sd


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def fit_bc(S, A, l2=1e-2, iters=400):
    st = _Std(S)
    Z = np.column_stack([np.ones(len(S)), st(S)])
    W = np.zeros((Z.shape[1], N_ACTIONS))
    Y = np.eye(N_ACTIONS)[A]
    for _ in range(iters):
        W -= 0.5 * (Z.T @ (_softmax(Z @ W) - Y) / len(S) + l2 * W)
    return lambda s: _softmax(np.r_[1.0, st(s[None])[0]] @ W)


def _features(st, S):
    Z = st(S)
    return np.column_stack([np.ones(len(Z)), Z, Z ** 2])


def fit_q(S, A, R, nxt, done, *, kind: str, bc=None, iters: int = 30, alpha: float = 1.0, tau: float = 0.7,
          lam: float = 1.0):
    """Linear-in-features fitted Q-iteration variants (CQL, IQL, TD3+BC, FQE-greedy)."""
    st = _Std(S)
    F, Fn = _features(st, S), _features(st, nxt)
    d = F.shape[1]
    W = np.zeros((N_ACTIONS, d))
    V = np.zeros(d)
    for _ in range(iters):
        q_next = Fn @ W.T
        if kind == "iql":
            target = R + np.where(done, 0.0, Fn @ V)
        elif kind == "td3bc" and bc is not None:
            logp = np.log(np.maximum(np.vstack([bc(s) for s in nxt]), 1e-6))
            target = R + np.where(done, 0.0, (q_next + lam * logp).max(1))
        else:
            target = R + np.where(done, 0.0, q_next.max(1))
        for a in range(N_ACTIONS):
            m = A == a
            if m.sum() < d:
                continue
            reg = 1e-2 * np.eye(d)
            W[a] = np.linalg.solve(F[m].T @ F[m] + reg, F[m].T @ target[m])
        if kind == "cql":
            q = F @ W.T
            lse = np.log(np.exp(q - q.max(1, keepdims=True)).sum(1)) + q.max(1)
            for a in range(N_ACTIONS):
                m = A == a
                if m.sum() >= d:
                    grad = F[m].T @ (np.exp(q[m, a] - lse[m]) - 1.0) / m.sum()
                    W[a] -= alpha * 1e-2 * grad
        if kind == "iql":
            qa = (F @ W.T)[np.arange(len(A)), A]
            for _ in range(5):
                u = qa - F @ V
                w = np.where(u > 0, tau, 1 - tau)
                V = np.linalg.solve((F * w[:, None]).T @ F + 1e-2 * np.eye(d), (F * w[:, None]).T @ qa)
    if kind == "iql":
        adv = (F @ W.T)[np.arange(len(A)), A] - F @ V
        wts = np.minimum(np.exp(3.0 * adv), 100.0)
        Z = np.column_stack([np.ones(len(S)), st(S)])
        Wp = np.zeros((Z.shape[1], N_ACTIONS))
        Y = np.eye(N_ACTIONS)[A]
        for _ in range(300):
            Wp -= 0.5 * (Z.T @ ((_softmax(Z @ Wp) - Y) * wts[:, None]) / wts.sum() + 1e-2 * Wp)
        return lambda s: _softmax(np.r_[1.0, st(s[None])[0]] @ Wp)
    return lambda s: _features(st, s[None])[0] @ W.T


def fit_decision(S, A, rtg, target_quantile: float = 0.9):
    """Return-conditioned policy: classifier on (state, return-to-go), deployed with a high registered target."""
    X = np.column_stack([S, rtg])
    bc = fit_bc(X, A)
    target = float(np.quantile(rtg, target_quantile))
    return lambda s: bc(np.r_[s, target])


def learners(eps: list[dict]) -> dict:
    S, A, Pm, R, nxt, done, rtg = _stack(eps)
    bc = fit_bc(S, A)
    return {"bc": bc, "cql": fit_q(S, A, R, nxt, done, kind="cql"), "iql": fit_q(S, A, R, nxt, done, kind="iql"),
            "td3bc": fit_q(S, A, R, nxt, done, kind="td3bc", bc=bc), "decision_transformer": fit_decision(S, A, rtg)}


# ----------------------------------------------------------------------------- OPE


def pi_probs(fn, s) -> np.ndarray:
    """Deterministic greedy target policy as a probability vector (argmax of the score)."""
    p = np.zeros(N_ACTIONS)
    p[int(np.argmax(fn(np.asarray(s, float))))] = 1.0
    return p


def ope(eps: list[dict], fn, fqe=None) -> dict:
    """PDIS, WPDIS and per-decision DR estimates of the target policy's episode return (higher = better)."""
    pdis, wpd_num, wpd_den, dr = [], [], [], []
    ood = []
    for e in eps:
        w, total, dr_ep = 1.0, 0.0, 0.0
        for k, (s, a, p, r) in enumerate(zip(e["s"], e["a"], e["p"], e["r"])):
            pt = pi_probs(fn, s)
            ood.append(p[int(np.argmax(pt))] < 0.05)
            w_prev = w
            w *= pt[a] / max(p[a], 1e-6)
            total += w * r
            if fqe is not None:
                q = fqe(np.asarray(s, float))
                v = float((pt * q).sum())
                dr_ep += w * (r - q[a]) + w_prev * v
        pdis.append(total)
        wpd_num.append(total)
        wpd_den.append(w)
        dr.append(dr_ep if fqe is not None else total)
    wpdis = float(np.sum(wpd_num) / max(np.sum(wpd_den), 1e-12) * 1.0) if np.sum(wpd_den) > 0 else None
    return {"pdis": float(np.mean(pdis)), "wpdis": wpdis, "dr": float(np.mean(dr)), "dr_episodes": dr,
            "ood_action_rate": float(np.mean(ood)) if ood else None}


def support(eps_train: list[dict], states: np.ndarray) -> float:
    S = np.asarray([s for e in eps_train for s in e["s"]], float)
    lo, hi = np.quantile(S, 0.005, 0), np.quantile(S, 0.995, 0)
    return float(((states >= lo) & (states <= hi)).all(1).mean())


def lcb(x: np.ndarray, seed: int, n_boot: int = 2000, level: float = 0.95) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    boots = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(np.quantile(boots, 1 - level))


def capability_report() -> dict:
    return {"real_data": {"IPS": "NOT_AVAILABLE", "SNIPS": "NOT_AVAILABLE", "DR": "NOT_AVAILABLE",
                          "FQE": "ASSUMPTION_DEPENDENT", "model_based": "ASSUMPTION_DEPENDENT (model risk)",
                          "reason": "public archives carry no behaviour-policy propensities"},
            "synthetic": {"IPS": "AVAILABLE", "SNIPS": "AVAILABLE", "DR": "AVAILABLE", "FQE": "AVAILABLE"}}
