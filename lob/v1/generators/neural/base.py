"""Shared machinery of the v1 neural generators (M80-M89): seeding, training loop, persistence, devices.

Every neural model is a ``NeuralModel`` with ``fit``, ``nll`` (held-out scoring, nats per item), ``sample`` and
``save`` / ``load`` (a ``.pt`` state dict plus a JSON config with the class name, hyper-parameters, training data
fingerprint and training history). Training is seeded (``torch.manual_seed`` and a seeded batch order), uses Adam with
gradient clipping and early stopping on a validation split taken from the *training* data only, and runs on the CPU
unless a device is requested. torch is an optional dependency (``pip install cleolob[neural]``).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def torch():
    import torch as _t
    return _t


def fingerprint(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


class NeuralModel:
    name = "neural"
    registry: dict[str, type] = {}

    def __init_subclass__(cls, **kw) -> None:
        super().__init_subclass__(**kw)
        NeuralModel.registry[cls.name] = cls

    def __init__(self, *, seed: int = 0, device: str = "cpu", **hp) -> None:
        self.seed, self.device, self.hp = int(seed), device, dict(hp)
        self.net = None
        self.history: list[dict] = []
        self.data_fingerprint: str | None = None

    # ------------------------------------------------------------------ training
    def _train(self, loss_fn, n_items: int, *, steps: int, batch: int, lr: float, val_frac: float = 0.15,
               patience: int = 8, eval_every: int = 50) -> None:
        T = torch()
        T.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n_items)
        n_val = max(1, int(round(val_frac * n_items))) if n_items > 10 else 0
        val, tr = idx[:n_val], idx[n_val:]
        opt = T.optim.Adam(self.net.parameters(), lr=lr)
        best, best_state, bad = math.inf, None, 0
        for step in range(1, steps + 1):
            b = tr[rng.integers(0, len(tr), size=min(batch, len(tr)))]
            self.net.train()
            loss = loss_fn(b)
            opt.zero_grad()
            loss.backward()
            T.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            opt.step()
            if n_val and (step % eval_every == 0 or step == steps):
                self.net.eval()
                with T.no_grad():
                    v = float(loss_fn(val))
                self.history.append({"step": step, "train": float(loss.detach()), "val": v})
                if v < best - 1e-6:
                    best, bad = v, 0
                    best_state = {k: t.detach().clone() for k, t in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()

    # ------------------------------------------------------------------ persistence
    def config(self) -> dict:
        return {"class": self.name, "seed": self.seed, "hp": self.hp, "fingerprint": self.data_fingerprint,
                "history": self.history[-20:], "extra": self.extra_state()}

    def extra_state(self) -> dict:
        return {}

    def load_extra(self, extra: dict) -> None:
        pass

    def save(self, path: Path) -> None:
        T = torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        T.save(self.net.state_dict(), path.with_suffix(".pt"))
        path.with_suffix(".json").write_text(json.dumps(self.config(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "NeuralModel":
        T = torch()
        path = Path(path)
        cfg = json.loads(path.with_suffix(".json").read_text())
        model_cls = NeuralModel.registry[cfg["class"]]
        m = model_cls(seed=cfg["seed"], **cfg["hp"])
        m.load_extra(cfg.get("extra", {}))
        m.build()
        m.net.load_state_dict(T.load(path.with_suffix(".pt"), weights_only=True))
        m.net.eval()
        m.history, m.data_fingerprint = cfg.get("history", []), cfg.get("fingerprint")
        return m

    def build(self) -> None:
        raise NotImplementedError


def mlp(sizes: list[int], act: str = "silu"):
    T = torch()
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [T.nn.Linear(a, b), T.nn.SiLU() if act == "silu" else T.nn.Tanh()]
    return T.nn.Sequential(*layers[:-1])


__all__ = ["NeuralModel", "mlp", "fingerprint", "torch"]
