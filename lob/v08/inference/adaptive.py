"""Online and adaptive calibration (M14).

A chronological sequence of windows is scored one step ahead. For window ``t+1``
each scheme supplies a generator built only from windows ``<= t``:

* ``frozen`` — fitted once on the first ``initial`` windows;
* ``rolling`` — refitted on the last ``window`` windows;
* ``expanding`` — refitted on every past window;
* ``forgetting`` — parameters exponentially smoothed: ``theta_t = (1 - rho) theta_{t-1} + rho theta_hat_t``
  (flattened, in log space for positive parameters);
* ``regime`` — one fit per development regime label; the next window is scored by
  the generator of the regime its first block is labelled with (labels use only
  information available at the window start);
* ``change_triggered`` — frozen until a CUSUM on the per-window predictive loss
  alarms, then refitted on the last ``window`` windows.

Loss is the negative predictive log-likelihood per valid bin; ``compare`` returns
per-window losses for every scheme so frozen-versus-adaptive differences can be
tested with paired statistics.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from ..generators.base import MarketGenerator
from ..market.frame import EventFrame
from . import abc  # noqa: F401  (kept for posterior updating hooks)

SCHEMES = ("frozen", "rolling", "expanding", "forgetting", "regime", "change_triggered")


def concat(frames: list[EventFrame]) -> EventFrame:
    """Concatenate frames (validity preserved; the first bin of every later frame is marked invalid so no
    predictive term crosses a window boundary)."""
    parts = []
    for i, f in enumerate(frames):
        v = f.valid.copy()
        if i > 0 and len(v):
            v[0] = False
        parts.append((f, v))
    mids = [parts[0][0].mid[:1]] + [f.mid[1:] for f, _ in parts]
    return EventFrame(np.concatenate([f.counts for f, _ in parts]), np.concatenate([f.state for f, _ in parts]),
                      np.concatenate([f.bid_px for f, _ in parts]), np.concatenate([f.ask_px for f, _ in parts]),
                      np.concatenate([f.depth for f, _ in parts]), np.concatenate([f.volume for f, _ in parts]),
                      np.concatenate(mids), np.concatenate([v for _, v in parts]),
                      samples={k: np.concatenate([np.asarray(f.samples.get(k, []), float) for f, _ in parts])
                               for k in ("market_size", "add_level", "add_size", "cancel_level", "cancel_frac")},
                      meta={"source": "concatenated", "windows": len(frames)})


def loss(generator: MarketGenerator, frame: EventFrame) -> float:
    ll = generator.loglik(frame)
    ok = np.isfinite(ll)
    return float(-ll[ok].mean()) if ok.any() else math.nan


def _smooth(previous: MarketGenerator, new: MarketGenerator, rho: float) -> MarketGenerator:
    a, b = previous.get_flat(), new.get_flat()
    if not a:
        return new
    theta = {}
    for k in a:
        x, y = a[k], b[k]
        theta[k] = math.exp((1 - rho) * math.log(x) + rho * math.log(y)) if x > 0 and y > 0 else (1 - rho) * x + rho * y
    return new.with_params(theta)


def compare(template: MarketGenerator, windows: list[EventFrame], *, initial: int = 2, window: int = 3,
            rho: float = 0.3, labeler=None, schemes: tuple[str, ...] = SCHEMES, cusum_h: float = 3.0,
            block: int = 300) -> dict:
    if len(windows) <= initial:
        raise ValueError("need more windows than the initial calibration span")
    schemes = tuple(s for s in schemes if s != "regime" or labeler is not None)
    fits: dict[tuple[int, ...], MarketGenerator] = {}

    def fit(idx: tuple[int, ...]) -> MarketGenerator:
        if idx not in fits:
            fits[idx] = template.fit(concat([windows[i] for i in idx]))
        return fits[idx]

    frozen = fit(tuple(range(initial)))
    smoothed = frozen
    current_trigger = frozen
    losses: dict[str, list[float]] = {s: [] for s in schemes}
    trigger_state = {"up": 0.0, "down": 0.0, "ref": [], "refits": []}
    regime_models = {}
    if labeler is not None:
        from ..regimes import block_features
        labelled: dict[int, list[EventFrame]] = {}
        for i in range(initial):
            feats = block_features(windows[i], block)
            labels = labeler.label(feats)
            for b, lab in enumerate(labels):
                if lab >= 0:
                    labelled.setdefault(int(lab), []).append(windows[i].slice(b * block, (b + 1) * block))
        regime_models = {lab: template.fit(concat(parts)) for lab, parts in labelled.items() if len(parts) >= 2}
    for t in range(initial, len(windows)):
        target = windows[t]
        past = tuple(range(t))
        if "frozen" in schemes:
            losses["frozen"].append(loss(frozen, target))
        if "rolling" in schemes:
            losses["rolling"].append(loss(fit(tuple(range(max(0, t - window), t))), target))
        if "expanding" in schemes:
            losses["expanding"].append(loss(fit(past), target))
        if "forgetting" in schemes:
            if t > initial:
                smoothed = _smooth(smoothed, fit((t - 1,)), rho)
            losses["forgetting"].append(loss(smoothed, target))
        if "regime" in schemes:
            from ..regimes import block_features
            first = labeler.label(block_features(windows[t - 1], block))
            lab = int(first[-1]) if len(first) and first[-1] >= 0 else -1
            model = regime_models.get(lab, frozen)
            losses["regime"].append(loss(model, target))
        if "change_triggered" in schemes:
            value = loss(current_trigger, target)
            losses["change_triggered"].append(value)
            ref = trigger_state["ref"]
            if len(ref) >= 2:
                mu, sd = float(np.mean(ref)), float(np.std(ref)) or 1e-6
                z = (value - mu) / sd
                trigger_state["up"] = max(0.0, trigger_state["up"] + z - 0.5)
                if trigger_state["up"] > cusum_h:
                    current_trigger = fit(tuple(range(max(0, t + 1 - window), t + 1)))
                    trigger_state.update(up=0.0, ref=[])
                    trigger_state["refits"].append(t)
                    continue
            ref.append(value)
    return {"schemes": list(schemes), "losses": losses, "windows_scored": len(windows) - initial,
            "initial": initial, "window": window, "rho": rho, "change_refits": trigger_state["refits"],
            "regime_models": sorted(regime_models)}


def copy_template(g: MarketGenerator) -> MarketGenerator:
    return deepcopy(g)
