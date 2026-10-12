"""Change-point and drift detection (M6) and calibration half-life (M7).

Detectors (all on a univariate monitoring series, e.g. block activity or a
calibration loss; reference statistics come from a development prefix):

* ``cusum`` — two-sided tabular CUSUM on standardized values (k, h);
* ``page_hinkley`` — Page-Hinkley test for mean increases and decreases;
* ``bocpd`` — Bayesian online change-point detection (Adams-MacKay) with a
  Normal-Gamma model and constant hazard; an alarm fires when the MAP run length
  drops;
* ``distribution`` — energy distance between a rolling window and the
  reference, with a permutation-calibrated threshold (seeded);
* ``support`` — rolling fraction of points outside the reference support.

``evaluate`` scores alarms against known change points: detection delay, false
alarms, missed changes. ``half_life`` estimates how fast a calibration degrades
with the horizon ``h`` by fitting ``D(h) = D_inf (1 - exp(-h / tau))`` to the
degradation curve; it returns NOT_ESTABLISHED when the curve is not
increasing or the fit is not informative (no half-life is claimed then).
"""
from __future__ import annotations

import math

import numpy as np

DETECTORS = ("cusum", "page_hinkley", "bocpd", "distribution", "support")


def cusum(x: np.ndarray, *, ref: int, k: float = 0.5, h: float = 5.0, rebase: int = 20) -> list[int]:
    """Two-sided CUSUM; after an alarm the reference is re-estimated from the next ``rebase`` points."""
    x = np.asarray(x, float)
    mu, sd = float(np.mean(x[:ref])), float(np.std(x[:ref])) or 1.0
    up = down = 0.0
    alarms = []
    i = ref
    while i < len(x):
        z = (x[i] - mu) / sd
        up, down = max(0.0, up + z - k), max(0.0, down - z - k)
        if up > h or down > h:
            alarms.append(i)
            window = x[i + 1:i + 1 + rebase]
            if len(window) >= 3:
                mu, sd = float(window.mean()), float(window.std()) or sd
            up = down = 0.0
            i += rebase
        i += 1
    return alarms


def page_hinkley(x: np.ndarray, *, ref: int, delta: float = 0.5, threshold: float = 8.0, rebase: int = 20) -> list[int]:
    """Page-Hinkley (both directions); re-baselined after each alarm like ``cusum``."""
    x = np.asarray(x, float)
    mu, sd = float(np.mean(x[:ref])), float(np.std(x[:ref])) or 1.0
    alarms = []
    m_up = m_down = min_up = max_down = 0.0
    i = ref
    while i < len(x):
        z = (x[i] - mu) / sd
        m_up += z - delta
        m_down += z + delta
        min_up, max_down = min(min_up, m_up), max(max_down, m_down)
        if m_up - min_up > threshold or max_down - m_down > threshold:
            alarms.append(i)
            window = x[i + 1:i + 1 + rebase]
            if len(window) >= 3:
                mu, sd = float(window.mean()), float(window.std()) or sd
            m_up = m_down = min_up = max_down = 0.0
            i += rebase
        i += 1
    return alarms


def bocpd(x: np.ndarray, *, ref: int, hazard: float = 1 / 50, drop: int = 5) -> list[int]:
    """Adams-MacKay BOCPD with Normal-Gamma prior from the reference; alarm when the MAP run length resets."""
    x = np.asarray(x, float)
    mu0, var0 = float(np.mean(x[:ref])), float(np.var(x[:ref])) or 1.0
    kappa0, alpha0, beta0 = 1.0, 1.0, var0
    mu, kappa, alpha, beta = (np.asarray([v]) for v in (mu0, kappa0, alpha0, beta0))
    r = np.asarray([1.0])
    alarms, previous_map = [], 0
    for i in range(ref, len(x)):
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        nu = 2 * alpha
        z = (x[i] - mu) / scale
        pred = np.exp(_lgamma((nu + 1) / 2) - _lgamma(nu / 2)) / (np.sqrt(nu * math.pi) * scale) * \
            (1 + z * z / nu) ** (-(nu + 1) / 2)
        growth = r * pred * (1 - hazard)
        change = float((r * pred * hazard).sum())
        r = np.r_[change, growth]
        r = r / r.sum()
        mu_new = (kappa * mu + x[i]) / (kappa + 1)
        beta_new = beta + kappa * (x[i] - mu) ** 2 / (2 * (kappa + 1))
        mu, kappa, alpha, beta = (np.r_[mu0, mu_new], np.r_[kappa0, kappa + 1], np.r_[alpha0, alpha + 0.5],
                                  np.r_[beta0, beta_new])
        if len(r) > 400:   # truncate negligible long run lengths for bounded cost
            r, mu, kappa, alpha, beta = r[:400], mu[:400], kappa[:400], alpha[:400], beta[:400]
            r = r / r.sum()
        current_map = int(np.argmax(r))
        if previous_map - current_map >= drop:
            alarms.append(i)
        previous_map = current_map
    return alarms


def _lgamma(v):
    return np.vectorize(math.lgamma)(v)


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = np.abs(a[:, None] - b[None]).mean()
    aa = np.abs(a[:, None] - a[None]).mean()
    bb = np.abs(b[:, None] - b[None]).mean()
    return float(2 * ab - aa - bb)


def distribution_drift(x: np.ndarray, *, ref: int, window: int = 20, alpha: float = 0.01, seed: int = 0,
                       permutations: int = 200) -> tuple[list[int], float]:
    """Energy-distance drift versus the reference; re-referenced to the post-alarm window after each alarm."""
    x = np.asarray(x, float)
    reference = x[:ref]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        perm = rng.permutation(reference)
        null.append(energy_distance(perm[:window], perm[window:]) if len(perm) > 2 * window else 0.0)
    threshold = float(np.quantile(null, 1 - alpha)) if null else math.inf
    alarms = []
    i = ref + window
    while i <= len(x):
        if energy_distance(x[i - window:i], reference) > threshold:
            alarms.append(i - 1)
            reference = x[i:i + 2 * window] if len(x[i:i + 2 * window]) >= window else reference
            i += 2 * window
        else:
            i += max(1, window // 2)
    return alarms, threshold


def support_failure(x: np.ndarray, *, ref: int, window: int = 20, margin: float = 0.1,
                    max_outside: float = 0.3) -> list[int]:
    x = np.asarray(x, float)
    lo, hi = float(np.min(x[:ref])), float(np.max(x[:ref]))
    pad = margin * (hi - lo or 1.0)
    outside = (x < lo - pad) | (x > hi + pad)
    return [i for i in range(ref + window, len(x) + 1) if outside[i - window:i].mean() > max_outside]


def detect(x: np.ndarray, *, ref: int, method: str, seed: int = 0) -> list[int]:
    if method == "cusum":
        return cusum(x, ref=ref)
    if method == "page_hinkley":
        return page_hinkley(x, ref=ref)
    if method == "bocpd":
        return bocpd(x, ref=ref)
    if method == "distribution":
        return distribution_drift(x, ref=ref, seed=seed)[0]
    if method == "support":
        return support_failure(x, ref=ref)
    raise ValueError(f"unknown detector {method!r}")


def evaluate(alarms: list[int], changes: list[int], *, tolerance: int, n: int) -> dict:
    """Delay to first alarm after each change (within ``tolerance``), false alarms and misses."""
    delays, matched = [], set()
    for c in changes:
        hits = [a for a in alarms if c <= a <= c + tolerance]
        if hits:
            delays.append(hits[0] - c)
            matched.update(a for a in alarms if c <= a <= c + tolerance)
    false = [a for a in alarms if a not in matched]
    pre_change_length = (changes[0] if changes else n)
    return {"detection_delay_mean": float(np.mean(delays)) if delays else None, "detected": len(delays),
            "missed": len(changes) - len(delays), "false_alarms": len(false),
            "false_alarm_rate_per_100": 100.0 * len(false) / max(n, 1),
            "false_alarms_before_first_change": sum(a < pre_change_length for a in false)}


def half_life(horizons, losses, baseline: float, *, min_points: int = 3) -> dict:
    """Fit D(h) = D_inf (1 - exp(-h / tau)) to the degradation D(h) = loss(h) - baseline (h > 0)."""
    h = np.asarray(horizons, float)
    d = np.asarray(losses, float) - float(baseline)
    ok = np.isfinite(h) & np.isfinite(d) & (h > 0)
    h, d = h[ok], d[ok]
    if len(h) < min_points:
        return {"status": "NOT_EVALUABLE", "reason": "fewer than three horizons", "half_life": None}
    if d.max() <= 0:
        return {"status": "NOT_ESTABLISHED", "reason": "no degradation relative to the in-sample baseline",
                "half_life": None, "degradation": d.tolist()}
    best = None
    for tau in np.geomspace(max(h.min() / 20, 1e-6), h.max() * 20, 400):
        basis = 1 - np.exp(-h / tau)
        dinf = float(np.dot(basis, d) / np.dot(basis, basis))
        sse = float(((dinf * basis - d) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, tau, dinf)
    sse, tau, dinf = best
    sst = float(((d - d.mean()) ** 2).sum()) or 1e-12
    r2 = 1 - sse / sst
    boundary = tau <= h.min() / 19 or tau >= h.max() * 19
    monotone = float(np.mean(np.diff(d[np.argsort(h)]) >= -1e-12)) if len(d) > 1 else 1.0
    informative = dinf > 0 and not boundary and r2 >= 0.5 and monotone >= 0.5
    return {"status": "ESTIMATED" if informative else "NOT_ESTABLISHED",
            "reason": None if informative else "degradation curve not informative (boundary tau, poor fit or "
                                               "non-monotone)",
            "half_life": float(tau * math.log(2)) if informative else None, "tau": float(tau), "d_inf": dinf,
            "r2": r2, "monotone_fraction": monotone, "degradation": d.tolist(), "horizons": h.tolist()}
