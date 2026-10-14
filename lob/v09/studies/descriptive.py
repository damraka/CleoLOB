"""Descriptive (non-confirmatory) v0.9 studies on explore days, synthetic worlds or sealed upstream runs.

None of these studies tests a registered hypothesis; their outputs are DESCRIPTIVE (or SYNTHETIC) characterizations
that feed the report, the model cards and the datasheets. They read only development/explore data (namespace
``explore`` / ``pilot``; ``dev`` with explicit development datasets for smoke runs) or sealed upstream runs, and
never confirmatory or vault data. Each command is a thin module (``reconstruction``, ``replay``, ...) over the task
and aggregation functions below.

Labels: every observational quantity on public L2 is a proxy (no order ids, queue positions or trader identities);
"toxicity" is a flow-risk proxy; "hidden liquidity" is executed volume beyond the displayed touch, not an
identification of iceberg orders; bounded replay values are HYPOTHETICAL.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from .. import protocol as pr
from ..data.daycache import BIN_MS, WINDOW_STARTS_S
from ..runner import Task, run_tasks
from . import common

ALL_STREAMS = ["bybit-linear:BTCUSDT", "bybit-linear:ETHUSDT", "binance-um:BTCUSDT", "binance-um:ETHUSDT"]
HORIZON_S, WARMUP_S = 300.0, 30.0
SIM_BINS = 3300


# ============================================================================= plumbing


def _days(args, protocol, streams=None) -> list[str]:
    return common.explore_units(args, protocol, streams)


def _design(args, analysis: str) -> str:
    return f"{args.namespace}-{analysis}"


def _load(d: str, root: str, design: str, purpose: str) -> dict:
    return common.load_cache(d, root, design, purpose)


def _per_day(args, analysis: str, fn: str, days: list[str], *, kwargs: dict | None = None,
             aggregate=None, config: dict | None = None, label: str = "DESCRIPTIVE") -> dict:
    root = common.root_of(args)
    out = common.start(args)
    design = _design(args, analysis)
    tasks = [Task(d, f"lob.v09.studies.descriptive:{fn}", {"dataset_id": d, "root": str(root), "design": design,
                                                           **(kwargs or {})}, namespace=args.namespace)
             for d in days]
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root / (args.out + ".checkpoint"), design={"analysis": analysis, "days": days},
                           inputs=days)
    by = {r["dataset"]: r for r in rows}
    result = {"days": by, "summary": aggregate(by) if aggregate else _streams_mean(by), "runner": info,
              "label": label}
    return common.finish(args, out, analysis=analysis, config={"days": days, **(config or {})}, result=result,
                         datasets=days)


def _stream_of(dataset_id: str) -> str:
    parts = dataset_id.split("-")
    return f"{parts[0]}-{parts[1]}:{parts[2].upper()}"


def _streams_mean(by: dict) -> dict:
    """Per-stream mean of every numeric top-level field."""
    out: dict = {}
    for d, row in by.items():
        s = out.setdefault(_stream_of(d), {})
        for k, v in row.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None and math.isfinite(v):
                s.setdefault(k, []).append(float(v))
    return {s: {k: float(np.mean(v)) for k, v in vals.items()} for s, vals in out.items()}


def _paths(mid: np.ndarray, spread: np.ndarray, valid: np.ndarray, starts) -> np.ndarray:
    from ..realism.path import LENGTH
    mid2 = np.rint(2 * mid).astype(np.int64)
    out = []
    for b in starts:
        if b + LENGTH + 1 > len(valid) or not valid[b:b + LENGTH + 1].all():
            continue
        out.append(np.column_stack([mid2[b:b + LENGTH + 1] - mid2[b], spread[b:b + LENGTH + 1]]))
    return np.asarray(out, float).reshape(-1, LENGTH + 1, 2)


def _sim_paths(frames) -> np.ndarray:
    from ..realism.path import LENGTH
    parts = [_paths(fr.mid[:len(fr.valid)].astype(float), (fr.ask_px - fr.bid_px).astype(float), fr.valid.astype(bool),
                    range(0, len(fr.valid) - LENGTH - 1, 300)) for fr in frames]
    parts = [p for p in parts if len(p)]
    return np.concatenate(parts) if parts else np.zeros((0, LENGTH + 1, 2))


def _sub(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(x) <= n:
        return x
    return x[np.sort(np.random.default_rng(seed).choice(len(x), n, replace=False))]


def _split(days: list[str]) -> dict[str, tuple[list[str], list[str]]]:
    plan: dict[str, list[str]] = {}
    for d in days:
        plan.setdefault(_stream_of(d), []).append(d)
    out = {}
    for s, v in plan.items():
        v = sorted(v, key=lambda d: d[-10:])
        out[s] = (v[: len(v) // 2], v[len(v) // 2:])
    return out


def _per_stream(args, analysis: str, fn: str, days: list[str], *, kwargs: dict | None = None,
                config: dict | None = None, label: str = "DESCRIPTIVE") -> dict:
    root = common.root_of(args)
    out = common.start(args)
    design = _design(args, analysis)
    plan = _split(days)
    tasks = [Task(s, f"lob.v09.studies.descriptive:{fn}", {"stream": s, "train": tr, "test": te, "root": str(root),
                                                           "design": design, "seed": args.seed, **(kwargs or {})},
                  namespace=args.namespace) for s, (tr, te) in plan.items()]
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root / (args.out + ".checkpoint"), design={"analysis": analysis, "plan": plan},
                           inputs=days)
    result = {"streams": {r["stream"]: r for r in rows}, "plan": plan, "runner": info, "label": label}
    return common.finish(args, out, analysis=analysis, config={"plan": plan, "seed": args.seed, **(config or {})},
                         result=result, datasets=days)


def _fit_frames(train: list[str], root: str, design: str, purpose: str, families, seed: int, sims: int = 4):
    from ..generators import classical as G
    caches = [_load(d, root, design, purpose) for d in train]
    frame = G.training_frame(caches)
    del caches
    book = G.book_model(frame)
    out = {}
    for k, name in enumerate(families):
        g = G.fit_family(name, frame, book)
        out[name] = (g, [g.simulate(SIM_BINS, seed + 1000 * k + j, warmup_bins=600) for j in range(sims)])
    return frame, out


# ============================================================================= M20-M21 event reconstruction


def reconstruction_day(dataset_id: str, root: str, design: str) -> dict:
    c = _load(dataset_id, root, design, "reconstruction audit")
    q = c["quality"]
    rec = {k: q["reconstruction"].get(k) for k in ("messages", "sequence_gaps", "duplicates", "out_of_order",
                                                    "crossed", "max_gap_ms")}
    return {"dataset": dataset_id, **rec, "valid_bin_fraction": float(q["valid_bin_fraction"]),
            "trades": int(len(c["trade_t"])), "status": q["decision"]["status"],
            "grid_anomalies": len(q.get("grid", {}).get("anomalies", [])),
            "levels": int(c["deep_q"].shape[-1]) if "deep_q" in c else None}


def run_reconstruction(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    return _per_day(args, "event-reconstruction-v09", "reconstruction_day", _days(args, protocol, ALL_STREAMS))


# ============================================================================= M22-M25 historical replay (bounded)


def replay_day(dataset_id: str, root: str, design: str, assumption_keys: list, notional: float) -> dict:
    from ..execution import policies as P
    from ..replay import engine as E
    book = E.DayBook(_load(dataset_id, root, design, "bounded replay audit"))
    grid = [E.Assumption(**json.loads(k)) for k in assumption_keys]
    out = {}
    for policy in ("twap", "passive_deadline"):
        widths, identified, n = [], [], 0
        for w in range(0, len(WINDOW_STARTS_S), 4):
            rows = [E.episode(book, P.make(policy), window=w, notional_usd=notional, assumption=a, context={"seed": w})
                    for a in grid]
            b = E.bounds(rows)
            if b["is_net_bps"] is None:
                continue
            n += 1
            widths.append(b["is_net_bps"][1] - b["is_net_bps"][0])
            identified.append(all(r.get("identified", True) for r in rows if r.get("status") == "OK"))
        out[policy] = {"windows": n, "median_width_bps": float(np.median(widths)) if widths else None,
                       "identified_fraction": float(np.mean(identified)) if identified else None}
    return {"dataset": dataset_id, "policies": out,
            "twap_width_bps": out["twap"]["median_width_bps"],
            "passive_width_bps": out["passive_deadline"]["median_width_bps"]}


def run_replay(args) -> dict:
    from ..replay import engine as E
    from .bounds import assumptions
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    keys = [json.dumps(E.assumption_dict(a), sort_keys=True) for a in assumptions(protocol)]
    days = _days(args, protocol)
    notional = protocol["mandates"]["classes"]["medium"]["notional_usd"]
    root_ = common.root_of(args)
    out = common.start(args)
    design = _design(args, "historical-replay-v09")
    tasks = [Task(d, "lob.v09.studies.descriptive:replay_day",
                  {"dataset_id": d, "root": str(root_), "design": design, "assumption_keys": keys,
                   "notional": float(notional[d.split("-")[2].upper()])}, namespace=args.namespace) for d in days]
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root_ / (args.out + ".checkpoint"), design={"days": days, "grid": keys},
                           inputs=days)
    by = {r["dataset"]: r for r in rows}
    result = {"days": by, "summary": _streams_mean(by), "assumptions": keys, "runner": info,
              "label": "BOUNDED, HYPOTHETICAL replay; width = assumption-grid interval of net implementation shortfall"}
    return common.finish(args, out, analysis="historical-replay-v09", config={"days": days, "grid": keys},
                         result=result, datasets=days)


# ============================================================================= M26-M27 queue reconstruction


def queue_day(dataset_id: str, root: str, design: str) -> dict:
    from ..queue import clearance as C, survival as SV
    c = _load(dataset_id, root, design, "queue reconstruction audit")
    ev = C.events(c)
    T, E_ = ev["T"], ev["E"]
    q = ev["X"][:, 0]
    cuts = np.quantile(q, [1 / 3, 2 / 3]) if len(q) else np.zeros(2)
    ter = np.digitize(q, cuts)
    km = SV.km_curve(T, E_) if len(T) else None
    rates = {}
    for h in (1.0, 5.0, 30.0):
        rates[f"cleared_{int(h)}s"] = float(np.mean((T <= h) & E_)) if len(T) else None
        for t in range(3):
            k = ter == t
            rates[f"cleared_{int(h)}s_tercile{t}"] = float(np.mean((T[k] <= h) & E_[k])) if k.any() else None
    return {"dataset": dataset_id, "probes": int(len(T)), "censored_fraction": float(1 - E_.mean()) if len(T) else None,
            **rates, "km": km.tolist() if km is not None else None, **C.queue_confidence(c)}


def run_queue(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    return _per_day(args, "queue-reconstruction-v09", "queue_day", _days(args, protocol),
                    label="DESCRIPTIVE; back-of-queue clearance of hypothetical resting orders (queue position unobserved)")


# ============================================================================= M63-M65 latent liquidity


def liquidity_day(dataset_id: str, root: str, design: str) -> dict:
    c = _load(dataset_id, root, design, "latent liquidity proxies")
    t, q, s = c["trade_t"], c["trade_q"].astype(float), c["trade_s"]
    b = np.clip((t * 1000 // BIN_MS).astype(np.int64) - 1, 0, len(c["valid"]) - 1)
    ok = c["valid"][b]
    passive = np.where(s > 0, c["depth"][b, 1, 0], c["depth"][b, 0, 0]).astype(float)
    touch = np.where(s > 0, c["ask_px"][b], c["bid_px"][b])
    after = np.clip(b + 2, 0, len(c["valid"]) - 1)
    touch_after = np.where(s > 0, c["ask_px"][after], c["bid_px"][after])
    beyond = ok & (q > passive) & (passive > 0)
    held = beyond & (touch_after == touch)
    d = c["depth"][:, 0, 0].astype(float)
    med = float(np.median(d[c["valid"] & (d > 0)])) if (c["valid"] & (d > 0)).any() else 0.0
    dep = np.flatnonzero((d[1:] < 0.1 * med) & (d[:-1] >= 0.1 * med) & c["valid"][1:])
    rec = []
    for i in dep[:5000]:
        j = np.flatnonzero(d[i + 1:i + 301] >= 0.5 * med)
        rec.append((j[0] + 1) * BIN_MS / 1000.0 if len(j) else np.nan)
    rec = np.asarray(rec, float)
    return {"dataset": dataset_id, "trades": int(ok.sum()),
            "beyond_display_fraction": float(beyond.sum() / max(ok.sum(), 1)),
            "beyond_display_price_held_fraction": float(held.sum() / max(beyond.sum(), 1)),
            "beyond_display_volume_share": float(q[beyond].sum() / max(q[ok].sum(), 1e-9)),
            "touch_depletions": int(len(dep)),
            "refill_median_s": float(np.nanmedian(rec)) if np.isfinite(rec).any() else None,
            "refill_within_30s_fraction": float(np.mean(np.isfinite(rec))) if len(rec) else None}


def run_liquidity(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    return _per_day(args, "latent-liquidity-v09", "liquidity_day", _days(args, protocol),
                    label="DESCRIPTIVE PROXIES; executed volume beyond the displayed touch is not an identification "
                          "of hidden or iceberg orders (100 ms snapshots, no order ids)")


# ============================================================================= M70 toxicity / flow-risk proxies


def toxicity_day(dataset_id: str, root: str, design: str) -> dict:
    from ..impact import models as IM
    c = _load(dataset_id, root, design, "flow-risk proxies")
    s = IM.seconds(c)
    v, mid, ok = s["v"], s["mid"], s["valid"]
    n = len(v)
    buy = np.where(v > 0, v, 0.0)
    sell = np.where(v < 0, -v, 0.0)
    total = float(np.abs(v).sum())
    vb = total / 50.0 if total > 0 else 1.0
    cum = np.cumsum(np.abs(v))
    bucket = np.minimum((cum / vb).astype(np.int64), 49)
    vpin = []
    for k in range(50):
        m = bucket == k
        if m.any():
            vpin.append(abs(buy[m].sum() - sell[m].sum()) / max(buy[m].sum() + sell[m].sum(), 1e-9))
    ofi = np.convolve(v, np.ones(30), "full")[:n]
    out = {"dataset": dataset_id, "vpin_mean": float(np.mean(vpin)) if vpin else None}
    for h in (5, 30):
        fwd = (mid[np.minimum(np.arange(n) + h, n)] - mid[:n]) / np.maximum(mid[:n], 1e-9) * 1e4
        k = ok & (np.arange(n) + h <= n) & (np.arange(n) >= 30)
        if k.sum() > 100 and np.std(ofi[k]) > 0 and np.std(fwd[k]) > 0:
            out[f"ofi30_fwd{h}s_corr"] = float(np.corrcoef(ofi[k], fwd[k])[0, 1])
            top = k & (np.abs(ofi) >= np.quantile(np.abs(ofi[k]), 0.9))
            # a passive fill against the flow (sell to buyers / buy from sellers): signed adverse move in bps
            out[f"adverse_{h}s_bps_top_decile"] = float(np.mean(np.sign(ofi[top]) * fwd[top])) if top.any() else None
    return out


def run_toxicity(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    return _per_day(args, "toxicity-study-v09", "toxicity_day", _days(args, protocol),
                    label="DESCRIPTIVE FLOW-RISK PROXIES; never an identification of informed traders")


# ============================================================================= regimes


def regime_day(dataset_id: str, root: str, design: str) -> dict:
    c = _load(dataset_id, root, design, "regime description")
    rows = []
    for w, s in enumerate(WINDOW_STARTS_S):
        a, b = int(s * 1000 // BIN_MS), int((s + HORIZON_S) * 1000 // BIN_MS)
        ok = c["valid"][a:b]
        if ok.mean() < 0.9:
            rows.append(None)
            continue
        m = c["mid"][a:b + 1:10].astype(float)
        r = np.diff(m) / np.maximum(m[:-1], 1e-9) * 1e4
        spread = (c["ask_px"][a:b] - c["bid_px"][a:b])[ok].astype(float)
        depth = c["depth"][a:b, :, 0][ok].astype(float).sum(1)
        rows.append({"vol_bps": float(np.std(r) * math.sqrt(len(r))), "spread_ticks": float(spread.mean()),
                     "touch_depth_lots": float(np.median(depth))})
    return {"dataset": dataset_id, "windows": rows}


def _regime_summary(by: dict) -> dict:
    out = {}
    for stream in sorted({_stream_of(d) for d in by}):
        rows = [(d, w, r) for d in sorted(by) if _stream_of(d) == stream for w, r in enumerate(by[d]["windows"]) if r]
        if len(rows) < 9:
            continue
        vol = np.asarray([r["vol_bps"] for _, _, r in rows])
        liq = np.asarray([r["touch_depth_lots"] for _, _, r in rows])
        vc, lc = np.quantile(vol, [1 / 3, 2 / 3]), np.quantile(liq, [1 / 3, 2 / 3])
        lab = 3 * np.digitize(vol, vc) + (2 - np.digitize(liq, lc))      # 0 = calm/deep ... 8 = volatile/thin
        occ = np.bincount(lab, minlength=9) / len(lab)
        trans = np.zeros((9, 9))
        for (d0, w0, _), (d1, w1, _), a, b in zip(rows[:-1], rows[1:], lab[:-1], lab[1:]):
            if d0 == d1 and w1 == w0 + 1:
                trans[a, b] += 1
        persist = float(np.trace(trans) / trans.sum()) if trans.sum() else None
        out[stream] = {"windows": len(rows), "occupancy": occ.tolist(), "persistence": persist,
                       "vol_terciles_bps": vc.tolist(), "depth_terciles_lots": lc.tolist(),
                       "labels": "3 x vol tercile + (2 - depth tercile): 0 calm/deep ... 8 volatile/thin"}
    return out


def run_regimes(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    return _per_day(args, "regime-study-v09", "regime_day", _days(args, protocol), aggregate=_regime_summary)


# ============================================================================= change points and drift


def _binary_segmentation(x: np.ndarray, penalty: float, min_size: int = 10, depth: int = 0) -> list[int]:
    n = len(x)
    if n < 2 * min_size or depth > 6:
        return []
    cs, cs2 = np.cumsum(x), np.cumsum(x * x)
    total = cs2[-1] - cs[-1] ** 2 / n
    best, arg = 0.0, None
    for k in range(min_size, n - min_size):
        left = cs2[k - 1] - cs[k - 1] ** 2 / k
        right = (cs2[-1] - cs2[k - 1]) - (cs[-1] - cs[k - 1]) ** 2 / (n - k)
        gain = total - left - right
        if gain > best:
            best, arg = gain, k
    if arg is None or best < penalty:
        return []
    return (_binary_segmentation(x[:arg], penalty, min_size, depth + 1) + [arg]
            + [arg + k for k in _binary_segmentation(x[arg:], penalty, min_size, depth + 1)])


def drift_day(dataset_id: str, root: str, design: str) -> dict:
    c = _load(dataset_id, root, design, "change-point description")
    m = c["mid"][::600].astype(float)                             # one-minute mid
    ok = c["valid"][::600][: len(m) - 1]
    r = np.diff(m) / np.maximum(m[:-1], 1e-9) * 1e4
    x = np.log(np.abs(r[ok]) + 0.1)
    sd = float(np.std(x)) or 1.0
    cps = _binary_segmentation((x - x.mean()) / sd, penalty=2.0 * math.log(max(len(x), 2)) * 2)
    return {"dataset": dataset_id, "change_points": len(cps), "change_minutes": [int(k) for k in cps],
            "profile": c["quality"].get("profile_v09")}


def _drift_summary(by: dict) -> dict:
    from .. import quality
    out = {"change_points": _streams_mean({d: {"change_points": r["change_points"]} for d, r in by.items()})}
    for stream in sorted({_stream_of(d) for d in by}):
        prof = {d: r["profile"] for d, r in by.items() if _stream_of(d) == stream and r.get("profile")}
        if len(prof) >= 3:
            out[stream] = quality.drift(prof, sorted(prof))
    return out


def run_drift(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    return _per_day(args, "change-point-study-v09", "drift_day", _days(args, protocol), aggregate=_drift_summary,
                    label="DESCRIPTIVE; binary segmentation of one-minute log absolute returns (BIC-type penalty)")


# ============================================================================= M52 path realism


FAMILY_SET = ("poisson", "hawkes", "queue_reactive")


def path_task(stream: str, train: list[str], test: list[str], root: str, design: str, seed: int) -> dict:
    from ..realism import path as RP
    _, fits = _fit_frames(train, root, design, "path realism", FAMILY_SET, seed)
    real = [RP.real_paths(_load(d, root, design, "path realism")) for d in test]
    real = [_sub(p, 150, seed + k) for k, p in enumerate(real) if len(p)]
    allreal = np.concatenate(real)
    scale = float(np.std(allreal[:, -1, 0])) or 1.0
    sig_real = [RP.signature(p, scale) for p in real]
    ref = []
    for i in range(len(sig_real)):
        for j in range(i + 1, len(sig_real)):
            ref.append(RP.mmd2(sig_real[i], sig_real[j]))
    out = {"stream": stream, "real_paths": int(len(allreal)), "real_real_mmd2": ref,
           "real_real_mmd2_p95": float(np.quantile(ref, 0.95)) if ref else None,
           "real_first_passage_median_s": float(np.median(RP.first_passage(allreal))),
           "real_spectrum": RP.spectrum(allreal).tolist(), "families": {}}
    sig_all = np.concatenate(sig_real)
    for name, (_, frames) in fits.items():
        sim = _sub(_sim_paths(frames), 300, seed)
        if not len(sim):
            out["families"][name] = {"status": "NOT_EVALUABLE", "reason": "no valid simulated paths"}
            continue
        sp = RP.spectrum(sim)
        out["families"][name] = {
            "mmd2": RP.mmd2(sig_all, RP.signature(sim, scale)),
            "first_passage_median_s": float(np.median(RP.first_passage(sim))),
            "spectrum_log_ratio": (np.log(np.maximum(sp, 1e-12)) - np.log(np.maximum(RP.spectrum(allreal), 1e-12))).tolist(),
            "paths": int(len(sim))}
        if ref:
            out["families"][name]["mmd2_over_real_p95"] = out["families"][name]["mmd2"] / max(out["real_real_mmd2_p95"], 1e-12)
    return out


def run_path_realism(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    return _per_stream(args, "path-realism-v09", "path_task", _days(args, protocol), config={"families": FAMILY_SET},
                       label="DESCRIPTIVE; level-3 path signatures (MMD), increment spectra and first-passage times "
                             "of classical generators vs held-out real windows")


# ============================================================================= M54 tail realism


def _hill(x: np.ndarray, frac: float = 0.02) -> float | None:
    x = np.sort(np.abs(x[np.isfinite(x) & (x != 0)]))[::-1]
    k = int(len(x) * frac)
    if k < 10:
        return None
    h = float(np.mean(np.log(x[:k] / x[k])))
    return 1.0 / h if h > 0 else None


def _tails(mid: np.ndarray, spread: np.ndarray, depth: np.ndarray, valid: np.ndarray) -> dict:
    m = mid[: len(valid) + 1:10].astype(float)
    r = np.diff(m)
    r = r[valid[: len(r) * 10:10][: len(r)]]
    d = depth[valid].astype(float)
    med = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
    return {"tail_index_1s": _hill(r), "spread_p999_ticks": float(np.quantile(spread[valid], 0.999)),
            "collapse_fraction": float(np.mean(d < 0.1 * med)), "kurtosis_1s": float(
                np.mean((r - r.mean()) ** 4) / max(np.var(r) ** 2, 1e-12)) if len(r) > 10 else None}


def tail_task(stream: str, train: list[str], test: list[str], root: str, design: str, seed: int) -> dict:
    _, fits = _fit_frames(train, root, design, "tail realism", FAMILY_SET, seed)
    real = []
    for d in test:
        c = _load(d, root, design, "tail realism")
        real.append(_tails(c["mid"], (c["ask_px"] - c["bid_px"]).astype(float), c["depth"][:, 0, 0], c["valid"]))
    out = {"stream": stream, "real": real, "families": {}}
    for name, (_, frames) in fits.items():
        sims = [_tails(fr.mid.astype(float), (fr.ask_px - fr.bid_px).astype(float), fr.depth[:, 0, 0],
                       fr.valid.astype(bool)) for fr in frames]
        out["families"][name] = {k: (float(np.mean([s[k] for s in sims if s[k] is not None]))
                                     if any(s[k] is not None for s in sims) else None) for k in sims[0]}
    out["real_mean"] = {k: (float(np.mean([s[k] for s in real if s[k] is not None]))
                            if any(s[k] is not None for s in real) else None) for k in real[0]}
    return out


def run_tail_realism(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    return _per_stream(args, "tail-realism-v09", "tail_task", _days(args, protocol), config={"families": FAMILY_SET},
                       label="DESCRIPTIVE; Hill tail index of 1 s mid changes, spread extremes and touch-depth collapse "
                             "frequency (simulated windows are short; extremes are compared, not tested)")


# ============================================================================= M44-M45 generator fit


def generator_fit_task(stream: str, train: list[str], test: list[str], root: str, design: str, seed: int) -> dict:
    from ...v08.market.frame import SUMMARY_NAMES, summaries
    from ..generators import classical as G
    frame, fits = _fit_frames(train, root, design, "generator fit",
                              ("poisson", "hawkes", "queue_reactive", "regime_switching", "state_dependent"), seed, sims=2)
    test_frame = G.training_frame([_load(d, root, design, "generator fit") for d in test])
    s_train, s_test = summaries(frame), summaries(test_frame)
    scale = np.maximum(np.abs(s_test), 1e-6)
    out = {"stream": stream, "summary_names": list(SUMMARY_NAMES), "train_summaries": s_train.tolist(),
           "test_summaries": s_test.tolist(),
           "train_test_rel_error": float(np.median(np.abs(s_train - s_test) / scale)), "families": {}}
    for name, (g, frames) in fits.items():
        sims = np.mean([summaries(fr) for fr in frames], axis=0)
        params = {}
        try:
            params = {k: float(v) for k, v in g.get_flat().items()}
        except (AttributeError, NotImplementedError, TypeError):
            params = {}
        out["families"][name] = {"params": params, "summaries": sims.tolist(),
                                 "median_rel_error_vs_test": float(np.median(np.abs(sims - s_test) / scale))}
    return out


def run_generator_fit(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    return _per_stream(args, "generator-fit-v09", "generator_fit_task", _days(args, protocol),
                       label="DESCRIPTIVE; classical-family calibration on the first half of the explore days, summary "
                             "fit on the second half (median relative error over the registered v0.8 summaries)")


# ============================================================================= identifiability (parameter recovery)


def identifiability_task(stream: str, train: list[str], test: list[str], root: str, design: str, seed: int) -> dict:
    from ..generators import classical as G
    out = {"stream": stream, "families": {}}
    caches = [_load(d, root, design, "identifiability") for d in train[:2]]
    frame = G.training_frame(caches)
    del caches
    book = G.book_model(frame)
    for name in ("hawkes", "queue_reactive"):
        g = G.fit_family(name, frame, book)
        try:
            truth = {k: float(v) for k, v in g.get_flat().items()}
        except (AttributeError, NotImplementedError, TypeError):
            out["families"][name] = {"status": "NOT_EVALUABLE", "reason": "family exposes no flat parameters"}
            continue
        fits = []
        for r in range(6):
            sim = g.simulate(SIM_BINS * 4, seed + 77 * (r + 1), warmup_bins=600)
            fits.append({k: float(v) for k, v in G.fit_family(name, sim, book).get_flat().items()})
        rows = {}
        for k, v in truth.items():
            est = np.asarray([f.get(k, np.nan) for f in fits], float)
            est = est[np.isfinite(est)]
            if not len(est):
                continue
            scale = max(abs(v), 1e-9)
            rows[k] = {"truth": v, "mean": float(est.mean()), "rel_bias": float((est.mean() - v) / scale),
                       "cv": float(est.std(ddof=1) / max(abs(est.mean()), 1e-9)) if len(est) > 1 else None}
        weak = [k for k, r in rows.items() if (r["cv"] is not None and r["cv"] > 0.5) or abs(r["rel_bias"]) > 0.5]
        out["families"][name] = {"parameters": rows, "weakly_identified": weak,
                                 "status": "IDENTIFIED" if not weak else ("PARTIALLY_IDENTIFIED" if len(weak) < len(rows)
                                                                          else "NOT_IDENTIFIED")}
    return out


def run_identifiability(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    return _per_stream(args, "identifiability-v5", "identifiability_task", _days(args, protocol),
                       label="SYNTHETIC parameter recovery (6 simulate-refit replicates at the fitted parameters): "
                             "practical identifiability of the calibrated families")


# ============================================================================= SBC (synthetic)


def sbc_task(seed: int, replicates: int = 400, draws: int = 99, n_obs: int = 300) -> dict:
    from ..inference.bayes import N_CLASSES
    rng = np.random.default_rng(np.random.SeedSequence([9, 59, int(seed)]))
    out = {}
    for name, inflate in (("correct", 1.0), ("overconfident_control", 3.0)):
        ranks = np.zeros((replicates, N_CLASSES), int)
        for r in range(replicates):
            theta = rng.dirichlet(np.ones(N_CLASSES))
            counts = rng.multinomial(n_obs, theta)
            post = rng.dirichlet(1.0 + inflate * counts, size=draws)
            ranks[r] = (post < theta).sum(0)
        bins = 10
        stats = []
        for k in range(N_CLASSES):
            h = np.bincount((ranks[:, k] * bins) // (draws + 1), minlength=bins)
            e = replicates / bins
            stats.append(float(((h - e) ** 2 / e).sum()))
        crit = 21.67                                              # chi2(9) 0.99 quantile
        out[name] = {"chi2_per_class": stats, "chi2_critical_0.99": crit,
                     "uniform_classes": int(sum(s <= crit for s in stats)),
                     "status": "CALIBRATED" if all(s <= crit for s in stats) else "MISCALIBRATED"}
    out["positive_control_detected"] = out["overconfident_control"]["status"] == "MISCALIBRATED"
    return out


def run_sbc(args) -> dict:
    out = common.start(args)
    res = sbc_task(args.seed)
    return common.finish(args, out, analysis="sbc-v09", config={"seed": args.seed, "replicates": 400, "draws": 99},
                         result={**res, "label": "SYNTHETIC simulation-based calibration of the conjugate "
                                                 "Dirichlet-categorical bucket posterior (with an overconfident "
                                                 "positive control)"})


# ============================================================================= posterior predictive checks


def posterior_task(stream: str, train: list[str], test: list[str], root: str, design: str, seed: int) -> dict:
    from ..inference import bayes as B
    from ..representations import sequence as S
    days = sorted(train + test, key=lambda d: d[-10:])
    data = []
    for d in days:
        X, y, _, _ = S.frame_features(_load(d, root, design, "posterior predictive check"))
        b = B.buckets(X, B.current_class(X))
        data.append(B.counts(b, y))
    post = B.sequential(data)
    rng = np.random.default_rng(np.random.SeedSequence([9, 58, int(seed)]))
    rows = []
    for k in range(1, len(days)):
        alpha, obs = post[k - 1], data[k]
        inside = total = 0
        for b in range(obs.shape[0]):
            n = int(obs[b].sum())
            if n < 50:
                continue
            th = rng.dirichlet(np.maximum(alpha[b], 1e-6), size=200)
            sims = np.asarray([rng.multinomial(n, t) for t in th]) / n
            lo, hi = np.quantile(sims, 0.05, axis=0), np.quantile(sims, 0.95, axis=0)
            f = obs[b] / n
            inside += int(((f >= lo) & (f <= hi)).sum())
            total += obs.shape[1]
        rows.append({"day": days[k], "coverage_90": inside / total if total else None, "cells": total})
    cov = [r["coverage_90"] for r in rows if r["coverage_90"] is not None]
    return {"stream": stream, "days": rows, "mean_coverage_90": float(np.mean(cov)) if cov else None}


def run_posterior(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    return _per_stream(args, "posterior-study-v09", "posterior_task", _days(args, protocol),
                       label="DESCRIPTIVE posterior predictive checks of the sequential bucket model "
                             "(next-day class frequencies inside the 90% predictive interval)")


# ============================================================================= M74 cross-impact


def cross_impact_task(date: str, root: str, design: str) -> dict:
    from ..impact import models as IM
    btc = _load(f"bybit-linear-btcusdt-{date}", root, design, "cross-impact")
    eth = _load(f"bybit-linear-ethusdt-{date}", root, design, "cross-impact")
    return {"date": date, "eth_on_btc_bursts": IM.cross_impact(eth, btc), "btc_on_eth_bursts": IM.cross_impact(btc, eth)}


def run_cross_impact(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    days = _days(args, protocol)
    dates = sorted({d[-10:] for d in days if f"bybit-linear-btcusdt-{d[-10:]}" in days
                    and f"bybit-linear-ethusdt-{d[-10:]}" in days})
    out = common.start(args)
    design = _design(args, "cross-impact-v09")
    tasks = [Task(dt, "lob.v09.studies.descriptive:cross_impact_task", {"date": dt, "root": str(root), "design": design},
                  namespace=args.namespace) for dt in dates]
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root / (args.out + ".checkpoint"), design={"dates": dates}, inputs=dates)
    result = {"dates": {r["date"]: r for r in rows}, "runner": info,
              "label": "DESCRIPTIVE, ASSUMPTION_DEPENDENT; same-second co-movement, not a causal cross-impact estimate"}
    used = [f"bybit-linear-{i}-{dt}" for dt in dates for i in ("btcusdt", "ethusdt")]
    return common.finish(args, out, analysis="cross-impact-v09", config={"dates": dates}, result=result, datasets=used)


# ============================================================================= M87 portfolio execution risk


def portfolio_task(date: str, root: str, design: str) -> dict:
    out = {"date": date, "windows": []}
    caches = {i: _load(f"bybit-linear-{i}usdt-{date}", root, design, "portfolio execution risk") for i in ("btc", "eth")}
    rho, vols = [], {"btc": [], "eth": []}
    for s in WINDOW_STARTS_S:
        a, b = int(s * 1000 // BIN_MS), int((s + HORIZON_S) * 1000 // BIN_MS)
        if not all(c["valid"][a:b].mean() > 0.9 for c in caches.values()):
            continue
        r = {}
        for i, c in caches.items():
            m = c["mid"][a:b + 1:50].astype(float)                # 5 s returns in bps
            r[i] = np.diff(m) / np.maximum(m[:-1], 1e-9) * 1e4
        if np.std(r["btc"]) > 0 and np.std(r["eth"]) > 0:
            rho.append(float(np.corrcoef(r["btc"], r["eth"])[0, 1]))
            for i in r:
                vols[i].append(float(np.std(r[i])))
    if not rho:
        return {**out, "status": "NOT_EVALUABLE"}
    p = float(np.mean(rho))
    s1, s2 = float(np.mean(vols["btc"])), float(np.mean(vols["eth"]))
    # TWAP liquidation of equal notionals: timing risk of the joint program relative to separate programs
    joint = math.sqrt(s1 * s1 + s2 * s2 + 2 * p * s1 * s2)
    opposite = math.sqrt(s1 * s1 + s2 * s2 - 2 * p * s1 * s2)
    return {**out, "windows": len(rho), "correlation_5s": p, "vol_5s_bps": {"btc": s1, "eth": s2},
            "same_side_diversification": 1.0 - joint / (s1 + s2),
            "opposite_side_hedge_reduction": 1.0 - opposite / (s1 + s2)}


def run_portfolio(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    days = _days(args, protocol)
    dates = sorted({d[-10:] for d in days if f"bybit-linear-btcusdt-{d[-10:]}" in days
                    and f"bybit-linear-ethusdt-{d[-10:]}" in days})
    out = common.start(args)
    design = _design(args, "portfolio-study-v09")
    tasks = [Task(dt, "lob.v09.studies.descriptive:portfolio_task", {"date": dt, "root": str(root), "design": design},
                  namespace=args.namespace) for dt in dates]
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root / (args.out + ".checkpoint"), design={"dates": dates}, inputs=dates)
    ok = [r for r in rows if r.get("correlation_5s") is not None]
    summary = {k: float(np.mean([r[k] for r in ok])) for k in ("correlation_5s", "same_side_diversification",
                                                               "opposite_side_hedge_reduction")} if ok else {}
    result = {"dates": {r["date"]: r for r in rows}, "summary": summary, "runner": info,
              "label": "DESCRIPTIVE timing-risk arithmetic of joint BTC/ETH TWAP programs (no impact interaction)"}
    used = [f"bybit-linear-{i}-{dt}" for dt in dates for i in ("btcusdt", "ethusdt")]
    return common.finish(args, out, analysis="portfolio-study-v09", config={"dates": dates}, result=result,
                         datasets=used)


# ============================================================================= M85 adversarial stress and counterexamples


STRESS_POLICIES = ("twap", "passive_deadline", "liquidity_pov", "mpc_cvar")


def _stress_world(g, z: np.ndarray):
    from copy import deepcopy
    w = deepcopy(g)
    w.params["market_scale"] *= float(np.exp(z[0]))
    w.params["limit_scale"] *= float(np.exp(z[1]))
    for k, a in enumerate(("momentum", "toxicity_proxy")):
        if a in w.params["agents"]:
            w.params["agents"][a] *= float(np.exp(z[2 + k]))
    return w


def adversarial_task(dataset_id: str, root: str, design: str, seed: int, notional: float, lot: float,
                     budget: int = 16, bound: float = math.log(3.0)) -> dict:
    from .. import synthetic as SY
    t0 = time.time()
    g, tick, _ = SY.calibrate([_load(dataset_id, root, design, "adversarial stress world")])
    qty = SY.qty_for(g, tick, lot, notional)
    rng = np.random.default_rng(np.random.SeedSequence([9, 85, int(seed)]))
    seeds = [int(seed) * 100 + k for k in range(4)]
    nominal = {p: SY._m(v) for p, v in SY.costs(g, STRESS_POLICIES, qty, seeds).items()}
    worst = {p: {"cost": nominal[p], "z": [0.0] * 4} for p in STRESS_POLICIES}
    envelope = []
    for _ in range(budget):
        z = rng.uniform(-bound, bound, 4)
        c = {p: SY._m(v) for p, v in SY.costs(_stress_world(g, z), STRESS_POLICIES, qty, seeds).items()}
        envelope.append({"z": z.tolist(), "cost": c})
        for p, v in c.items():
            if v is not None and (worst[p]["cost"] is None or v > worst[p]["cost"]):
                worst[p] = {"cost": v, "z": z.tolist()}
    cat = {p: float(np.mean([e["cost"][p] is not None and e["cost"][p] > SY.CATASTROPHE_BPS for e in envelope]))
           for p in STRESS_POLICIES}
    return {"dataset": dataset_id, "qty_lots": qty, "nominal": nominal, "worst": worst,
            "catastrophe_fraction": cat, "envelope": envelope, "seconds": time.time() - t0,
            "search": {"budget": budget, "bound_log": bound, "axes": ["market_scale", "limit_scale", "momentum",
                                                                     "toxicity_proxy"]}}


def counterexample_task(dataset_id: str, root: str, design: str, seed: int, notionals: dict, lot: float,
                        conclusions: dict, budget: int = 12, bound: float = math.log(3.0)) -> dict:
    from .. import synthetic as SY, twin
    from ..execution import policies as P
    g, tick, _ = SY.calibrate([_load(dataset_id, root, design, "counterexample search world")])
    inst = dataset_id.split("-")[2].upper()
    rng = np.random.default_rng(np.random.SeedSequence([9, 86, int(seed)]))
    seeds = [int(seed) * 100 + k for k in range(4)]
    worlds = [np.zeros(4)] + [rng.uniform(-bound, bound, 4) for _ in range(budget)]
    out = {}
    for cid, c in conclusions.items():
        if c["endpoint"] not in ("mean_is_bps", "cvar90_is_bps"):
            out[cid] = {"status": "NOT_EVALUABLE", "reason": "endpoint needs historical passive markouts"}
            continue
        qty = SY.qty_for(g, tick, lot, notionals[c["mandate"]][inst])
        signs = []
        for z in worlds:
            w = _stress_world(g, z)
            vals = {}
            for p in (c["treatment"], c["control"]):
                res = [twin.synthetic_episode(w, P.make(p), qty=qty, seed=s, side=1 if k % 2 == 0 else -1)
                       for k, s in enumerate(seeds)]
                v = np.asarray([r["is_net_bps"] for r in res if r["status"] == "OK" and r["is_net_bps"] is not None])
                vals[p] = (float(np.sort(v)[-max(1, len(v) // 10):].mean()) if c["endpoint"] == "cvar90_is_bps"
                           else float(v.mean())) if len(v) else None
            if vals[c["treatment"]] is None or vals[c["control"]] is None:
                signs.append(None)
                continue
            d = vals[c["control"]] - vals[c["treatment"]]
            signs.append(float(d if c["direction"] == "decrease" else -d))
        flips = [i for i, s in enumerate(signs[1:], 1) if s is not None and signs[0] is not None
                 and np.sign(s) != np.sign(signs[0])]
        out[cid] = {"nominal_effect_bps": signs[0], "effects_bps": signs, "flip_worlds": flips,
                    "flip_fraction": len(flips) / max(sum(s is not None for s in signs[1:]), 1),
                    "smallest_flip": (min((float(np.abs(worlds[i]).max()) for i in flips)) if flips else None)}
    return {"dataset": dataset_id, "conclusions": out, "worlds": [w.tolist() for w in worlds]}


def _stress_days(args, protocol) -> list[str]:
    days = _days(args, protocol)
    out = []
    for inst in ("btcusdt", "ethusdt"):
        out += sorted([d for d in days if inst in d], key=lambda d: d[-10:])[:2]
    return out


def run_adversarial(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    days = _stress_days(args, protocol)
    notional = protocol["mandates"]["classes"]["medium"]["notional_usd"]
    base = common.seeds(args, "adversarial-study-v09", len(days))
    out = common.start(args)
    design = _design(args, "adversarial-study-v09")
    tasks = []
    for d, s in zip(days, base):
        inst = d.split("-")[2].upper()
        lot = float(next(x for x in protocol["datasets"] if x["instrument"] == inst)["lot_size"])
        tasks.append(Task(d, "lob.v09.studies.descriptive:adversarial_task",
                          {"dataset_id": d, "root": str(root), "design": design, "seed": int(s),
                           "notional": float(notional[inst]), "lot": lot}, seed=int(s), namespace=args.namespace))
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root / (args.out + ".checkpoint"), design={"days": days}, inputs=days)
    result = {"worlds": {r["dataset"]: r for r in rows}, "runner": info,
              "label": "SYNTHETIC bounded adversarial search (16 perturbations within x/÷3 of the calibrated ecology); "
                       "failure envelopes are reported, not tested"}
    return common.finish(args, out, analysis="adversarial-study-v09", config={"days": days, "policies": STRESS_POLICIES},
                         result=result, datasets=days)


def run_counterexample(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    days = _stress_days(args, protocol)
    notionals = {k: v["notional_usd"] for k, v in protocol["mandates"]["classes"].items()}
    base = common.seeds(args, "counterexample-study-v09", len(days))
    out = common.start(args)
    design = _design(args, "counterexample-study-v09")
    concl = protocol["conclusions"]["set"]
    tasks = []
    for d, s in zip(days, base):
        inst = d.split("-")[2].upper()
        lot = float(next(x for x in protocol["datasets"] if x["instrument"] == inst)["lot_size"])
        tasks.append(Task(d, "lob.v09.studies.descriptive:counterexample_task",
                          {"dataset_id": d, "root": str(root), "design": design, "seed": int(s),
                           "notionals": notionals, "lot": lot, "conclusions": concl}, seed=int(s),
                          namespace=args.namespace))
    rows, info = run_tasks(tasks, workers=args.workers, contract=common.contract(args),
                           checkpoint=root / (args.out + ".checkpoint"), design={"days": days}, inputs=days)
    summary = {}
    for cid in concl:
        fr = [r["conclusions"][cid].get("flip_fraction") for r in rows if "flip_fraction" in r["conclusions"][cid]]
        summary[cid] = {"mean_flip_fraction": float(np.mean(fr)) if fr else None, "worlds_searched": len(fr)}
    result = {"worlds": {r["dataset"]: r for r in rows}, "summary": summary, "runner": info,
              "label": "SYNTHETIC counterexample search: perturbed ecologies in which a conclusion's direction flips"}
    return common.finish(args, out, analysis="counterexample-study-v09", config={"days": days}, result=result,
                         datasets=days)


# ============================================================================= upstream-run syntheses


def run_ope(args) -> dict:
    up = common.read_input(args, "rl")
    out = common.start(args)
    rows = {}
    for day, w in up["worlds"].items():
        truth = w["true_mean_cost"]
        rows[day] = {name: {"dr_value": e["dr"], "pdis_value": e["pdis"], "wpdis_value": e["wpdis"],
                            "ood_action_rate": e["ood_action_rate"], "true_cost": truth.get(name)}
                     for name, e in w["estimates"].items()}
    err = {}
    for est in ("dr_value", "pdis_value", "wpdis_value"):
        pairs = [(-r[est], r["true_cost"]) for v in rows.values() for r in v.values()
                 if r.get(est) is not None and r.get("true_cost") is not None]
        if len(pairs) > 2:
            a, b = np.asarray(pairs).T
            err[est] = {"spearman_with_truth": float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]),
                        "pairs": len(pairs)}
    result = {"worlds": rows, "estimator_quality": err, "historical": {
        "status": "NOT_AVAILABLE", "reason": "historical data carry no behaviour propensities; IPS/SNIPS/DR are not "
                                             "identified on real days"},
              "upstream": common.inputs(args)["rl"], "label": "SYNTHETIC (E1) off-policy evaluation"}
    return common.finish(args, out, analysis="ope-study-v09", config={"input": "rl"}, result=result)


def run_ranking(args) -> dict:
    up = common.read_input(args, "bounds")
    out = common.start(args)
    ranking = up["ranking"]
    by_inst: dict[str, dict[str, list]] = {}
    for d, pairs in ranking.items():
        inst = d.split("-")[2]
        for k, r in pairs.items():
            if r["status"] == "IDENTIFIED":
                by_inst.setdefault(inst, {}).setdefault(k, []).append(1 if r["lower"] > 0 else -1)
    stab = {}
    for inst, pairs in by_inst.items():
        cons = [abs(np.mean(v)) for v in pairs.values() if len(v) >= 2]
        stab[inst] = {"identified_pairs": len(pairs), "mean_direction_consistency": float(np.mean(cons)) if cons else None}
    insts = sorted(by_inst)
    cross = None
    if len(insts) == 2:
        common_pairs = set(by_inst[insts[0]]) & set(by_inst[insts[1]])
        agree = [np.sign(np.mean(by_inst[insts[0]][k])) == np.sign(np.mean(by_inst[insts[1]][k])) for k in common_pairs]
        cross = {"pairs": len(common_pairs), "agreement": float(np.mean(agree)) if agree else None}
    identified = [r["status"] == "IDENTIFIED" for pairs in ranking.values() for r in pairs.values()]
    result = {"within_instrument": stab, "cross_instrument": cross,
              "identified_fraction": float(np.mean(identified)) if identified else None,
              "upstream": common.inputs(args)["bounds"],
              "label": "BOUNDED; stability of identified pairwise policy directions across days and instruments"}
    return common.finish(args, out, analysis="ranking-transfer-v09", config={"input": "bounds"}, result=result)


def run_uncertainty(args) -> dict:
    up = common.read_input(args, "bounds")
    out = common.start(args)
    rows = {}
    for cid, c in up["conclusions"].items():
        units = list(c["units"].values())
        if len(units) < 2:
            continue
        lo = np.asarray([u["lower"] for u in units])
        hi = np.asarray([u["upper"] for u in units])
        mid = (lo + hi) / 2
        rows[cid] = {"assumption_half_width_bps": float(np.mean(hi - lo) / 2),
                     "sampling_se_bps": float(mid.std(ddof=1) / math.sqrt(len(mid))),
                     "between_day_sd_bps": float(mid.std(ddof=1)), "units": len(units)}
        rows[cid]["assumption_share"] = rows[cid]["assumption_half_width_bps"] / max(
            rows[cid]["assumption_half_width_bps"] + rows[cid]["sampling_se_bps"], 1e-12)
    result = {"conclusions": rows, "upstream": common.inputs(args)["bounds"],
              "label": "BOUNDED; assumption (partial-identification) vs sampling uncertainty of each conclusion"}
    return common.finish(args, out, analysis="uncertainty-decomposition-v09", config={"input": "bounds"}, result=result)


def run_voi(args) -> dict:
    up = common.read_input(args, "bounds")
    out = common.start(args)
    rows = {}
    for cid, c in up["conclusions"].items():
        units = list(c["units"].values())
        if not units:
            continue
        lo = np.asarray([u["lower"] for u in units])
        hi = np.asarray([u["upper"] for u in units])
        robust = np.where(lo > 0, lo, 0.0)                    # adopt only when every assumption favours treatment
        oracle_hi = np.maximum(hi, 0.0)                       # best case of resolving the assumption
        rows[cid] = {"evpi_upper_bps": float(np.mean(oracle_hi - robust)),
                     "robust_value_bps": float(np.mean(robust)), "units": len(units)}
    result = {"conclusions": rows, "upstream": common.inputs(args)["bounds"],
              "label": "BOUNDED; upper bound on the value of resolving the replay assumptions (perfect information)"}
    return common.finish(args, out, analysis="value-of-information-v09", config={"input": "bounds"}, result=result)


# ============================================================================= local workload benchmark


def benchmark_task(dataset_id: str, root: str, design: str, seed: int) -> dict:
    from ..execution import policies as P
    from ..generators import classical as G
    from ..replay import engine as E
    from ..runtime import peak_memory_mb
    timings = {}
    t = time.perf_counter()
    c = _load(dataset_id, root, design, "local workload benchmark")
    timings["load_day_cache_s"] = time.perf_counter() - t
    t = time.perf_counter()
    book = E.DayBook(c)
    timings["daybook_s"] = time.perf_counter() - t
    a = E.Assumption("neutral", 0.05, 0.0, "transient")
    for p in ("twap", "passive_deadline", "mpc_cvar"):
        t = time.perf_counter()
        E.episode(book, P.make(p), window=0, notional_usd=1_000_000.0, assumption=a, context={"seed": 0})
        timings[f"episode_{p}_s"] = time.perf_counter() - t
    t = time.perf_counter()
    frame = G.training_frame([c])
    g = G.fit_family("hawkes", frame)
    timings["fit_hawkes_s"] = time.perf_counter() - t
    t = time.perf_counter()
    g.simulate(SIM_BINS, seed, warmup_bins=300)
    timings["simulate_3300_bins_s"] = time.perf_counter() - t
    return {"dataset": dataset_id, "timings": timings, "peak_memory_mb": peak_memory_mb()}


def run_benchmark(args) -> dict:
    root = common.root_of(args)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    days = _days(args, protocol)[:1]
    out = common.start(args)
    row = benchmark_task(days[0], str(root), _design(args, "benchmark-v09"), args.seed)
    from ..runtime import fingerprint
    result = {**row, "fingerprint": fingerprint(),
              "label": "LOCAL RESEARCH-WORKLOAD TIMINGS on one machine; not latency, throughput or HFT benchmarks"}
    return common.finish(args, out, analysis="benchmark-v09", config={"days": days, "seed": args.seed},
                         result=result, datasets=days)
