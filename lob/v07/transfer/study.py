"""M16: historical execution transfer v3 on the final transfer holdout (H12, H13; workstreams 5, 92).

predict  -> results/v07/m16/predictions: simulated costs of the learned policies (both algorithms, both
            variants, every training seed) in their training worlds (G0 point; 16 posterior draws).
seal     -> configs/v07/transfer-design.json + ledger ``seal_design`` reading ETH 2021-01-01: the policy
            evaluation lock (policies, predictions, H8 classification, mandate, rules) before any access.
evaluate -> results/v07/m16/<dataset>: bounded replay of every primary classical agent and every learned
            policy under the conservative and optimistic fill modes; H12 and H13.

Replay is ``lob.historical_sim.HistoricalSimulator`` (unchanged): displayed liquidity is reset to history,
passive fills are bounded, the hypothetical parent has no market impact, and no exact FIFO or exact
historical fill is claimed. Conservative and optimistic modes are separate self-consistent paths.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass, field
import gzip
from itertools import combinations
import json
import math
import os
from pathlib import Path

import numpy as np

from ...config import canonical_json
from ...engine import SimConfig
from ...experiments.registry import PROJECT_ROOT, sha256_file
from ...historical_sim import FILL_MODES, HistoricalEpisode, HistoricalSimulator
from ...replay.l2 import L2Replay
from ...v06.identifiability import clean
from ...v06.policies import load_policy
from ...v06.worlds import World, episode_params
from ..data import access
from ..data.schema import venue_spec
from ..data.tape import READER_LIMITS
from ..evidence.runs import finalize, new_run, verify_run, write_json
from ..execution import episode as ep
from ..generators.study import v06_inputs
from ..protocol import core as pr

DESIGN = "m16-transfer"
DESIGN_PATH = "configs/v07/transfer-design.json"
DATASET = "deribit-eth-perp-2021-01-01"
LEARNED = ("ppo:single", "dqn:single", "ppo:ensemble", "dqn:ensemble")
EPISODES = 144
WINDOW_S = ep.MANDATE["warmup_s"] + ep.MANDATE["horizon_s"] + ep.MANDATE["settlement_timeout_s"] + 1.0
PREDICTION_MARKETS = 64
SAMPLES = 2000
COST = ep.COST


class HistoricalWorld:
    """A replay 'world': ``build(seed)`` returns the frozen HistoricalSimulator for one episode and fill mode."""

    def __init__(self, config: dict, episode: HistoricalEpisode, fill_mode: str) -> None:
        self.config, self.episode, self.fill_mode = config, episode, fill_mode

    def build(self, seed: int) -> HistoricalSimulator:
        fields = {k: v for k, v in self.config.items() if k in SimConfig.__dataclass_fields__}
        cfg = SimConfig(**{**fields, "seed": seed, "record_events": False, "check_invariants": False})
        return HistoricalSimulator(cfg, self.episode, self.fill_mode)


@dataclass
class CompactEpisode:
    """Memory-compact form of a ``HistoricalEpisode``: snapshots packed into contiguous float64 arrays.

    ``bids``/``asks`` hold (tick, qty) columns of every snapshot back to back, delimited by ``*_offsets``. Ticks
    are integers far below 2**53, so they round-trip exactly; ``expand`` rebuilds the identical episode (same keys,
    values and key order). Resource control only: replay semantics are unchanged.
    """

    start_s: float
    times: np.ndarray
    bids: np.ndarray            # (2, total bid levels)
    bid_offsets: np.ndarray     # len(times) + 1
    asks: np.ndarray
    ask_offsets: np.ndarray
    prints: list
    tick_size: float
    clock_ratio: float
    label: str = ""
    meta: dict = field(default_factory=dict)

    @staticmethod
    def pack(updates: list) -> tuple:
        """[(t, bids (2, n), asks (2, m))] -> (times, bids, bid_offsets, asks, ask_offsets)."""
        out = [np.asarray([u[0] for u in updates], dtype=np.float64)]
        for side in (1, 2):
            sizes = [u[side].shape[1] for u in updates]
            out += [np.concatenate([u[side] for u in updates], axis=1),
                    np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)]
        return tuple(out)

    def expand(self) -> HistoricalEpisode:
        def book(levels, offsets, i):
            ticks, qty = levels[:, offsets[i]:offsets[i + 1]]
            return {int(k): float(q) for k, q in zip(ticks, qty)}
        updates = [(float(t), book(self.bids, self.bid_offsets, i), book(self.asks, self.ask_offsets, i))
                   for i, t in enumerate(self.times)]
        return HistoricalEpisode(self.start_s, updates, self.prints, self.tick_size, self.clock_ratio,
                                 label=self.label, meta=self.meta)


def _expanded(episode):
    return episode.expand() if isinstance(episode, CompactEpisode) else episode


def _side(levels, tick: float, lots_per_native: float, compact: bool):
    if compact:
        return np.asarray([[round(float(p) / tick) for p, _ in levels], [float(q) * lots_per_native for _, q in levels]],
                          dtype=np.float64).reshape(2, -1)
    return {int(round(float(p) / tick)): float(q) * lots_per_native for p, q in levels}


def _window_updates(window: list, start: float) -> list | None:
    """The v0.6 window rule: book state at ``start`` followed by the later updates; None if not covered."""
    updates = [u for u in window if u[0] <= start + WINDOW_S]
    if not updates or updates[0][0] > start:
        return None
    head = [u for u in updates if u[0] <= start][-1]
    return [(start, head[1], head[2])] + [u for u in updates if u[0] > start]


def extract_episodes(files: dict, *, tick: float, lots_per_native: float, label: str,
                     compact: bool = False) -> tuple[list, dict]:
    """The v0.6 episode rule (first capture + 300 s + 600 s k, clock ratio 1) with the v0.7 reader limits.

    ``compact=True`` packs each window as soon as replay time passes its end (``CompactEpisode``), bounding peak
    memory on busy days; it yields the same episodes.
    """
    replay = L2Replay(files["l2"], depth=20, **READER_LIMITS)
    starts = windows = first = previous = None
    finished: dict[int, tuple | None] = {}
    for state in replay:
        t = state.local_timestamp_us / 1e6
        if starts is None:
            first = t
            starts = [first + 300.0 + 600.0 * k for k in range(EPISODES)]
            windows = [[] for _ in starts]
        snapshot = (t, _side(state.bids, tick, lots_per_native, compact), _side(state.asks, tick, lots_per_native, compact))
        k = int((t - first - 300.0) // 600.0)
        for j in (k, k + 1):
            if 0 <= j < len(starts) and starts[j] - 5.0 <= t <= starts[j] + WINDOW_S:
                if not windows[j] and previous is not None:
                    windows[j].append(previous)
                windows[j].append(snapshot)
        previous = snapshot
        if compact:   # no later update can enter a window whose end has passed
            while len(finished) < len(starts) and t > starts[len(finished)] + WINDOW_S:
                j = len(finished)
                updates = _window_updates(windows[j], starts[j])
                finished[j] = None if updates is None else CompactEpisode.pack(updates)
                windows[j] = []
    if not replay.stats["complete"]:
        raise ValueError("L2 source incomplete")
    prints = [[] for _ in starts]
    opener = gzip.open if str(files["trades"]).endswith(".gz") else open
    with opener(files["trades"], "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            side = row["side"].lower()
            if side not in {"buy", "sell"}:
                continue
            t = int(row["local_timestamp"]) / 1e6
            k = int((t - first - 300.0) // 600.0)
            if 0 <= k < len(starts) and starts[k] < t <= starts[k] + WINDOW_S:
                prints[k].append((t, int(round(float(row["price"]) / tick)), float(row["amount"]) * lots_per_native,
                                  "BUY" if side == "buy" else "SELL"))
    episodes, skipped = [], []
    for k, start in enumerate(starts):
        meta = {"label": f"{label}#{k}", "meta": {"index": k}}
        if compact:
            packed = finished[k] if k in finished else (
                None if (u := _window_updates(windows[k], start)) is None else CompactEpisode.pack(u))
            if packed is None:
                skipped.append(k)
                continue
            episodes.append(CompactEpisode(start, *packed, prints[k], tick, 1.0, **meta))
            continue
        updates = _window_updates(windows[k], start)
        if updates is None:
            skipped.append(k)
            continue
        episodes.append(HistoricalEpisode(start, updates, prints[k], tick, 1.0, **meta))
    return episodes, {"episodes": len(episodes), "planned": len(starts), "skipped": skipped, "window_s": WINDOW_S}


# ----------------------------------------------------------------------------- learned-policy rows


def _learned_rows(task: tuple) -> list[dict]:
    """Learned policy on simulated worlds (``world_dict``) or historical episodes (``historical``)."""
    world_dict, agent, training_seed, seeds, policy_dir, historical = task
    import torch
    torch.set_num_threads(1)
    from ...runner import run_episode
    algorithm, variant = agent.split(":")
    model = load_policy(Path(policy_dir), algorithm, variant, training_seed)
    normalization = json.loads((Path(policy_dir) / f"normalization-{variant}.json").read_text(
        encoding="utf-8"))["normalization"]
    world = World.from_dict(world_dict)
    rows = []
    for item in (historical or [(None, s) for s in seeds]):
        episode_obj, seed = item
        episode_obj = _expanded(episode_obj) if historical else episode_obj
        modes = FILL_MODES if historical else (None,)
        for mode in modes:
            try:
                params = episode_params(world, ep.MANDATE, seed if not historical else 900_000 + episode_obj.meta["index"],
                                        normalization=normalization)
                if historical:
                    params["flow_extensions"] = None
                    params["historical"] = {"episode": episode_obj, "fill_mode": mode}
                row = run_episode("ppo", params, model=model)
            except Exception as exc:  # retained, never rerun
                row = {"status": "INVALID", "error": f"{type(exc).__name__}: {exc}", COST: None}
            keep = {"agent": agent, "training_seed": training_seed, "world": world.name, "status": row.get("status"),
                    COST: row.get(COST), "error": row.get("error")}
            if historical:
                keep.update(episode=episode_obj.meta["index"], fill_mode=mode)
            else:
                keep.update(seed=seed)
            rows.append(json.loads(canonical_json(keep)))
    return rows


def _classical_rows(task: tuple) -> list[dict]:
    episodes, agent, config, profile = task
    rows = []
    for e in episodes:
        e = _expanded(e)   # one episode expanded at a time
        for mode in FILL_MODES:
            row = ep.run(HistoricalWorld(config, e, mode), agent, ep.MANDATE, 900_000 + e.meta["index"], profile=profile)
            rows.append({"agent": agent, "training_seed": None, "episode": e.meta["index"], "fill_mode": mode,
                         "status": row.get("status"), COST: row.get(COST), "error": row.get("error")})
    return rows


def _workers() -> int:
    """Resource control only: rows are keyed by episode with per-episode seeds, so results do not depend on it."""
    cap = int(os.environ.get("CLEOLOB_WORKERS") or 14)
    return max(1, min(cap, 14, (os.cpu_count() or 2) - 2))


def predict(out: str | Path, *, policy_run: str, root: Path = PROJECT_ROOT, markets: int = PREDICTION_MARKETS) -> dict:
    report = verify_run(root / policy_run, root=root)
    if not report["valid"]:
        raise ValueError(f"{policy_run} is not valid: {report['issues']}")
    plan = json.loads((root / policy_run / "registration.json").read_text(encoding="utf-8"))
    summary = json.loads((root / policy_run / "training_summary.json").read_text(encoding="utf-8"))
    trained = {(m["algorithm"], m["variant"], m["seed"]) for m in summary["models"] if "failed" not in m}
    start = pr.load_protocol(root / pr.PROTOCOL_PATH)["seeds"]["execution"]["market_seed_start"] + 200_000
    seeds = list(range(start, start + markets))
    out = new_run(root / out)
    tasks = []
    for agent in LEARNED:
        algorithm, variant = agent.split(":")
        for world in plan["training_worlds"][variant]:
            for s in plan["training_seeds"]:
                if (algorithm, variant, s) in trained:
                    tasks.append((world, agent, s, seeds, str(root / policy_run), None))
    with ProcessPoolExecutor(max_workers=_workers()) as executor:
        rows = [r for part in executor.map(_learned_rows, tasks, chunksize=1) for r in part]
    with (out / "rows.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    means = {}
    for agent in LEARNED:
        for s in plan["training_seeds"]:
            values = [r[COST] for r in rows if r["agent"] == agent and r["training_seed"] == s and r[COST] is not None]
            means[f"{agent}|{s}"] = float(np.mean(values)) if values else None
    result = {"markets": markets, "rows": len(rows), "invalid_rows": sum(r[COST] is None for r in rows),
              "training_seed_means": means, "untrained": sorted(set((a, v, s) for a in ("ppo", "dqn")
                                                                    for v in ("single", "ensemble")
                                                                    for s in plan["training_seeds"]) - trained)}
    finalize(out, analysis="m16-learned-predictions", dataset_ids=[], config={"policy_run": policy_run, "markets": markets},
             result=clean(result), root=root, seeds={"market_start": start})
    return result


# ----------------------------------------------------------------------------- statistics


def _episode_means(rows: list[dict], agent: str, mode: str) -> dict[int, float]:
    """Per episode: cost averaged over training seeds (NaN when any seed is INVALID)."""
    values: dict[int, list[float]] = {}
    for r in rows:
        if r["agent"] == agent and r["fill_mode"] == mode:
            values.setdefault(int(r["episode"]), []).append(r[COST] if r[COST] is not None else math.nan)
    return {k: float(np.mean(v)) if all(map(math.isfinite, v)) else math.nan for k, v in values.items()}


def h12(rows: list[dict], predictions: list[dict], *, alpha: float, samples: int, seed: int) -> dict:
    """|predicted - historical| for the posterior (``ensemble``) minus the single variant, per algorithm and mode."""
    rng = np.random.default_rng(seed)
    members = {}
    for algorithm in ("ppo", "dqn"):
        for mode in FILL_MODES:
            hist, pred = {}, {}
            for variant in ("single", "ensemble"):
                agent = f"{algorithm}:{variant}"
                by_seed: dict = {}
                for r in rows:
                    if r["agent"] == agent and r["fill_mode"] == mode:
                        by_seed.setdefault(r["training_seed"], {})[int(r["episode"])] = (
                            r[COST] if r[COST] is not None else math.nan)
                hist[variant] = by_seed
                sim: dict = {}
                for r in predictions:
                    if r["agent"] == agent and r[COST] is not None:
                        sim.setdefault(r["training_seed"], []).append(r[COST])
                pred[variant] = {s: np.asarray(v) for s, v in sim.items()}
            seeds = sorted(set(hist["single"]) & set(hist["ensemble"]) & set(pred["single"]) & set(pred["ensemble"]))
            episodes = sorted(set.intersection(*[set(hist[v][s]) for v in hist for s in seeds])) if seeds else []
            matrix = {v: np.asarray([[hist[v][s][e] for e in episodes] for s in seeds]) for v in hist}
            if len(seeds) < 2 or len(episodes) < 10 or not all(np.isfinite(m).all() for m in matrix.values()):
                members[f"{algorithm}@{mode}"] = {"status": "NOT_AVAILABLE", "reason": "missing or INVALID rows"}
                continue

            def gap(v, seed_idx, ep_idx, sim_draw=None):
                hist_mean = matrix[v][np.ix_(seed_idx, ep_idx)].mean()
                predicted = np.mean([(sim_draw[v][s] if sim_draw else pred[v][seeds[s]].mean()) for s in seed_idx])
                return abs(predicted - hist_mean)
            all_s, all_e = np.arange(len(seeds)), np.arange(len(episodes))
            point = {v: gap(v, all_s, all_e) for v in matrix}
            draws = np.empty(samples)
            for b in range(samples):
                s_idx = rng.integers(0, len(seeds), len(seeds))
                e_idx = rng.integers(0, len(episodes), len(episodes))
                sim_draw = {v: {i: pred[v][seeds[i]][rng.integers(0, len(pred[v][seeds[i]]), len(pred[v][seeds[i]]))].mean()
                                for i in set(s_idx.tolist())} for v in matrix}
                draws[b] = gap("ensemble", s_idx, e_idx, sim_draw) - gap("single", s_idx, e_idx, sim_draw)
            low, high = float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))
            status = "ESTABLISHED" if high < 0 else "FAILED" if low > 0 else "NOT_ESTABLISHED"
            members[f"{algorithm}@{mode}"] = {"gap_single_bps": point["single"], "gap_posterior_bps": point["ensemble"],
                                              "difference": point["ensemble"] - point["single"], "ci": [low, high],
                                              "alpha": alpha, "training_seeds": len(seeds), "episodes": len(episodes),
                                              "status": status}
    statuses = [m["status"] for m in members.values()]
    summary = ("ESTABLISHED" if "ESTABLISHED" in statuses else "NOT_AVAILABLE" if all(s == "NOT_AVAILABLE" for s in statuses)
               else "NOT_ESTABLISHED")
    return {"members": members, "status": summary, "rule": "ESTABLISHED per member if the Bonferroni interval of "
            "gap(posterior) - gap(single) lies below zero"}


def h13(rows: list[dict], robust: dict, *, alpha: float, samples: int, seed: int) -> dict:
    """For each H8-robust pair: replay difference under both fill bounds; survives if same sign and determinate."""
    if not robust:
        return {"status": "INCONCLUSIVE", "reason": "no H8-robust pair (NOT_EVALUABLE)", "pairs": {}}
    adjusted = alpha / (2 * len(robust))
    rng = np.random.default_rng(seed)
    out = {}
    for pair, edge in robust.items():
        a, b = pair.split("|")
        expected = -1 if edge == "ROBUSTLY_BETTER" else 1
        modes = {}
        for mode in FILL_MODES:
            x, y = _episode_means(rows, a, mode), _episode_means(rows, b, mode)
            keys = sorted(set(x) & set(y))
            d = np.asarray([x[k] - y[k] for k in keys])
            d = d[np.isfinite(d)]
            if len(d) < 10:
                modes[mode] = {"status": "NOT_AVAILABLE"}
                continue
            means = d[rng.integers(0, len(d), (samples, len(d)))].mean(1)
            low, high = float(np.quantile(means, adjusted / 2)), float(np.quantile(means, 1 - adjusted / 2))
            direction = -1 if high < 0 else 1 if low > 0 else 0
            modes[mode] = {"mean_bps": float(d.mean()), "ci": [low, high], "direction": direction, "episodes": len(d)}
        dirs = [m.get("direction") for m in modes.values()]
        status = ("SURVIVES" if all(x == expected for x in dirs) else
                  "FAILED" if any(x == -expected for x in dirs) else
                  "NOT_AVAILABLE" if None in dirs else "NOT_ESTABLISHED")
        out[pair] = {"expected_direction": expected, "modes": modes, "status": status}
    statuses = [p["status"] for p in out.values()]
    summary = ("ESTABLISHED" if statuses and all(s == "SURVIVES" for s in statuses) else
               "FAILED" if any(s == "FAILED" for s in statuses) else "NOT_ESTABLISHED")
    return {"status": summary, "adjusted_alpha": adjusted, "pairs": out, "kind": "conditional descriptive"}


# ----------------------------------------------------------------------------- lock and evaluation


def seal(*, policy_run: str, prediction_run: str, execution_run: str, root: Path = PROJECT_ROOT) -> dict:
    for run_dir in (policy_run, prediction_run, execution_run):
        report = verify_run(root / run_dir, root=root)
        if not report["valid"]:
            raise ValueError(f"{run_dir} is not valid: {report['issues']}")
    path = root / DESIGN_PATH
    if path.exists():
        raise FileExistsError("transfer design already written; it is immutable")
    families = json.loads((root / pr.CONFIG_DIR / "statistical-families.json").read_text(encoding="utf-8"))["families"]
    execution = json.loads((root / execution_run / "result.json").read_text(encoding="utf-8"))
    robust = {k: v["edge"] for k, v in execution["H8"]["edges"].items() if k in execution["H8"]["robust_pairs"]}
    summary = json.loads((root / policy_run / "training_summary.json").read_text(encoding="utf-8"))
    document = {"schema": "cleolob-v07-transfer-design-1", "dataset": DATASET, "policy_run": policy_run,
                "policy_models_sha256": {f"{m['algorithm']}-{m['variant']}-{m['seed']}": m.get("model_sha256")
                                         for m in summary["models"]},
                "prediction_run": prediction_run,
                "prediction_result_sha256": pr.file_sha256(root / prediction_run / "result.json"),
                "prediction_rows_sha256": sha256_file(root / prediction_run / "rows.jsonl"),
                "execution_run": execution_run,
                "execution_result_sha256": pr.file_sha256(root / execution_run / "result.json"),
                "H8_robust_pairs": robust, "agents_classical": list(ep.PRIMARY), "agents_learned": list(LEARNED),
                "vwap_profile": execution["vwap_profile"], "mandate": ep.MANDATE, "ac_parameters": ep.AC_V06,
                "episodes": {"count": EPISODES, "rule": "first capture + 300 s + 600 s k", "window_s": WINDOW_S},
                "fill_modes": list(FILL_MODES), "H12_alpha": families["F12_transfer"]["adjusted_alpha"],
                "H13_alpha": 0.05, "samples": SAMPLES,
                "seed": pr.load_protocol(root / pr.PROTOCOL_PATH)["seeds"]["bootstrap"]["transfer"],
                "reader_limits": READER_LIMITS,
                "lock": "policies, predictions, the H8 classification, mandate and rules are fixed before access"}
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    pr.append_event(root / pr.LEDGER_PATH, "seal_design", design=DESIGN, root=root,
                    protocol=pr.load_protocol(root / pr.PROTOCOL_PATH),
                    reason="policy evaluation lock: transfer design sealed before the final transfer holdout",
                    payload={"design_sha256": pr.document_sha256(document), "reads": [DATASET], "posthoc": False,
                             "design_path": DESIGN_PATH})
    return document


def evaluate(out: str | Path, *, root: Path = PROJECT_ROOT, downloader=None) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    document = json.loads((root / DESIGN_PATH).read_text(encoding="utf-8"))
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), protocol)
    if state.designs.get(DESIGN, {}).get("design_sha256") != pr.document_sha256(document):
        raise ValueError("transfer design is not sealed in the ledger")
    if sha256_file(root / document["prediction_run"] / "rows.jsonl") != document["prediction_rows_sha256"]:
        raise ValueError("predictions differ from the sealed design")
    inputs = v06_inputs(root)
    frozen = inputs["frozen"]
    out = new_run(root / out)
    run_name = out.relative_to(root).as_posix()
    result: dict = {"dataset": DATASET, "label": "FINAL_TRANSFER"}
    try:
        files = access.acquire(DATASET, use="evaluate", design=DESIGN, purpose="final historical transfer", root=root,
                               **({"downloader": downloader} if downloader else {}))
    except access.DataNotAvailable as exc:
        result.update(status="NOT_AVAILABLE", reason=str(exc))
        finalize(out, analysis="m16-transfer", dataset_ids=[DATASET], config=document, result=result, root=root)
        return result
    declaration = pr.dataset_declaration(protocol, DATASET)
    tick = float(venue_spec(declaration["venue"], declaration["instrument"]).tick)
    episodes, extraction = extract_episodes(files, tick=tick, lots_per_native=1.0 / frozen["scale_native_per_lot"],
                                            label=DATASET, compact=True)
    pr.append_event(root / pr.LEDGER_PATH, "access", dataset=DATASET, role=declaration["role"], design=DESIGN,
                    reason="replay episodes extracted", protocol=protocol, root=root,
                    payload={"use": "evaluate", "stage": "parsed", "episodes": extraction["episodes"]})
    result["extraction"] = extraction
    selected = inputs["selection"]["selected"]
    workers = _workers()
    tasks = [(episodes[i::workers], agent, selected["config"], document["vwap_profile"])
             for agent in document["agents_classical"] for i in range(workers) if episodes[i::workers]]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = [r for part in executor.map(_classical_rows, tasks, chunksize=1) for r in part]
    plan = json.loads((root / document["policy_run"] / "registration.json").read_text(encoding="utf-8"))
    world = World("G0_point", "selected", selected["config"], selected["extensions"]).to_dict()
    learned_tasks = [(world, agent, s, None, str(root / document["policy_run"]),
                      [(e, None) for e in episodes[i::workers]])
                     for agent in LEARNED for s in plan["training_seeds"] for i in range(workers) if episodes[i::workers]]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows += [r for part in executor.map(_learned_rows, learned_tasks, chunksize=1) for r in part]
    with (out / "rows.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    predictions = [json.loads(line) for line in (root / document["prediction_run"] / "rows.jsonl").read_text(
        encoding="utf-8").splitlines()]
    result["H12"] = h12(rows, predictions, alpha=document["H12_alpha"], samples=document["samples"], seed=document["seed"])
    result["H13"] = h13(rows, document["H8_robust_pairs"], alpha=document["H13_alpha"], samples=document["samples"],
                        seed=document["seed"] + 13)
    result["means"] = {f"{mode}|{agent}": float(np.nanmean(list(_episode_means(rows, agent, mode).values())))
                       for mode in FILL_MODES for agent in (*document["agents_classical"], *LEARNED)}
    result["pairs_all"] = {}
    for a, b in combinations(document["agents_classical"], 2):
        entry = {}
        for mode in FILL_MODES:
            x, y = _episode_means(rows, a, mode), _episode_means(rows, b, mode)
            d = np.asarray([x[k] - y[k] for k in sorted(set(x) & set(y))])
            d = d[np.isfinite(d)]
            entry[mode] = float(d.mean()) if len(d) else None
        result["pairs_all"][f"{a}|{b}"] = entry
    result["invalid_rows"] = sum(r[COST] is None for r in rows)
    result["interpretation"] = ("Bounded historical replay with no market impact of the hypothetical parent; no exact "
                                "fill, FIFO, profitability or live claim.")
    write_json(out, "summary.json", {"rows": len(rows), "invalid_rows": result["invalid_rows"]})
    access.mark_evaluated(DATASET, use="evaluate", design=DESIGN, run=run_name, root=root)
    finalize(out, analysis="m16-transfer", dataset_ids=[DATASET], config=document, result=clean(result), root=root,
             seeds={"bootstrap": document["seed"]})
    access.mark_inspected(DATASET, run=run_name, root=root)
    return result
