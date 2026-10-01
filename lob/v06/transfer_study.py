"""M14/M24 runs: sealed transfer design and bounded historical execution (H11, H12, sources A-D).

seal     -> configs/v06/transfer-design.json + ledger ``m14-transfer-design`` (reads the transfer holdout):
            frozen environment, registered policies, the execution-evaluation result that supplies the
            simulated predictions, episode rule, agents, fill modes, families, alphas and seeds.
evaluate -> results/v06/m14/<dataset>: every agent (learned: every training seed) on every episode under
            both fill modes, then H12, H11 and the source comparison. Retrospective datasets use the
            same frozen design with the 'retrospective' use and are labelled retrospective.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from ..config import canonical_json
from ..experiments.registry import PROJECT_ROOT, sha256_file
from ..historical_sim import FILL_MODES
from . import protocol as pr
from .data import acquire
from .evidence import finalize, new_run, write_json
from .execution import CLASSICAL, COMPLETION, COST, LEARNED, cells
from .execution_study import _workers, read_freeze
from .history import TICKS
from .policies import load_policy
from .realism_study import read_design
from .calibration_study import DESIGN_RUN
from .transfer import PAIRS, extract_episodes, fill_semantics, historical_pairs, source_comparison, transfer_gap
from .worlds import World, episode_params

ANALYSIS = "m14-transfer-design"
DESIGN_PATH = "configs/v06/transfer-design.json"


def seal(*, policy_run: str, evaluation_run: str, root: Path = PROJECT_ROOT) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    freeze_doc = read_freeze(root)
    if (root / DESIGN_PATH).exists():
        raise FileExistsError("transfer design already written; it is immutable")
    evaluation = root / evaluation_run
    plan = json.loads((root / policy_run / "registration.json").read_text(encoding="utf-8"))
    worlds = [World.from_dict(w) for w in freeze_doc["worlds"]]
    members = [w.name for w in worlds if w.kind in {"selected", "member"}]
    interventions = [w.name for w in worlds if w.kind == "intervention"]
    regime = [w.name for w in worlds if w.kind == "regime"]
    families = protocol["statistics"]["families"]
    document = {
        "schema": "cleolob-v06-transfer-design-1", "freeze_sha256": pr.document_sha256(freeze_doc),
        "policy_registration_sha256": pr.document_sha256(plan), "policy_run": policy_run,
        "evaluation_run": evaluation_run, "evaluation_result_sha256": pr.file_sha256(evaluation / "result.json"),
        "evaluation_episodes_sha256": sha256_file(evaluation / "episodes.jsonl"),
        "datasets": {"registered": "deribit-eth-perp-2020-10-01", "retrospective": "deribit-eth-perp-2020-08-01"},
        "episodes": {"count": 144, "start": "first capture + 300 s + 600 s * k", "book_levels": 20,
                     "window_s": freeze_doc["mandate"]["warmup_s"] + freeze_doc["mandate"]["horizon_s"]
                     + freeze_doc["mandate"]["settlement_timeout_s"] + 1.0, "clock_ratio": 1.0,
                     "lots_per_native": "1 / development depth scale"},
        "agents": list(CLASSICAL) + list(LEARNED), "fill_modes": list(FILL_MODES), "pairs": [list(p) for p in PAIRS],
        "ac_parameters": freeze_doc["ac_parameters"]["selected"]["selected"],
        "sources": {"A_single": ["selected"], "B_ensemble": members, "C_regime": regime or None,
                    "D_robust": members + interventions},
        "H12": {"alpha": families["F9_fill_semantics"]["adjusted_alpha"], "family_size": 30,
                "note": "per pair and mode two-sided interval at 0.05/30"},
        "H11": {"alpha": families["F8_transfer_learning"]["adjusted_alpha"], "reference": "twap",
                "single_worlds": ["selected"], "ensemble_worlds": members},
        "sources_alpha": 0.05, "samples": 5000, "seed": protocol["seeds"]["bootstrap"]["transfer"],
        "never_claimed": protocol["transfer"]["never_claimed"]}
    (root / DESIGN_PATH).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    pr.append_event(root / pr.LEDGER_PATH, "seal_design", {
        "analysis": ANALYSIS, "design_sha256": pr.document_sha256(document),
        "reads": [document["datasets"]["registered"]],
        "note": "Historical transfer v2 design sealed after the policy evaluation lock and before transfer access."},
        protocol=protocol)
    return document


def _chunk(task: tuple) -> list[dict]:
    episodes, dataset, agent, training_seed, mandate, ac, policy_dir, world_dict = task
    import torch
    torch.set_num_threads(1)
    from ..runner import run_episode
    model, normalization = None, None
    if ":" in agent:
        algorithm, variant = agent.split(":")
        model = load_policy(Path(policy_dir), algorithm, variant, training_seed)
        normalization = json.loads((Path(policy_dir) / f"normalization-{variant}.json").read_text(
            encoding="utf-8"))["normalization"]
    world = World.from_dict(world_dict)
    rows = []
    for episode in episodes:
        for mode in FILL_MODES:
            try:
                params = episode_params(world, mandate, 900_000 + episode.meta["index"], normalization=normalization,
                                        ac=ac if agent == "ac" else None)
                params["flow_extensions"] = None
                params["historical"] = {"episode": episode, "fill_mode": mode}
                row = run_episode("ppo" if model is not None else agent, params, model=model)
            except Exception as exc:  # retained, never rerun
                row = {"status": "INVALID", "error": f"{type(exc).__name__}: {exc}", COST: None, COMPLETION: None}
            keep = {k: row.get(k) for k in (COST, COMPLETION, "status", "error", "mandate_final_settlement_completion",
                                             "mandate_fill_fraction_at_horizon")}
            keep.update(dataset=dataset, episode=episode.meta["index"], agent=agent, training_seed=training_seed,
                        fill_mode=mode)
            rows.append(json.loads(canonical_json(keep)))
    return rows


def evaluate(dataset_id: str, out: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    document = json.loads((root / DESIGN_PATH).read_text(encoding="utf-8"))
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), protocol)
    if state.designs.get(ANALYSIS, {}).get("design_sha256") != pr.document_sha256(document):
        raise ValueError("transfer design is not sealed in the ledger")
    freeze_doc = read_freeze(root)
    if pr.document_sha256(freeze_doc) != document["freeze_sha256"]:
        raise ValueError("environment freeze differs from the transfer design")
    evaluation = root / document["evaluation_run"]
    if pr.file_sha256(evaluation / "result.json") != document["evaluation_result_sha256"] or \
            sha256_file(evaluation / "episodes.jsonl") != document["evaluation_episodes_sha256"]:
        raise ValueError("simulated predictions differ from the transfer design")
    declaration = pr.dataset_declaration(protocol, dataset_id)
    use = "retrospective" if declaration["role"] == "retrospective" else "evaluate"
    frozen = read_design(root / DESIGN_RUN, root=root)
    out = new_run(out)
    files = acquire(dataset_id, use=use, analysis=ANALYSIS if use == "evaluate" else f"{ANALYSIS}-retrospective",
                    purpose="historical execution transfer v2", root=root)
    episodes, extraction = extract_episodes(files, tick=TICKS[declaration["instrument"]],
                                            lots_per_native=1.0 / frozen["scale_native_per_lot"],
                                            window_s=document["episodes"]["window_s"], label=dataset_id)
    policy_dir = root / document["policy_run"]
    plan = json.loads((policy_dir / "registration.json").read_text(encoding="utf-8"))
    summary = json.loads((policy_dir / "training_summary.json").read_text(encoding="utf-8"))
    trained = {(m["algorithm"], m["variant"], m["seed"]) for m in summary["models"] if "failed" not in m}
    selected = next(w for w in freeze_doc["worlds"] if w["name"] == "selected")
    workers = _workers()
    tasks = []
    for agent in document["agents"]:
        for seed in (plan["training_seeds"] if ":" in agent else [None]):
            if ":" in agent and (*agent.split(":"), seed) not in trained:
                continue
            for i in range(workers):
                chunk = episodes[i::workers]
                if chunk:
                    tasks.append((chunk, dataset_id, agent, seed, freeze_doc["mandate"], document["ac_parameters"],
                                  str(policy_dir), selected))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = [row for part in executor.map(_chunk, tasks, chunksize=1) for row in part]
    rows.sort(key=lambda r: (r["agent"], str(r["training_seed"]), r["episode"], r["fill_mode"]))
    with (out / "episodes.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    pairs = historical_pairs(rows, alpha=document["H12"]["alpha"], samples=document["samples"],
                             seed=document["seed"])
    simulated = [json.loads(line) for line in (evaluation / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    lock = json.loads((evaluation / "evaluation_lock.json").read_text(encoding="utf-8"))
    cell_map = cells(simulated, lock["market_seeds"])
    sources = {k: v for k, v in document["sources"].items() if v}
    result = {"dataset": dataset_id, "role": declaration["role"], "retrospective": use == "retrospective",
              "extraction": extraction, "rows": len(rows),
              "invalid_rows": sum(r.get("status") == "INVALID" for r in rows),
              "H12_fill_semantics": fill_semantics(pairs), "historical_pairs": pairs,
              "sources": source_comparison(cell_map, sources, pairs, alpha=document["sources_alpha"],
                                           samples=2000, seed=document["seed"] + 7),
              "H11": {algorithm: transfer_gap(rows, cell_map, algorithm=algorithm,
                                              single_worlds=document["H11"]["single_worlds"],
                                              ensemble_worlds=document["H11"]["ensemble_worlds"],
                                              alpha=document["H11"]["alpha"], samples=document["samples"],
                                              seed=document["seed"] + 11 + i)
                      for i, algorithm in enumerate(("ppo", "dqn"))},
              "means": _means(rows),
              "interpretation": ("Bounded historical replay with no market impact of the hypothetical parent; "
                                 "conservative and optimistic modes are separate self-consistent paths; no exact fill, "
                                 "FIFO, profitability or live claim.")}
    write_json(out, "summary.json", {k: result[k] for k in ("dataset", "rows", "invalid_rows")})
    finalize(out, analysis=f"m14-transfer-{dataset_id}", dataset_ids=[dataset_id],
             config={"design_sha256": pr.document_sha256(document)}, result=result, root=root,
             seeds={"bootstrap": document["seed"]})
    return result


def _means(rows: list[dict]) -> dict:
    from .transfer import episode_matrix
    import numpy as np
    out = {}
    for mode in FILL_MODES:
        for agent in sorted({r["agent"] for r in rows}):
            values = list(episode_matrix(rows, mode, agent).values())
            finite = [v for v in values if v == v]
            completion = [r.get(COMPLETION) for r in rows if r["agent"] == agent and r["fill_mode"] == mode]
            out[f"{mode}|{agent}"] = {"mean_cost_bps": float(np.mean(finite)) if finite else None,
                                      "episodes": len(values), "valid_episodes": len(finite),
                                      "within_horizon_completion_rate": float(np.mean([c is True for c in completion]))
                                      if completion else None}
    return out
