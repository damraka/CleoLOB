"""M8-M11/M14-M16 runs: environment freeze, policy registration/training and world evaluation.

freeze   -> configs/v06/environment-freeze.json (committed) + ledger seal ``environment-freeze``:
            mandate, worlds (selected, ensemble members, structural interventions, regime models,
            v0.5 control), agents, seeds, hyperparameters, AC parameters per world and the hashes
            of every execution-path source file. Written before any policy training.
train    -> results/v06/m10/policies: registration seal (``policy-registration``) before fitting,
            observation normalization sealed per variant, 16 models, failures retained.
evaluate -> results/v06/m10/evaluation: evaluation lock, every world x agent x market seed,
            then H5, H6, H7, rank stability and model-risk decomposition from the frozen rules.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

from ..config import canonical_json
from ..experiments.registry import PROJECT_ROOT, sha256_file, utc_now
from . import protocol as pr
from .calibration import run_tasks
from .calibration_study import DESIGN_RUN, read_selection, record_use
from .evidence import finalize, new_run, write_json
from .execution import (CLASSICAL, COMPLETION, COST, LEARNED, PRIMARY, cells, decomposition, disagreement,
                        equifinality, execution_sensitive_realism, rank_stability, summarize_cell)
from .misspecification import interventions
from .observables import FAMILIES, pool
from .policies import ALGORITHMS, VARIANTS, fit_observation_normalization, load_policy, train_one
from .realism_study import load_sketches, read_design
from .worlds import World, episode_params

FREEZE_PATH = "configs/v06/environment-freeze.json"
EXECUTION_FILES = ("lob/runner.py", "lob/rl_env.py", "lob/engine.py", "lob/sim_v2.py", "lob/execution.py",
                   "lob/controls.py", "lob/completion.py", "lob/mandate.py", "lob/settlement.py", "lob/accounting.py",
                   "lob/risk.py", "lob/observations.py", "lob/simulators.py", "lob/historical_sim.py",
                   "lob/fill_bounds.py", "lob/policy_study.py", "lob/core_study.py",
                   "lob/v06/worlds.py", "lob/v06/policies.py", "lob/v06/execution.py", "lob/v06/misspecification.py")
PPO = {"n_steps": 256, "batch_size": 64, "n_epochs": 5, "learning_rate": 0.0003, "gamma": 0.999, "ent_coef": 0.005,
       "device": "cpu"}
DQN = {"buffer_size": 10000, "learning_starts": 64, "batch_size": 64, "train_freq": 4, "gradient_steps": 1,
       "target_update_interval": 128, "exploration_fraction": 0.4, "exploration_initial_eps": 1.0,
       "exploration_final_eps": 0.05, "learning_rate": 0.0003, "gamma": 0.999, "device": "cpu"}


def _workers() -> int:
    return int(os.environ.get("CLEOLOB_WORKERS", max(1, min(14, (os.cpu_count() or 2) - 2))))


# ----------------------------------------------------------------------------- freeze


def _ac_task(task: tuple) -> dict:
    world_dict, mandate, seeds = task
    from ..controls import estimate_ac_parameters
    from ..runner import cfg_from_params
    world = World.from_dict(world_dict)
    params = episode_params(world, mandate, seeds[0])
    try:
        fit = estimate_ac_parameters(cfg_from_params(params), seeds, horizon=mandate["horizon_s"],
                                     sample_dt=mandate["horizon_s"] / 10, execution_interval=mandate["decision_dt_s"],
                                     warmup_seconds=mandate["warmup_s"], flow_extensions=world.extensions or None)
    except Exception as exc:  # retained as an identification failure
        fit = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    selected = {"temp_impact": params["temp_impact"], "sigma": params["sigma"], "risk_aversion": params["risk_aversion"]}
    if fit.get("status") == "PASS":
        selected.update(temp_impact=fit["temp_impact"], sigma=fit["sigma"])
        if fit["sigma"] > 0:
            selected["risk_aversion"] = fit["temp_impact"] / (fit["sigma"] * mandate["horizon_s"]) ** 2
    return {"world": world.name, "fit": json.loads(canonical_json(fit)), "selected": selected}


def build_worlds(selection: dict, regime: dict | None) -> list[World]:
    selected = World("selected", "selected", selection["selected"]["config"], selection["selected"]["extensions"],
                     f"calibration v3 selected ({selection['selected']['key']})")
    worlds = [selected]
    for member in selection["ensemble"][1:]:
        worlds.append(World(f"member:{member['key']}", "member", member["config"], member["extensions"],
                            "materially distinct near-optimal parameter vector"))
    worlds += [World(f"intervention:{w.name}", w.kind, w.config, w.extensions, w.note) for w in interventions(selected)]
    for name, model in (regime or {}).items():
        worlds.append(World(f"regime:{name}", "regime", model["config"], model["extensions"],
                            f"regime-conditioned calibration ({name} volatility)"))
    control = selection["control"]
    worlds.append(World("control:v05", "control", control["config"], control["extensions"],
                        "v0.5 selected calibration-v2 model (unchanged)"))
    return worlds


def freeze(*, selection_run: str, regime: dict | None, root: Path = PROJECT_ROOT, seal: bool = True) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    if (root / FREEZE_PATH).exists():
        raise FileExistsError("environment freeze already written; it is immutable")
    selection = read_selection(root / selection_run, root=root)
    design_run = read_design(root / DESIGN_RUN, root=root)
    from .history import block_raws, load
    from .observables import concatenate
    dev_id = pr.datasets_by_role(protocol, "development")[0]
    tape, _ = load(dev_id, use="develop", analysis="environment-freeze", purpose="mandate quantity (median L1 depth)",
                   root=root, scale=design_run["scale_native_per_lot"])
    quantity = int(round(float(np.median(concatenate(block_raws(tape)[0], "depth_l1")))))
    del tape
    spec = protocol["execution"]["mandate"]
    mandate = {"side": spec["side"], "quantity": quantity, "horizon_s": spec["horizon_s"],
               "decision_dt_s": spec["decision_dt_s"], "warmup_s": spec["warmup_s"],
               "settlement_timeout_s": spec["settlement_timeout_s"], "fees": spec["fees"],
               "terminal_penalty_bps": spec["terminal_penalty_bps"], "completion_urgency_fraction": 0.8,
               "pov_participation": spec["pov_participation"]}
    worlds = build_worlds(selection, regime)
    seeds = protocol["seeds"]["execution"]
    with ProcessPoolExecutor(max_workers=_workers()) as executor:
        ac = list(executor.map(_ac_task, [(w.to_dict(), mandate, seeds["identification_seeds"]) for w in worlds]))
    document = {
        "schema": "cleolob-v06-environment-freeze-1", "frozen_at_utc": utc_now(), "mandate": mandate,
        "worlds": [w.to_dict() for w in worlds], "ac_parameters": {a["world"]: a for a in ac},
        "agents": {"classical": list(CLASSICAL), "learned": list(LEARNED), "primary_pairs_agents": list(PRIMARY)},
        "seeds": seeds, "training": {"timesteps": 16384, "algorithms": list(ALGORITHMS), "variants": list(VARIANTS),
                                     "hyperparameters": {"ppo": PPO, "dqn": DQN}},
        "observation": "v0.4 observation contract; normalization fitted per variant on training-world exploration",
        "endpoints": {"cost": COST, "completion": COMPLETION},
        "selection_sha256": pr.document_sha256({"selected": selection["selected"], "ensemble": selection["ensemble"]}),
        "regime_models_sha256": pr.document_sha256(regime) if regime else None,
        "implementation_sha256": {name: pr.text_sha256(root / name) for name in EXECUTION_FILES},
        "semantics": {"completion": "within-horizon and settlement completion reported separately (v0.5 M5)",
                      "settlement": "after the horizon only cancellations and exchange events; post-horizon fills are "
                                    "settlement fills, never mandate completion",
                      "cost": "mandate completion-adjusted cost in bps of arrival price; residual valued by the terminal "
                              "book walk, never an actual fill"}}
    path = root / FREEZE_PATH
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    if seal:
        pr.append_event(root / pr.LEDGER_PATH, "seal_design", {
            "analysis": "environment-freeze", "design_sha256": pr.document_sha256(document), "reads": [],
            "note": "Mandate, worlds, agents, seeds and execution implementation frozen before any policy training."},
            protocol=protocol)
    return document


def read_freeze(root: Path = PROJECT_ROOT) -> dict:
    document = json.loads((root / FREEZE_PATH).read_text(encoding="utf-8"))
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), pr.load_protocol(root / pr.PROTOCOL_PATH))
    sealed = state.designs.get("environment-freeze")
    if sealed is None or sealed["design_sha256"] != pr.document_sha256(document):
        raise ValueError("environment freeze is missing from, or differs from, the ledger seal")
    for name, digest in document["implementation_sha256"].items():
        if pr.text_sha256(root / name) != digest:
            raise ValueError(f"frozen execution implementation changed: {name}")
    return document


# ----------------------------------------------------------------------------- policies


def train(out: str | Path, *, root: Path = PROJECT_ROOT, pilot: dict | None = None) -> dict:
    freeze_doc = read_freeze(root)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    worlds = [World.from_dict(w) for w in freeze_doc["worlds"]]
    members = [w for w in worlds if w.kind in {"selected", "member"}]
    seeds = freeze_doc["seeds"]
    training = freeze_doc["training"]
    plan = {"schema": "cleolob-v06-policy-registration-1", "registered_at": utc_now(),
            "freeze_sha256": pr.document_sha256(freeze_doc), "mandate": freeze_doc["mandate"],
            "training_worlds": {"single": [members[0].to_dict()], "ensemble": [w.to_dict() for w in members]},
            "training_seeds": seeds["training_seeds"][: (pilot or {}).get("seeds", len(seeds["training_seeds"]))],
            "timesteps": (pilot or {}).get("timesteps", training["timesteps"]), "hyperparameters": training["hyperparameters"],
            "normalization_seeds": seeds["normalization_seeds"],
            "domain_randomization_seed_offset": seeds["domain_randomization_seed_offset"],
            "evidence_level": "pilot" if pilot else "registered",
            "checkpoint": "final fixed-budget checkpoint only; no early stopping or held-out selection"}
    out = new_run(out)
    (out / "models").mkdir()
    write_json(out, "registration.json", plan)
    if not pilot:
        pr.append_event(root / pr.LEDGER_PATH, "seal_design", {
            "analysis": "policy-registration", "design_sha256": pr.document_sha256(plan), "reads": [],
            "note": "PPO/DQN single and ensemble variants registered on the frozen environment before fitting."},
            protocol=protocol)
    for variant in VARIANTS:
        fitted = fit_observation_normalization(variant, [World.from_dict(w) for w in plan["training_worlds"][variant]],
                                               plan["mandate"], plan["normalization_seeds"])
        write_json(out, f"normalization-{variant}.json", {"normalization": fitted,
                                                            "plan_sha256": pr.document_sha256(plan)})
    tasks = [(str(out), plan, a, v, s) for a in ALGORITHMS for v in VARIANTS for s in plan["training_seeds"]]
    with ProcessPoolExecutor(max_workers=_workers()) as executor:
        models = list(executor.map(train_one, tasks))
    summary = {"models": models, "failed": [m for m in models if "failed" in m],
               "total_timesteps": sum(m.get("timesteps", 0) for m in models)}
    write_json(out, "training_summary.json", summary)
    finalize(out, analysis=f"policy-training-{plan['evidence_level']}", dataset_ids=[], config=plan, result=summary,
             root=root, seeds={"training_seeds": plan["training_seeds"]})
    return summary


# ----------------------------------------------------------------------------- evaluation


def _episode_chunk(task: tuple) -> list[dict]:
    world_dict, agent, training_seed, markets, mandate, ac, policy_dir = task
    import torch
    torch.set_num_threads(1)
    from ..runner import run_episode
    world = World.from_dict(world_dict)
    model, normalization = None, None
    if ":" in agent:
        algorithm, variant = agent.split(":")
        model = load_policy(Path(policy_dir), algorithm, variant, training_seed)
        normalization = json.loads((Path(policy_dir) / f"normalization-{variant}.json").read_text(
            encoding="utf-8"))["normalization"]
    rows = []
    for market in markets:
        try:
            params = episode_params(world, mandate, market, normalization=normalization,
                                    ac=ac if agent == "ac" else None)
            row = run_episode("ppo" if model is not None else agent, params, model=model)
            row.pop("audit", None)
        except Exception as exc:  # retained, never rerun
            row = {"status": "INVALID", "error": f"{type(exc).__name__}: {exc}", COST: None, COMPLETION: None}
        keep = {k: row.get(k) for k in (COST, COMPLETION, "status", "error", "mandate_final_settlement_completion",
                                         "mandate_fill_fraction_at_horizon", "mandate_realized_fill_cost_bps",
                                         "invalid_reasons")}
        keep.update(world=world.name, agent=agent, training_seed=training_seed, seed=market)
        rows.append(json.loads(canonical_json(keep)))
    return rows


def episode_tasks(worlds: list[World], agents: list[str], training_seeds: list[int], markets: list[int],
                  mandate: dict, ac: dict, policy_dir: str, chunk: int = 25) -> list[tuple]:
    tasks = []
    for world in worlds:
        for agent in agents:
            for seed in (training_seeds if ":" in agent else [None]):
                for i in range(0, len(markets), chunk):
                    tasks.append((world.to_dict(), agent, seed, markets[i:i + chunk], mandate,
                                  ac[world.name]["selected"], policy_dir))
    return tasks


def world_realism(worlds: list[World], *, root: Path, seconds: float, seeds: list[int]) -> dict[str, dict]:
    """Selection-day family errors of every world (H7 input), with the sealed observable design."""
    design_run = read_design(root / DESIGN_RUN, root=root)
    target = pool(load_sketches(root / DESIGN_RUN / "selection-sketches.npz"))
    context = {"design": design_run["design"], "scales": design_run["objective_scales"],
               "targets": {"selection": target}}
    tasks = [(w.name, {"config": w.config, "extensions": w.extensions}, seeds, seconds, "selection") for w in worlds]
    return {r["key"]: {"objective": r["objective"], "families": dict(zip(FAMILIES, r["families"])), "error": r["error"]}
            for r in run_tasks(tasks, context)}


def evaluate(policy_dir: str | Path, out: str | Path, *, root: Path = PROJECT_ROOT, markets: int | None = None,
             label: str = "registered") -> dict:
    freeze_doc = read_freeze(root)
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    policy_dir = Path(policy_dir)
    plan = json.loads((policy_dir / "registration.json").read_text(encoding="utf-8"))
    if label == "registered" and plan["evidence_level"] != "registered":
        raise ValueError("registered evaluation requires registered policies")
    summary = json.loads((policy_dir / "training_summary.json").read_text(encoding="utf-8"))
    for model in summary["models"]:
        if "failed" not in model:
            stem = f"{model['algorithm']}-{model['variant']}-{model['seed']}"
            if sha256_file(policy_dir / "models" / f"{stem}.zip") != model["model_sha256"]:
                raise ValueError(f"checkpoint changed: {stem}")
    worlds = [World.from_dict(w) for w in freeze_doc["worlds"]]
    seeds = freeze_doc["seeds"]
    market_seeds = list(range(seeds["evaluation_market_seed_start"],
                              seeds["evaluation_market_seed_start"] + (markets or seeds["evaluation_market_seeds"])))
    trained = {(m["algorithm"], m["variant"], m["seed"]) for m in summary["models"] if "failed" not in m}
    agents = list(CLASSICAL) + list(LEARNED)
    out = new_run(out)
    lock = {"locked_at": utc_now(), "freeze_sha256": pr.document_sha256(freeze_doc),
            "registration_sha256": pr.document_sha256(plan), "market_seeds": market_seeds, "agents": agents,
            "worlds": [w.name for w in worlds], "label": label}
    write_json(out, "evaluation_lock.json", lock)
    if label == "registered":
        pr.append_event(root / pr.LEDGER_PATH, "seal_design", {
            "analysis": "execution-evaluation-lock", "design_sha256": pr.document_sha256(lock), "reads": [],
            "note": "World x agent x market-seed evaluation locked before any registered execution outcome."},
            protocol=protocol)
    tasks = episode_tasks(worlds, agents, plan["training_seeds"], market_seeds, freeze_doc["mandate"],
                          freeze_doc["ac_parameters"], str(policy_dir))

    def untrained(task: tuple) -> bool:
        return ":" in task[1] and (*task[1].split(":"), task[2]) not in trained

    missing = sorted({(t[1], t[2]) for t in tasks if untrained(t)})
    tasks = [t for t in tasks if not untrained(t)]
    with ProcessPoolExecutor(max_workers=_workers()) as executor:
        rows = [row for chunk in executor.map(_episode_chunk, tasks, chunksize=1) for row in chunk]
    for agent, seed in missing:
        for world in worlds:
            for market in market_seeds:
                rows.append({"status": "INVALID", "error": "training failed", COST: None, COMPLETION: None,
                             "world": world.name, "agent": agent, "training_seed": seed, "seed": market})
    with (out / "episodes.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    record_use(pr.datasets_by_role(protocol, "selection")[0], "select", "execution-world-realism",
               "selection-day family errors of every world (sealed sketches)", root)
    realism = world_realism(worlds, root=root, seconds=3600.0,
                            seeds=protocol["seeds"]["calibration"]["selection_seeds"])
    result = analyse(rows, worlds, market_seeds, realism, protocol)
    result.update(label=label, episodes=len(rows), invalid_episodes=sum(r.get("status") == "INVALID" for r in rows))
    finalize(out, analysis=f"execution-evaluation-{label}", dataset_ids=[], config=lock, result=result, root=root,
             seeds={"market_seeds": [market_seeds[0], market_seeds[-1]], "bootstrap": protocol["seeds"]["bootstrap"]})
    return result


def analyse(rows: list[dict], worlds: list[World], market_seeds: list[int], realism: dict, protocol: dict) -> dict:
    alpha, samples = 0.05, 5000
    boot = protocol["seeds"]["bootstrap"]
    cell_map = cells(rows, market_seeds)
    names = [w.name for w in worlds]
    members = [w.name for w in worlds if w.kind in {"selected", "member"}]
    intervention_names = [w.name for w in worlds if w.kind == "intervention"]
    regime_names = [w.name for w in worlds if w.kind == "regime"]
    summaries = {f"{w}|{a}": summarize_cell(c, alpha=alpha, samples=2000, seed=boot["execution"])
                 for (w, a), c in sorted(cell_map.items())}
    h5 = equifinality(cell_map, members, alpha=alpha, samples=samples, seed=boot["execution"])
    h6 = rank_stability(cell_map, members, PRIMARY, alpha=alpha, samples=samples, seed=boot["execution"],
                        reference="selected")
    sensitivity_worlds = members + intervention_names
    instability = disagreement(cell_map, sensitivity_worlds, "selected", PRIMARY, alpha=alpha, samples=samples,
                               seed=boot["sensitivity"])
    families = {w: realism[w]["families"] for w in sensitivity_worlds if w in realism}
    h7 = execution_sensitive_realism(families, {w: v for w, v in instability.items() if w != "selected"},
                                     families=tuple(FAMILIES), alpha=alpha, samples=samples, seed=boot["sensitivity"])
    all_worlds = rank_stability(cell_map, names, PRIMARY, alpha=alpha, samples=2000, seed=boot["execution"] + 1,
                                reference="selected")
    risk = decomposition(cell_map, selected="selected", members=members, interventions=intervention_names,
                         regime_worlds=regime_names, agents=tuple(CLASSICAL) + LEARNED)
    return {"summaries": summaries, "H5_equifinality": h5, "H6_rank_stability": h6,
            "H7_execution_sensitive_realism": h7, "instability": instability, "world_realism": realism,
            "rank_stability_all_worlds_descriptive": all_worlds, "model_risk": risk,
            "worlds": {w.name: {"kind": w.kind, "note": w.note, "sha256": w.sha256} for w in worlds},
            "interpretation": ("Simulator-only evidence across calibrated and perturbed worlds; no historical, live or "
                               "profitability claim. No global best policy is declared.")}
