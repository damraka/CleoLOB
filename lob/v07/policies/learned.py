"""M14: learned execution policies trained in one world or across the G0 posterior (workstreams 32-34).

Reuses the frozen v0.6 training stack unchanged (``lob.v06.policies``: the v0.5
environment, v0.4 observation contract, v0.5 M7 hyperparameters, final
fixed-budget checkpoints, retained failures). The two variants keep the v0.6
internal names: ``single`` trains in the G0 point world; ``ensemble`` draws one of
the 16 G0 posterior-predictive worlds uniformly at every episode reset
(posterior domain randomization; v0.6 used its 2-member ensemble). Training
seeds are the v0.7 protocol seeds. Offline RL on real data is NOT_AVAILABLE:
no logged historical actions exist.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

from ...experiments.registry import PROJECT_ROOT, utc_now
from ...v06.policies import ALGORITHMS, VARIANTS, fit_observation_normalization, train_one
from ...v06.worlds import World
from ..evidence.runs import finalize, new_run, verify_run, write_json
from ..execution.episode import MANDATE
from ..generators.study import v06_inputs
from ..posterior.study import posterior_specs
from ..protocol import core as pr

OFFLINE_RL = "NOT_AVAILABLE: no logged historical actions exist for offline reinforcement learning"


def training_worlds(posterior_run: str, root: Path) -> dict[str, list[World]]:
    selected = v06_inputs(root)["selection"]["selected"]
    single = World("G0_point", "selected", selected["config"], selected["extensions"], "v0.6 selected point model")
    posterior = [World(f"post_{i:02d}", "member", s.config, s.extensions, "G0 posterior-predictive draw")
                 for i, s in enumerate(posterior_specs(root / posterior_run, root))]
    return {"single": [single], "ensemble": posterior}


def plan_document(posterior_run: str, root: Path, *, pilot: dict | None = None) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    freeze = json.loads((root / "configs/v06/environment-freeze.json").read_text(encoding="utf-8"))
    worlds = training_worlds(posterior_run, root)
    seeds = protocol["seeds"]["execution"]["training_seeds"]
    return {"schema": "cleolob-v07-policy-registration-1", "registered_at": utc_now(), "mandate": MANDATE,
            "posterior_run": posterior_run,
            "training_worlds": {v: [w.to_dict() for w in ws] for v, ws in worlds.items()},
            "variant_meaning": {"single": "G0 point world", "ensemble": "uniform draw among 16 G0 posterior worlds"},
            "training_seeds": seeds[: (pilot or {}).get("seeds", len(seeds))],
            "timesteps": (pilot or {}).get("timesteps", freeze["training"]["timesteps"]),
            "hyperparameters": freeze["training"]["hyperparameters"],
            "normalization_seeds": freeze["seeds"]["normalization_seeds"],
            "domain_randomization_seed_offset": freeze["seeds"]["domain_randomization_seed_offset"],
            "evidence_level": "pilot" if pilot else "registered", "offline_rl": OFFLINE_RL,
            "checkpoint": "final fixed-budget checkpoint only; no early stopping or held-out selection"}


def train(out: str | Path, *, posterior_run: str, root: Path = PROJECT_ROOT, pilot: dict | None = None,
          workers: int | None = None) -> dict:
    report = verify_run(root / posterior_run, root=root)
    if not report["valid"]:
        raise ValueError(f"{posterior_run} is not valid: {report['issues']}")
    plan = plan_document(posterior_run, root, pilot=pilot)
    out = new_run(root / out)
    (out / "models").mkdir()
    write_json(out, "registration.json", plan)
    if not pilot:
        pr.append_event(root / pr.LEDGER_PATH, "seal_design", design="m14-policy-registration", root=root,
                        protocol=pr.load_protocol(root / pr.PROTOCOL_PATH),
                        reason="PPO/DQN single and posterior variants registered before fitting",
                        payload={"design_sha256": pr.document_sha256(plan), "reads": [], "posthoc": False})
    for variant in VARIANTS:
        fitted = fit_observation_normalization(variant, [World.from_dict(w) for w in plan["training_worlds"][variant]],
                                               plan["mandate"], plan["normalization_seeds"])
        write_json(out, f"normalization-{variant}.json", {**fitted, "plan_sha256": pr.document_sha256(plan)})
    tasks = [(str(out), plan, a, v, s) for a in ALGORITHMS for v in VARIANTS for s in plan["training_seeds"]]
    workers = workers or max(1, min(14, (os.cpu_count() or 2) - 2))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        models = list(executor.map(train_one, tasks))
    summary = {"models": models, "failed": [m for m in models if "failed" in m],
               "total_timesteps": sum(m.get("timesteps", 0) for m in models), "offline_rl": OFFLINE_RL}
    write_json(out, "training_summary.json", summary)
    finalize(out, analysis=f"m14-policy-training-{plan['evidence_level']}", dataset_ids=[], config=plan,
             result=summary, root=root, seeds={"training_seeds": plan["training_seeds"]})
    return summary
