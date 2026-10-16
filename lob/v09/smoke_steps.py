"""smoke-v09 steps: offline, generated data only; functionality evidence, never scientific validation.

Steps (each sealed as a run in the isolated protocol root when the repository configs exist, otherwise sealed with
byte-integrity checksums only): deterministic runner across worker counts, classical generator simulation,
synthetic execution episodes for registered policies, path-signature MMD, queue-survival models on simulated
durations, the promotion gate on generated pilot records, confirmatory decision rules with Holm, SBC with its
positive control, backup-slot substitution in a sealed confirmatory design, and verification of every step.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..artifacts import seal_artifacts, verify_artifacts


def run_smoke(out: Path, *, workers: int = 1) -> dict:
    from ..v08.generators import HawkesGenerator, PoissonGenerator
    from . import artifacts, decide as D, promotion as PM, protocol as pr, smoke, twin
    from .execution import policies as P
    from .queue import survival as SV
    from .realism import path as RP
    from .runner import Task, run_tasks
    from .studies.descriptive import _sim_paths, sbc_task

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    root = smoke.isolated_root(out)
    mode = "protocol_bound" if root is not None else "standalone"
    steps: dict[str, dict] = {}
    runs: list[Path] = []

    def seal(name: str, result: dict) -> None:
        if root is not None:
            target = artifacts.new_run(root / "results/v09/smoke" / name)
            artifacts.finalize(target, analysis=f"smoke-{name}", config={"step": name}, result=result, root=root,
                               check_source=False)
        else:
            target = out / "runs" / name
            target.mkdir(parents=True)
            (target / "result.json").write_text(json.dumps(result, sort_keys=True, default=str) + "\n",
                                                encoding="utf-8", newline="\n")
            seal_artifacts(target)
        runs.append(target)
        steps[name] = {"status": "OK", "keys": sorted(result)[:8]}

    tasks = [Task(f"t{k}", "lob.v09.selftest:draw", {"seed": k}) for k in range(6)]
    one, _ = run_tasks(tasks, workers=1)
    many, _ = run_tasks(tasks, workers=max(1, workers))
    seal("runner", {"deterministic_across_workers": one == many, "tasks": len(tasks)})

    g = PoissonGenerator()
    h = HawkesGenerator()
    fa, fb = [g.simulate(3300, 1)], [h.simulate(3300, 2)]
    seal("generators", {"poisson_bins": int(len(fa[0].valid)), "hawkes_bins": int(len(fb[0].valid)),
                        "poisson_trades": float(fa[0].counts[:, :2].sum())})

    rows = {p: twin.synthetic_episode(g, P.make(p), qty=20, seed=3, side=1) for p in ("twap", "passive_deadline",
                                                                                      "mpc_cvar")}
    seal("execution", {p: {"status": r["status"], "is_net_bps": r["is_net_bps"]} for p, r in rows.items()})

    pa, pb = _sim_paths(fa), _sim_paths(fb)
    if len(pa) > 3 and len(pb) > 3:
        scale = float(np.std(pa[:, -1, 0])) or 1.0
        seal("path_realism", {"mmd2": RP.mmd2(RP.signature(pa, scale), RP.signature(pb, scale)),
                              "paths": [int(len(pa)), int(len(pb))]})
    else:
        seal("path_realism", {"status": "NOT_EVALUABLE", "paths": [int(len(pa)), int(len(pb))]})

    rng = np.random.default_rng(7)
    X = rng.normal(size=(400, 3))
    T = rng.exponential(np.exp(-0.5 * X[:, 0]) * 5.0)
    E = T < 30.0
    T = np.minimum(T, 30.0)
    km, cox = SV.KaplanMeier().fit(X, T, E), SV.CoxPH().fit(X, T, E)
    seal("survival", {"ibs_km": SV.integrated_brier(km.predict(X), T, E),
                      "ibs_cox": SV.integrated_brier(cox.predict(X), T, E)})

    cand = {"test": "superiority", "threshold": 0.1, "max_units": 80, "required_capabilities": [], "family": "x"}
    pilot = {"data_available": True, "capabilities": {}, "units": 12, "effect": 0.6, "sd": 0.3,
             "odd_values": list(np.linspace(0.5, 0.7, 12)), "even_values": list(np.linspace(0.52, 0.68, 12)),
             "compute_hours_per_unit": 0.01}
    rules = {"min_reliability": 0.5, "min_pilot_units": 6, "min_power": 0.8, "shrinkage_se": 1.0,
             "min_confirm_units": 8, "compute_hours_ceiling": 40.0, "reliability_rule": "split_half_consistency"}
    dec = PM.decide("HX", cand, pilot, rules=rules, family={"alpha": 0.05}, family_size=1)
    seal("promotion", {"outcome": dec["outcome"], "required_units": dec.get("required_units")})

    st = {"A": D.evaluate("superiority", rng.normal(0.5, 0.2, 20), threshold=0.1, seed=1),
          "B": D.evaluate("superiority", rng.normal(0.0, 0.2, 20), threshold=0.1, seed=2)}
    fam = D.family(st)
    seal("decisions", {k: v["status"] for k, v in fam.items()})

    s = sbc_task(0, replicates=120)
    seal("sbc", {"correct": s["correct"]["status"], "positive_control_detected": s["positive_control_detected"]})

    if root is not None:
        from . import confirm as CF
        protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
        prim, back = CF.slots(protocol, "confirm", "bybit-linear:BTCUSDT")
        state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), protocol)
        state.attempts.append({"dataset": prim[0], "outcome": "NOT_AVAILABLE"})
        doc, mapping = CF.substitute({"test_datasets": prim[:3]}, protocol, state)
        seal("substitution", {"mapping": mapping, "uses_first_backup": doc["test_datasets"][0] == back[0]})

    verdicts = {}
    for target in runs:
        verdicts[target.name] = (artifacts.verify_run(target, root=root)["valid"] if root is not None
                                 else not verify_artifacts(target)["issues"])
    protocol_check = pr.verify_protocol_files(root)["valid"] if root is not None else None
    checks = [steps and all(verdicts.values()), protocol_check is not False]
    return {"valid": all(checks), "mode": mode, "steps": steps, "verification": verdicts,
            "protocol_valid": protocol_check,
            "protocol_binding": "BOUND (isolated smoke ledger)" if root else "NOT_AVAILABLE (no repository configs)",
            "label": "functionality evidence only; not scientific validation"}
