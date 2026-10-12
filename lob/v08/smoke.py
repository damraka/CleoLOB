"""Offline generated-data v0.8 smoke (brief section 32): functionality evidence, not scientific validation.

The smoke never touches real data and never writes to the repository ledger. When the repository configs are
available it builds an isolated protocol root inside the output directory (copies of the v0.7 records and the v0.8
protocol/configs, a fresh v0.8 ledger) so every step is sealed and bound exactly like a study run; otherwise
(e.g. an installed wheel without the repository) it runs in standalone mode with byte-integrity seals only and
reports protocol binding as NOT_AVAILABLE.

Steps: data validation, regime detection, generator fitting, posterior inference (tiny ABC-SMC), model averaging,
uncertainty decomposition, OOD, conformal, completion-constrained execution, robust selection, counterfactual
bounds (synthetic twin), a transfer comparison, a claim graph and verification.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from ..artifacts import seal_artifacts, verify_artifacts
from ..experiments.registry import PROJECT_ROOT

V07_FILES = ("configs/v07/protocol.json", "configs/v07/consumption-ledger.jsonl")
V08_FILES = ("protocol.json", "hypotheses.json", "statistical-families.json", "dataset-registry.json",
             "compute-budget.json", "claim-registry.json")


def isolated_root(out: Path) -> Path | None:
    from . import protocol as pr
    repo = PROJECT_ROOT
    needed = [repo / f for f in V07_FILES] + [repo / pr.CONFIG_DIR / f for f in V08_FILES]
    if not all(p.is_file() for p in needed):
        return None
    root = out / "root"
    for f in V07_FILES:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / f, root / f)
    for f in V08_FILES:
        (root / pr.CONFIG_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / pr.CONFIG_DIR / f, root / pr.CONFIG_DIR / f)
    pr.initialize_ledger(root)
    return root


def run(out: Path, *, workers: int = 1) -> dict:
    from . import artifacts, conformal, counterfactual as cf, model_averaging as ma, quality, regimes, robust
    from .execution import policies as P
    from .execution.completion import completion_summary
    from .execution.simulator import Mandate, run_episode
    from .generators import (HawkesGenerator, PoissonGenerator, QueueReactiveGenerator, RegimeSwitchingGenerator,
                             StateDependentGenerator)
    from .inference import abc
    from .market.frame import summaries
    from .ood import Reference, threshold
    from .transfer import compare
    from .uncertainty import decompose
    from .worlds import World
    from . import mpc_registration  # noqa: F401
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    root = isolated_root(out)
    mode = "protocol_bound" if root is not None else "standalone"
    steps: dict[str, dict] = {}
    runs: list[Path] = []

    def seal(name: str, result: dict) -> None:
        if root is not None:
            target = artifacts.new_run(root / "results/v08/smoke" / name)
            artifacts.finalize(target, analysis=f"smoke_{name}", config={"step": name}, result=result, root=root,
                               check_source=False)
        else:
            target = out / "runs" / name
            target.mkdir(parents=True)
            (target / "result.json").write_text(json.dumps(result, sort_keys=True, default=str) + "\n",
                                                encoding="utf-8", newline="\n")
            seal_artifacts(target)
        runs.append(target)
        steps[name] = {"status": "OK", "keys": sorted(result)[:8]}

    g = PoissonGenerator()
    frame = g.simulate(1200, 1)
    report = quality.validate(quality.from_frame(frame, tick=1.0), tick=1.0)
    seal("validation", {"status": report["status"], "score": report["score"]})
    rs = RegimeSwitchingGenerator()
    feats_a, feats_b = regimes.block_features(rs.simulate(3000, 2), 100), regimes.block_features(rs.simulate(3000, 3), 100)
    seal("regimes", {m: {"persistence": v["persistence_blocks"]} for m, v in regimes.compare(feats_a, feats_b).items()})
    fits = {cls.name: cls().fit(frame) for cls in (PoissonGenerator, HawkesGenerator, QueueReactiveGenerator,
                                                   RegimeSwitchingGenerator, StateDependentGenerator)}
    seal("generators", {k: v.score(frame)["loglik_per_bin"] for k, v in fits.items()})
    prior = abc.Prior.from_space({"market": (0.3, 3.0), "add": (1.0, 12.0)}, log={"market", "add"})
    problem = abc.Problem(PoissonGenerator(), prior, summaries(frame), n_bins=300, seed=5, pilot=8)
    post = abc.smc(problem, particles=16, generations=2)
    seal("posterior", {"summary": post["summary"], "simulations": post["simulations"]})
    test = g.simulate(600, 9)
    ll = np.column_stack([f.loglik(test) for f in fits.values()])
    ok = np.all(np.isfinite(ll), 1)
    seal("model_averaging", {"stacking": ma.stacking_weights(ll[ok]).tolist(),
                             "material": ma.material(ma.stacking_weights(ll[ok]))})
    mandate = Mandate(qty=20, horizon_s=30.0, warmup_s=10.0, settlement_s=5.0)
    worlds = [World(f"w{i}", PoissonGenerator({**PoissonGenerator.default_params(), "market": m}),
                    PoissonGenerator().book) for i, m in enumerate((0.6, 1.0, 1.6))]
    rows = [run_episode(w, P.make(p), mandate, s) for w in worlds for p in ("twap", "heuristic") for s in (1, 2)]
    y = np.asarray([[r["economic_cost_bps"] for r in rows if r["world_id"] == w.world_id and r["policy"] == "twap"]
                    for w in worlds])
    seal("uncertainty", decompose(y, {"model_family": ["a", "a", "b"]}, n_boot=50))
    ref = Reference(np.asarray([summaries(g.simulate(600, 20 + k)) for k in range(12)]))
    cal = ref.mahalanobis(np.asarray([summaries(g.simulate(600, 40 + k)) for k in range(12)]))
    seal("ood", {"threshold": threshold(cal, 0.05)})
    costs = np.asarray([r["economic_cost_bps"] for r in rows])
    seal("conformal", conformal.coverage(costs[6:], *[np.asarray(v) for v in (
        lambda s: (s["lower"], s["upper"]))(conformal.split(costs[:6], [costs[:6].mean()] * 6,
                                                            [costs[:6].mean()] * (len(costs) - 6), 0.2))], nominal=0.8))
    soft = [run_episode(worlds[0], P.make("heuristic"), mandate.replace(completion="none"), s) for s in (1, 2, 3)]
    hard = [run_episode(worlds[0], P.make("heuristic"), mandate.replace(completion="hard"), s) for s in (1, 2, 3)]
    seal("completion", {"none": completion_summary(soft), "hard": completion_summary(hard)})
    seal("robust", robust.select_all(["a", "b"], np.asarray([[1.0, 3.0], [2.0, 2.0]]), np.ones(2)))
    cov = cf.synthetic_coverage(worlds[1], lambda: P.make("twap"), mandate, [1, 2],
                                cf.grid((0.0, 0.1), 0.0, impacts=("none", "persistent")))
    seal("counterfactual", cov["summary"])
    seal("transfer", compare({"twap": 1.0, "ac": 2.0, "mpc": 3.0}, {"twap": 1.5, "ac": 1.0, "mpc": 4.0}))
    seal("claim_graph", {"claims": [{"claim": "SMOKE", "step": name, "run": p.name} for name, p in zip(steps, runs)]})
    verdicts = {}
    for target in runs:
        if root is not None:
            verdicts[target.name] = artifacts.verify_run(target, root=root)["valid"]
        else:
            verdicts[target.name] = not verify_artifacts(target)["issues"]
    protocol_check = None
    if root is not None:
        from . import protocol as pr
        protocol_check = pr.verify_protocol_files(root)["valid"]
    valid = all(verdicts.values()) and (protocol_check is not False)
    return {"valid": valid, "mode": mode, "steps": steps, "verification": verdicts, "protocol_valid": protocol_check,
            "protocol_binding": "BOUND (isolated smoke ledger)" if root else "NOT_AVAILABLE (no repository configs)",
            "label": "functionality evidence only; not scientific validation"}
