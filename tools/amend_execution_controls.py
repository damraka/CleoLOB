"""Execute a finite, preregistered execution-grid amendment and one confirmation.

Only execution quantity, horizon and decision interval change. The historical
calibration model, economic accounting, fees and positive-control gate stay fixed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import pandas as pd

from lob.controls import _control_episode, _paired, finalize_execution_config
from lob.experiments.registry import sha256_file, write_json


GRID = [(multiplier, horizon, interval)
        for interval in (2., 5., .5)
        for horizon in (60., 120., 240.)
        if interval <= horizon / 20.
        for multiplier in (2., 1.5, 3.)]
SMALL_GRID = [(multiplier, horizon, interval)
              for horizon, interval in ((60., 3.), (120., 6.), (240., 12.),
                                        (60., 2.), (120., 5.), (240., 5.))
              for multiplier in (.5, .75, 1.)]
AGENTS = ("twap", "random", "all_wait", "all_market")


class FixedPolicy:
    def __init__(self, action):
        self.action = action

    def predict(self, _observation, deterministic=True):
        return self.action, None


def episode(task):
    agent, seed, params = task
    selected = {**params, "seed": seed, "sim": {**params["sim"], "seed": seed}}
    try:
        if agent.startswith("fixed_"):
            from lob.runner import run_episode
            row = run_episode("ppo", selected, model=FixedPolicy(int(agent[-1])))
            return {**row, "agent": agent}
        return _control_episode(agent, selected)
    except (RuntimeError, ValueError) as exc:
        return {"agent": agent, "seed": seed, "status": "ERROR", "effective_bps": None,
                "error": str(exc)}


def fully_priced(rows):
    return bool(rows) and all(row.get("status") in {"VALID", "WARNING"}
                             and row.get("effective_bps") is not None
                             and math.isfinite(float(row["effective_bps"])) for row in rows)


def passing(comparison):
    return (comparison["status"] == "AVAILABLE"
            and abs(comparison["delta_mean_bps"]) >= .5
            and (comparison["ci_low_bps"] > 0 or comparison["ci_high_bps"] < 0))


def run(out: Path, workers: int, resume_registered_plan: bool = False,
        small_quantity_extension: bool = False):
    original = Path("examples/studies/core/execution-controls/compute-continuation/positive-control")
    original_plan = json.loads((original / "plan.json").read_text())
    model = Path("examples/studies/core/calibration/zi-eth-amended-fit/model.json")
    params = original_plan["params"]
    grid = SMALL_GRID if small_quantity_extension else GRID
    previous_amendment = None
    if small_quantity_extension:
        previous_result = Path("examples/studies/core/execution-controls/execution-amendment/result.json")
        previous_amendment = json.loads(previous_result.read_text())
        if (previous_amendment["status"] != "FAIL"
                or previous_amendment["confirmation"]["status"] != "NOT_RUN_NO_DIAGNOSTIC_PASS"
                or len(previous_amendment["diagnostics"]) != len(GRID)):
            raise ValueError("Small-quantity extension requires a completed failed diagnostic grid and unopened confirmation")
    plan = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "amendment": "Execution-only diagnostic amendment after original grid failed; all original artifacts retained.",
        "original_result_sha256": sha256_file(original / "result.json"),
        "model_file_sha256": sha256_file(model), "params": params,
        "median_top5_depth": original_plan["median_top5_depth"],
        "grid_columns": ["depth_multiplier", "horizon_seconds", "decision_dt_seconds"],
        "grid": grid,
        "diagnostic_seeds": list(range(50000, 50128) if small_quantity_extension else range(47000, 47032)),
        "confirmation_seeds": list(range(51000, 51128) if small_quantity_extension else range(48000, 48064)),
        "capacity_preflight_seeds": list(range(52000, 52512) if small_quantity_extension else range(49000, 49064)),
        "capacity_preflight_agents": ["random", "fixed_0", "fixed_1", "fixed_2", "fixed_3", "fixed_4"],
        "unchanged_gate": "All raw outcomes finite and non-INVALID, absolute Random-minus-TWAP mean >= 0.5 bps and paired 95% bootstrap CI excludes zero; all-wait and all-market capacity controls fully priced.",
        "min_effect_bps": .5, "bootstrap_samples": 5000,
        "selection": "Evaluate the finite diagnostic grid in the recorded order. Select the first cell passing the unchanged gate AND every additional fixed-action/random capacity preflight episode. Stop diagnostic search once selected. A preflight failure rejects that diagnostic cell, retaining every observation.",
        "confirmation": "Exactly one confirmation of the selected setting on the declared independent seeds, with the unchanged gate. Never reselect or search after confirmation. Failed confirmation produces no final study configuration.",
        "preflight_scope": "Fresh simulator seeds outside training, power-diagnostic and final test blocks; all five constant actions and random actions. This empirical coverage is not a proof for every learned policy path.",
        "frozen_constraints": "Frozen ETH physical simulator, 1 bps taker fee, zero maker fee, 30-second warmup, existing strict terminal accounting and 0.5 bps effect gate. check_invariants=False only disables assertions in both train/evaluation.",
    }
    if previous_amendment is not None:
        plan["previous_amendment_result_sha256"] = sha256_file(previous_result)
        plan["amendment"] = "Second prospective execution-only amendment: the entire previous larger-quantity diagnostic grid failed before any confirmation opened. Extend only to smaller 0.5/0.75/1.0 depth quantities and declared intervals. Increase diagnostic sample to128, independent confirmation to128, and six-policy capacity preflight to512seeds. All old failures remain; no physical simulator or fee changes."
    if resume_registered_plan:
        if {p.name for p in out.iterdir()} != {"plan.json"}:
            raise ValueError("Only a registered plan with zero observed episodes can resume")
        previous = json.loads((out / "plan.json").read_text())
        plan["registered_at_utc"] = previous["registered_at_utc"]
        if json.loads(json.dumps(plan)) != previous:
            raise ValueError("Registered plan changed before execution")
        plan = previous
    else:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "plan.json", plan)
    started = time.monotonic()
    rows, diagnostics, selected = [], [], None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for cell, (multiplier, horizon, interval) in enumerate(grid):
            chosen = {**params, "qty": round(plan["median_top5_depth"] * multiplier),
                      "horizon": horizon, "dt": interval}
            group = list(pool.map(episode, [(agent, seed, chosen)
                                           for seed in plan["diagnostic_seeds"] for agent in AGENTS]))
            comparison = _paired(group, "twap", "random", n_boot=plan["bootstrap_samples"])
            capacity = fully_priced(group)
            candidate = capacity and passing(comparison)
            detail = {"cell": cell, "depth_multiplier": multiplier, "horizon": horizon,
                      "decision_dt": interval, "quantity": chosen["qty"], "comparison": comparison,
                      "all_wait_and_market_capacity_pass": capacity, "diagnostic_pass": candidate}
            rows.extend({**r, "phase": "diagnostic", "cell": cell} for r in group)
            if candidate:
                preflight = list(pool.map(episode, [(agent, seed, chosen)
                                                   for seed in plan["capacity_preflight_seeds"]
                                                   for agent in plan["capacity_preflight_agents"]]))
                rows.extend({**r, "phase": "capacity_preflight", "cell": cell} for r in preflight)
                detail["capacity_preflight_pass"] = fully_priced(preflight)
                detail["capacity_preflight_episodes"] = len(preflight)
                detail["capacity_preflight_invalids"] = sum(not fully_priced([r]) for r in preflight)
                if detail["capacity_preflight_pass"]:
                    selected = chosen
            diagnostics.append(detail)
            write_json(out / f"diagnostic-cell-{cell}.json", detail)
            pd.DataFrame(rows).to_csv(out / "episodes.csv", index=False)
            print(json.dumps({**detail, "elapsed_seconds": time.monotonic() - started}), flush=True)
            if selected is not None:
                break
        confirmation = {"status": "NOT_RUN_NO_DIAGNOSTIC_PASS"}
        confirmation_capacity = False
        if selected is not None:
            group = list(pool.map(episode, [(agent, seed, selected)
                                           for seed in plan["confirmation_seeds"] for agent in AGENTS]))
            rows.extend({**r, "phase": "confirmation", "cell": diagnostics[-1]["cell"]} for r in group)
            confirmation = _paired(group, "twap", "random", n_boot=plan["bootstrap_samples"])
            confirmation_capacity = fully_priced(group)
    result = {"status": "PASS" if confirmation_capacity and passing(confirmation) else "FAIL",
              "diagnostics": diagnostics, "selected_params": selected, "confirmation": confirmation,
              "confirmation_all_wait_and_market_capacity_pass": confirmation_capacity,
              "elapsed_seconds": time.monotonic() - started,
              "interpretation": "Execution discrimination under a frozen, unvalidated synthetic simulator; not historical fidelity or a PPO result."}
    pd.DataFrame(rows).to_csv(out / "episodes.csv", index=False)
    write_json(out / "result.json", result)
    print(json.dumps(result), flush=True)
    if result["status"] == "PASS":
        final = finalize_execution_config(model, out, out / "final-ac-fit", "configs/core-study.json")
        print(json.dumps(final), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="examples/studies/core/execution-controls/execution-amendment")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume-registered-plan", action="store_true")
    parser.add_argument("--small-quantity-extension", action="store_true")
    args = parser.parse_args()
    run(Path(args.out), args.workers, args.resume_registered_plan, args.small_quantity_extension)
