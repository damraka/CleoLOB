"""Simulator identification, independent execution controls and missing-price sensitivity.

Fitted AC impact is a local linear approximation to the executable book, not a
claim that this ZI exchange obeys the Almgren--Chriss price process. Sensitivity
prices are declared stress assumptions, never replacements for raw observations.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .engine import ExchangeSimulator, Side, SimConfig
from .experiments.registry import sha256_file, write_json
from .runner import run_episode
from .stats import adjust_pvalues, bootstrap_ci, sign_test_p


def _seeds(values: Sequence[int]) -> list[int]:
    result = list(values)
    if len(result) < 2 or len(set(result)) != len(result):
        raise ValueError("at least two distinct seeds required")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 2**32 for v in result):
        raise ValueError("seeds must be unsigned 32-bit integers")
    return result


def estimate_ac_parameters(cfg: SimConfig, seeds: Sequence[int], *, horizon: float = 60,
                           sample_dt: float = 1, execution_interval: float = 1,
                           warmup_seconds: float = 5,
                           fractions: Sequence[float] = (.1, .25, .5, 1., 2.)) -> dict:
    """Identify eta and sigma on no-parent simulator paths only.

    Each sample evaluates both sides of the frozen current book at quantities
    proportional to its own top-five depth. OLS includes a spread intercept:
    adverse VWAP - midpoint = intercept + eta * (quantity / slice_seconds).
    Consequently eta must be refit if the AC slice interval changes. Sigma is
    the pooled within-path demeaned arithmetic price-increment standard
    deviation divided by sqrt(sample_dt). No fills or prices are invented.
    """
    seeds = _seeds(seeds)
    if any(getattr(cfg, name) is not None for name in ("market_seed", "order_flow_seed", "latency_seed",
                                                     "cancellation_seed", "resilience_seed")):
        raise ValueError("component seed overrides prevent independent simulator identification paths")
    if any(not math.isfinite(x) or x <= 0 for x in (horizon, sample_dt, execution_interval)):
        raise ValueError("positive finite time intervals required")
    if not math.isfinite(warmup_seconds) or warmup_seconds < 0:
        raise ValueError("warmup_seconds must be finite and nonnegative")
    if horizon < 2 * sample_dt or not fractions or any(not math.isfinite(f) or f <= 0 for f in fractions):
        raise ValueError("at least two increments and positive quantity fractions required")
    probes, increments, depths, spreads = [], [], [], []
    attempted = 0
    for seed in seeds:
        sim = ExchangeSimulator(replace(cfg, seed=seed, record_events=False))
        sim.step(warmup_seconds)
        previous = sim.book.mid() * cfg.tick_size
        path_increments = []
        for _ in range(int(horizon // sample_dt)):
            sim.step(sample_dt)
            mid = sim.book.mid() * cfg.tick_size
            path_increments.append(mid - previous)
            previous = mid
            bids, asks = sim.book.depth(5)
            if not bids or not asks:
                continue
            spread = (asks[0][0] - bids[0][0]) * cfg.tick_size
            spreads.append(spread)
            for side, levels in ((Side.SELL, bids), (Side.BUY, asks)):
                depth = sum(q for _, q in levels)
                depths.append(depth)
                for fraction in fractions:
                    attempted += 1
                    quantity = max(cfg.lot_size, round(depth * fraction / cfg.lot_size) * cfg.lot_size)
                    vwap, fillable = sim.book.walk_cost(side, quantity)
                    if vwap is None or fillable != quantity:
                        continue
                    adverse = (vwap * cfg.tick_size - mid) * (1 if side is Side.BUY else -1)
                    probes.append((quantity / execution_interval, adverse))
        returns = np.asarray(path_increments, dtype=float)
        increments.extend((returns - returns.mean()).tolist())
    if len(probes) < 20 or len(increments) <= len(seeds):
        raise ValueError("insufficient executable depth/increments for AC identification")
    observations = np.asarray(probes)
    design = np.column_stack((np.ones(len(probes)), observations[:, 0]))
    intercept, slope = np.linalg.lstsq(design, observations[:, 1], rcond=None)[0]
    residuals = observations[:, 1] - design @ np.array([intercept, slope])
    total_variation = float(np.sum((observations[:, 1] - observations[:, 1].mean()) ** 2))
    r_squared = 1 - float(np.sum(residuals**2)) / total_variation if total_variation else 0.
    sigma = math.sqrt(float(np.sum(np.square(increments))) / (len(increments) - len(seeds)) / sample_dt)
    coverage = len(probes) / attempted
    passed = bool(slope > 0 and intercept >= 0 and r_squared >= .2 and coverage >= .8)
    return {
        "method": "pooled frozen-book executable VWAP OLS with spread intercept; no-parent arithmetic volatility",
        "status": "PASS" if passed else "FAIL", "sim_config": asdict(cfg), "seeds": seeds,
        "horizon_seconds": horizon, "sample_dt_seconds": sample_dt,
        "warmup_seconds": warmup_seconds,
        "execution_interval_seconds": execution_interval, "quantity_depth_fractions": list(fractions),
        "temp_impact": float(slope), "sigma": sigma,
        "impact_units": "currency * seconds / quantity_unit",
        "sigma_units": "currency / sqrt(second)", "intercept_currency": float(intercept),
        "mean_half_spread_currency": float(np.mean(spreads) / 2),
        "r_squared": r_squared, "probe_coverage": coverage, "probe_count": len(probes),
        "probe_attempts": attempted, "median_top5_depth": float(np.median(depths)),
        "gate": {"positive_slope": True, "nonnegative_intercept": True, "min_r_squared": .2,
                 "min_probe_coverage": .8},
        "limitations": ["Local linear depth approximation; permanent impact is not separately identified.",
                        "Impact is conditional on the registered child interval and quantity range.",
                        "Fitting AC parameters does not establish simulator fidelity to historical markets."],
    }


def _paired(rows: list[dict], reference: str, agent: str, *, n_boot: int, seed: int = 7919) -> dict:
    indexed = {(r["agent"], r["seed"]): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate agent/seed observations")
    seed_set = sorted({r["seed"] for r in rows})
    delta = []
    for market_seed in seed_set:
        left, right = indexed.get((agent, market_seed)), indexed.get((reference, market_seed))
        if left is None or right is None or any(r.get("status") not in {"VALID", "WARNING"}
                                                or r.get("effective_bps") is None
                                                or not math.isfinite(float(r["effective_bps"]))
                                                for r in (left, right)):
            return {"status": "WITHHELD_FOR_THIS_COMPARISON", "planned_pairs": len(seed_set),
                    "reason": "Missing/nonfinite/INVALID raw economic outcome"}
        delta.append(float(left["effective_bps"]) - float(right["effective_bps"]))
    if not np.isfinite(delta).all():
        raise ValueError("nonfinite paired outcomes")
    lo, hi = bootstrap_ci(delta, n_boot=n_boot, seed=seed)
    wins, losses = sum(v < 0 for v in delta), sum(v > 0 for v in delta)
    return {"status": "AVAILABLE", "n": len(delta), "agent": agent, "reference": reference,
            "delta_mean_bps": float(np.mean(delta)), "ci_low_bps": lo, "ci_high_bps": hi,
            "sd_paired_bps": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.,
            "p_sign": sign_test_p(wins, losses), "delta_definition": "agent minus reference"}


class AllWaitPolicy:
    """Fixed action-zero capacity control, explicitly not a trained PPO model."""

    def predict(self, _observation, deterministic=True):
        return 0, None


class AllMarketPolicy:
    """Fixed most-aggressive action capacity control, not a trained PPO model."""

    def predict(self, _observation, deterministic=True):
        return 4, None


def _control_episode(agent: str, params: dict) -> dict:
    if agent in {"all_wait", "all_market"}:
        model = AllWaitPolicy() if agent == "all_wait" else AllMarketPolicy()
        row = run_episode("ppo", params, model=model)
        return {**row, "agent": agent, "label": f"{agent} capacity control",
                "implementation": "fixed_action_zero" if agent == "all_wait" else "fixed_action_four"}
    return run_episode(agent, params)


def positive_control(params: dict, diagnostic_seeds: Sequence[int], confirmation_seeds: Sequence[int],
                     out: str | Path, *, median_top5_depth: float,
                     grid: Sequence[tuple[float, float]] = ((2., 30.), (5., 60.), (10., 120.)),
                     min_effect_bps: float = .5, n_boot: int = 2000) -> dict:
    """Register a finite diagnostic grid, select first passing cell, confirm once.

    Two-sided separation is the gate: this tests whether execution policies can
    produce distinguishable economic costs, without assuming Random must lose.
    A failed independent confirmation is final and never triggers more search.
    """
    diagnostics, confirmation = _seeds(diagnostic_seeds), _seeds(confirmation_seeds)
    if set(diagnostics) & set(confirmation):
        raise ValueError("diagnostic and confirmation seed blocks must be disjoint")
    if (not math.isfinite(median_top5_depth) or median_top5_depth <= 0
            or not math.isfinite(min_effect_bps) or min_effect_bps <= 0):
        raise ValueError("positive depth and minimum effect required")
    if not grid or any(not math.isfinite(m) or not math.isfinite(h) or m <= 0 or h <= 0 for m, h in grid):
        raise ValueError("positive multiplier/horizon grid required")
    path = Path(out)
    path.mkdir(parents=True, exist_ok=False)
    plan = {"params": params, "diagnostic_seeds": diagnostics, "confirmation_seeds": confirmation,
            "median_top5_depth": median_top5_depth, "grid": [list(v) for v in grid],
            "selection": "first grid cell with no invalids, CI excluding zero, absolute effect >= threshold",
            "capacity_gate": "All-wait and all-market policies must have fully priced economic outcomes on every diagnostic and confirmation seed.",
            "confirmation": "one independent two-sided comparison; do not search after its result",
            "min_effect_bps": min_effect_bps, "bootstrap_samples": n_boot}
    write_json(path / "plan.json", plan)
    started = time.monotonic()
    lot = int(params.get("sim", {}).get("lot_size", 1))
    rows, summaries, selected = [], [], None
    for cell, (multiplier, horizon) in enumerate(grid):
        chosen = {**params, "qty": max(lot, round(median_top5_depth * multiplier / lot) * lot),
                  "horizon": horizon}
        group = []
        for market_seed in diagnostics:
            p = {**chosen, "seed": market_seed, "sim": {**chosen.get("sim", {}), "seed": market_seed}}
            for agent in ("twap", "random", "all_wait", "all_market"):
                try:
                    row = _control_episode(agent, p)
                except (RuntimeError, ValueError) as exc:
                    row = {"agent": agent, "seed": market_seed, "status": "ERROR", "effective_bps": None,
                           "error": str(exc)}
                group.append({**row, "cell": cell, "phase": "diagnostic"})
        comparison = _paired(group, "twap", "random", n_boot=n_boot)
        capacity_pass = all(r.get("status") in {"VALID", "WARNING"}
                            and r.get("effective_bps") is not None
                            and math.isfinite(float(r["effective_bps"]))
                            for r in group if r["agent"] in {"all_wait", "all_market"})
        passing = (comparison["status"] == "AVAILABLE" and capacity_pass
                   and (comparison["ci_low_bps"] > 0 or comparison["ci_high_bps"] < 0)
                   and abs(comparison["delta_mean_bps"]) >= min_effect_bps)
        summaries.append({"cell": cell, "depth_multiplier": multiplier, "horizon": horizon,
                          "quantity": chosen["qty"], "comparison": comparison,
                          "all_wait_and_market_capacity_pass": capacity_pass, "diagnostic_pass": passing})
        rows.extend(group)
        pd.DataFrame(rows).to_csv(path / "episodes.csv", index=False)
        write_json(path / f"diagnostic-cell-{cell}.json", summaries[-1])
        print(json.dumps({"positive_control_cell": cell, "elapsed_seconds": time.monotonic() - started,
                          **summaries[-1]}), flush=True)
        if passing and selected is None:
            selected = chosen
    confirmation_result = {"status": "NOT_RUN_NO_DIAGNOSTIC_PASS"}
    confirmation_capacity = False
    if selected is not None:
        group = []
        for market_seed in confirmation:
            p = {**selected, "seed": market_seed, "sim": {**selected.get("sim", {}), "seed": market_seed}}
            for agent in ("twap", "random", "all_wait", "all_market"):
                try:
                    row = _control_episode(agent, p)
                except (RuntimeError, ValueError) as exc:
                    row = {"agent": agent, "seed": market_seed, "status": "ERROR", "effective_bps": None,
                           "error": str(exc)}
                group.append({**row, "phase": "confirmation"})
        rows.extend(group)
        confirmation_result = _paired(group, "twap", "random", n_boot=n_boot)
        confirmation_capacity = all(r.get("status") in {"VALID", "WARNING"}
                                    and r.get("effective_bps") is not None
                                    and math.isfinite(float(r["effective_bps"]))
                                    for r in group if r["agent"] in {"all_wait", "all_market"})
    passed = (confirmation_result["status"] == "AVAILABLE" and confirmation_capacity
              and (confirmation_result["ci_low_bps"] > 0 or confirmation_result["ci_high_bps"] < 0)
              and abs(confirmation_result["delta_mean_bps"]) >= min_effect_bps)
    result = {"status": "PASS" if passed else "FAIL", "diagnostics": summaries,
              "selected_params": selected, "confirmation": confirmation_result,
              "confirmation_all_wait_and_market_capacity_pass": confirmation_capacity,
              "elapsed_seconds": time.monotonic() - started,
              "interpretation": "Execution-cost discrimination only; not a calibration or PPO superiority test."}
    pd.DataFrame(rows).to_csv(path / "episodes.csv", index=False)
    write_json(path / "result.json", result)
    return result


def _reasons(value: Any) -> set[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    return set(value or [])


def residual_sensitivity(episodes: pd.DataFrame, *, quantity: int | dict[str, int],
                         reference: str = "twap", haircuts_bps: Sequence[float] = (100., 500.),
                         n_boot: int = 2000) -> dict:
    """Per-comparison raw inference and explicitly labeled residual-price proxies.

    For insufficient-terminal-depth INVALID rows only, value *all* remaining
    quantity at arrival plus/minus the adverse haircut. Preserve actual filled
    shortfall and fees; add no fictitious terminal fee. This is a conditional
    stress proxy, not a guaranteed bound: unconstrained prices have no finite
    worst case. Other INVALIDs remain withheld only within affected pairs.
    """
    if not len(episodes) or episodes.duplicated(["scenario", "agent", "seed"]).any():
        raise ValueError("nonempty unique scenario/agent/seed episodes required")
    if not haircuts_bps or any(not math.isfinite(h) or h <= 0 for h in haircuts_bps):
        raise ValueError("positive finite residual haircuts required")
    raw_rows = episodes.to_dict("records")
    comparisons = []
    for haircut in (None, *haircuts_bps):
        arm = []
        imputed = 0
        for source in raw_rows:
            row = dict(source)
            if haircut is not None and row.get("status") == "INVALID":
                reasons = _reasons(row.get("invalid_reasons"))
                if reasons == {"insufficient_terminal_depth"}:
                    target = quantity[row["scenario"]] if isinstance(quantity, dict) else quantity
                    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
                        raise ValueError("positive integer parent quantity required for each scenario")
                    arrival, fill_frac = float(row["arrival"]), float(row["fill_frac"])
                    if arrival <= 0 or not math.isfinite(arrival) or not 0 <= fill_frac <= 1:
                        raise ValueError("invalid arrival/fill fraction inputs")
                    actual = (float(row["gross_cost"]) + float(row["total_fees"])) / (target * arrival) * 10000
                    proxy = actual + (1 - fill_frac) * haircut
                    if arrival <= 0 or not 0 <= fill_frac <= 1 or not math.isfinite(proxy):
                        raise ValueError("invalid actual filled-cost inputs")
                    row.update(effective_bps=proxy, status="WARNING")
                    imputed += 1
            arm.append(row)
        family = []
        for scenario in dict.fromkeys(r["scenario"] for r in arm):
            group = [r for r in arm if r["scenario"] == scenario]
            for agent in dict.fromkeys(r["agent"] for r in group):
                if agent == reference:
                    continue
                subset = [r for r in group if r["agent"] in {reference, agent}]
                comp = _paired(subset, reference, agent, n_boot=n_boot)
                family.append({**comp, "scenario": scenario, "agent": agent, "reference": reference})
        # Unidentified tests stay in the planned family as p=1, so valid pairs
        # remain testable without silently shrinking multiplicity correction.
        adjusted = adjust_pvalues([r.get("p_sign", 1.) for r in family], "holm")
        for row, corrected in zip(family, adjusted):
            row.update(p_adjusted=float(corrected), family_size=len(family), correction="holm")
        comparisons.append({"arm": "raw" if haircut is None else f"residual_stress_proxy_{haircut:g}bps",
                            "haircut_bps": haircut, "imputed_rows": imputed, "comparisons": family})
    return {"raw_status_counts": episodes.status.value_counts().to_dict(), "episodes": len(episodes),
            "interpretation": "Proxy arms assume adverse residual liquidation relative to arrival; these are not observed prices or guaranteed worst-case bounds.",
            "terminal_proxy_fee": "No added fee on hypothetical residual; actual fees preserved.",
            "arms": comparisons}


def run_stress_sensitivity(source: str | Path, out: str | Path, *, quantity: int = 600,
                           haircuts_bps: Sequence[float] = (100., 500.), n_boot: int = 2000) -> Path:
    """Seal a separate sensitivity study without modifying the original evidence."""
    from .robustness import _new_run, _seal, verify_study

    source = Path(source).resolve(strict=True)
    verification = verify_study(source)
    if not verification["valid"]:
        raise ValueError(f"source stress study failed verification: {verification['issues']}")
    episode_path = source / "episodes.csv"
    plan = {"kind": "invalid_residual_sensitivity", "source_study": source.as_posix(),
            "source_episodes_sha256": sha256_file(episode_path),
            "source_manifest_sha256": sha256_file(source / "manifest.json"),
            "quantity": quantity, "haircuts_bps": list(haircuts_bps), "bootstrap_samples": n_boot,
            "reference": "twap", "registration_status": "post_hoc_existing_stress_sensitivity",
            "multiplicity": "Holm over all planned comparisons separately per declared sensitivity arm"}
    run, _ = _new_run(out, plan)
    episodes = pd.read_csv(episode_path)
    result = residual_sensitivity(episodes, quantity=quantity, haircuts_bps=haircuts_bps, n_boot=n_boot)
    if sha256_file(episode_path) != plan["source_episodes_sha256"]:
        raise ValueError("source episodes changed during sensitivity analysis")
    write_json(run / "result.json", result)
    rows = []
    for arm in result["arms"]:
        for comparison in arm["comparisons"]:
            rows.append({"arm": arm["arm"], "imputed_rows_in_arm": arm["imputed_rows"], **comparison})
    pd.DataFrame(rows).to_csv(run / "comparisons.csv", index=False)
    report = ["# Residual-price sensitivity", "", result["interpretation"], "",
              "This is a post-hoc analysis of previously sealed stress evidence. The original statuses and files are preserved.", "",
              "| Arm | Imputed rows | Available comparisons | Withheld comparisons |",
              "| --- | ---: | ---: | ---: |"]
    for arm in result["arms"]:
        available = sum(r["status"] == "AVAILABLE" for r in arm["comparisons"])
        report.append(f"| {arm['arm']} | {arm['imputed_rows']} | {available} | {len(arm['comparisons']) - available} |")
    report.extend(["", "Unavailable comparisons remain in each planned Holm family with p=1. Valid scenarios no longer inherit another scenario's INVALID veto.", "",
                   "The proxy applies 100/500 bps adverse arrival-relative prices to all unfilled quantity; it preserves actual filled shortfall and actual fees. It adds no invented terminal fee, and does not price unresolved orders.", ""])
    (run / "report.md").write_text("\n".join(report), encoding="utf-8")
    _seal(run)
    return run


def execution_design(model_path: str | Path, out: str | Path,
                     config_out: str | Path) -> dict:
    """Fit AC, run registered controls, and freeze one complete execution config.

    A failed control does not produce a study configuration. It remains a
    preserved failed study; any diagnostic amendment must be explicit.
    """
    from .config import MarketSettings, ResearchConfig
    from .zi_calibration import load_model

    model_path = Path(model_path).resolve(strict=True)
    model = load_model(model_path)
    path = Path(out)
    path.mkdir(parents=True, exist_ok=False)
    config_path = Path(config_out)
    if config_path.exists() or config_path.with_name(config_path.stem + "-ac-risk.json").exists():
        raise FileExistsError("execution configuration already exists")
    cfg = SimConfig(**model["simulator_config"])
    fit_seeds = list(range(43000, 43016))
    plan = {
        "model_path": model_path.as_posix(), "model_file_sha256": sha256_file(model_path),
        "model_sha256": model["model_sha256"], "ac_fit_seeds": fit_seeds,
        "ac_fit_horizon_seconds": 120., "ac_sample_dt_seconds": 1., "warmup_seconds": 30.,
        "diagnostic_seeds": list(range(44000, 44016)),
        "confirmation_seeds": list(range(45000, 45032)),
        "primary_risk_aversion": 0.,
        "primary_objective": "Expected net execution cost, matching PPO; risk-neutral AC is the analytical TWAP limit.",
        "secondary_risk_aversion": "eta / (sigma * horizon)^2, giving kappa * horizon = 1; no selection using test outcomes",
        "decision_dt_seconds": .5, "taker_fee_bps": 1., "terminal_penalty_bps": 0.,
        "control_grid": [[2., 30.], [5., 60.], [10., 120.]],
        "control_min_effect_bps": .5, "bootstrap_samples": 2000,
    }
    write_json(path / "plan.json", plan)
    initial = estimate_ac_parameters(cfg, fit_seeds, horizon=120., warmup_seconds=30.)
    write_json(path / "ac-one-second-probes.json", initial)
    print(json.dumps({"ac_initial_fit_status": initial["status"],
                      "median_top5_depth": initial["median_top5_depth"]}), flush=True)
    market = {name: getattr(cfg, name) for name in MarketSettings.model_fields}
    base = ResearchConfig(name="ppo_vs_simulator_fitted_ac", market=market,
                          execution={"warmup_seconds": 30., "terminal_penalty_bps": 0.,
                                     "risk_aversion": 0., "temp_impact": initial["temp_impact"],
                                     "sigma": initial["sigma"]},
                          evaluation={"agents": ("ac", "twap", "random"), "reference": "ac",
                                      "seeds": (61000,)},
                          resources={"max_episodes": 10000, "max_runtime_seconds": 7200.,
                                     "max_estimated_events": 1_000_000_000,
                                     "max_events_per_episode": 2_000_000})
    params = base.runner_params(61000)
    params["sim"]["record_events"] = False
    control = positive_control(params, plan["diagnostic_seeds"], plan["confirmation_seeds"],
                               path / "positive-control", median_top5_depth=initial["median_top5_depth"])
    result = {"status": control["status"], "positive_control": control,
              "calibration_train_status": model["training_comparison"]["status"]}
    if control["status"] == "PASS":
        selected = control["selected_params"]
        fitted = estimate_ac_parameters(cfg, fit_seeds, horizon=120.,
                                        execution_interval=selected["horizon"] / 20., warmup_seconds=30.)
        write_json(path / "ac-fit.json", fitted)
        document = base.model_dump(mode="json")
        document["execution"].update(quantity=selected["qty"], horizon=selected["horizon"],
                                     temp_impact=fitted["temp_impact"], sigma=fitted["sigma"])
        resolved = ResearchConfig.model_validate(document)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(config_path, resolved.model_dump(mode="json"))
        risk_lambda = (fitted["temp_impact"] / (fitted["sigma"] * selected["horizon"]) ** 2
                       if fitted["sigma"] > 0 else None)
        sensitivity = {"risk_aversion": risk_lambda, "target_kappa_horizon": 1.,
                       "status": "AVAILABLE" if risk_lambda is not None else "UNAVAILABLE_ZERO_VOLATILITY",
                       "temp_impact": fitted["temp_impact"], "sigma": fitted["sigma"],
                       "horizon": selected["horizon"], "primary_risk_aversion": 0.,
                       "interpretation": "Prespecified risk-sensitive AC comparator; primary AC remains risk neutral to match the PPO expected-cost objective."}
        write_json(config_path.with_name(config_path.stem + "-ac-risk.json"), sensitivity)
        result.update(config_path=config_path.as_posix(), ac_fit_status=fitted["status"],
                      risk_sensitive_ac=sensitivity)
    write_json(path / "result.json", result)
    return result


def finalize_execution_config(model_path: str | Path, control_path: str | Path,
                              out: str | Path, config_out: str | Path) -> dict:
    """Freeze a conditional study config with independently refitted AC inputs.

    Failed scientific gates remain failed. A configuration may still support a
    clearly labeled synthetic experiment if the fixed capacity controls pass.
    No final market or PPO training seeds enter this selection or identification.
    """
    from .config import MarketSettings, ResearchConfig
    from .zi_calibration import load_model

    model_path = Path(model_path).resolve(strict=True)
    model = load_model(model_path)
    control_path = Path(control_path).resolve(strict=True)
    control = json.loads((control_path / "result.json").read_text(encoding="utf-8"))
    control_plan = json.loads((control_path / "plan.json").read_text(encoding="utf-8"))
    selected = control["selected_params"]
    selection = "Previously selected diagnostic cell; independent confirmation is not used for reselection."
    if selected is None:
        cells = [cell for cell in control["diagnostics"]
                 if cell["all_wait_and_market_capacity_pass"]
                 and cell["comparison"]["status"] == "AVAILABLE"]
        if not cells:
            raise ValueError("No fully priced capacity cell; a registered capacity amendment is required")
        cell = cells[0]
        selected = {**control_plan["params"], "qty": cell["quantity"], "horizon": cell["horizon"]}
        selection = "First fully priced diagnostic capacity cell, irrespective of effect size; exploratory synthetic configuration only."
    path, config_path = Path(out), Path(config_out)
    if config_path.exists() or config_path.with_name(config_path.stem + "-ac-risk.json").exists():
        raise FileExistsError("final execution configuration already exists")
    path.mkdir(parents=True, exist_ok=False)
    plan = {
        "model_file_sha256": sha256_file(model_path), "model_sha256": model["model_sha256"],
        "control_result_sha256": sha256_file(control_path / "result.json"),
        "control_status": control["status"], "selected_params": selected, "selection": selection,
        "ac_fit_seeds": list(range(46000, 46016)), "ac_fit_horizon_seconds": 120.,
        "warmup_seconds": 30., "ac_sample_dt_seconds": 1.,
        "ac_execution_interval_seconds": selected["horizon"] / 20.,
        "seed_amendment": "Initial identification seeds 43001/43002 overlapped calibration holdout simulator seeds. Preserve that exploratory fit, then refit final AC inputs on independent 46000..46015 before PPO registration. Historical holdout observations never enter AC identification.",
        "primary_risk_aversion": 0., "secondary_kappa_times_horizon": 1.,
        "runtime_check_invariants": False,
        "runtime_check_interpretation": "Disable exhaustive after-event assertions in both training and evaluation. This changes checks only; a path-identity regression verifies identical exchange events and fills.",
        "interpretation": "Synthetic execution comparison conditional on a frozen simulator; failed real-data calibration or execution-discrimination gates are not changed by this configuration.",
    }
    write_json(path / "plan.json", plan)
    cfg = SimConfig(**model["simulator_config"])
    fitted = estimate_ac_parameters(cfg, plan["ac_fit_seeds"], horizon=120.,
                                    execution_interval=plan["ac_execution_interval_seconds"], warmup_seconds=30.)
    write_json(path / "ac-fit.json", fitted)
    if fitted["temp_impact"] <= 0:
        raise ValueError("AC identification has nonpositive slope; no valid parameterized AC comparator")
    market = {name: getattr(cfg, name) for name in MarketSettings.model_fields}
    resolved = ResearchConfig(
        name="ppo_vs_simulator_fitted_ac", market=market,
        execution={"quantity": selected["qty"], "horizon": selected["horizon"],
                   "decision_dt": selected["dt"], "warmup_seconds": 30.,
                   "terminal_penalty_bps": 0., "risk_aversion": 0.,
                   "temp_impact": fitted["temp_impact"], "sigma": fitted["sigma"]},
        evaluation={"agents": ("ac", "twap", "random"), "reference": "ac", "seeds": (61000,)},
        resources={"max_episodes": 10000, "max_runtime_seconds": 7200.,
                   "max_estimated_events": 1_000_000_000, "max_events_per_episode": 2_000_000,
                   "check_invariants": False})
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(config_path, resolved.model_dump(mode="json"))
    risk_lambda = (fitted["temp_impact"] / (fitted["sigma"] * selected["horizon"]) ** 2
                   if fitted["sigma"] > 0 else None)
    sensitivity = {"risk_aversion": risk_lambda, "target_kappa_horizon": 1.,
                   "status": "AVAILABLE" if risk_lambda is not None else "UNAVAILABLE_ZERO_VOLATILITY",
                   "temp_impact": fitted["temp_impact"], "sigma": fitted["sigma"],
                   "horizon": selected["horizon"], "primary_risk_aversion": 0.}
    write_json(config_path.with_name(config_path.stem + "-ac-risk.json"), sensitivity)
    result = {"positive_control_status": control["status"], "ac_fit_status": fitted["status"],
              "calibration_train_status": model["training_comparison"]["status"],
              "config_path": config_path.as_posix(), "config_sha256": sha256_file(config_path),
              "selection": selection, "risk_sensitive_ac": sensitivity,
              "interpretation": plan["interpretation"]}
    write_json(path / "result.json", result)
    return result


def continue_execution_controls(initial_path: str | Path, out: str | Path) -> dict:
    """Repeat interrupted controls with check-only overhead disabled, preserving evidence."""
    initial_path = Path(initial_path).resolve(strict=True)
    original = initial_path / "positive-control"
    plan = json.loads((original / "plan.json").read_text(encoding="utf-8"))
    path = Path(out)
    path.mkdir(parents=True, exist_ok=False)
    amendment = {
        "kind": "compute_only_continuation", "original_plan_sha256": sha256_file(original / "plan.json"),
        "original_episodes_sha256": sha256_file(original / "episodes.csv"),
        "change": "Disable exhaustive invariant assertions; original grid, seeds, objective and effect threshold unchanged.",
        "reason": "Assertions dominate execution time. Regression confirms identical simulator paths when toggled.",
        "verification": "Every recorded economic and execution outcome of completed original diagnostic cells must match exactly after CSV round-trip.",
    }
    write_json(path / "continuation.json", amendment)
    params = {**plan["params"], "sim": {**plan["params"]["sim"], "check_invariants": False}}
    result = positive_control(params, plan["diagnostic_seeds"], plan["confirmation_seeds"],
                              path / "positive-control", median_top5_depth=plan["median_top5_depth"],
                              grid=plan["grid"], min_effect_bps=plan["min_effect_bps"],
                              n_boot=plan["bootstrap_samples"])
    previous = pd.read_csv(original / "episodes.csv")
    current = pd.read_csv(path / "positive-control" / "episodes.csv")
    current = current[(current.phase == "diagnostic") & current.cell.isin(previous.cell.unique())]
    pd.testing.assert_frame_equal(previous.reset_index(drop=True), current.reset_index(drop=True),
                                  check_exact=True, check_dtype=False)
    write_json(path / "path-equivalence.json", {
        "status": "PASS", "compared_episode_rows": len(previous),
        "comparison": "Exact equality of all recorded outcomes from completed original diagnostic cells."})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulator-fitted AC and independent execution controls")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config-out", default="configs/core-study.json")
    parser.add_argument("--continuation-from")
    args = parser.parse_args()
    if args.continuation_from:
        result = continue_execution_controls(args.continuation_from, args.out)
    else:
        result = execution_design(args.model, args.out, args.config_out)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
