"""``final-analysis-v08``: apply the registered H1-H21 rules to the sealed runs.

No hypothesis is evaluated by judgement: every status below is computed from stored numbers by the rule text in
``configs/v08/hypotheses.json``. Primary-mandate results decide statuses; the registered secondary (400-lot)
mandate is reported beside each member (amendment 1). This step also performs the registered acquisition attempt
for the final real holdout (H20) and, only when an execution conclusion is carried from H8/H9, opens the sealed
synthetic holdout namespace once and evaluates H21.

Inputs (``--input KEY=RUN``): drift, regime, switch, recovery, uncertainty, calutil, completion, robust, mpc,
selection, abstention, counterfactual, conformal, transfer_fresh, ranking (optional), worlds (for H21).
"""
from __future__ import annotations

import numpy as np

from .. import protocol as pr, realdata
from ..statistics import hierarchical_paired, paired_bootstrap
from ..taxonomy import Result
from . import common as C

KEYS = ("drift", "regime", "switch", "recovery", "uncertainty", "calutil", "completion", "robust", "mpc", "selection",
        "abstention", "counterfactual", "conformal")
FINAL_HOLDOUT = ("deribit-eth-perp-2021-03-01", "deribit-btc-perp-2021-03-01", "bitmex-xbtusd-2021-03-01")


def _ci_rule(est, lo, hi, threshold, *, lower_is_better=False) -> str:
    if est is None or lo is None or hi is None:
        return "NOT_EVALUABLE"
    if lo > 0 and est >= threshold:
        return "ESTABLISHED"
    if hi < 0:
        return "FAILED"
    return "NOT_ESTABLISHED"


def _units(values, alpha, boot, seed):
    v = [x for x in values if x is not None and np.isfinite(x)]
    if len(v) < 2:
        return None
    return paired_bootstrap(v, alpha=alpha, n_boot=boot, seed=seed)


def _row(hid, status, *, estimate=None, lo=None, hi=None, threshold=None, n=None, alpha=None, qualifiers=("SYNTHETIC",
         "CONFIRMATORY"), reason="", run=None, members=None, secondary=None) -> dict:
    Result(hid, hid, status, estimate=estimate, ci_low=lo, ci_high=hi, threshold=threshold, alpha=alpha, n=n,
           qualifiers=tuple(qualifiers), reason=reason or ("" if status not in ("NOT_AVAILABLE", "INVALID", "NOT_EVALUABLE",
                                                                                  "INCONCLUSIVE") else "see members"))
    return {"status": status, "estimate": estimate, "ci_low": lo, "ci_high": hi, "threshold": threshold, "n": n,
            "alpha": alpha, "qualifiers": list(qualifiers), "reason": reason, "run": run, "members": members or {},
            "secondary_mandate": secondary}


def run(args) -> dict:
    root = C.root_of(args)
    out = C.start(args)
    boot = 4000
    runs = {k: C.input_run(args, k) for k in KEYS if k in args.inputs}
    res = {k: C.read_result(v) for k, v in runs.items()}
    rel = {k: v.relative_to(root).as_posix() for k, v in runs.items()}
    H = {}

    def missing(hid, key):
        H[hid] = _row(hid, "NOT_AVAILABLE", reason=f"sealed run for {key} not provided", qualifiers=())

    # ---------------------------------------------------------------- predictive family
    for hid, key, field, start in (("H1", "drift", "h1", None), ("H2", "regime", "h2", None), ("H3", "switch", "h3", None)):
        if key not in res:
            missing(hid, key)
            continue
        units = res[key]["units"][field] if key == "drift" else res[key][field]
        iv = _units(units, 0.0125, boot, 1)
        st = _ci_rule(iv.estimate, iv.low, iv.high, 0.01) if iv else "NOT_EVALUABLE"
        H[hid] = _row(hid, st, estimate=iv.estimate if iv else None, lo=iv.low if iv else None,
                      hi=iv.high if iv else None, threshold=0.01, n=iv.n if iv else 0, alpha=0.0125, run=rel[key],
                      reason="" if st != "NOT_EVALUABLE" else "fewer than two units")
    if "drift" in res:
        hl = res["drift"]["half_life"]
        a_ok = all(hl.get(s, {}).get("status") == "ESTIMATED" for s in ("gradual_rates", "gradual_mix"))
        iv = _units(res["drift"]["units"]["h18b"], 0.0125, boot, 2)
        b = _ci_rule(iv.estimate, iv.low, iv.high, 0.01) if iv else "NOT_EVALUABLE"
        st = "ESTABLISHED" if a_ok and b == "ESTABLISHED" else "FAILED" if b == "FAILED" else "NOT_ESTABLISHED"
        H["H18"] = _row("H18", st, estimate=iv.estimate if iv else None, lo=iv.low if iv else None,
                        hi=iv.high if iv else None, threshold=0.01, n=iv.n if iv else 0, alpha=0.0125, run=rel["drift"],
                        members={"half_life": {s: {k: hl[s].get(k) for k in ("status", "half_life", "r2", "reason")}
                                               for s in hl}, "post_change": b})
    else:
        missing("H18", "drift")
    # ---------------------------------------------------------------- H4
    if "recovery" in res:
        summary = res["recovery"]["summary"]
        params = {f"{f}.{p}": v for f, ps in summary.items() for p, v in ps.items()}
        ok = all(v["coverage"] >= 0.8 and v["wilson"][0] >= 0.6 and v["median_relative_error"] <= 0.15
                 for v in params.values())
        fail = any(v["wilson"][1] < 0.8 for v in params.values())
        st = "ESTABLISHED" if ok else "FAILED" if fail else "NOT_ESTABLISHED"
        worst = min(params.values(), key=lambda v: v["coverage"]) if params else None
        H["H4"] = _row("H4", st, estimate=worst["coverage"] if worst else None, lo=worst["wilson"][0] if worst else None,
                       hi=worst["wilson"][1] if worst else None, threshold=0.8, n=worst["n"] if worst else 0,
                       run=rel["recovery"], members={k: {kk: v[kk] for kk in ("coverage", "wilson",
                                                                             "median_relative_error", "n")}
                                                     for k, v in params.items()},
                       reason="estimate = lowest per-parameter coverage")
    else:
        missing("H4", "recovery")
    # ---------------------------------------------------------------- H5
    if "uncertainty" in res:
        H["H5"] = _h5(res["uncertainty"], rel["uncertainty"])
    else:
        missing("H5", "uncertainty")
    # ---------------------------------------------------------------- H6
    if "calutil" in res:
        h6 = res["calutil"]["h6"]
        lo, hi = (h6["ci"] or [None, None])
        st = ("ESTABLISHED" if lo is not None and lo > 0 and h6["spearman"] >= 0.2 else "FAILED" if hi is not None and
              hi < 0 else "NOT_ESTABLISHED") if h6["spearman"] is not None else "NOT_EVALUABLE"
        H["H6"] = _row("H6", st, estimate=h6["spearman"], lo=lo, hi=hi, threshold=0.2, n=h6["n"], alpha=h6["alpha"],
                       run=rel["calutil"], reason="" if st != "NOT_EVALUABLE" else "fewer than five pairs")
    else:
        missing("H6", "calutil")
    # ---------------------------------------------------------------- H7
    H["H7"] = _h7(res["completion"], rel["completion"]) if "completion" in res else None
    if H["H7"] is None:
        missing("H7", "completion")
    # ---------------------------------------------------------------- H8
    if "robust" in res:
        h8 = res["robust"]["mandates"]["primary"]["h8"]
        sec = res["robust"]["mandates"].get("large", {}).get("h8")
        if "estimate_cvar_regret_nominal_minus_cvar" not in h8:
            H["H8"] = _row("H8", h8.get("status", "INVALID"), reason=h8.get("reason", ""), run=rel["robust"], secondary=sec)
        elif h8["same_choice"]:
            H["H8"] = _row("H8", "NOT_ESTABLISHED", estimate=0.0, lo=h8["ci_low"], hi=h8["ci_high"], threshold=0.25,
                           run=rel["robust"], reason="nominal and CVaR criteria selected the same candidate",
                           secondary=sec, alpha=h8["alpha"])
        else:
            st = _ci_rule(h8["estimate_cvar_regret_nominal_minus_cvar"], h8["ci_low"], h8["ci_high"], 0.25)
            H["H8"] = _row("H8", st, estimate=h8["estimate_cvar_regret_nominal_minus_cvar"], lo=h8["ci_low"],
                           hi=h8["ci_high"], threshold=0.25, alpha=h8["alpha"], run=rel["robust"], secondary=sec,
                           members={"nominal_choice": h8["nominal_choice"], "cvar_choice": h8["cvar_choice"]})
    else:
        missing("H8", "robust")
    # ---------------------------------------------------------------- H9
    if "mpc" in res:
        H["H9"] = _h9(res["mpc"], rel["mpc"])
    else:
        missing("H9", "mpc")
    # ---------------------------------------------------------------- H10-H12
    if "selection" in res:
        h = res["selection"]["mandates"]["primary"]["h10"]
        st = _ci_rule(h.get("estimate"), h.get("ci_low"), h.get("ci_high"), 0.1)
        H["H10"] = _row("H10", st, estimate=h.get("estimate"), lo=h.get("ci_low"), hi=h.get("ci_high"), threshold=0.1,
                        n=h.get("n"), alpha=0.01, run=rel["selection"],
                        secondary=res["selection"]["mandates"].get("large", {}).get("h10"),
                        reason="" if st != "NOT_EVALUABLE" else "no paired cases")
    else:
        missing("H10", "selection")
    if "abstention" in res:
        H["H11"], H["H12"] = _h11_h12(res["abstention"], rel["abstention"])
    else:
        missing("H11", "abstention")
        missing("H12", "abstention")
    # ---------------------------------------------------------------- H14
    if "counterfactual" in res:
        H["H14"] = _h14(res["counterfactual"], rel["counterfactual"])
    else:
        missing("H14", "counterfactual")
    # ---------------------------------------------------------------- H19
    if "conformal" in res:
        cov = res["conformal"]["mandates"]["primary"]["pooled_exchangeable"]
        lo, hi = cov["ci"]
        st = "ESTABLISHED" if lo >= 0.85 else "FAILED" if hi < 0.85 else "NOT_ESTABLISHED"
        H["H19"] = _row("H19", st, estimate=cov["coverage"], lo=lo, hi=hi, threshold=0.85, n=cov["n"], alpha=0.05,
                        run=rel["conformal"], members={p: {"exchangeable": v.get("exchangeable", {}).get("coverage"),
                                                           "shifted": v.get("shifted", {}).get("coverage")}
                                                       for p, v in res["conformal"]["mandates"]["primary"]["per_policy"].items()},
                        secondary=res["conformal"]["mandates"].get("large", {}).get("pooled_exchangeable"))
    else:
        missing("H19", "conformal")
    # ---------------------------------------------------------------- fresh real data (H13, H15-H17, H20)
    fresh = C.read_result(C.input_run(args, "transfer_fresh")) if "transfer_fresh" in args.inputs else None
    ranking = C.read_result(C.input_run(args, "ranking")) if "ranking" in args.inputs else None
    acquisition = {"fresh_transfer": fresh.get("acquisition") if fresh else "NOT_ATTEMPTED"}
    blocked = fresh is not None and fresh.get("status") == "NOT_AVAILABLE"
    for hid in ("H13", "H15", "H16", "H17"):
        if blocked:
            H[hid] = _row(hid, "NOT_AVAILABLE", qualifiers=("DATA_ACQUISITION_BLOCKED",),
                          reason="fresh holdout acquisition blocked (registered probe failed); no holdout byte requested",
                          run=C.input_run(args, "transfer_fresh").relative_to(root).as_posix(),
                          members={"retrospective_exploratory": (ranking or {}).get("retrospective") if hid == "H13"
                                   else (fresh or {}).get("retrospective_pointer")})
        else:
            H[hid] = _row(hid, "NOT_AVAILABLE", qualifiers=(), reason="fresh transfer phase not run")
    final_acq = _final_holdout(args, root)
    acquisition["final_holdout"] = final_acq
    H["H20"] = _row("H20", "NOT_AVAILABLE" if final_acq["status"] == "NOT_AVAILABLE" else "NOT_EVALUABLE",
                    qualifiers=("DATA_ACQUISITION_BLOCKED",) if final_acq["status"] == "NOT_AVAILABLE" else (),
                    reason=final_acq.get("reason", ""))
    # ---------------------------------------------------------------- H21
    carried = _carried(H)
    H["H21"] = _h21(args, root, carried, out)
    order = [f"H{k}" for k in range(1, 22)]
    statuses = {h: H[h]["status"] for h in order}
    result = {"hypotheses": {h: H[h] for h in order}, "statuses": statuses, "acquisition": acquisition,
              "carried_conclusions": carried, "inputs": rel,
              "protocol_sha256": pr.verify_protocol_files(root)["protocol_sha256"],
              "rule_source": "configs/v08/hypotheses.json (status_rule) applied mechanically by lob.v08.studies.final"}
    inputs = list(rel.values()) + [x for x in (args.inputs.get("transfer_fresh"), args.inputs.get("ranking")) if x]
    return C.finish(args, out, analysis="final_analysis", config={"keys": sorted(args.inputs)},
                    result=C.json_safe(result), inputs=inputs)


def _h5(r, run) -> dict:
    pairs = r["mandates"]["primary"]["decomposition"]["pairs"]
    members, passing, invalid = {}, [], []
    for k, v in pairs.items():
        if v.get("status") == "INVALID":
            invalid.append(k)
            members[k] = {"status": "INVALID"}
            continue
        members[k] = {"epistemic_share": v["epistemic_share"], "ci": v["epistemic_share_ci"],
                      "epistemic_sd": v["epistemic_sd"]}
        if v["epistemic_share_ci"][0] >= 0.25:
            passing.append(k)
    valid = [k for k in members if k not in invalid]
    if passing:
        st = "ESTABLISHED"
    elif invalid and not valid:
        st = "INVALID"
    elif valid and all(members[k]["ci"][1] < 0.10 for k in valid):
        st = "FAILED"
    else:
        st = "NOT_ESTABLISHED"
    best = max(valid, key=lambda k: members[k]["ci"][0]) if valid else None
    sec = r["mandates"].get("large", {}).get("decomposition", {}).get("pairs")
    return _row("H5", st, estimate=members[best]["epistemic_share"] if best else None,
                lo=members[best]["ci"][0] if best else None, hi=members[best]["ci"][1] if best else None, threshold=0.25,
                alpha=0.0125, run=run, members=members, reason=f"estimate from the pair with the highest lower bound ({best})",
                secondary={k: {"epistemic_share": v.get("epistemic_share"), "ci": v.get("epistemic_share_ci")}
                           for k, v in (sec or {}).items()})


def _h7(r, run) -> dict:
    members = r["mandates"]["primary"]["members"]
    considered, out, passing, inferior = [], {}, [], []
    for p, v in members.items():
        if v.get("status") == "INVALID":
            out[p] = {"status": "INVALID"}
            continue
        unc = v["unconstrained_completion"]
        ci, cd = v["completion_increase"], v["cost_difference_hard_minus_none"]
        out[p] = {"unconstrained_completion": unc, "completion_increase": ci["estimate"],
                  "completion_increase_ci": [ci["ci_low"], ci["ci_high"]], "cost_difference_bps": cd["estimate"],
                  "cost_difference_one_sided_upper": cd["ci_high"], "considered": unc is not None and unc < 0.95}
        if out[p]["considered"]:
            considered.append(p)
            if ci["estimate"] >= 0.10 and ci["ci_low"] > 0.05 and cd["ci_high"] < 3.0:
                passing.append(p)
            if cd["ci_low"] > 3.0:
                inferior.append(p)
    if not considered:
        st = "NOT_EVALUABLE"
    elif inferior:
        st = "FAILED"
    elif set(passing) == set(considered):
        st = "ESTABLISHED"
    else:
        st = "NOT_ESTABLISHED"
    worst = min((out[p] for p in considered), key=lambda v: v["completion_increase"]) if considered else None
    sec = r["mandates"].get("large", {}).get("members")
    return _row("H7", st, estimate=worst["completion_increase"] if worst else None,
                lo=worst["completion_increase_ci"][0] if worst else None,
                hi=worst["completion_increase_ci"][1] if worst else None, threshold=0.1, alpha=0.0025, run=run,
                members=out, reason="no member was materially incomplete" if st == "NOT_EVALUABLE" else
                "estimate = smallest completion increase among considered members",
                secondary={p: {"completion_increase": v.get("completion_increase", {}).get("estimate"),
                               "cost_difference_bps": v.get("cost_difference_hard_minus_none", {}).get("estimate"),
                               "unconstrained_completion": v.get("unconstrained_completion")}
                           for p, v in (sec or {}).items()})


def _h9(r, run) -> dict:
    members = r["mandates"]["primary"]["h9_members"]
    out, established, valid = {}, [], []
    for k, v in members.items():
        if "cvar_difference_twap_minus_member" not in v:
            out[k] = {"status": v.get("status", "INVALID")}
            continue
        s = v["cvar_difference_twap_minus_member"]
        valid.append(k)
        out[k] = {"policy": v["policy"], "estimate": s["estimate"], "ci": [s["ci_low"], s["ci_high"]],
                  "p_value": s["p_value"], "holm_reject": v.get("holm_reject")}
        if v.get("holm_reject") and s["ci_low"] > 0 and s["estimate"] >= 0.25:
            established.append(k)
    st = "ESTABLISHED" if established else "FAILED" if valid and all(out[k]["ci"][1] < 0 for k in valid) else \
        "NOT_ESTABLISHED" if valid else "INVALID"
    best = max(valid, key=lambda k: out[k]["estimate"]) if valid else None
    sec = r["mandates"].get("large", {}).get("h9_members")
    return _row("H9", st, estimate=out[best]["estimate"] if best else None, lo=out[best]["ci"][0] if best else None,
                hi=out[best]["ci"][1] if best else None, threshold=0.25, alpha=0.01, run=run, members=out,
                reason=f"estimate from the member with the largest CVaR improvement ({best})",
                secondary={k: (v.get("cvar_difference_twap_minus_member") or {}) for k, v in (sec or {}).items()})


def _h11_h12(r, run):
    p = r["mandates"]["primary"]
    h11 = p["h11"]
    st11 = _ci_rule(h11["estimate"], h11["ci_low"], h11["ci_high"], 0.05)
    row11 = _row("H11", st11, estimate=h11["estimate"], lo=h11["ci_low"], hi=h11["ci_high"], threshold=0.05,
                 n=h11["n"], alpha=0.025, run=run, members={"without": p["without_abstention"],
                                                            "with": p["with_abstention"]},
                 secondary=r["mandates"].get("large", {}).get("h11"))
    h12 = p["h12"]
    if not h12.get("n_ood_cases"):
        return row11, _row("H12", "NOT_EVALUABLE", reason="no OOD cases", run=run)
    wl, wu = h12["with_fallback_wilson"]
    red = h12["reduction"] or {}
    if wu <= 0.10 and (red.get("ci_low") is None or red.get("ci_low") >= 0):
        st = "ESTABLISHED"
    elif wl > 0.10:
        st = "FAILED"
    else:
        st = "NOT_ESTABLISHED"
    vacuous = h12["without_fallback_wilson"][1] <= 0.10
    quals = ("SYNTHETIC", "CONFIRMATORY", "VACUOUS") if st == "ESTABLISHED" and vacuous else ("SYNTHETIC", "CONFIRMATORY")
    row12 = _row("H12", st, estimate=h12["with_fallback_rate"], lo=wl, hi=wu, threshold=0.10, n=h12["n_ood_cases"],
                 alpha=0.025, run=run, qualifiers=quals,
                 members={"without_fallback_rate": h12["without_fallback_rate"],
                          "without_fallback_wilson": h12["without_fallback_wilson"], "reduction": red},
                 reason="VACUOUS: large degradation was already rare without fallback" if "VACUOUS" in quals else "",
                 secondary=r["mandates"].get("large", {}).get("h12"))
    return row11, row12


def _h14(r, run) -> dict:
    p = r["mandates"]["primary"]
    cov = p["coverage"]
    rejected = {k for k, v in cov.items() if v.get("economic_cost_bps", {}).get("expected_value_coverage_rejected")}
    det = {k: v for k, v in p["pairs"].items() if v.get("status") in ("A_LOWER", "B_LOWER")}
    clean = [k for k in det if not (set(k.split("|")) & rejected)]
    if clean:
        st = "ESTABLISHED"
    elif det:
        st = "ASSUMPTION_DEPENDENT"
    else:
        st = "NOT_ESTABLISHED"
    return _row("H14", st, estimate=float(len(det)), threshold=1.0, n=len(p["pairs"]), alpha=0.025, run=run,
                members={"pairs": {k: {kk: v.get(kk) for kk in ("status", "lower", "upper", "n")} for k, v in
                                   p["pairs"].items()}, "coverage_rejected_policies": sorted(rejected),
                         "per_path_coverage": {k: v.get("economic_cost_bps", {}).get("per_path_coverage")
                                               for k, v in cov.items()}},
                reason="estimate = number of determinate registered pairs",
                secondary={k: v.get("status") for k, v in r["mandates"].get("large", {}).get("pairs", {}).items()})


def _final_holdout(args, root) -> dict:
    """Registered acquisition attempt for the final real holdout (probe first; never substitutes data)."""
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    state = pr.replay_ledger(pr.read_ledger(root / pr.LEDGER_PATH), protocol)
    design = "final-holdout-h20"
    if design not in state.designs:
        pr.seal_design(root, design, {"hypotheses": ["H20"], "reads": list(FINAL_HOLDOUT),
                                      "rule": "probe, then bounded replays of carried conclusions"},
                       reads=list(FINAL_HOLDOUT), reason="seal the final real holdout reader before any access")
    recorded = {a["dataset"]: a["index"] for a in state.attempts
                if a["design"] == design and a.get("outcome") == "NOT_AVAILABLE" and a["dataset"] in FINAL_HOLDOUT}
    if set(recorded) == set(FINAL_HOLDOUT):
        # the registered probe already failed for every dataset: reuse that ledgered outcome, never probe again
        return {"status": "NOT_AVAILABLE", "datasets": {ds: f"NOT_AVAILABLE: ledger entry {i}" for ds, i in
                                                        recorded.items()},
                "reason": "the registered availability probe failed (ledgered); no holdout byte was requested; "
                          "the holdout stays fresh"}
    outcomes = {}
    for ds in FINAL_HOLDOUT:
        try:
            realdata.acquire(ds, use="evaluate", design=design, purpose="final holdout (H20)", root=root)
            outcomes[ds] = "ACQUIRED"
        except realdata.DataNotAvailable as exc:
            outcomes[ds] = f"NOT_AVAILABLE: {exc}"
    if all(v == "ACQUIRED" for v in outcomes.values()):
        return {"status": "ACQUIRED", "datasets": outcomes,
                "reason": "acquired; evaluation of carried conclusions on real holdout frames is required"}
    return {"status": "NOT_AVAILABLE", "datasets": outcomes,
            "reason": "the registered availability probe failed; no holdout byte was requested; the holdout stays fresh"}


def _carried(H: dict) -> list[dict]:
    out = []
    if H.get("H8", {}).get("status") == "ESTABLISHED":
        out.append({"source": "H8", "kind": "robust_vs_nominal", **H["H8"]["members"]})
    for k, v in H.get("H9", {}).get("members", {}).items():
        if isinstance(v, dict) and v.get("holm_reject") and v.get("ci", [0])[0] > 0 and v.get("estimate", 0) >= 0.25:
            out.append({"source": "H9", "kind": "member_vs_twap", "member": k, "policy": v["policy"]})
    return out


def _h21(args, root, carried: list[dict], out) -> dict:
    if not carried:
        return _row("H21", "NOT_EVALUABLE", qualifiers=("SYNTHETIC_HOLDOUT",),
                    reason="no execution conclusion was ESTABLISHED by H8 or H9, so nothing is carried; the synthetic "
                           "holdout namespace stays unopened")
    from .holdout import evaluate_holdout
    return evaluate_holdout(args, root, carried, out, _row)


__all__ = ["run", "hierarchical_paired"]
