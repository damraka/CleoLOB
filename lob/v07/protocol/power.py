"""Prospective power and minimum detectable effects (workstream 64), sealed before the registered studies.

Normal-approximation MDE for a two-sided test at (adjusted) alpha with power 1 - beta:
``MDE = (z_{1-alpha/2} + z_{1-beta}) * SE``. Variance inputs are pilot estimates from sealed v0.6
results (retrospective and v0.6-fresh data; never a v0.7 holdout), stated with their source. Where
the v0.7 design differs (world count, seeds), the SE is rescaled by the registered sample sizes.
These are design approximations, not guarantees.
"""
from __future__ import annotations

from math import sqrt
from statistics import NormalDist

Z = NormalDist()


def z(p: float) -> float:
    return Z.inv_cdf(p)


def mde(se: float, *, alpha: float, power: float = 0.8, sides: int = 2) -> float:
    return (z(1 - alpha / sides) + z(power)) * se


def required_n(sd: float, effect: float, *, alpha: float, power: float = 0.8, sides: int = 2) -> int:
    return int((((z(1 - alpha / sides) + z(power)) * sd / effect) ** 2) + 0.999999)


def pooled_world_se(within_sd: float, between_sd: float, worlds: int, seeds: int) -> float:
    return sqrt(within_sd ** 2 / (worlds * seeds) + between_sd ** 2 / worlds)


def design(families: dict) -> dict:
    """The v0.7 prospective power document (inputs, assumptions, MDEs per family)."""
    a = {k: v["adjusted_alpha"] for k, v in families.items()}
    out = {}
    # Realism contrasts (H1, H4, H5, H10, H11): v0.6 objective-difference bootstrap SE on 144-block days.
    realism_se = 0.07   # v0.6 H1-H3 Bonferroni CIs had half-widths 0.09-0.15 at alpha 0.05/3 -> SE ~0.04-0.08
    out["F1_posterior"] = {"se": realism_se, "alpha": a["F1_posterior"], "mde_objective": mde(realism_se, alpha=a["F1_posterior"]),
                           "source": "v0.6 m13 holdout contrasts (SE 0.04-0.08 objective units); upper value used"}
    out["F4_cross_market"] = {"se": realism_se, "alpha": a["F4_cross_market"],
                              "mde_objective": mde(realism_se, alpha=a["F4_cross_market"]), "source": "as F1"}
    out["F10_execution_realism"] = {"se": 0.15, "alpha": a["F10_execution_realism"],
                                    "mde_es_objective": mde(0.15, alpha=a["F10_execution_realism"]),
                                    "source": "assumed 2x the generic SE (8 components instead of 9 families)"}
    out["F11_noninferiority"] = {"se": realism_se, "alpha": a["F11_noninferiority"],
                                 "one_sided_resolution": mde(realism_se, alpha=a["F11_noninferiority"], sides=1),
                                 "note": "delta = 0.10 x G0 selection objective (about 0.25) exceeds this resolution"}
    # Domain gap and support (H2, H3): AUC and coverage differences on ~720 test windows per day.
    out["F2_distinguishability"] = {"se": 0.02, "alpha": a["F2_distinguishability"],
                                    "mde_auc": mde(0.02, alpha=a["F2_distinguishability"]),
                                    "caveat": "near the AUC ceiling (v0.6 AUC ~1) the attainable effect is bounded; "
                                              "H2 then reports VACUOUS"}
    out["F3_support"] = {"se": 0.02, "alpha": a["F3_support"], "mde_coverage": mde(0.02, alpha=a["F3_support"]),
                         "margin": 0.05}
    # Execution (H7, H8): v0.6 per-episode cost SD 5.3-8.6 bps; paired-difference SD assumed 6 bps.
    within, between = 6.0, 1.0
    se8 = pooled_world_se(within, between, worlds=21, seeds=256)
    out["F8_robust_pairs"] = {"within_world_paired_sd_bps": within, "between_world_sd_bps": between,
                              "worlds": 21, "seeds": 256, "se": se8, "alpha": a["F8_robust_pairs"],
                              "mde_bps": mde(se8, alpha=a["F8_robust_pairs"]), "margin_bps": 1.0,
                              "source": "v0.6 m10 market-seed SD 5.3-8.6 bps; between-world SD from v0.6 sensitivity "
                                        "(1-4.8 bps, lower end assumed)",
                              "caveat": "with between-world SD of 4.8 bps the MDE rises to "
                                        f"{mde(pooled_world_se(within, 4.8, 21, 256), alpha=a['F8_robust_pairs']):.2f} bps"}
    out["F7_model_uncertainty"] = {"world_mean_se_bps": 7.5 / sqrt(256), "material_between_world_sd_bps": 0.5,
                                   "alpha_first_holm_step": a["F7_model_uncertainty"],
                                   "note": "R > 1 is detectable when the between-world SD exceeds about "
                                           f"{(1 + mde(0.15, alpha=a['F7_model_uncertainty'])) * 7.5 / 16:.2f} bps "
                                           "(approximate: SE of R ~0.15 with 21 worlds)"}
    # Transfer (H12): v0.6 H11 gap-difference CIs had half-width ~1.5-1.8 bps at alpha 0.025.
    se12 = 3.47 / (2 * z(1 - 0.025 / 2))
    out["F12_transfer"] = {"se": se12, "alpha": a["F12_transfer"], "mde_bps": mde(se12, alpha=a["F12_transfer"]),
                           "episodes": 144, "training_seeds": 4,
                           "source": "v0.6 m14 transfer (ETH 2020-10-01, retrospective in v0.7) H11 intervals"}
    return {"schema": "cleolob-v07-power-1", "power": 0.8, "families": out,
            "rule": "MDE = (z_{1-alpha/2} + z_{power}) * SE; inputs are sealed v0.6 pilot estimates",
            "interpretation": "Effects below the MDE are unlikely to be resolved; NOT_ESTABLISHED below the MDE is "
                              "not evidence of absence."}
