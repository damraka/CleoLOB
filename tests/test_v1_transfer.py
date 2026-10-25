"""v1 transfer across time / instrument / venue with heterogeneity (M137)."""
import numpy as np

from lob.v1.transfer import study as TS


def dates(n, off=0):
    return [f"2026-0{1 + (k + off) // 28}-{1 + (k + off) % 28:02d}" for k in range(n)]


def test_transfer_decides_per_domain_and_flags_heterogeneity():
    rng = np.random.default_rng(0)
    vals = {"late": {"C1": dict(zip(dates(12), 5 + rng.standard_normal(12))),
                     "C3": dict(zip(dates(12), 1.0 + 0.2 * rng.standard_normal(12)))},
            "sol": {"C1": dict(zip(dates(12, 30), 5 + rng.standard_normal(12))),
                    "C3": dict(zip(dates(12, 30), 0.0 + 0.2 * rng.standard_normal(12)))}}
    out = TS.transfer(vals, {"C1": 1.0, "C3": 0.25})
    assert out["domains"]["late"]["status"] == "ESTABLISHED"
    assert out["domains"]["sol"]["conclusions"]["C3"]["status"] != "ESTABLISHED"
    assert out["transfers_everywhere"]["C1"] and not out["transfers_everywhere"]["C3"]
    assert out["heterogeneity"]["C3"]["i2"] > out["heterogeneity"]["C1"]["i2"]
