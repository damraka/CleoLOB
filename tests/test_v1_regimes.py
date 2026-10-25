"""v1 regime ontology, discovery and drift detection (M54-M56)."""
import numpy as np

from lob.v1.regimes import regimes as RG


def test_ontology_labels_and_majority():
    rng = np.random.default_rng(0)
    blocks = np.column_stack([rng.random(30), rng.random(30), rng.random(30), np.ones(30)])
    cuts = RG.cuts_from([blocks])
    lab = RG.label(blocks, cuts)
    assert all(x.count("|") == 2 for x in lab)
    assert {x.split("|")[0] for x in lab} == {"calm", "normal", "volatile"}
    blocks[0, 0] = np.nan
    assert RG.label(blocks, cuts)[0] is None
    assert RG.majority(["calm|deep|quiet", "calm|thin|quiet", "volatile|thin|active"]) == "calm"


def test_discovery_recovers_two_regimes():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 1, (300, 3)), rng.normal(6, 1, (300, 3))])
    d = RG.discover(X, ks=range(1, 5), seed=2)
    assert d["k"] == 2
    lab = np.asarray(d["labels"])
    assert (lab[:300] == lab[0]).mean() > 0.97 and (lab[300:] != lab[0]).mean() > 0.97


def test_drift_detectors_find_a_mean_shift():
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(0, 1, 400), rng.normal(2.5, 1, 400)]
    alarms = RG.cusum(x, threshold=8.0)
    assert alarms and 395 <= alarms[0] <= 430
    cps = RG.bocpd(x)["change_points"]
    assert any(395 <= c <= 420 for c in cps) and len(cps) <= 3
    assert RG.bocpd(rng.normal(0, 1, 800))["change_points"] == [] or len(RG.bocpd(rng.normal(0, 1, 800))["change_points"]) <= 1
