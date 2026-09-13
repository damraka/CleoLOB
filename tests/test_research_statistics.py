"""Correction reference values, complete-pair discipline and finite samples."""
import numpy as np
import pandas as pd
import pytest

from lob.stats import adjust_pvalues, bootstrap_ci, paired_vs_reference, sign_test_p


@pytest.mark.parametrize("method,expected", [
    ("bonferroni", [0.04, 0.16, 0.12, 0.008]),
    ("holm", [0.03, 0.06, 0.06, 0.008]),
    ("fdr_bh", [0.02, 0.04, 0.04, 0.008]),
])
def test_corrections_match_hand_computed_ordered_hypotheses(method, expected):
    np.testing.assert_allclose(adjust_pvalues([0.01, 0.04, 0.03, 0.002], method), expected)


def test_bootstrap_rejects_nan_invalid_alpha_and_replicates():
    for values, kwargs in [([1, float("nan")], {}), ([1, 2], {"alpha": 1}), ([1, 2], {"n_boot": 0})]:
        with pytest.raises(ValueError):
            bootstrap_ci(values, **kwargs)
    with pytest.raises(ValueError):
        sign_test_p(-1, 3)
    with pytest.raises(ValueError):
        adjust_pvalues([1.1])


def sample():
    return pd.DataFrame([dict(scenario="calm", agent=agent, label=agent, seed=seed, effective_bps=seed + cost)
                         for agent, cost in [("twap", 0), ("test", -1)] for seed in range(10)])


def test_missing_and_duplicate_pairs_cannot_be_silently_intersected():
    data = sample()
    with pytest.raises(ValueError, match="incomplete"):
        paired_vs_reference(data.iloc[:-1], reference="twap")
    with pytest.raises(ValueError, match="duplicate"):
        paired_vs_reference(pd.concat([data, data.iloc[:1]]), reference="twap")
    data.loc[0, "effective_bps"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        paired_vs_reference(data, reference="twap")


def test_correction_family_includes_every_scenario():
    data = sample()
    data = pd.concat([data, data.assign(scenario="thin")], ignore_index=True)
    result = paired_vs_reference(data, reference="twap")
    assert list(result.family_size) == [2, 2]
    assert (result.p_adjusted >= result.p_sign).all()
    only_reference = data[data.agent == "twap"]
    empty = paired_vs_reference(only_reference, reference="twap")
    assert empty.empty and "scenario" in empty and "p_adjusted" in empty
