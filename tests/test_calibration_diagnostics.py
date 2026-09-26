import json

import numpy as np
import pytest

from lob.calibration_diagnostics import diagnose_period, mean_drift_interval, report_markdown
from lob.generalization import synthetic_fixture


def fixture(seed, start, shift=1):
    return synthetic_fixture({"seed": seed, "rows": 1024, "start_us": start, "shift": shift}, 1_000_000)


def test_shift_diagnostic_retains_failed_assessment_and_unsupported_causes():
    result = diagnose_period(fixture(1, 0), fixture(2, 2_000_000_000, 2),
                             interval_us=1_000_000, original_status="FAIL", block_rows=32, repetitions=199)
    assert result["overall_status"] == "FAIL"
    assert result["status_changed"] is False
    assert result["causal_attribution"] == "NOT_ESTABLISHED"
    assert "queue_position" in result["unsupported_observability"]
    assert result["observables"]["spread_bps"]["normalized_quantile_distance"] > .5
    assert result["observables"]["spread_bps"]["attribution"] == "distribution_shift_association"
    assert "Original assessment: **FAIL**" in report_markdown(result)
    json.dumps(result, allow_nan=False)


def test_block_intervals_are_deterministic_multiplicity_aware_and_directional():
    left = np.sin(np.arange(1024))
    a = mean_drift_interval(left, left + 4, block_rows=32, repetitions=299, seed=7)
    b = mean_drift_interval(left, left + 4, block_rows=32, repetitions=299, seed=7)
    assert a == b
    assert a["estimate"] == pytest.approx(4)
    assert a["interval"][0] > 3.9
    assert a["correction"] == "Bonferroni" and a["family_size"] == 8


def test_insufficient_blocks_do_not_get_spurious_confidence():
    result = mean_drift_interval(np.arange(30.), np.arange(30.) + 100, block_rows=10,
                                 repetitions=99, seed=1)
    assert result["status"] == "INCONCLUSIVE" and result["interval"] is None


def test_constant_marginals_are_finite_and_undefined_correlations_are_null():
    train, target = fixture(1, 0), fixture(2, 2_000_000_000)
    train["spread_bps"] = target["spread_bps"] = 1.
    result = diagnose_period(train, target, interval_us=1_000_000,
                             original_status="FAIL", block_rows=32, repetitions=99)
    assert result["reference_dynamics"]["lag1_autocorrelation"]["spread_bps"] is None
    assert result["observables"]["spread_bps"]["variance_ratio"] is None
    json.dumps(result, allow_nan=False)


def test_transition_counts_exclude_sampling_gaps():
    train, target = fixture(1, 0), fixture(2, 2_000_000_000)
    target = target.drop(index=500).reset_index(drop=True)
    target.loc[500, "log_return"] = np.nan
    result = diagnose_period(train, target, interval_us=1_000_000,
                             original_status="PASS", block_rows=32, repetitions=99)
    assert np.asarray(result["target_dynamics"]["spread_transition_counts"]).sum() == len(target) - 2


@pytest.mark.parametrize("block,repetitions", [(True, 99), (1, 99), (32, 5), (32, True)])
def test_invalid_bootstrap_design_rejected(block, repetitions):
    with pytest.raises(ValueError):
        mean_drift_interval(np.arange(100.), np.arange(100.), block_rows=block,
                             repetitions=repetitions, seed=1)
