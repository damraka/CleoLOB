"""Chronology, causal observables, frozen selection and honest failed holdouts."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from lob import generalization as g


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / "configs/v03-calibration.json").read_text())


@pytest.fixture
def frame(config):
    return g.synthetic_fixture(config["development"], config["interval_us"])


def model(frame, family="spread_markov"):
    return g.fit_observable_model(g._section(frame, 0, 500), family=family,
                                  interval_us=1_000_000, seed=7, max_pool_rows=64)


def test_rolling_and_expanding_folds_have_true_elapsed_time_embargo(frame):
    selection = g._section(frame, 0, 840)
    for window in g.WINDOWS:
        folds = g.chronological_folds(selection, initial_train_rows=360, validation_rows=120,
                                     folds=4, window=window, embargo_us=2_000_000)
        assert len(folds) == 4
        for split in folds:
            assert split["train"].timestamp_us.max() < split["validation"].timestamp_us.min() - 2_000_000
            assert np.isnan(split["validation"].log_return.iloc[0])
        counts = [len(split["train"]) for split in folds]
        assert counts == ([358, 478, 598, 718] if window == "expanding" else [358, 360, 360, 360])
        assert len(set(np.concatenate([split["validation"].timestamp_us for split in folds]))) == 480


def test_fold_design_cannot_silently_drop_tail(frame):
    with pytest.raises(ValueError, match="entire selection"):
        g.chronological_folds(frame, initial_train_rows=360, validation_rows=120,
                              folds=4, window="rolling", embargo_us=0)


def test_causal_features_cannot_see_modified_future(frame):
    before = g.observe(frame, 1_000_000)
    changed = frame.copy()
    changed.loc[500:, "spread_bps"] *= 3
    after = g.observe(changed, 1_000_000)
    np.testing.assert_allclose(before.iloc[:500], after.iloc[:500], equal_nan=True)
    assert before.rms_return_10.iloc[:10].isna().all()
    assert np.isfinite(before.rms_return_10.iloc[10])


def test_invalid_rows_and_gaps_reset_all_temporal_features(frame):
    frame.loc[40, "valid"] = False
    frame.loc[40, ["mid_price", *g.FEATURES]] = np.nan
    frame.loc[41, "log_return"] = np.nan
    observations = g.observe(frame, 1_000_000)
    assert observations.loc[40:41, "spread_change_bps"].isna().all()
    assert observations.loc[40:50, "rms_return_10"].isna().all()
    frame = frame.drop(index=70)
    frame.loc[71, "log_return"] = np.nan
    assert np.isnan(g.observe(frame, 1_000_000).loc[71, "relative_depth_change"])


def test_forged_noncausal_return_is_rejected(frame):
    frame.loc[10, "log_return"] = .8
    with pytest.raises(ValueError, match="match consecutive"):
        g.observe(frame, 1_000_000)


@pytest.mark.parametrize("missing", ["invalid", "gap"])
def test_supplied_return_cannot_bridge_missing_history(frame, missing):
    if missing == "invalid":
        frame.loc[40, "valid"] = False
        frame.loc[40, ["mid_price", *g.FEATURES]] = np.nan
    else:
        frame = frame.drop(index=40)
    with pytest.raises(ValueError, match="must not bridge"):
        g.observe(frame, 1_000_000)


def test_markov_model_learns_train_spread_persistence_and_is_reproducible(frame):
    fitted = model(frame)
    probabilities = np.asarray(fitted["transition_probabilities"])
    assert probabilities[0, 0] > .6 and probabilities[1, 1] > .6
    assert fitted == model(frame)
    before = copy.deepcopy(fitted)
    a = g.generate_observables(fitted, 1000, 101)
    b = g.generate_observables(fitted, 1000, 101)
    np.testing.assert_allclose(a, b, equal_nan=True)
    assert fitted == before


def test_baseline_is_distinct_autonomous_generator(frame):
    baseline, markov = model(frame, "iid_joint"), model(frame)
    assert len(baseline["pools"]) == 1 and len(markov["pools"]) == 2
    iid = g.generate_observables(baseline, 4000, 101)
    dependent = g.generate_observables(markov, 4000, 101)
    assert dependent.spread_bps.autocorr() > iid.spread_bps.autocorr() + .3


def test_model_hash_rejects_mutation(frame):
    fitted = model(frame)
    fitted["transition_probabilities"][0][0] = .2
    with pytest.raises(ValueError, match="model hash"):
        g.generate_observables(fitted, 100, 88)


def test_evaluation_preserves_model_and_reports_shift_failure(frame, config):
    fitted = model(frame)
    before = copy.deepcopy(fitted)
    external = g.synthetic_fixture(config["external"], config["interval_us"])
    score = g.evaluate_observable_model(fitted, external, seeds=[100], simulation_rows=512)
    assert score["status"] == "FAIL"
    assert score["metrics"]["spread_bps"]["train_relative_quantile_drift"] > 0
    assert score["failure_attribution"]
    assert fitted == before


def test_evaluation_rejects_training_overlap_and_reused_seeds(frame):
    fitted = model(frame)
    with pytest.raises(ValueError, match="strictly later"):
        g.evaluate_observable_model(fitted, frame, seeds=[100], simulation_rows=128)
    with pytest.raises(ValueError, match="disjoint"):
        g.evaluate_observable_model(fitted, g._section(frame, 600, 800), seeds=[7], simulation_rows=128)


def test_missing_observations_are_failures_not_removed_from_denominator(frame):
    fitted = model(frame)
    target = g._section(frame, 600, 800)
    target.loc[:, "valid"] = False
    target.loc[:, ["mid_price", *g.FEATURES]] = np.nan
    score = g.evaluate_observable_model(fitted, target, seeds=[100], simulation_rows=128)
    assert score["status"] == "FAIL" and score["valid_fraction"] == 0
    assert all(metric["reason"] == "insufficient_observable_coverage" for metric in score["metrics"].values())


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c.update(fit_seed=c["external_seeds"][0]), "disjoint"),
    (lambda c: c["external"].update(seed=c["development"]["seed"]), "disjoint"),
    (lambda c: c["external"].update(start_us=0), "later than development"),
    (lambda c: c.update(embargo_us=-1), "embargo_us"),
    (lambda c: c.update(folds=True), "folds"),
    (lambda c: c["development"].update(rows=100), "development size"),
])
def test_invalid_registration_is_rejected_before_creating_output(config, tmp_path, mutation, match):
    mutation(config)
    with pytest.raises(ValueError, match=match):
        g.run_study(config, tmp_path / "study")
    assert not (tmp_path / "study").exists()


def test_study_registers_and_seals_before_external_loader_and_preserves_fail(config, tmp_path, monkeypatch):
    out = tmp_path / "study"
    original = g._load
    calls = []

    def checked_load(cfg, phase):
        assert (out / "plan.json").is_file()
        calls.append(phase)
        if phase == "external":
            assert (out / "selection-seal.json").is_file()
            assert (out / "internal.json").is_file()
            g._check_seal(out)
        return original(cfg, phase)

    monkeypatch.setattr(g, "_load", checked_load)
    result = g.run_study(config, out)
    assert calls == ["development", "external"]
    assert result["external_status"] == "FAIL"
    assert result["external_used_for_selection"] is False
    assert result["real_market_generalization"] == "NOT_ESTABLISHED"
    assert g.verify_study(out) == {"valid": True, "issues": []}
    validation = json.loads((out / "validation.json").read_text())
    assert len(validation["candidates"]) == 4
    assert all(len(candidate["folds"]) == 4 for candidate in validation["candidates"])
    for candidate in validation["candidates"]:
        for fold in candidate["folds"]:
            assert fold["valid_fraction"] == fold["train_valid_fraction"] == 1
            assert fold["train_complete_rows"] > 64
            assert set(fold["metrics"]) == set(g.OBSERVABLES)
            assert all("normalized_quantile_distance" in metric and "coverage_error" in metric
                       for metric in fold["metrics"].values())
    with pytest.raises(FileExistsError):
        g.run_study(config, out)
    (out / "external.json").write_text("{}")
    assert not g.verify_study(out)["valid"]


def test_external_changes_cannot_change_selection(config, tmp_path):
    config["simulation_rows"] = 128
    first = g.run_study(config, tmp_path / "first")
    config["external"]["shift"] = 3.0
    second = g.run_study(config, tmp_path / "second")
    assert first["model_sha256"] == second["model_sha256"]
    assert (tmp_path / "first" / "validation.json").read_bytes() == (tmp_path / "second" / "validation.json").read_bytes()


def test_internal_test_changes_cannot_change_selection_or_refit(config, tmp_path, monkeypatch):
    config["simulation_rows"] = 128
    first = g.run_study(config, tmp_path / "first")
    original = g._load

    def shifted_internal(cfg, phase):
        observations = original(cfg, phase)
        if phase == "development":
            observations.loc[len(observations) - cfg["internal_rows"]:, "spread_bps"] *= 3
        return observations

    monkeypatch.setattr(g, "_load", shifted_internal)
    second = g.run_study(config, tmp_path / "second")
    assert first["model_sha256"] == second["model_sha256"]
    assert (tmp_path / "first" / "validation.json").read_bytes() == (tmp_path / "second" / "validation.json").read_bytes()
    assert (tmp_path / "first" / "internal.json").read_bytes() != (tmp_path / "second" / "internal.json").read_bytes()


def test_seal_tampering_stops_external_evaluation(config, tmp_path, monkeypatch):
    out = tmp_path / "study"
    original = g._load

    def corrupt(cfg, phase):
        if phase == "external":
            (out / "selection.json").write_text("{}")
        return original(cfg, phase)

    monkeypatch.setattr(g, "_load", corrupt)
    with pytest.raises(ValueError, match="selection seal"):
        g.run_study(config, out)
    assert not (out / "external.json").exists()
    assert (out / "internal.json").is_file()


def test_failed_candidates_retained_and_disqualified(config, tmp_path, monkeypatch):
    original = g.fit_observable_model

    def fail_markov(*args, **kwargs):
        if kwargs["family"] == "spread_markov":
            raise ValueError("declared model failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(g, "fit_observable_model", fail_markov)
    g.run_study(config, tmp_path / "study")
    candidates = json.loads((tmp_path / "study" / "validation.json").read_text())["candidates"]
    failed = [c for c in candidates if c["family"] == "spread_markov"]
    assert all(not c["eligible"] and c["mean_validation_loss"] is None for c in failed)
    assert all("declared model failure" in fold["error"] for c in failed for fold in c["folds"])


def test_csv_input_requires_registered_hash_and_portable_path(config, tmp_path, monkeypatch):
    frame = g.synthetic_fixture(config["development"], config["interval_us"])
    monkeypatch.setattr(g, "PROJECT_ROOT", tmp_path)
    frame.to_csv(tmp_path / "features.csv", index=False)
    config["kind"] = "feature_csv"
    for phase in ("development", "external"):
        config[phase].update(path="features.csv", sha256=g.sha256_file(tmp_path / "features.csv"))
    g._validate_config(config)
    loaded = g._load(config, "development")
    assert len(loaded) == len(frame)
    (tmp_path / "features.csv").write_text("changed")
    with pytest.raises(ValueError, match="source hash"):
        g._load(config, "development")
    config["external"]["path"] = "../private.csv"
    with pytest.raises(ValueError, match="portable"):
        g._validate_config(config)


# --- v0.3 calibration boundary regression tests ---


def test_v03_invalid_finite_measurements_are_rejected_before_fit():
    import pytest

    from lob.generalization import fit_observable_model, synthetic_fixture

    interval_us = 1_000_000
    frame = synthetic_fixture(
        {
            "rows": 128,
            "seed": 913,
            "start_us": 1_700_000_000_000_000,
            "shift": 1.0,
        },
        interval_us,
    )

    # Invalid rows are required to carry NaN measurements. A caller cannot
    # smuggle a finite numeric payload into the fitted distribution merely by
    # toggling valid=False.
    frame.loc[64, "valid"] = False

    with pytest.raises(
        ValueError,
        match="invalid samples must have NaN measurements",
    ):
        fit_observable_model(
            frame,
            family="iid_joint",
            interval_us=interval_us,
            seed=17,
            max_pool_rows=128,
        )


def test_v03_finite_return_across_timestamp_gap_is_rejected():
    import pytest

    from lob.generalization import observe, synthetic_fixture

    interval_us = 1_000_000
    frame = synthetic_fixture(
        {
            "rows": 128,
            "seed": 914,
            "start_us": 1_700_100_000_000_000,
            "shift": 1.0,
        },
        interval_us,
    )

    gap = 64

    # Introduce one missing grid interval while deliberately leaving the
    # supplied return finite. The registered feature contract must reject it
    # rather than silently accepting or repairing it.
    frame.loc[gap:, "timestamp_us"] += interval_us

    with pytest.raises(
        ValueError,
        match="returns must not bridge invalid samples or sampling gaps",
    ):
        observe(frame, interval_us)


def test_v03_invalid_sample_resets_return_and_rms_chain():
    import numpy as np

    from lob.calibration import FEATURES
    from lob.generalization import observe, synthetic_fixture

    interval_us = 1_000_000
    frame = synthetic_fixture(
        {
            "rows": 128,
            "seed": 915,
            "start_us": 1_700_200_000_000_000,
            "shift": 1.0,
        },
        interval_us,
    )

    invalid = 64
    frame.loc[invalid, "valid"] = False

    # The calibration schema requires every measurement on an invalid sample
    # to be NaN.
    frame.loc[invalid, ["mid_price", *FEATURES]] = np.nan

    # The first row after an invalid sample cannot claim a causal one-step
    # return because its predecessor is unavailable.
    frame.loc[invalid + 1, "log_return"] = np.nan

    observed = observe(frame, interval_us)

    assert observed.loc[invalid, list(FEATURES)].isna().all()
    assert np.isnan(observed.loc[invalid + 1, "log_return"])

    # The trailing ten-return statistic must rebuild from fresh consecutive
    # observations rather than bridge the invalid boundary.
    assert observed.loc[invalid:invalid + 10, "rms_return_10"].isna().all()
    assert np.isfinite(observed.loc[invalid + 11, "rms_return_10"])


def test_v03_partition_start_resets_causal_return():
    import numpy as np

    from lob.generalization import _section, observe, synthetic_fixture

    interval_us = 1_000_000
    frame = synthetic_fixture(
        {
            "rows": 160,
            "seed": 916,
            "start_us": 1_700_300_000_000_000,
            "shift": 1.0,
        },
        interval_us,
    )

    partition = _section(frame, 40, 140)
    observed = observe(partition, interval_us)

    assert np.isnan(partition.loc[0, "log_return"])
    assert np.isnan(observed.loc[0, "log_return"])

    # Ten fresh returns are required after the partition boundary.
    assert observed.loc[0:9, "rms_return_10"].isna().all()
    assert np.isfinite(observed.loc[10, "rms_return_10"])


# --- v0.3 strict holdout-isolation regression tests ---


def test_internal_payload_cannot_change_sealed_selection_artifacts(
    config,
    tmp_path,
    monkeypatch,
):
    config["simulation_rows"] = 128

    first_out = tmp_path / "first"
    second_out = tmp_path / "second"

    g.run_study(config, first_out)

    original_load = g._load

    def mutate_only_internal(cfg, phase):
        observations = original_load(cfg, phase)

        if phase == "development":
            start = len(observations) - cfg["internal_rows"]

            # Only the untouched internal holdout is changed.
            observations.loc[start:, "spread_bps"] *= 2.5

        return observations

    monkeypatch.setattr(g, "_load", mutate_only_internal)

    g.run_study(config, second_out)

    # Selection and frozen-refit artifacts must be byte-identical.
    for name in (
        "validation.json",
        "model.json",
        "selection.json",
        "selection-seal.json",
    ):
        assert (
            (first_out / name).read_bytes()
            == (second_out / name).read_bytes()
        ), name

    # Internal scoring itself must observe the changed holdout.
    assert (
        (first_out / "internal.json").read_bytes()
        != (second_out / "internal.json").read_bytes()
    )


def test_external_payload_cannot_change_selection_artifacts(
    config,
    tmp_path,
):
    config["simulation_rows"] = 128

    first_out = tmp_path / "first"
    second_out = tmp_path / "second"

    g.run_study(config, first_out)

    config["external"]["shift"] = 3.0
    g.run_study(config, second_out)

    # External registration changes plan.json, so the seal itself may
    # legitimately differ. The actual selection outputs must not.
    for name in (
        "validation.json",
        "model.json",
        "selection.json",
    ):
        assert (
            (first_out / name).read_bytes()
            == (second_out / name).read_bytes()
        ), name

    assert (
        (first_out / "external.json").read_bytes()
        != (second_out / "external.json").read_bytes()
    )


def test_selection_json_hashes_only_selection_period(
    config,
    tmp_path,
):
    config["simulation_rows"] = 128

    out = tmp_path / "study"
    g.run_study(config, out)

    artifact = json.loads(
        (out / "selection.json").read_text()
    )

    assert "selection_frame_sha256" in artifact
    assert "development_frame_sha256" not in artifact

    development = g.synthetic_fixture(
        config["development"],
        config["interval_us"],
    )

    selection_stop = (
        len(development) - config["internal_rows"]
    )

    expected = g._section(
        development,
        0,
        selection_stop,
    )

    assert (
        artifact["selection_frame_sha256"]
        == g._frame_hash(expected)
    )


def test_feature_csv_external_first_return_cannot_cross_partition(
    config,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(g, "PROJECT_ROOT", tmp_path)

    interval_us = config["interval_us"]

    development = g.synthetic_fixture(
        config["development"],
        interval_us,
    )
    external = g.synthetic_fixture(
        config["external"],
        interval_us,
    )

    # Finite first return would necessarily rely on an observation outside
    # the registered external partition.
    external.loc[0, "log_return"] = 0.001

    development_path = tmp_path / "development.csv"
    external_path = tmp_path / "external.csv"

    development.to_csv(development_path, index=False)
    external.to_csv(external_path, index=False)

    config["kind"] = "feature_csv"

    config["development"].update(
        path="development.csv",
        sha256=g.sha256_file(development_path),
    )

    config["external"].update(
        path="external.csv",
        sha256=g.sha256_file(external_path),
    )

    g._validate_config(config)

    with pytest.raises(
        ValueError,
        match="first feature row must have NaN log_return",
    ):
        g._load(config, "external")
