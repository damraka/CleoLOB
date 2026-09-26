import copy
import json
from pathlib import Path

import pytest

from lob import multiperiod as m
from lob.generalization import synthetic_fixture
from lob.experiments.registry import sha256_file


@pytest.fixture
def config():
    value = json.loads((Path(__file__).parents[1] / "configs/v04-multiperiod-smoke.json").read_text())
    value["bootstrap_repetitions"] = 99
    value["simulation_rows"] = 128
    return value


def csv_config(config, root):
    result = copy.deepcopy(config)
    result["kind"] = "feature_csv"
    for index, period in enumerate(result["periods"]):
        path = root / (period["id"] + ".csv")
        synthetic_fixture(period, config["interval_us"]).to_csv(path, index=False)
        period.update(path=path.name, sha256=sha256_file(path), source_sha256=f"{index + 1:064x}",
                      independent_period_id=f"fixture-day-{index}", inspection="consumed",
                      provenance={"scope": "synthetic CSV ingestion test, not historical evidence"})
    return result


def test_internal_and_external_are_never_read_before_seal(config, tmp_path, monkeypatch):
    original = m._load_period
    visited = []
    def checked(config, period, data_root):
        visited.append(period["id"])
        if period["role"] in {"internal", "external"}:
            assert (tmp_path / "study/selection-seal.json").is_file()
        return original(config, period, data_root)
    monkeypatch.setattr(m, "_load_period", checked)
    result = m.run_study(config, tmp_path / "study")
    assert visited == [p["id"] for p in config["periods"]]
    assert result["fresh_independent_external_periods"] == 0
    assert result["real_market_generalization"] == "NOT_ESTABLISHED"
    assert m.verify_study(tmp_path / "study")["valid"]


def test_changed_or_missing_holdout_bytes_do_not_change_frozen_selection(config, tmp_path):
    config = csv_config(config, tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    m.run_study(config, first, data_root=tmp_path)
    # The registered descriptors are unchanged; unreadable/mutated final files
    # must produce INVALID results only after identical selection has been sealed.
    (tmp_path / "internal.csv").write_text("invalid")
    (tmp_path / "external-1.csv").unlink()
    result = m.run_study(config, second, data_root=tmp_path)
    for name in ("model.json", "selection.json", "validation.json", "selection-seal.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert [p["status"] for p in result["periods"]][:2] == ["INVALID", "INVALID"]
    assert m.verify_study(second)["valid"]


def test_consumed_registry_blocks_relabeling_and_source_aliases(config, tmp_path):
    config = csv_config(config, tmp_path)
    target = config["periods"][-1]
    target["inspection"] = "fresh"
    config["consumed_periods"] = [{"source_sha256": target["source_sha256"]}]
    with pytest.raises(ValueError, match="consumed"):
        m.validate_plan(config)
    config["consumed_periods"] = [{"independent_period_id": target["independent_period_id"]}]
    with pytest.raises(ValueError, match="consumed"):
        m.validate_plan(config)


@pytest.mark.parametrize("alias", ["source_sha256", "independent_period_id"])
def test_fresh_alias_cannot_inflate_independent_period_count(config, tmp_path, alias):
    config = csv_config(config, tmp_path)
    first, second = config["periods"][-2:]
    first["inspection"] = second["inspection"] = "fresh"
    # Canonical bytes/timestamps differ, but the raw source or period identity is shared.
    second[alias] = first[alias]
    with pytest.raises(ValueError, match="distinct source identities"):
        m.validate_plan(config)


@pytest.mark.parametrize("mutation", ["chronology", "duplicate", "seed", "synthetic_fresh"])
def test_invalid_design_rejected_without_data_access(config, mutation):
    if mutation == "chronology":
        config["periods"][-1]["start_us"] = 1
    elif mutation == "duplicate":
        config["periods"][-1]["id"] = config["periods"][0]["id"]
    elif mutation == "seed":
        config["external_seeds"] = config["selection_seeds"]
    else:
        config["periods"][-1]["inspection"] = "fresh"
    with pytest.raises(ValueError):
        m.validate_plan(config)


def test_distinct_files_and_hashes_required(config, tmp_path):
    config = csv_config(config, tmp_path)
    config["periods"][-1]["path"] = config["periods"][0]["path"]
    with pytest.raises(ValueError, match="distinct portable"):
        m.validate_plan(config)
    config["periods"][-1]["sha256"] = config["periods"][0]["sha256"]
    with pytest.raises(ValueError, match="duplicate canonical"):
        m.validate_plan(config)


def test_write_once_and_hash_tamper_detection(config, tmp_path):
    m.run_study(config, tmp_path / "study")
    with pytest.raises(FileExistsError):
        m.run_study(config, tmp_path / "study")
    (tmp_path / "study/result.json").write_text("{}")
    assert not m.verify_study(tmp_path / "study")["valid"]
