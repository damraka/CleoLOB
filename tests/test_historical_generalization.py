import copy
import json

import pytest

from lob import historical_generalization as h
from lob.calibration import evaluate_calibration, fit_calibration, save_calibration
from lob.generalization import synthetic_fixture
from lob.experiments.registry import sha256_file


def test_historical_runner_preserves_model_failures_and_freezes_before_fresh_access(tmp_path, monkeypatch):
    april = synthetic_fixture({"rows": 1024, "seed": 1, "start_us": 0}, 1_000_000)
    may = synthetic_fixture({"rows": 512, "seed": 2, "start_us": 2_000_000_000, "shift": 2}, 1_000_000)
    june = synthetic_fixture({"rows": 512, "seed": 3, "start_us": 4_000_000_000, "shift": 2}, 1_000_000)
    for name in ("april", "may"):
        (tmp_path / name).write_text(name)
    metadata = {"exchange": "deribit", "symbol": "ETH-PERPETUAL", "sample_interval_us": 1_000_000,
                "source_sha256": sha256_file(tmp_path / "april")}
    may_meta = {**metadata, "source_sha256": sha256_file(tmp_path / "may")}
    train = h._section(april, 0, 511_000_000)
    model = fit_calibration(train, metadata)
    save_calibration(model, tmp_path / "model.json")
    original = {"validation": evaluate_calibration(model, h._section(april, 512_000_000, 767_000_000), metadata),
                "internal_test": evaluate_calibration(model, h._section(april, 768_000_000, 1_023_000_000), metadata),
                "test": evaluate_calibration(model, may, may_meta)}
    (tmp_path / "original.json").write_text(json.dumps(original))
    config = {"schema_version": 1, "kind": "frozen_2020_local_diagnostics", "model_path": "model.json",
              "model_file_sha256": sha256_file(tmp_path / "model.json"), "original_result_path": "original.json",
              "original_result_sha256": sha256_file(tmp_path / "original.json"), "development_path": "april",
              "development_sha256": metadata["source_sha256"], "consumed_external_path": "may",
              "consumed_external_sha256": may_meta["source_sha256"], "generation_seed": 27,
              "diagnostic_seed": 37, "simulation_rows": 128, "block_rows": 32, "bootstrap_repetitions": 99,
              "fresh_source": {"exchange": "deribit", "symbol": "ETH-PERPETUAL", "date": "2020-06-01",
                               "data_type": "incremental_book_L2", "directory": "raw"}}
    before = copy.deepcopy(model.to_dict())
    accessed = []
    def extract(path):
        accessed.append(path.name)
        if path.name == "april":
            return april, metadata
        if path.name == "may":
            return may, may_meta
        assert (tmp_path / "out/fresh-intake.json").is_file()
        assert (tmp_path / "out/plan.json").is_file()
        return june, {**metadata, "source_sha256": sha256_file(path)}
    def download(exchange, symbol, date, kind, directory, **kwargs):
        assert (tmp_path / "out/plan.json").is_file()
        directory.mkdir()
        path = directory / f"{exchange}_{kind}_{date}_{symbol}.csv.gz"
        path.write_text("synthetic test raw source")
        (directory / (path.name + ".provenance.json")).write_text("{}")
        return path
    monkeypatch.setattr(h, "extract_features", extract)
    monkeypatch.setattr(h, "download_tardis_sample", download)
    result = h.run_historical(config, tmp_path / "out", data_root=tmp_path)
    assert result["old_failures_preserved"] and result["model_unchanged"]
    assert result["periods"][2]["status"] == original["test"]["status"] == "FAIL"
    assert result["fresh_external_status"] == "FAIL"
    assert model.to_dict() == before
    assert len(accessed) == 3
    with pytest.raises(ValueError, match="already present or consumed"):
        h.run_historical(config, tmp_path / "repeat", data_root=tmp_path)


def test_historical_runner_rejects_model_tamper_before_any_source_access(tmp_path):
    (tmp_path / "model.json").write_text("{}")
    with pytest.raises(ValueError, match="immutable original"):
        h.run_historical({"schema_version": 1, "kind": "frozen_2020_local_diagnostics",
                          "model_path": "model.json", "original_result_path": "absent.json",
                          "model_file_sha256": "0" * 64}, tmp_path / "out", data_root=tmp_path)
    assert not (tmp_path / "out").exists()
