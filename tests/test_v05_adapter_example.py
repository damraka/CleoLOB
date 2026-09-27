"""M12: the documented order-level adapter template runs and refuses undeclared input (synthetic)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lob.mbo_sources import validate_source
from lob.order_lifecycle import LifecycleInputError, LifecycleSemantics

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "adapters" / "custom_mbo_adapter.py"


def _module():
    spec = importlib.util.spec_from_file_location("custom_mbo_adapter", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_template_adapter_validates_its_synthetic_fixture(tmp_path):
    module = _module()
    path = tmp_path / "fixture.csv"
    path.write_text(module.SYNTHETIC_FIXTURE, encoding="utf-8")
    result = validate_source(module.ExampleCsvAdapter(path), LifecycleSemantics(census_complete=False))
    assert result["deterministic_replay"] is True
    assert result["reference_comparison"]["exact_fraction"] == 1.0
    assert result["contract"]["fifo_established"] is False


def test_template_adapter_refuses_undeclared_rows_and_inexact_prices(tmp_path):
    module = _module()
    path = tmp_path / "bad.csv"
    path.write_text("1,MODIFY,b1,BUY,99.99,10\n", encoding="utf-8")
    with pytest.raises(LifecycleInputError):
        module.ExampleCsvAdapter(path).build()
    path.write_text("1,ADD,b1,BUY,99.995,10\n", encoding="utf-8")
    with pytest.raises(LifecycleInputError):
        module.ExampleCsvAdapter(path).build()
