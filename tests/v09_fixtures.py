"""Shared v0.9 test helpers: isolated protocol roots (never the repository ledger)."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lob.v09 import protocol as pr
from lob.v09.smoke import isolated_root


@pytest.fixture(scope="module")
def v09_template(tmp_path_factory) -> Path:
    r = isolated_root(tmp_path_factory.mktemp("v09root"))
    assert r is not None, "repository v0.9 configs are required for protocol tests"
    return r


@pytest.fixture
def v09_root(v09_template, tmp_path) -> Path:
    target = tmp_path / "root"
    shutil.copytree(v09_template, target)
    return target


def first(protocol: dict, *, role: str | None = None, pool: str | None = None, venue: str | None = None) -> str:
    for d in pr.datasets_by(protocol, role=role, pool=pool):
        if venue is None or d["venue"] == venue:
            return d["id"]
    raise LookupError("no such dataset")


def promote_dummy(root: Path, promoted: list[str] | None = None) -> dict:
    """A syntactically valid promote event (for firewall tests only; real promotions come from a sealed run)."""
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    return pr.append_event(root / pr.LEDGER_PATH, "promote", reason="test promotion", protocol=protocol, root=root,
                           payload={"confirmatory_sha256": "a" * 64, "candidates_sha256": "b" * 64,
                                    "promotion_run_result_sha256": "c" * 64, "promoted": promoted or ["H1"],
                                    "decisions": {}})


def access(root: Path, dataset: str, *, use: str, design: str, stage: str = "downloaded",
           sources: dict | None = None) -> dict:
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    role = pr.dataset_declaration(protocol, dataset)["role"]
    return pr.append_event(root / pr.LEDGER_PATH, "access", dataset=dataset, role=role, design=design,
                           reason="test access", protocol=protocol, root=root,
                           payload={"use": use, "stage": stage, "source_sha256": sources or {}})
