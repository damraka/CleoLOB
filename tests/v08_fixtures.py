"""Shared v0.8 test helpers: isolated protocol roots (never the repository ledger) and tiny worlds."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lob.v08.generators import PoissonGenerator
from lob.v08.market.sim import BookModel
from lob.v08.smoke import isolated_root
from lob.v08.worlds import World


@pytest.fixture(scope="module")
def v08_template(tmp_path_factory) -> Path:
    r = isolated_root(tmp_path_factory.mktemp("v08root"))
    assert r is not None
    return r


@pytest.fixture
def v08_root(v08_template, tmp_path) -> Path:
    target = tmp_path / "root"
    shutil.copytree(v08_template, target)
    return target


def tiny_world(world_id: str = "w", market: float = 1.0, refill: float = 8.0) -> World:
    book = BookModel(refill_mean=refill)
    g = PoissonGenerator({**PoissonGenerator.default_params(), "market": market}, book)
    return World(world_id, g, book)
