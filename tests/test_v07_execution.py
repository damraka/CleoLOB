"""v0.7 execution episodes: every agent on v0.6 and generator worlds, determinism and invariants."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lob.sim_v2 import SimulatorSpec
from lob.v07.execution import episode
from lob.v07.generators import counts as cm
from lob.v07.generators.engine import EmpiricalTables, GeneratorSpec
from lob.v07.generators.events import extract
from lob.v07.data.tape import build_tape
from tests.v07_fixtures import write_tardis

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {"tick_size": 100 / 2640, "initial_mid_ticks": 2640, "target_level_vol": 26, "limit_rate": 2.9,
          "market_rate": 0.14, "cancel_rate": 0.03, "offset_p": 0.05, "limit_qty_mean": 35, "market_qty_mean": 40,
          "resilience": 0.001, "max_events": 3_000_000}
MANDATE = {**episode.MANDATE, "horizon_s": 60.0, "warmup_s": 20.0}
AGENTS = episode.PRIMARY + episode.EXPLORATORY


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    files = write_tardis(tmp_path_factory.mktemp("exec"), seconds=900.0)
    tape, _ = build_tape(files["l2"], files["trades"], tick=0.05)
    b = extract(tape.rescaled(10.0))
    g3 = GeneratorSpec("G3", {"tick_size": 100 / 2640, "initial_mid_ticks": 2640, "target_level_vol": 20,
                              "max_events": 2_000_000}, cm.ConditionalCounts.fit(b.counts, b.state, b.valid),
                       EmpiricalTables(b.sizes, b.offsets))
    return {"v06": SimulatorSpec(CONFIG, {}), "g3": g3}


@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("world", ["v06", "g3"])
def test_every_agent_runs_on_every_world_type(worlds, agent, world) -> None:
    row = episode.run(worlds[world], agent, MANDATE, 11)
    assert row["status"] in {"VALID", "WARNING"}, row
    assert row[episode.COST] is not None and row["fill_frac"] == pytest.approx(1.0)
    assert row == episode.run(worlds[world], agent, MANDATE, 11)


def test_invalid_rows_are_returned_not_raised() -> None:
    starved = SimulatorSpec({**CONFIG, "max_events": 50}, {})
    row = episode.run(starved, "twap", MANDATE, 1)
    assert row["status"] == "INVALID" and row[episode.COST] is None and "max_events" in row["error"]
    with pytest.raises(KeyError):
        episode.run(SimulatorSpec(CONFIG, {}), "unknown-agent", MANDATE, 1)


def test_vwap_profile_is_a_distribution(worlds) -> None:
    profile = episode.volume_profile(worlds["g3"], 3, MANDATE, days=2, warmup=10.0)
    assert len(profile) == 20 and abs(sum(profile) - 1) < 1e-9 and min(profile) >= 0


def test_mandate_matches_v06_freeze() -> None:
    frozen = json.loads((ROOT / "configs/v06/environment-freeze.json").read_text(encoding="utf-8"))
    assert frozen["mandate"] == episode.MANDATE
    assert frozen["ac_parameters"]["selected"]["selected"] == episode.AC_V06
