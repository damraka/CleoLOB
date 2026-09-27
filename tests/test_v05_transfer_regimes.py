"""M8 regime sensitivity: episode labelling by frozen M6 thresholds (synthetic blocks)."""
from __future__ import annotations

from lob.regimes import DIMENSIONS
from lob.transfer_regimes import episode_labels, episode_starts

THRESHOLDS = {name: {"median": 1.0, "p90": 2.0} for name in DIMENSIONS}


def _block(start, value):
    return {"start_s": start, **{name: value for name in DIMENSIONS}}


def test_episode_starts_follow_the_frozen_schedule():
    assert episode_starts(100.0, 3) == {0: 400.0, 1: 1000.0, 2: 1600.0}


def test_episode_is_labelled_by_the_block_containing_its_window_midpoint():
    blocks = [_block(0.0, 0.5), _block(300.0, 3.0), _block(600.0, 0.5)]
    # start 0.05 s before the block boundary: the midpoint decides, not the start
    labels = episode_labels({0: 299.95, 1: 610.0}, 35.0, blocks, THRESHOLDS)
    assert labels[0]["volatility"] == "high" and labels[0]["stress"] == "stress"
    assert labels[1]["volatility"] == "low" and labels[1]["stress"] == "normal"


def test_episode_without_a_valid_block_is_unlabelled():
    labels = episode_labels({0: 5000.0}, 35.0, [_block(0.0, 0.5)], THRESHOLDS)
    assert labels[0] is None
