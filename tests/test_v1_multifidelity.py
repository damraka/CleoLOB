"""v1 multi-fidelity hub helpers (M70-M75 decision quantities)."""
import pytest

from lob.v1.research.studies import multifidelity_hub as MF


def test_multifidelity_regret_is_zero_for_best_and_none_for_missing():
    costs = {"twap": [1.0, 3.0], "sweep": [5.0, None], "pov": [None, None]}
    assert MF.regret(costs, "twap") == pytest.approx(0.0)
    assert MF.regret(costs, "sweep") == pytest.approx(3.0)
    assert MF.regret(costs, "pov") is None


def test_multifidelity_directions_cover_every_registered_pair():
    from lob.v09.studies.bounds import RANKING_POLICIES
    costs = {p: [float(i)] * 8 for i, p in enumerate(RANKING_POLICIES)}
    d = MF._dirs(costs)
    n = len(RANKING_POLICIES)
    assert len(d) == n * (n - 1) // 2 and set(d.values()) <= {-1, 0, 1}
