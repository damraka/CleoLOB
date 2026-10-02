"""Read-only re-exports of the frozen v0.6 measurement layer used by v0.7.

v0.7 never edits these modules (they are hash-bound by the v0.6 environment
freeze and observable-design seal); importing them through one place makes the
dependency explicit and auditable.
"""
from __future__ import annotations

from ..v06.observables import BINS, COMPONENTS, FAMILIES, measure, pool, sketch  # noqa: F401
from ..v06.realism import bootstrap_realism, improvement_status  # noqa: F401
from ..v06.tape import (LEVELS, MAX_STALENESS_S, SAMPLE_DT, TAPE_FIELDS, WARMUP_S, BookTape,  # noqa: F401
                        depth_scale, tape_from_simulator, tick_bps)
