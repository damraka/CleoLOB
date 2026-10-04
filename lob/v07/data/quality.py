"""Data-quality validation for historical tapes (registered thresholds; failures are INVALID, never repaired).

A quality report is computed while the tape is built and stored with every
result that uses the tape. ``status`` is PASS, WARN (usable; the warning is
reported) or FAIL (the dataset's dependent results are INVALID).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class QualityThresholds:
    min_valid_fraction: float = 0.90          # FAIL below
    warn_valid_fraction: float = 0.98         # WARN below
    max_crossed_fraction: float = 0.01        # FAIL above (fraction of 100 ms samples)
    max_local_gap_s: float = 300.0            # WARN above (capture gap)
    max_presnapshot_fraction: float = 0.01    # WARN above
    max_unknown_trade_side_fraction: float = 0.01  # WARN above
    min_duration_s: float = 6 * 3600.0        # FAIL below
    max_duplicate_trade_ids: int = 0          # WARN above


def assess(tape, l2_stats: dict, trade_stats: dict, samples: dict, thresholds: QualityThresholds) -> dict:
    fails, warns = [], []
    valid = float(tape.valid.mean()) if len(tape.t) else 0.0
    n = max(samples["samples"], 1)
    crossed = samples["crossed_samples"] / n
    gap_s = l2_stats.get("max_local_gap_us", 0) / 1e6
    pre = l2_stats.get("presnapshot_rows", 0) / max(l2_stats.get("rows", 1), 1)
    unknown = trade_stats["unknown_side"] / max(trade_stats["rows"], 1)
    if not l2_stats.get("complete"):
        fails.append("L2 replay incomplete")
    if valid < thresholds.min_valid_fraction:
        fails.append(f"valid fraction {valid:.4f} < {thresholds.min_valid_fraction}")
    elif valid < thresholds.warn_valid_fraction:
        warns.append(f"valid fraction {valid:.4f} < {thresholds.warn_valid_fraction}")
    if crossed > thresholds.max_crossed_fraction:
        fails.append(f"crossed-sample fraction {crossed:.4f} > {thresholds.max_crossed_fraction}")
    if tape.duration < thresholds.min_duration_s:
        fails.append(f"duration {tape.duration:.0f} s < {thresholds.min_duration_s:.0f} s")
    if gap_s > thresholds.max_local_gap_s:
        warns.append(f"largest capture gap {gap_s:.1f} s > {thresholds.max_local_gap_s} s")
    if pre > thresholds.max_presnapshot_fraction:
        warns.append(f"pre-snapshot row fraction {pre:.4f}")
    if unknown > thresholds.max_unknown_trade_side_fraction:
        warns.append(f"unknown aggressor-side fraction {unknown:.4f}")
    if trade_stats["duplicate_ids"] > thresholds.max_duplicate_trade_ids:
        warns.append(f"{trade_stats['duplicate_ids']} duplicate trade ids")
    if l2_stats.get("exchange_clock_regressions", 0):
        warns.append(f"{l2_stats['exchange_clock_regressions']} exchange clock regressions")
    mid = tape.mid[tape.valid]
    spread_ticks = ((tape.ap[:, 0] - tape.bp[:, 0]) / tape.tick)[tape.valid]
    off_grid = float(np.mean(np.abs(spread_ticks - np.round(spread_ticks)) > 1e-6)) if len(spread_ticks) else None
    if off_grid is not None and off_grid > 0.001:
        warns.append(f"{off_grid:.4f} of spreads are off the declared tick grid")
    status = "FAIL" if fails else "WARN" if warns else "PASS"
    return {"status": status, "fails": fails, "warnings": warns, "thresholds": asdict(thresholds),
            "valid_fraction": valid, "crossed_sample_fraction": crossed, "stale_sample_fraction":
            samples["stale_samples"] / n, "duration_s": float(tape.duration), "max_local_gap_s": gap_s,
            "trade_events": int(len(tape.trade_t)), "spread_off_grid_fraction": off_grid,
            "median_mid": float(np.median(mid)) if len(mid) else None,
            "l2": {k: l2_stats.get(k) for k in ("rows", "groups", "snapshots", "presnapshot_rows", "crossed_groups",
                                                 "one_sided_groups", "empty_groups", "missing_level_deletes",
                                                 "exchange_clock_regressions", "max_levels_observed", "file_bytes",
                                                 "expanded_bytes", "exchange", "symbol")},
            "trades": trade_stats}
