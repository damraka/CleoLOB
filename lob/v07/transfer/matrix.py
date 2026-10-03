"""Transfer matrices and hierarchical calibration (workstreams 25-27). Retrospective/development data only.

For every consumed L2 dataset whose role permits fitting (development, validation, retrospective; the selection
day is excluded because its role permits only selection), the generator families G1 (state-conditioned Hawkes)
and G3 (conditional resampling) are refitted, simulated (4 seeds x 3600 s) and scored with the frozen v0.6
objective against every dataset's historical sketch: a source x target matrix. Pairs are classified as
same-instrument temporal (ETH -> ETH), cross-instrument (ETH <-> BTC, same venue). "Related instrument"
(e.g. BTC perpetual vs BTC future) and venue x venue fitting are NOT_AVAILABLE: no such consumed data exist,
and the only second venue (BitMEX) is a fresh holdout that may not be fitted on.

Hierarchical calibration: the G1 coefficient vectors per dataset are decomposed into between-instrument and
within-instrument (between-month) variance; a coefficient is "stable" when its between-dataset spread is
small relative to its fit uncertainty proxy, "local" otherwise. Levels global -> instrument -> month are
supported; the venue level is NOT_AVAILABLE (single venue with fit-eligible L2 data).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import gc
from pathlib import Path

import numpy as np

from ...experiments.registry import PROJECT_ROOT
from ...v06.history import block_raws
from ...v06.identifiability import clean
from ..data import access
from ..evidence.runs import finalize, new_run
from ..generators import counts as cm
from ..generators.engine import EmpiricalTables, GeneratorSpec
from ..generators.events import extract
from ..generators.study import v06_inputs
from ..realism.scoring import simulate_sketches
from ..v06_compat import pool, sketch

SOURCES = {"deribit-eth-perp-2020-04-01": "develop", "deribit-eth-perp-2020-06-01": "validate",
           "deribit-eth-perp-2020-07-01": "retrospective", "deribit-eth-perp-2020-08-01": "retrospective",
           "deribit-eth-perp-2020-09-01": "retrospective", "deribit-eth-perp-2020-10-01": "retrospective",
           "deribit-btc-perp-2020-07-01": "retrospective", "deribit-btc-perp-2020-09-01": "retrospective"}
DESIGN = "m12-transfer-matrix"
SEEDS = (79501, 79502, 79503, 79504)
TABLE_LIMIT = 100_000


def _simulate(task):
    key, spec, design = task
    return key, pool(simulate_sketches(spec, list(SEEDS), 3600.0, design))


def variance_decomposition(vectors: dict[str, np.ndarray], groups: dict[str, str]) -> dict:
    """Per coefficient: between-group and within-group variance (one-way random-effects moments)."""
    names = sorted(vectors)
    x = np.asarray([vectors[n] for n in names])
    g = np.asarray([groups[n] for n in names])
    out = []
    for j in range(x.shape[1]):
        col = x[:, j]
        means = {k: col[g == k].mean() for k in np.unique(g)}
        within = float(np.mean([col[g == k].var(ddof=1) for k in means if (g == k).sum() > 1])) if any(
            (g == k).sum() > 1 for k in means) else 0.0
        between = float(np.var(list(means.values()), ddof=1)) if len(means) > 1 else 0.0
        total = float(col.var(ddof=1)) if len(col) > 1 else 0.0
        out.append({"between_group_var": between, "within_group_var": within, "total_var": total,
                    "between_share": between / (between + within) if between + within > 0 else None})
    return {"coefficients": out, "groups": sorted(set(g))}


def run(out: str | Path, *, root: Path = PROJECT_ROOT, workers: int = 8) -> dict:
    inputs = v06_inputs(root)
    frozen = inputs["frozen"]
    out = new_run(root / out)
    targets, specs, coefs, activity = {}, {}, {}, {}
    rng = np.random.default_rng(SEEDS[0])
    for dataset, use in SOURCES.items():
        tape, meta = access.load_tape(dataset, use=use, design=DESIGN, purpose="transfer matrix and hierarchical fit",
                                      root=root, scale=frozen["scale_native_per_lot"])
        raws, _ = block_raws(tape)
        targets[dataset] = pool([sketch(r, frozen["design"]) for r in raws])
        b = extract(tape)
        del tape, raws
        gc.collect()
        sub = {k: (v if len(v) <= TABLE_LIMIT else v[rng.choice(len(v), TABLE_LIMIT, replace=False)])
               for k, v in b.offsets.items()}
        tables = EmpiricalTables({k: (v if len(v) <= TABLE_LIMIT else v[rng.choice(len(v), TABLE_LIMIT, replace=False)])
                                  for k, v in b.sizes.items()}, sub)
        base = {**inputs["base"], "target_level_vol": 174}
        g1 = cm.HawkesCounts.fit(b.counts, b.state, b.valid)
        g3 = cm.ConditionalCounts.fit(b.counts, b.state, b.valid)
        coefs[dataset] = g1.coef.ravel()
        activity[dataset] = float(b.counts[b.valid].sum(1).mean())
        specs[f"G1|{dataset}"] = GeneratorSpec("G1_state_hawkes", base, g1, tables)
        specs[f"G3|{dataset}"] = GeneratorSpec("G3_conditional_ar", base, g3, tables)
        access.mark_evaluated(dataset, use=use, design=DESIGN, run=out.relative_to(root).as_posix(), root=root)
        del b
        gc.collect()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        simulated = dict(executor.map(_simulate, [(k, s, frozen["design"]) for k, s in specs.items()]))
    from ...v06.realism import compare
    matrix = {}
    for key, sim in simulated.items():
        family, source = key.split("|")
        for target, hist in targets.items():
            matrix.setdefault(family, {}).setdefault(source, {})[target] = compare(
                hist, sim, frozen["design"], frozen["objective_scales"])["objective"]

    def kind(s, t):
        si, ti = s.split("-")[1], t.split("-")[1]
        return "in_sample" if s == t else ("same_instrument_temporal" if si == ti else "cross_instrument")
    summary = {}
    for family, rows in matrix.items():
        by_kind: dict[str, list[float]] = {}
        sim_vs_activity = []
        for s, row in rows.items():
            for t, v in row.items():
                by_kind.setdefault(kind(s, t), []).append(v)
                sim_vs_activity.append((abs(np.log(activity[s] / activity[t])), v))
        a = np.asarray(sim_vs_activity)
        summary[family] = {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in by_kind.items()}
        summary[family]["corr_objective_vs_activity_dissimilarity"] = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    groups = {d: d.split("-")[1] for d in coefs}
    result = {"matrix": matrix, "summary": summary, "activity_events_per_bin": activity,
              "hierarchy": variance_decomposition(coefs, groups),
              "not_available": {"related_instrument": "no consumed related-instrument data (e.g. BTC futures)",
                                "venue_level": "no fit-eligible second-venue L2 data; BitMEX is a fresh holdout",
                                "selection_day": "excluded: the selection role permits only selection"},
              "label": "RETROSPECTIVE (consumed data); EXPLORATORY transfer description"}
    finalize(out, analysis="m12-transfer-matrix", dataset_ids=list(SOURCES), config={"seeds": list(SEEDS)},
             result=clean(result), root=root, seeds={"simulation": list(SEEDS)})
    return result

