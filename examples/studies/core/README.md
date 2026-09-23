# CleoLOB Core Execution Study Evidence

This directory contains a compact, reviewable evidence package for the
CleoLOB core execution research study.

The full study produced substantially larger intermediate artifacts, including
model checkpoints, generated HTML reports, episode-level datasets, diagnostic
dumps, and CI logs. Those large generated outputs are intentionally excluded
from this repository snapshot.

The committed evidence focuses on the artifacts required to understand the
study design, reproduce its declared configuration, inspect final results, and
audit the main research conclusions.

## Included Evidence

### Study Design

- `DATA_PLAN.json`
- `PARALLEL_REPLICATION.json`

These files document the intended dataset usage and replication structure.

### Calibration

The calibration evidence includes:

- cohort definitions,
- external-evaluation plans,
- validation and holdout scorecards,
- frozen-model checksums,
- selected finalist configurations,
- calibration protocols,
- and external-result summaries.

The committed artifacts preserve final diagnostic outputs while excluding
large intermediate search dumps.

### Execution Controls

The execution-control evidence includes:

- control-study plans,
- continuation checks,
- path-equivalence checks,
- positive controls,
- execution-amendment results,
- fitted Almgren-Chriss controls,
- and one-second probe diagnostics.

These artifacts are intended to document whether the execution comparisons
behave as expected under controlled interventions.

### PPO Study

The compact PPO evidence includes:

- `ppo-final-20260922/result.json`
- `ppo-final-20260922/training_summary.json`

Model checkpoint ZIP files, full episode dumps, and generated report HTML files
are intentionally excluded from this repository snapshot.

### Stress Sensitivity

The retained stress-sensitivity evidence includes the declared plan and final
result for the registered residual-inventory sensitivity analysis.

## Evidence Policy

The repository retains final plans, declared protocols, checksums, summaries,
and result files that are useful for review and reproduction.

The following generated artifacts are intentionally excluded:

- PPO checkpoint ZIP files,
- multi-megabyte generated HTML reports,
- raw CI job logs,
- large episode-level CSV/JSONL files,
- calibration candidate-search dumps,
- archived source snapshots,
- and repetitive diagnostic-cell outputs.

This keeps the repository reviewable while preserving the main research
evidence and provenance.

## Path Sanitization

Absolute local filesystem paths have been removed from committed artifacts.

Dataset identifiers, relative repository paths, hashes, study parameters, and
diagnostic values are retained where they contribute to reproducibility.

## Interpretation

These artifacts document research procedures and observed experimental
results. They should not be interpreted as evidence of:

- live trading performance,
- historical alpha,
- guaranteed strategy profitability,
- complete market-impact identification,
- counterfactual historical fills,
- FIFO queue reconstruction from aggregate L2 data,
- or universal generalization across market regimes.

Synthetic execution experiments, historical reconstruction, calibration
diagnostics, and learned-policy evaluation are treated as separate forms of
evidence.

## Reproduction

The corresponding implementation, configuration, and research protocol are
available in:

- `configs/core-study.json`
- `configs/core-study-ac-risk.json`
- `docs/core-research-protocol.md`
- `docs/publication-integrity.md`
- `lob/core_study.py`
- `train_rl.py`

Large regenerated artifacts should be produced from the declared study
configuration rather than committed directly to the repository.