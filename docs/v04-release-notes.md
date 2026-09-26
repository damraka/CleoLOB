# CleoLOB v0.4.0

## External Validity and Real-Market Evidence

v0.4 strengthens CleoLOB's research-validity boundaries, historical evaluation,
execution studies, dataset contracts and reproducibility infrastructure.

### Main additions

- formal trades/L1/L2/MBO market-data capability contracts
- identity-preserving dataset adapters and MBO validation infrastructure
- chronological multi-period evaluation with consumed-holdout tracking
- calibration failure diagnostics
- completion-constrained PPO/DQN and classical execution studies
- training-only observation normalization and bounded v0.4 observations
- stronger artifact provenance and verification
- scaling and performance studies across 45 workload configurations
- reproducible v0.4 evidence bundles and clean-wheel workflows

## Empirical results

A fresh June 2020 Deribit ETH-PERPETUAL holdout failed the registered
cross-regime calibration gate.

Cross-regime simulator generalization therefore remains NOT_ESTABLISHED.

The registered v0.4 execution study trained 24 final models for 49,152 total
training steps and retained 1,080 evaluation episodes.

1,061 episodes completed during the decision horizon.
19 completed during post-horizon settlement.
All eventual completions were actual simulator fills; hypothetical residual
valuation was never converted into an actual fill.

Of 48 registered economic contrasts, 47 multiplicity-adjusted intervals
included zero.

The single exception was stress DQN minus POV, but no registered joint
cost-and-completion success gate passed.

Learned-policy superiority therefore remains NOT_ESTABLISHED.

## MBO status

CleoLOB now contains stronger identity-preserving MBO adapters, capability
contracts, deterministic replay and validation machinery.

However, no suitable genuine historical MBO dataset was available for this
release.

Real historical MBO validation therefore remains NOT_AVAILABLE.

Synthetic MBO fixtures are explicitly labeled and are not presented as
real-market queue evidence.

## Validation

The v0.4 development branch passed:

- 817 tests
- Ruff
- compileall
- dependency checks
- Windows and Linux CI
- Python 3.11 through 3.14
- container smoke tests
- wheel installation outside the source checkout
- CLI smoke tests
- v0.3 evidence verification
- v0.4 compact evidence verification

## Evidence boundaries

v0.4 does not establish:

- live alpha
- historical profitability
- learned-policy superiority
- broad cross-regime calibration validity
- real historical MBO queue validity
- hidden liquidity reconstruction
- exact passive-fill counterfactuals from aggregate L2
- production HFT or exchange-colocation performance

Negative and null findings remain part of the published research record.

See:

- `docs/v04-final-report.md`
- `docs/v04-reproduction.md`
- `docs/v04-calibration.md`
- `docs/v04-rl.md`
- `docs/v04-performance.md`
