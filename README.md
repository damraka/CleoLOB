<h1 align="center">CleoLOB</h1>

<p align="center">
  <strong>Reproducible market-microstructure and execution research.</strong>
</p>

<p align="center">
  CleoLOB studies execution strategies under explicit constraints around historical reconstruction,
  calibration, queue observability, incomplete execution, and out-of-sample evaluation.
</p>

<p align="center">
  <a href="https://github.com/damraka/CleoLOB/actions/workflows/ci.yml"><img src="https://github.com/damraka/CleoLOB/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/damraka/CleoLOB/actions/workflows/research.yml"><img src="https://github.com/damraka/CleoLOB/actions/workflows/research.yml/badge.svg" alt="Research"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/github/v/release/damraka/CleoLOB" alt="Release">
  <a href="https://pypi.org/project/cleolob/"><img src="https://img.shields.io/pypi/v/cleolob" alt="PyPI"></a>
  <img src="https://img.shields.io/github/license/damraka/CleoLOB" alt="License">
</p>

<p align="center">
  <img src="docs/assets/cleolob-dashboard.png" alt="CleoLOB market microstructure dashboard" width="100%">
</p>

---

## What it is

CleoLOB is an open-source quantitative research platform for limit order books, historical replay,
execution algorithms, empirical calibration, reinforcement learning, and reproducible experiments.

Its design goal is simple: make invalid assumptions, incomplete fills, data leakage, unsupported
market-data inference, and selective reporting harder to mistake for evidence.

It is a research framework, not a production trading system.

## Current evidence

| Question | Current result |
|---|---|
| Historical L2 reconstruction | Across the preserved Deribit April/May validation samples, 5,972,671 L2 updates and 2,081,479 published top-five snapshots were processed; all compared top-five snapshots matched exactly. |
| Real queue / order-identity research | Identity-preserving MBO replay, lifecycle validation, queue semantics and MBO-to-L2 aggregation are implemented. Genuine historical MBO validation remains unavailable. |
| Cross-regime calibration | Not established. The preserved v0.3 external failure remains, and the fresh June 2020 Deribit ETH-PERPETUAL holdout also failed its frozen acceptance gate. |
| Completion-constrained execution | In the registered v0.4 policy study, 1,061 / 1,080 episodes completed within the decision horizon and 19 completed during post-horizon settlement. |
| PPO / DQN superiority | Not established. Of 48 registered v0.4 economic contrasts, 47 multiplicity-adjusted intervals included zero; one stress DQN-vs-POV contrast excluded zero, but zero joint cost-and-completion success gates passed. |
| Performance | Registered local Python research benchmarks only; no production/HFT or exchange-colocation latency claim. |

Negative, null, and failed results are retained rather than removed from the research record.

The [v0.4 final report](docs/v04-final-report.md) documents the latest external-validity work,
including formal market-data capability contracts, completion-constrained execution, stronger
artifact verification, multi-period diagnostics, scaling measurements, and the failed fresh
historical holdout.

## Core capabilities

- deterministic FIFO synthetic exchange with limit/market orders, partial fills, cancellation and modification
- historical aggregate-L2 reconstruction and snapshot validation
- formal trades/L1/L2/MBO/simulator capability contracts
- identity-preserving MBO replay with lifecycle, queue and priority-reset semantics
- chronological calibration with sealed holdouts and consumed-dataset tracking
- calibration-failure diagnostics across distributions, tails, coverage, persistence and dependence
- TWAP, VWAP, POV, Almgren-Chriss and heuristic execution controls
- Stable-Baselines3 PPO and discrete-action DQN research workflows
- completion-constrained execution with actual-fill and residual-valuation separation
- paired bootstrap inference and multiplicity-aware registered comparisons
- experiment provenance, source/config/data/model hashes and artifact verification
- risk, settlement and residual-inventory accounting
- registered performance and scaling studies
- FastAPI/Three.js and Streamlit exploration interfaces

## Quick start

Requires **Python 3.11+**.

Install the latest release from PyPI:

```bash
python -m pip install cleolob
```

Verify:

```bash
cleo --help
```

For development:

```bash
git clone https://github.com/damraka/CleoLOB.git
cd CleoLOB
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the development environment:

```bash
python -m pip install -e '.[dev]'
```

Optional RL and web dependencies:

```bash
python -m pip install -e '.[dev,rl,web]'
```

Run the test suite:

```bash
python -m pytest -q
```

Example commands:

```bash
cleo config validate configs/research.yaml
cleo validate-data examples/data/canonical-events.jsonl
cleo replay examples/data/canonical-events.jsonl
cleo validate-mbo examples/studies/v04/mbo-input/synthetic-events.jsonl \
  --manifest examples/studies/v04/mbo-input/adapter-manifest.json \
  --references examples/studies/v04/mbo-input/synthetic-aggregate.jsonl \
  --out results/mbo-check
cleo multiperiod-study --config configs/v04-multiperiod-smoke.json --out results/multiperiod
cleo scaling-study --out results/scaling
cleo smoke --out results/smoke
cleo verify-artifact results/smoke
```

See [docs/reproduction.md](docs/reproduction.md) and
[docs/v04-reproduction.md](docs/v04-reproduction.md) for the full reproduction workflows.

## Historical reconstruction

CleoLOB has been evaluated against public Deribit ETH-PERPETUAL Level-2 data.

Across the preserved April/May validation samples:

| Metric | Result |
|---|---:|
| L2 updates processed | 5,972,671 |
| Published top-five snapshots compared | 2,081,479 |
| Exact top-five matches | 2,081,479 |
| Trades checked | 46,195 |

One registered approximately 24-hour April stream contained 2,321,160 incremental rows,
1,531,713 reconstructed states and 988,235 exact top-five snapshot matches.

This demonstrates reconstruction consistency with published source snapshots. It does not establish
exchange truth, hidden liquidity, exact passive-fill counterfactuals, or strategy profitability.

See [historical validation](examples/studies/historical/README.md) and
[public market-data documentation](docs/public-market-data.md).

## Market-data capabilities, MBO and queue research

Aggregate L2 does not reveal individual order identities or exact FIFO queue positions, so CleoLOB
keeps L2 and MBO evidence mechanically separated.

The v0.4 capability layer distinguishes trades, L1, aggregate L2, genuine order-level MBO, and
simulator-internal information. Consumers are expected to fail explicitly when they request evidence
the underlying source cannot establish.

The MBO layer supports source-native identities where available, ADD/MODIFY/CANCEL/EXECUTE events,
deterministic replay, lifecycle validation, queue-ahead tracking, priority resets, observed maker
executions, censoring and MBO-to-L2 aggregation.

The bundled MBO fixtures are synthetic. **Genuine historical MBO validation remains NOT_AVAILABLE.**

See [docs/mbo.md](docs/mbo.md) and
[docs/v04-data-contracts.md](docs/v04-data-contracts.md).

## Calibration and historical generalization

CleoLOB uses chronological rather than random train/test evaluation and supports frozen selection,
sealed holdouts, consumed-dataset tracking and observable-level failure diagnostics.

The preserved v0.3 external calibration failure remains part of the evidence record. v0.4 also
evaluated the immutable historical model against a fresh June 2020 Deribit ETH-PERPETUAL period
under the frozen design. That fresh holdout failed as well.

CleoLOB therefore does not currently claim that its calibrated synthetic market generalizes across
unseen real-market regimes.

The diagnostics can describe associations with distribution shift, tails, coverage, persistence,
dependence and regime behavior, but they do not by themselves establish a unique causal explanation
for calibration failure.

See [docs/calibration-v03.md](docs/calibration-v03.md) and
[docs/v04-calibration.md](docs/v04-calibration.md).

## Execution and reinforcement learning

The registered v0.4 policy study applies the same completion mechanism to PPO, DQN and the classical
control family.

It trained **24 final models**, totaling **49,152 training steps**, and retained **1,080 registered
evaluation episodes** across three regimes, four training seeds and twelve evaluation market seeds.

Observed completion:

| Metric | Result |
|---|---:|
| Completed within decision horizon | 1,061 / 1,080 |
| Completed during post-horizon settlement | 19 / 1,080 |
| Final unresolved residuals | 0 / 1,080 |

No new orders are submitted after the decision horizon. Post-horizon settlement is reported
separately, and hypothetical residual valuation never creates an actual fill.

Of the 48 registered economic contrasts, **47 multiplicity-adjusted intervals included zero**.
The single exception was stress DQN minus POV, but **zero joint cost-and-completion success gates
passed**. Learned-policy superiority, equivalence, adequate power, convergence and live
profitability therefore remain **NOT_ESTABLISHED**.

The v0.3 PPO/DQN study remains preserved separately rather than being overwritten.

See [docs/rl-v03.md](docs/rl-v03.md) and [docs/v04-rl.md](docs/v04-rl.md).

## Performance and scaling

v0.4 includes a registered study covering **45 workload configurations** across replay,
simulation, parsing, calibration, serialization and related research paths.

The results characterize local Python research workloads. They are not production trading,
exchange-colocation or HFT latency measurements.

See [docs/v04-performance.md](docs/v04-performance.md).

## Reproducibility

Registered experiments can retain:

- normalized configuration
- source identity and hashes
- dataset identity and hashes
- model/checkpoint identity
- normalization identity
- random seeds
- study registration
- evaluation locks
- outcomes and validity status
- statistical results
- provenance metadata

Committed evidence can be verified with:

```bash
cleo verify-artifact examples/studies/v03
cleo verify-artifact examples/studies/v04/evidence
```

The v0.4 release passed **817 tests**, cross-platform GitHub CI on Windows/Linux with
Python 3.11-3.14, package builds, clean-wheel installation, CLI smoke tests and compact-evidence
verification.

Byte-integrity verification establishes artifact consistency, not independent scientific replication.

## Limitations

CleoLOB does not currently establish:

- genuine historical MBO queue validity
- hidden liquidity
- exact passive-fill counterfactuals from aggregate L2
- successful broad cross-regime simulator calibration
- PPO/DQN or learned-policy superiority
- adequate power for broad policy-superiority claims
- live alpha or profitability
- production HFT or exchange-colocated performance
- universal cross-instrument or cross-venue transfer

These are explicit research boundaries rather than hidden assumptions.

## Roadmap

v0.5 focuses on **real-market execution validation**.

Planned work includes:

- genuine historical MBO validation where legally and practically available
- bounded historical counterfactual execution rather than invented exact fills
- calibration v2 and explicit model-class comparison
- market-impact and resilience validation
- within-horizon completion as a formal execution endpoint
- multi-regime and cross-instrument external validity
- stronger registered classical/RL execution studies
- historical-vs-synthetic policy-transfer analysis
- paper-style research reporting
- stronger evidence provenance, verification and scaling analysis

See the full [v0.5 roadmap](docs/v05-roadmap.md).

## Documentation

- [Architecture](docs/architecture.md)
- [MBO semantics](docs/mbo.md)
- [Public market data](docs/public-market-data.md)
- [Research methodology](docs/research-methodology.md)
- [Reproduction guide](docs/reproduction.md)
- [v0.3 calibration protocol](docs/calibration-v03.md)
- [v0.3 RL protocol](docs/rl-v03.md)
- [v0.3 final report](docs/v03-final-report.md)
- [v0.4 roadmap](docs/v04-roadmap.md)
- [v0.4 data contracts](docs/v04-data-contracts.md)
- [v0.4 calibration](docs/v04-calibration.md)
- [v0.4 RL study](docs/v04-rl.md)
- [v0.4 performance](docs/v04-performance.md)
- [v0.4 reproduction](docs/v04-reproduction.md)
- [v0.4 final report](docs/v04-final-report.md)
- [v0.5 roadmap](docs/v05-roadmap.md)

## Citation

Academic users can use the metadata in [`CITATION.cff`](CITATION.cff).

## License

CleoLOB is licensed under the **GNU Lesser General Public License v3.0 or later
(LGPL-3.0-or-later)**.

## Disclaimer

CleoLOB is provided for research and educational purposes. Nothing in this repository constitutes
investment advice, a recommendation to trade, or evidence of future trading performance.
