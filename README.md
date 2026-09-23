<div align="center">

# CleoLOB

### A research-grade laboratory for limit order books, market microstructure, execution algorithms, and reinforcement learning.

**CleoLOB provides a reproducible environment for simulating, reconstructing, calibrating, and analyzing limit order book markets — from synthetic experiments to historical L2 market data.**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Research](https://img.shields.io/badge/focus-market%20microstructure-blueviolet)
![LOB](https://img.shields.io/badge/limit%20order%20book-simulation-orange)
![RL](https://img.shields.io/badge/reinforcement%20learning-execution-green)
![License](https://img.shields.io/github/license/damraka/CleoLOB)

</div>

---

## Overview

CleoLOB is an open-source quantitative research platform for studying modern electronic markets.

It combines a configurable limit order book simulator with historical market-data reconstruction, empirical calibration, execution algorithms, reinforcement-learning environments, risk analysis, and interactive visualization.

The project is designed for research into questions such as:

- How do execution strategies behave under different market regimes?
- How accurately can historical L2 order books be reconstructed?
- How does order-book imbalance relate to short-term market dynamics?
- How do classical execution algorithms compare with learned policies?
- How sensitive are strategies to latency, impact, liquidity, fees, and market structure?
- Can simulated market dynamics reproduce empirical characteristics observed in real markets?

> **CleoLOB is a research and experimentation framework, not a production trading system.**

---

## Core Capabilities

| Area | Capabilities |
|---|---|
| Limit Order Book | Event-driven matching, bids/asks, market and limit orders |
| Market Simulation | Configurable synthetic order-flow environments |
| Historical Data | L2 reconstruction and snapshot validation |
| Calibration | Empirical parameter estimation from market data |
| Execution | TWAP, VWAP, POV, Almgren-Chriss and configurable policies |
| Reinforcement Learning | Execution environments and trainable agents |
| Risk | Execution cost, slippage, inventory and impact analysis |
| Research | Reproducible experiments, seeds, configs and reporting |
| Visualization | Interactive market-microstructure dashboards |

---

[![Research foundations](https://github.com/damraka/CleoLOB/actions/workflows/research.yml/badge.svg)](https://github.com/damraka/CleoLOB/actions/workflows/research.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Version](https://img.shields.io/badge/version-0.2.0-blue)
![Research](https://img.shields.io/badge/focus-market%20microstructure-informational)

---

## Highlights

|                               |                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------ |
| **521 tests**                 | Integrated unit, randomized-invariant, integration and regression suite        |
| **5,972,671**                 | Real Deribit L2 updates processed                                              |
| **2,081,479**                 | Top-five order-book snapshots matched exactly                                  |
| **46,195**                    | Public trade records checked                                                   |
| **Deterministic experiments** | Separate random streams, persistent clocks and reproducible run manifests      |
| **Execution research**        | TWAP, VWAP, POV, Almgren–Chriss, heuristics, random policies and PPO interface |
| **Statistical inference**     | Paired bootstrap, exact sign tests and multiple-testing correction             |
| **Historical reconstruction** | Exact event replay without inventing counterfactual fills                      |

CleoLOB combines a deterministic synthetic exchange with real-market reconstruction and research infrastructure designed to make experiments **auditable, reproducible and difficult to accidentally overstate**.

![CLEO liquidity-canyon visualization](docs/3d-graph.png)

---

## What CleoLOB Is

CleoLOB is intended for research into:

* limit-order-book mechanics,
* optimal execution,
* market microstructure,
* execution cost and risk,
* reinforcement-learning execution agents,
* historical L2 reconstruction,
* calibration and walk-forward validation,
* stress testing,
* portfolio exposure,
* and reproducible strategy comparison.

It is **not** a live trading system.

There is no broker integration, real-money order submission, or claim of demonstrated live alpha.

---

## Research Philosophy

Market-microstructure experiments are extremely easy to overstate.

A strategy can appear profitable because of:

* unrealistic fills,
* unpriced residual inventory,
* look-ahead bias,
* inconsistent randomness,
* missing fees,
* survivorship of successful experiments,
* poorly calibrated synthetic order flow,
* or repeated hypothesis testing without correction.

CleoLOB is designed to expose these failure modes instead of hiding them.

The framework therefore treats experiment validity, provenance and failure reporting as first-class parts of the research process.

---

# Core Capabilities

## Deterministic FIFO Exchange

The synthetic exchange supports:

* integer price ticks and order quantities,
* FIFO price-time priority,
* GTC orders,
* IOC orders,
* FOK orders,
* GTD orders,
* post-only orders,
* partial fills,
* cancellation,
* modification,
* conditional cancel/replace,
* explicit order states,
* queue positions,
* and invariant checks.

Persistent event clocks provide deterministic tie-breaking while separate random streams isolate independent sources of stochasticity.

Delayed messages and cancellation races are explicitly modeled.

Changing the observation frequency of a simulation does not change its underlying exogenous event path.

---

## Execution Algorithms

Built-in execution strategies include:

* **TWAP**
* **synthetic-profile VWAP**
* **POV**
* **Almgren–Chriss**
* heuristic policies,
* random policies,
* and a Stable-Baselines3 **PPO** interface.

Registered research commands require an actual PPO checkpoint when PPO is selected, preventing an unavailable learned policy from silently falling back to another strategy.

---

## Accounting and Execution Risk

CleoLOB tracks:

* cash,
* signed inventory,
* average-cost PnL,
* realized execution effects,
* maker/taker fees,
* rebates,
* working orders,
* and in-flight reservations.

Risk controls include:

* child-order limits,
* position limits,
* notional limits,
* loss limits,
* conservative reservations,
* and a latched kill switch.

Outstanding orders are reconciled through a bounded post-horizon settlement phase.

Late fills and fees are included when they actually occur.

Unresolved orders can invalidate final economic metrics rather than being ignored.

---

# Real-Market L2 Reconstruction

CleoLOB includes a separate historical reconstruction pipeline for recorded market data.

Supported functionality includes:

* bounded CSV/JSONL ingestion,
* exact order IDs,
* snapshots,
* incremental events,
* source hashing,
* data-quality reports,
* pause/resume/reset,
* deterministic reconstruction,
* and stepwise replay.

Historical events are reconstructed separately from the synthetic matching engine so that recorded fills are not accidentally modified by synthetic exchange mechanics.

## Public L2 Validation

The public-data pipeline supports bounded Tardis samples with:

* download verification,
* gzip/checksum validation,
* exact decimal price handling,
* price-level reconstruction,
* snapshot comparison,
* trade checks,
* and causal one-second descriptive statistics.

A real-data validation study processed Deribit ETH perpetual samples from:

* **April 1, 2020**
* **May 1, 2020**

with the following results:

| Validation metric               |        Result |
| ------------------------------- | ------------: |
| L2 updates processed            | **5,972,671** |
| Top-five snapshots compared     | **2,081,479** |
| Exact top-five snapshot matches | **2,081,479** |
| Trades checked                  |    **46,195** |

This validates **aggregate historical reconstruction**.

It does **not** establish historical strategy profitability because aggregate L2 data alone does not identify the counterfactual queue position and fills that an unobserved strategy would have received.

See:

* [`examples/studies/historical/README.md`](examples/studies/historical/README.md)
* [`docs/public-market-data.md`](docs/public-market-data.md)

---

# Calibration and Validation

Historical observable models support:

* causal feature construction,
* frozen calibration artifacts,
* chronological train/test separation,
* purged splits,
* expanding walk-forward folds,
* later-date validation,
* drift diagnostics,
* coverage diagnostics,
* and deterministic observable generation.

Calibration artifacts can therefore be separated from later evaluation periods rather than allowing future information to leak backward into a study.

The current aggregate-L2 calibration layer does **not** identify queue-level arrival and cancellation dynamics.

The synthetic FIFO order-flow model should therefore not be interpreted as a fully calibrated representation of a real exchange.

---

# Reproducible Experiments

Every registered experiment can produce an isolated run directory containing evidence such as:

* normalized configuration,
* configuration hash,
* source snapshot,
* source hash,
* checkpoint identity,
* random seed manifest,
* episode outcomes,
* order logs,
* fill logs,
* risk events,
* validity information,
* statistics,
* and offline HTML reports.

Runs can be verified and compared rather than relying solely on final summary tables.

Example:

```bash
cleo reproduce examples/studies/foundation/20260913T191627-0268a8fa8cb3
cleo verify examples/studies/foundation/20260913T191627-0268a8fa8cb3
cleo experiment diff PATH_TO_FIRST_RUN PATH_TO_SECOND_RUN
```

Reproduction creates a new result directory.

It does not execute archived Python code or overwrite the original experiment.

---

# Statistical Research Design

CleoLOB includes tools for paired experimental designs and uncertainty-aware comparison.

Implemented methods include:

* paired bootstrap intervals,
* exact sign tests,
* Holm correction,
* Bonferroni correction,
* Benjamini–Hochberg correction,
* complete-pair validation,
* and bounded sampling of large finite parameter spaces.

Planned experiments that fail or produce economically unpriceable outcomes are retained.

If the required observations for a valid comparison do not exist, comparative inference can be withheld rather than calculated from a selectively surviving subset.

---

# Foundation Study

The included foundation study compares six baseline/heuristic execution policies using ten paired seeds:

**60 total episodes**

with:

* 2,000-share sell orders,
* 10-second horizons,
* 1 bps taker fees,
* full run evidence,
* configuration snapshots,
* source identity,
* and audit logs.

No PPO or SAC comparison was run in this study.

Every candidate-minus-TWAP bootstrap interval contains zero, and all five Holm-adjusted sign-test p-values are `1.0`.

Therefore:

> **The foundation study does not establish that any tested strategy outperforms TWAP.**

The experiment was performed in one uncalibrated synthetic market and should not be interpreted as evidence of live trading performance.

See:

[`examples/studies/foundation/README.md`](examples/studies/foundation/README.md)

Older 100-seed results generated under different market mechanics are retained separately for provenance:

[`results/baselines-100seeds/`](results/baselines-100seeds/)

They are archived results, not validation of the current engine.

---

# Stress Testing

Registered stress families allow strategies to be evaluated across multiple controlled scenarios while preserving:

* source identity,
* model identity,
* configuration identity,
* complete outcome retention,
* and family-wide statistical correction.

This allows sensitivity analysis without treating every parameter variation as an unrelated experiment.

Example:

```bash
cleo stress --config configs/robustness.yaml
```

---

# Portfolio Risk

CleoLOB includes linear multi-asset portfolio accounting with support for:

* multiple currencies,
* joint asset/FX scenarios,
* conservative reservations,
* exposure limits,
* loss limits,
* and portfolio-level stress evaluation.

Example:

```bash
cleo portfolio --config configs/portfolio_example.json
```

The current implementation is a **linear portfolio risk framework**.

It is not a nonlinear derivatives pricing or Greeks engine.

---

# Architecture

```mermaid
flowchart TD

    Config[Validated Immutable Configuration]

    Config --> Clock[Persistent Event Clock]

    Clock --> Exchange[FIFO Exchange]

    Agents[Baseline / RL Policy] --> Risk[Risk + Reservations]

    Risk --> Exchange

    Exchange --> Fills[Recorded Fills]
    Exchange --> State[Observable Book State]

    State --> Agents

    Fills --> Ledger[Cash / Inventory / Fees / PnL]

    Ledger --> Metrics[Economic Outcomes + Validity]

    Metrics --> Registry[Immutable Experiment Record]

    Registry --> Stats[Paired Statistics + Corrections]

    Stats --> Report[Evidence Report]

    Data[Historical Market Events] --> Validate[Data Validation]

    Validate --> Replay[Exact Historical Reconstruction]

    Replay --> Calibration[Calibration + Walk-Forward Validation]

    Calibration --> Config
```

## Main Modules

| Module                 | Responsibility                                    |
| ---------------------- | ------------------------------------------------- |
| `lob/engine.py`        | Order lifecycle, FIFO book and synthetic exchange |
| `lob/accounting.py`    | Cash, inventory, fees and PnL                     |
| `lob/risk.py`          | Execution limits and reservations                 |
| `lob/execution.py`     | Execution strategies                              |
| `lob/rl_env.py`        | Reinforcement-learning environment                |
| `lob/runner.py`        | Simulation and strategy orchestration             |
| `lob/replay/`          | Historical schemas, validation and reconstruction |
| `lob/config.py`        | Strict research configuration                     |
| `configs/`             | Reusable experiment presets                       |
| `lob/experiments/`     | Experiment provenance and registry                |
| `lob/stats.py`         | Statistical comparison                            |
| `lob/cli.py`           | Research command-line interface                   |
| `lob/benchmarks.py`    | Matching microbenchmarks                          |
| `server.py`, `static/` | FastAPI / Three.js exploration interface          |
| `app.py`               | Legacy Streamlit exploration interface            |
| `tests/`               | Unit, invariant, integration and regression tests |

Correctness is currently prioritized over acceleration.

No Rust, C++, GPU or low-latency production performance claim is made.

---

# Installation

CleoLOB requires **Python 3.11+**.

Clone the repository:

```bash
git clone https://github.com/damraka/CleoLOB.git
cd CleoLOB
```

Create a virtual environment:

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the core development environment:

```bash
python -m pip install -e '.[dev]'
```

Install the optional RL and web components:

```bash
python -m pip install -e '.[dev,rl,web]'
```

---

# Quickstart

Run the test suite:

```bash
python -m pytest -q
```

Validate a research configuration:

```bash
cleo config validate configs/research.yaml
```

Run an evaluation:

```bash
cleo evaluate --config configs/research.yaml
```

Validate historical events:

```bash
cleo validate-data examples/data/canonical-events.jsonl
```

Replay them:

```bash
cleo replay examples/data/canonical-events.jsonl
```

Run the matching benchmark:

```bash
cleo benchmark --pairs 2000 --repeats 3
```

Run registered stress scenarios:

```bash
cleo stress --config configs/robustness.yaml
```

Run a portfolio-risk example:

```bash
cleo portfolio --config configs/portfolio_example.json
```

`python -m lob.cli` is equivalent to the `cleo` command.

The original standalone demonstration remains available through:

```bash
python -m lob
```

---

# Web Visualization

CleoLOB retains both its FastAPI/Three.js interface and its original Streamlit dashboard.

Start the FastAPI application with:

```bash
python server.py --no-browser
```

Then open:

```text
http://127.0.0.1:8000
```

The web application is intended primarily for **exploration and visualization**.

Registered research experiments use the stricter CLI workflow and immutable experiment records.

---

# Configuration

Research configuration uses typed YAML/JSON with support for:

* strict validation,
* inheritance,
* environment overrides,
* CLI overrides,
* schema export,
* normalized hashing,
* and configuration diffs.

The experiment configuration is part of the evidence record rather than an implicit collection of runtime parameters.

Example:

```bash
cleo config validate configs/research.yaml
```

---

# Economic Validity

CleoLOB distinguishes actual execution from hypothetical residual valuation.

`effective_bps` includes:

* actual fill costs,
* and hypothetical terminal liquidation costs/fees,

only when sufficient depth exists to price the remaining quantity.

It does **not** include the separate RL completion penalty.

If residual inventory cannot be economically priced, the metric becomes null and the episode is marked **INVALID**.

A hypothetical mark or terminal liquidation is never recorded as an actual fill.

This prevents unfinished execution from appearing artificially profitable simply because remaining inventory disappeared at the end of an episode.

---

# Current Limitations

CleoLOB deliberately documents what it does **not** currently establish.

### Historical strategy performance

Historical L2 reconstruction is implemented and validated, but counterfactual historical execution is not yet established.

Aggregate L2 data does not uniquely determine:

* queue position,
* hidden liquidity,
* strategy-induced market response,
* or the fills an unobserved agent would have received.

Therefore the repository currently makes **no historical alpha claim**.

### Synthetic order flow

The FIFO exchange mechanics are explicit and tested, but the queue-level synthetic order-flow model remains uncalibrated.

Historical aggregate L2 calibration does not by itself identify queue-level arrival/cancellation processes.

### Latency

Message latency is modeled.

Full market-data dissemination latency is not.

### Reinforcement learning

A Stable-Baselines3 PPO interface is available.

The project currently does not claim a completed PPO-vs-baseline result establishing superiority.

SAC is not implemented.

### Market coverage

The project does not currently provide:

* an exchange-native production feed adapter,
* a complete outage engine,
* a complete flash-crash model,
* nonlinear derivatives portfolio risk,
* or production live-trading connectivity.

---

# Verification

The integrated project suite currently passes:

* **521 automated tests**
* Ruff checks
* Python compilation

CI is configured on Ubuntu for:

* Python **3.11**
* Python **3.13**

The current matching benchmark is a local **matching-engine microbenchmark**.

It should not be interpreted as full simulation throughput or production exchange performance.

---

# Documentation

Detailed documentation is available in:

* [Architecture audit](docs/architecture.md)
* [Configuration, CLI and provenance](docs/configuration.md)
* [Historical replay](docs/historical-replay.md)
* [Public L2 data and validation](docs/public-market-data.md)
* [Calibration, stress, settlement and portfolio workflows](docs/validation-and-risk.md)
* [Research methodology and limitations](docs/research-methodology.md)
* [Implementation status](docs/implementation-status.md)

Executed validation evidence:

* [`examples/studies/validation/README.md`](examples/studies/validation/README.md)

Foundation study:

* [`examples/studies/foundation/README.md`](examples/studies/foundation/README.md)

Historical validation:

* [`examples/studies/historical/README.md`](examples/studies/historical/README.md)

---

# Roadmap

The highest-value remaining research work is centered on connecting historical market observations to increasingly realistic strategy evaluation.

Priority areas include:

* calibrated queue-level order-flow models,
* counterfactual historical execution methodology,
* transaction-cost attribution,
* empirical latency modeling,
* richer regime and stress models,
* PPO evaluation against strong execution baselines,
* additional RL policies,
* market-making research,
* multi-instrument microstructure,
* and larger out-of-sample studies.

The goal is not to maximize the number of implemented algorithms.

The goal is to make increasingly strong research claims while preserving explicit assumptions, reproducibility and falsifiability.

---

# Project Status

CleoLOB is an active research project.

The framework currently provides substantial infrastructure for:

**simulation → execution → accounting → risk → replay → calibration → experiments → statistics → verification**

but it should still be treated as a research environment rather than a production trading system.

---

## Disclaimer

CleoLOB is provided for research and educational purposes.

Nothing in this repository constitutes investment advice, a recommendation to trade, or evidence of future trading performance.
