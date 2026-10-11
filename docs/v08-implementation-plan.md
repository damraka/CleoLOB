# v0.8 implementation plan (M0 audit)

Branch `research/v0.8-decision-uncertainty`, development version `0.8.0.dev0`. This document
records the repository audit done before any v0.8 code was written, and the plan that
follows from it. It is a plan. It is not evidence, and it claims no results.

## 1. Audit of the existing repository

**Layout.**
- `lob/` holds the v0.1–v0.5 modules at the top level, plus `lob/v06/` and `lob/v07/`.
- Each version has its own protocol, ledger, evidence namespace (`results/vNN`, git-ignored),
  public bundle (`examples/studies/vNN`) and documents (`docs/vNN-*`).
- Earlier implementation files are hash-bound by environment freezes and seals. They are
  imported, never edited.

**Version and CLI.**
- `pyproject.toml` is at `0.7.0`, with `requires-python >= 3.11`. CI runs 3.11–3.14 on Ubuntu
  and Windows.
- The core dependencies are numpy, pandas, gymnasium, pydantic and PyYAML. The extras are
  `rl`, `plots`, `web` and `dev`.
- `cleo` is `lob.cli:main`. It builds one argparse tree and delegates to `lob.v06.cli` and
  `lob.v07.cli` through `add_commands`/`dispatch`. v0.8 follows the same pattern
  (`lob.v08.cli`).

**v0.7 protocol and ledger.**
- `lob/v07/protocol/core.py` keeps an append-only, hash-chained JSONL ledger. Its events are
  `import_consumed`, `declare_fresh`, `freeze`, `seal_design`, `access`, `inspect`, `attempt`
  and `amend`.
- Data access is role-gated and staged: downloaded, opened, parsed, evaluated.
- Evidence binding records protocol, ledger anchor, dataset identity, config, result and
  source hashes.
- `cleo protocol-v07 status` verifies 196 entries:
  - protocol `63fb6e56…`;
  - head `8a1cb71d…`;
  - **no fresh holdout remains**. ETH 2020-11/12, BTC 2020-11, BitMEX 2020-11 and ETH
    2021-01 were all consumed by v0.7.

**v0.7 evidence and claims.**
- `lob/v07/evidence/runs.py` creates write-once run directories with `config`, `result`,
  `provenance`, `binding` and `checksums`. Source hashes use CRLF→LF normalization.
- `lob/v07/reports/registry.py` builds the hypothesis table, the claim graph
  (`[C-Hn]` markers) and the term audit.
- `lob/artifacts.py` provides `seal_artifacts` and `verify_artifacts`, which give byte
  integrity.

**v0.7 market and execution layer.**
- The frozen v0.5 `ExchangeSimulator` is a Python event-driven matching engine with
  order-level FIFO. v0.7's `BundleSimulator` drives it with 100 ms count models.
- Episodes run through `lob/v07/execution/episode.py` with v0.5 agents. One 120 s episode
  costs tens to hundreds of milliseconds, depending on the world.
- Historical data comes as 100 ms `BookTape` grids from Tardis incremental L2 plus trades,
  about 285 MB per day as a pickle.
- `lob/v07/generators/events.py` derives six 100 ms event counts from a tape.

**v0.7 results that motivate v0.8 (unchanged here):**
- **FAILED:** H1 (posterior calibration on fresh data), H4/H5 (selected generator cross-instrument/venue), H11.
- **NOT_ESTABLISHED:** H2/H3 (realism), H8 (robust policy comparison), H10, H12 (transfer).
- **INCONCLUSIVE:** H9 and H13.
- **ASSUMPTION_DEPENDENT:** posterior recovery (2 of 3 recovery coverages below 0.80).
- **ESTABLISHED:** H6 (multimodal, diffuse posterior) and H7 for POV only.

**Tests.** There are about 85 test modules. They are bounded fixtures (`tests/v07_fixtures.py`)
and do not depend on local `results/`.

**Environment observations (this machine).**
- Windows 11, i7-13620H (16 logical CPUs), 32 GB RAM, Python 3.14.
- The active shell has the ignored `cleo-playground` venv on `PATH`. v0.8 work uses a separate
  ignored `.venv` with the `dev` extra installed.
- **Tardis public samples cannot be downloaded from this machine.**
  `datasets.tardis.dev` returns HTTP 403 (a Cloudflare challenge) even for an
  *already-consumed* file. It was probed with headers only, so no new period was requested.
  Any v0.8 result that needs new real market data therefore depends on an external
  acquisition that is currently blocked (see §6).

## 2. Reuse map

| v0.8 need | Reused component (imported, never edited) |
|---|---|
| canonical JSON / hashing | `lob.config.canonical_json`, `lob.preregistration.document_sha256`, `period_key` |
| byte-integrity sealing | `lob.artifacts.seal_artifacts`, `verify_artifacts`, `portable_provenance` |
| source manifest (CRLF-normalized) | `lob.v07.evidence.runs.source_manifest` / `normalized_sha256` |
| v0.7 history pinning | `lob.v07.protocol.core` (load, replay, protocol hash) |
| historical tapes | `lob.v07.data.tape.cached_build`, `lob.v07.data.schema.venue_spec`, `lob.v06.tape.BookTape` |
| real 100 ms event counts | `lob.v07.generators.events.extract` (six marks + state) |
| result vocabulary | v0.7 taxonomy statuses (v0.8 adds `NOT_EVALUABLE` and generator-gate statuses) |

**Incompatible assumptions (why v0.8 adds its own market core).**
1. The v0.5 engine has no separate *exchange-truth / observed / decision* state, and no feed
   dissemination delay at event level. v0.7's `DelayedFeed` was a wrapper around snapshots.
2. Synthetic **ground truth for queue position** is needed to validate counterfactual bounds
   (§38 of the brief): run the same world with and without the strategy, then check the
   aggregate-only bounds against the truth. The v0.5 engine stores orders but gives no
   aggregate-tape/no-strategy twin.
3. Robust optimization, MPC and uncertainty decomposition need on the order of 10^5 episodes.
   At v0.7 per-episode costs that exceeds the budget.

v0.8 therefore adds `lob/v08/market`, a compact **aggregate-queue book**:
- integer tick grid;
- K visible levels per side;
- background liquidity as level queues;
- strategy orders tracked by explicit queue position (`ahead`, `own`, `behind`).

Generators drive this book in 100 ms bins. The book is a *model component*, not a
replacement for the v0.5 engine, which stays frozen. Its semantics are documented and tested:
FIFO consumption, a uniform background-cancel position, price shifts on depletion, and refill
from a registered depth distribution.

## 3. Component map

```
lob/v08/
  taxonomy.py        result vocabulary (9 statuses), generator-gate statuses, Result validation
  protocol.py        frozen protocol, append-only ledger, design seals, evidence binding (M1)
  datasets.py        dataset registry v3 + capability matrix + fail-closed capability gates (M2)
  quality.py         market-data quality engine on columnar event tables (M3)
  clocks.py          nine-clock timestamp normalization; event/volume time (M4)
  realdata.py        ledger-gated access to consumed tapes; windows -> EventFrames
  market/            aggregate-queue book, EventFrame, summaries, simulation driver
  generators/        plugin interface + Poisson, Hawkes, queue-reactive, regime-switching,
                     state-dependent, mixture (M8); ecology (M9); endogenous response (M10)
  impact.py          exponential / power-law / multi-timescale kernels + diagnostics (M11)
  inference/         ABC rejection, ABC-SMC, importance reweighting, surrogate likelihood (M12),
                     hierarchical pooling (M13), adaptive/online calibration (M14),
                     identifiability v3 (M15), synthetic recovery + misspecification (§14)
  regimes.py         threshold / volatility / liquidity / spread / OFI / k-means / change-point (M5)
  drift.py           CUSUM, Page-Hinkley, BOCPD, distribution drift, support failure, half-life (M6, M7)
  ppc.py             posterior predictive checks at multiple horizons (M16)
  model_averaging.py posterior model probabilities, information/holdout/dynamic weights (M17)
  uncertainty.py     nested variance decomposition across eight sources (M18)
  ood.py             support, Mahalanobis, kNN density, ensemble disagreement, surprise, conformal (M19)
  conformal.py       split / rolling / Mondrian conformal with coverage records (M20)
  worlds.py          registered plausible world sets (generator x posterior x queue x latency x ...)
  decision_calibration.py  calibration loss vs regret / ranking accuracy (M21)
  execution/         simulator (truth/observed/decision state, latencies), accounting + metrics (§11),
                     policy plugin interface (§10), baselines, adaptive variants (M26),
                     completion constraints (M22), Pareto analysis (M23), MPC (M25)
  robust.py          nominal, posterior-expected, CVaR, minimax-regret, worst-case, KL-DRO (M24)
  selection.py       contextual/regime-aware selection, mixture of experts, abstention, fallback (M27)
  stress.py          13 structured stress scenarios + bounded adversarial search (M28)
  counterfactual.py  aggregate-L2 historical execution bounds; synthetic coverage benchmark (M29)
  latency.py         latency/stale-book study (M30)
  portfolio.py       cross-venue / cross-instrument / SOR / basket / cross-impact (M31)
  transfer.py        ranking transfer: Kendall, Spearman, pairwise, top-k, regret, dominance (M32)
  statistics.py      bootstrap, sign, TOST, non-inferiority, multiplicity, MDE/power, sequential (M33)
  planning.py        resource and sample-size planning; budgets tiny..exhaustive (M34)
  runner.py          deterministic parallel tasks, checkpoint/resume, duplicate prevention (M34)
  artifacts.py       write-once runs, lineage, DAG, manifests (M35)
  claims.py          claim graph + fail-closed claim audit (M35)
  figures.py         dependency-free SVG figures with CSV/JSON beside them (M35)
  reporting.py       hypothesis table, final analysis, report, evidence export (M35)
  studies/           one module per CLI study
  cli.py             30 `*-v08` commands (+ `identifiability-v3`, `inference-v08`)
```

## 4. Dependency graph (build order)

```
taxonomy -> protocol -> datasets -> artifacts/runner/planning/statistics
market/book -> market/frame -> generators/* -> inference/* -> worlds
execution/* (needs market, generators) -> robust, selection, stress, counterfactual, latency, portfolio
regimes, drift, ppc, model_averaging, uncertainty, ood, conformal (need generators + statistics)
decision_calibration, transfer (need execution + inference)
studies/* -> claims/reporting/figures -> cli
```

## 5. Milestones and scientific design

Every milestone M0–M35 maps to a module and a CLI command. The hypotheses H1–H21 are frozen
in `configs/v08/hypotheses.json` **before** any confirmatory seed namespace is used. The
seeding rules are:
- Development and testing use seed namespace `dev`.
- Confirmatory simulation studies use `confirm`. It can only be opened after the protocol
  freeze, through a sealed design.
- The sealed synthetic multi-axis holdout uses `holdout`. It is opened once, by the final
  design.

Leakage tests assert these gates.

**Evidence levels.**
- Synthetic confirmatory studies, where ground truth is known, are labelled `SYNTHETIC`.
- Analyses on the consumed v0.5–v0.7 days are capped at `EXPLORATORY` with qualifier
  `RETROSPECTIVE`. They are chronological and informative, but the data are not fresh.
- Fresh real holdouts are declared:
  - ETH/BTC/BitMEX 2021-02-01: temporal, instrument and venue transfer;
  - ETH/BTC/BitMEX 2021-03-01: final multi-axis holdout.
  If acquisition is blocked, the dependent hypotheses are `NOT_AVAILABLE`.

## 6. Scientific risks

1. **Fresh real data unavailable** (403 at audit time).
   - Mitigation: a registered acquisition rule. Before any holdout request, the downloader
     probes an already-consumed control file. If the probe fails, the attempt is ledgered
     as NOT_AVAILABLE and the holdout stays unread (and fresh).
   - Consequence: no fabricated external validation.
2. **Synthetic-only confirmatory evidence** says nothing direct about real markets. Every
   such result carries `SYNTHETIC`, and the documents state this beside each claim.
3. **Weak identifiability** (inherited from v0.7) can make posteriors diffuse. H4/H5 are
   designed to report that rather than force recovery.
4. **Aggregate-queue book misspecification.** The book is itself a model. The misspecification
   benchmark (§34) and the generator rejection gate (§13) are designed to expose this.
5. **Multiplicity.** There are 21 hypotheses with sub-endpoints. Families and corrections
   are registered (Holm within family; BH for exploratory screens).
6. **Retrospective reuse.** Consumed days were seen by earlier versions. v0.8 never selects
   models on them and never upgrades their results beyond EXPLORATORY.

## 7. Engineering risks

- **Runtime.** Episode cost is controlled through bin-level vectorization and budget tiers.
  The planner predicts events, RAM, disk and wall clock before a study starts.
- **Memory.** One day's tape is about 285 MB, and v0.7 had low-memory aborts. Real-data
  windows are extracted once into compact EventFrame caches (`data/v08/cache`, ignored).
  Studies hold at most one tape at a time.
- **Nondeterminism under parallelism.** Every task seed is derived from
  `SeedSequence([namespace, study, task-id])` and never from scheduling. Serial and parallel
  runs are compared byte-for-byte in tests.
- **Windows/Linux numeric drift.** Artifact hashes are compared on the same platform.
  Cross-platform comparisons use documented tolerances.
- **Legacy compatibility.** No v0.5–v0.7 file is edited. `lob/cli.py` gains one
  `add_commands`/`dispatch` hook. The `pyproject` version changes to `0.8.0.dev0`.

## 8. Migration and compatibility rules

- v0.7 protocol hash, ledger head and entry count are pinned in the v0.8 protocol
  `history`. `protocol-v08 verify` fails if any of them changes.
- Old tags `v0.1.0`–`v0.7.0` are not touched. A test checks that the v0.7 ledger still
  replays.
- New optional extras:
  - `inference` (scipy, used only where the numpy fallback exists);
  - `neural` (torch);
  - `distributed` (none required; the process pool is the default);
  - `all`.
  Core installs stay numpy/pandas only. A missing optional dependency raises an error that
  names the extra.

## 9. Expected artifacts

- `configs/v08/`: protocol, hypotheses, statistical families, dataset registry, compute
  budget, claim registry, designs and the ledger.
- `results/v08/<study>/<run>/` (ignored): sealed runs with lineage.
- `examples/studies/v08/` (committed): the compact public bundle. It contains the protocol,
  hashes, results summaries, hypothesis table, claim graph, figures with CSV/JSON, transfer
  matrices, counterfactual bounds and verification metadata.
- `docs/v08-*.md`: protocol, calibration, inference, generators, uncertainty, completion,
  robust execution, counterfactual, transfer, statistics, reproduction, final report and
  paper.
