# v0.5 research protocol (M0 freeze)

The machine-readable protocol is [`configs/v05/protocol.json`](../configs/v05/protocol.json).
It was frozen before any v0.5 empirical result was produced, on branch
`research/v0.5-real-market-validation` from base commit `224415d`. Its canonical
SHA-256 is recorded in the first `freeze` entry of the append-only
[consumption ledger](../configs/v05/consumption-ledger.jsonl):

```text
protocol_sha256 = feeec468fff3fd417ab024711759e19a7d10da56ae064edd8a07e3827bb32de7
```

Verify the freeze, the referenced configuration hashes and current holdout freshness:

```sh
cleo protocol verify
```

The scope follows the [v0.5 roadmap](v05-roadmap.md). This document summarizes the
JSON; where they differ, the JSON (bound by hash) is authoritative.

## Evidence statuses

Every v0.5 conclusion carries exactly one of: `ESTABLISHED`, `NOT_ESTABLISHED`,
`FAILED`, `NOT_AVAILABLE`, `INVALID`, `INCONCLUSIVE`. `INVALID` means integrity or
execution failure (unknown outcome), never a research failure. `NOT_AVAILABLE` means
nothing was substituted. Legacy PASS/WARNING/FAIL gates retain their v0.3/v0.4 meaning.

## Research questions

| ID | Milestone | Question |
|---|---|---|
| RQ1 | M1 | Do identity-preserving MBO semantics reproduce a genuine venue order-level feed? |
| RQ2 | M2 | How determinate are hypothetical passive fills from aggregate L2 versus order-level data? |
| RQ3 | M3 | Does a justified richer simulator family improve out-of-sample fit and pass frozen gates? |
| RQ4 | M4 | Do simulator impact/resilience responses agree with history at preregistered horizons? |
| RQ5 | M5 | How often is completion within horizon versus only after settlement? |
| RQ6 | M6 | Which conclusions survive regimes, an unseen period and a different instrument? |
| RQ7 | M7 | Does main-arm PPO/DQN pass the joint cost and within-horizon completion gate? |
| RQ8 | M8 | Do simulator policy conclusions survive bounded historical execution? |

## Datasets and freshness

| Dataset | Role | Freshness at freeze |
|---|---|---|
| Deribit ETH-PERPETUAL 2020-04-01 (L2, trades, top-5) | development | consumed (v0.3/v0.4) |
| Deribit ETH-PERPETUAL 2020-05-01 (L2, trades) | selection | consumed (v0.3/v0.4) |
| Deribit ETH-PERPETUAL 2020-06-01 (L2, trades) | sealed internal holdout | consumed by v0.4; **retrospective only** |
| Deribit ETH-PERPETUAL 2020-07-01 (L2, trades) | external holdout: unseen period | fresh |
| Deribit BTC-PERPETUAL 2020-07-01 (L2, trades) | external holdout: different instrument | fresh |
| Deribit ETH-PERPETUAL 2020-08-01 (L2, trades) | M8 transfer holdout only | fresh |
| Bitstamp btcusd live capture, 900 s | MBO adapter development | fresh (to be recorded) |
| Bitstamp btcusd live capture, 1,800 s | MBO validation, after adapter freeze | fresh (to be recorded) |

The June 2020 period cannot provide fresh confirmation: v0.4 evaluated it and its
failure is known. It is used only as a sealed-from-v0.5-selection internal holdout
and every result on it is labeled retrospective. The July/August/September 2026
cohort and April–June 2020 ETH periods are imported into the ledger as consumed.

Bitstamp data are genuine venue order-level messages recorded from the public API
and replayed offline. They are not a vendor-certified historical MBO archive, and
Bitstamp's price-time priority is not established by the source feed itself.
LOBSTER samples now require registration and terms acceptance; Databento requires
credentials. Neither was used.

## Access policy

- Freshness is derived only from the hash-chained ledger. A dataset becomes consumed
  at first access (download), and the ledger entry is written before the download.
- `declare_fresh` is refused for any consumed dataset or venue/instrument/date alias.
- An external or transfer holdout cannot be accessed until a `seal_design` entry
  lists it, so every analysis reading it is frozen before first access.
- Amendments must be appended before first access of every dataset they affect.
- Raw provider data stay under ignored `data/` directories. Tardis terms restrict
  redistribution of raw and derived data; public evidence carries hashes and summaries.

## Endpoints

Primary endpoints E1–E7 are defined exactly in the JSON. In summary:

- **E1 (RQ1):** the fraction of published top-10 aggregate books exactly matched by the replayed MBO state.
- **E2 (RQ2):** the determinate share of hypothetical orders under conservative bounds, plus the normalized bound width.
- **E3 (RQ3):** the mean family error between simulated and historical observables, with a per-family gate of 0.60.
- **E4 (RQ4):** sign agreement and the magnitude ratio of conditional mid-price responses.
- **E5 (RQ5/RQ7):** within-horizon completion. A fill at exactly the horizon counts; settlement fills never do.
- **E6 (RQ7):** completion-adjusted cost in basis points. Realized fill cost and hypothetical residual valuation are reported separately.
- **E7 (RQ8):** the transfer classification.

Secondary endpoints include:

- settlement completion, residuals at the horizon and after settlement, and post-horizon fills;
- lateness, fill fractions, fees, participation, implementation shortfall and tail cost;
- seed and regime sensitivity;
- lifecycle anomaly counts;
- impact/resilience quantities and Almgren–Chriss proxies.

## Statistics and gates

| Study | Family | Correction | Gate |
|---|---|---|---|
| M3 | 3 holdouts, selected v2 versus v0.4 baseline | Bonferroni | Improvement: interval of loss difference < 0. Generalization: every family ≤ 0.60 with ≥ 95% valid samples and ≥ 1,000 samples |
| M4 | bucket × horizon × dataset | Bonferroni | Sign agreement and magnitude ratio in [0.5, 2] for ≥ 2/3 of buckets at each horizon |
| M7 | 4 regimes × 2 algorithms × 8 references × 2 endpoints = 128 | Bonferroni | Cost upper bound < 0 **and** completion lower bound ≥ −0.05 |
| M8 | policy pairs × datasets | Bonferroni | Classification only |

**M7 completion inference.** The v0.4 study had a degenerate-interval problem: constant
completion differences made every completion contrast `INCONCLUSIVE`. v0.5 preregisters
a distribution-free bound that avoids it:

- For each market seed m, Z_m = 1 if any training seed of the policy misses within-horizon
  completion while the reference completes.
- The expected completion difference is at least −π, where π is the probability that Z_m = 1.
- π is bounded above by a one-sided Clopper–Pearson limit at α = 0.05/128 across independent
  market seeds.
- With 160 market seeds and no discordance, the bound is 1 − α^(1/160) ≈ 0.048, inside the
  0.05 margin.

The bound conditions on the eight trained models. Cost uses the crossed bootstrap from v0.4.
Missing or INVALID cells are withheld and never shrink the family.

## Frozen semantics

**Completion.** No order may be submitted at or after the horizon. Within-horizon completion
counts only fills at or before the horizon end. Settlement processes only cancellations and
exchange events, and its fills are post-horizon. Final settlement completion is reported
separately and never called mandate completion. Hypothetical residual valuation never creates
a fill.

**Historical fill uncertainty.** Every hypothetical order receives one of these classes:
`OBSERVED_FILL` (source orders only), `GUARANTEED_FILL`, `POSSIBLE_FILL`,
`GUARANTEED_NON_FILL`, `INDETERMINATE` or `UNSUPPORTED`. Classification uses three bound sets:

- **Conservative:** only prints strictly through the order's price fill it.
- **Observable-FIFO:** displayed queue-ahead under price-time priority, with non-trade
  decreases attributed ahead or behind; this set assumes no hidden liquidity.
- **Optimistic:** the order is at the front of its level.

The hypothetical order is assumed to have no market impact. Aggregate L2 never yields exact
FIFO positions, order identities or hidden liquidity.

## Seeds, models and budgets

**Seeds:**
- M3 search seed 45100.
- M3 fit, selection and holdout simulation seeds: 41101–41103, 42101–42103 and 43101–43105.
- M4 simulator seeds 46101–46110.
- M7 seeds, in [`configs/v05/policy-study.json`](../configs/v05/policy-study.json):
  - training seeds 85001–85008
  - 160 evaluation market seeds from 77000
  - identification seeds 56001–56004
  - normalization seeds 1900200–1900207
  - statistics seed 95001

**Model families.**
- M3 starts from the unchanged v0.4 zero-intelligence (ZI) simulator search as the control, plus
  an IID resampling control.
- The M3 candidate extensions are: empirical size distributions, spread-conditioned arrivals,
  depth-conditioned cancellation, imbalance-conditioned market flow, and a Markov activity
  regime.
- Self-exciting market orders are included only if development trade interarrivals have a
  dispersion index above 1.5.
- One combined model joins the individually improving extensions.
- M7 retains PPO, DQN, TWAP, VWAP, POV, AC, heuristic and random controls, each under the
  `main`, `no_book` and `no_terminal` arms.

**Budgets:**

| Study | Budget |
|---|---|
| M3 | ≤ 48 candidates per family, 900 simulated seconds per seed |
| M4 | 1,800 simulated seconds per seed |
| M7 | ≤ 48 models × 16,384 steps; ≤ 40,000 evaluation episodes |
| M8 | 144 episodes per dataset per policy |

Reduced runs are labeled pilot or smoke and are never presented as the registered study.

**Normalization sources.**
- All M3, M4, M6 and M8 scales, thresholds and unit mappings come from the development
  period only.
- M7 observation normalization comes from registered training-domain exploration seeds only.

## Evidence binding

`lob.preregistration.bind_evidence` records the following, and `verify_binding` detects
changes to any of them:
- the protocol hash and a ledger anchor
- per-dataset identity hashes, which include exact source hashes from the ledger
- config, result and implementation-source hashes
- the git commit

Changes to the protocol, dataset identity, config, result or source, and ledger truncation,
are all detected. Hashes provide consistency, not an external timestamp. They cannot prove
non-inspection outside this workflow.
