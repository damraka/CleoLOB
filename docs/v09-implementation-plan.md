# CleoLOB v0.9 implementation plan (M0)

Branch `research/v0.9-reality-calibrated-twins`, development version `0.9.0.dev0`, base commit `c42be48` (the
`v0.8.0` release merge). This plan was written before any v0.9 data access. Section 3 records only header-level
availability probes, which read no market data.

> Which execution conclusions remain scientifically defensible when the following are considered simultaneously:
> market reconstruction uncertainty, latent liquidity, self-impact, strategic response, causal ambiguity,
> multi-fidelity simulation, distribution shift, tail events, model uncertainty, online adaptation and real
> historical validation?

## 1. v0.8 postmortem and failure taxonomy

| Class | What happened in v0.8 | v0.9 countermeasure |
|---|---|---|
| Data access (single point of failure) | The only provider was the Tardis free sample. Both real holdouts were blocked (HTTP 403), so H13, H15–H17 and H20 were NOT_AVAILABLE. Every inference was SYNTHETIC. | A header-only multi-provider probe before freeze (§3), registered backups, and an L2 source with ~2.5 years of daily archives. Data roles are fixed before access (M10, M29 of the brief). |
| Measurement capability | Aggregate L2 at 10 visible levels; queue position never observed. | Full incremental L2 (500 levels, update IDs and cross sequence) plus trade prints with IDs. Observable *queue-clearance* events give an exact fill condition for a resting order regardless of queue position (§6, M22/M66). |
| Global generator gate | One binary gate gave NO_ACCEPTABLE_MODEL, and nothing could be said per capability. | A capability-specific gate (M55): ACCEPTABLE / PARTIALLY_ACCEPTABLE / REJECTED per capability, and studies fail closed on the capabilities they need. |
| Underpowered confirmatory hypotheses | Several hypotheses ended NOT_ESTABLISHED with intervals wider than the practical threshold. | A promotion gate (M2): pilot variance → MDE and power at the preregistered threshold; only powered hypotheses are sealed. |
| Abstention and fallback | OOD-aware abstention and TWAP fallback made decisions worse (H11/H12 FAILED). | Abstention v3 evaluates signals jointly with predicted regret and tests a contextual fallback ladder (M83, M84). No assumption that abstention helps. |
| Engineering invalidity | Fractional queue positions, `log(0)` in a fixed-share mixture, claim resolution picking a retained failed attempt, holdout namespace opened before input validation. | Regression tests carried forward (fixed-share underflow, M47 of the brief). Inputs are validated before every ledger write. An incident ledger (M4/§50) is kept. |
| Resource pressure | A host reboot and memory pressure interrupted studies. | Registered resource contracts per study (RAM ceiling, worker cap, checkpoint interval) and RESOURCE_EXHAUSTED semantics (M9, §41). Raw files stream and are never held whole in memory. |

## 2. Status vocabulary

These are the brief's vocabularies. Each is fixed in `configs/v09/protocol.json`:
- engineering: PLANNED, IMPLEMENTED, TESTED, VERIFIED, COMPLETE, BLOCKED;
- data: AVAILABLE, NOT_AVAILABLE, CAPABILITY_MISSING, QUALITY_REJECTED;
- hypothesis lifecycle: CANDIDATE → FEASIBILITY_PASS/FAIL → PROMOTED/NOT_PROMOTED_* → SEALED → a terminal status
  (ESTABLISHED, FAILED, NOT_ESTABLISHED, NOT_AVAILABLE, NOT_EVALUABLE, INCONCLUSIVE);
- realism;
- historical and causal identification;
- evidence levels E0–E10.

## 3. Pre-freeze availability probe (2026-10-13, headers only)

| Source | Capability | Availability (probed) |
|---|---|---|
| Tardis.dev free samples (v0.8 source) | aggregate L2 + trades | **404** — retired |
| Bybit linear perpetual `ob500` archive | incremental L2, 500 levels, snapshot + delta, update id `u`, cross sequence `seq` | 2023-02 → 2025-08 |
| Bybit spot `ob200` archive | incremental L2, 200 levels | 2025-05 → present |
| Bybit public trades | trade id, ms timestamp, aggressor side, price, size | all probed dates |
| Binance USD-M `bookTicker` | L1 every change, update id, transaction + event time | 2023-07 → 2024-03-30 |
| Binance USD-M trades, `bookDepth` | trades with buyer-maker flag; aggregated ±% depth bands | all probed dates |
| OKX trades | trades with side | all probed dates; no public L2 archive found |

No Bybit, Binance or OKX data, and no 2023–2025 period, appears in the v0.5–v0.8 consumption ledgers.

## 4. Data chronology and roles (fixed before access)

| Role | Venue / instrument | Period | Purpose |
|---|---|---|---|
| development | Bybit linear BTCUSDT | 2023-07-15 | parser, reconstruction and replay engineering only |
| explore | Bybit linear BTCUSDT, ETHUSDT; Binance UM BTCUSDT, ETHUSDT | 2023-08-01 → 2023-10-31 (registered day list) | pilots: variance, effect size, feasibility, compute |
| confirm | Bybit linear BTCUSDT, ETHUSDT; Binance UM BTCUSDT, ETHUSDT (L1) | 2023-11-01 → 2024-03-28 (registered randomized day list; promotion fixes the prefix length) | confirmatory tests, cross-venue (L1-identifiable conclusions) |
| confirm_late | Bybit linear BTCUSDT, ETHUSDT | 2024-06-01 → 2024-12-31 | cross-time transfer |
| final vault | Bybit linear SOLUSDT, BTCUSDT; Bybit spot BTCUSDT | 2025-05-01 → 2025-08-28 | one-time multi-axis holdout: new time, new instrument, new market segment |

Day lists are drawn by a registered SeedSequence rule from each pool before any access. A confirmatory day may
be read only after the promotion freeze, and only by a sealed design that lists it. Final-vault days are read
exactly once, by the final-holdout design, after every confirmatory run has sealed. If a registered source fails
(HTTP error, archive missing, quality rejection), the registered backup day for that pool slot is tried. If every
registered option fails, the outcome is NOT_AVAILABLE. Consumed data are never substituted.

## 5. Fidelity ladder and evidence levels

| Fidelity | Implementation | Evidence level reachable |
|---|---|---|
| F0 raw observations | Bybit / Binance / OKX archives, hashed at download | — |
| F1 historical replay | event-sourced L2 reconstruction (M20), observational statistics | E4–E9 (observational) |
| F2 bounded counterfactual replay | strategy injected into the reconstructed book under an assumption grid (queue, latency, hidden liquidity, impact); interval outcomes | E4/E5 when the interval separates |
| F3 hybrid twin | historical backbone + strategy + calibrated local stochastic response (resilience, impact) | E3 |
| F4 classical generators | Poisson, marked Poisson, Hawkes, marked Hawkes, queue-reactive, regime/semi-Markov, mixtures | E1/E2 |
| F5 neural generators | neural TPP (RMTPP-like, Transformer TPP), GRU/LSTM/causal-Transformer sequence models, latent state-space | E1/E2 |
| F6 strategic ecology | ten behavioural proxies with endogenous response | E1/E2 |

The highest evidence level ever reported is the level its sources support. A conclusion reaching E9 is reported
as having survived the registered final multi-axis holdout, never as working live.

## 6. Measurement design on public L2

- **Exact book reconstruction (M20).** Snapshot + delta replay with update-id continuity and cross-sequence
  checks. Gaps, duplicates, out-of-order messages and crossed books are flagged, never repaired silently (M13–M16).
- **Queue-clearance events (M22, M66, M67).** For a level present at time t, a print *through* that price (or the
  level disappearing via trades) is an exact fill condition for any resting order at that level. A trade *at* the
  price is necessary but not sufficient. So the clearance time bounds the fill time from above for every queue
  position, and the first trade at the price bounds it from below. Survival models are fitted to the observable
  clearance time, and every fill claim is stated as bounded.
- **Bounded counterfactual replay (M28).** This reuses the registered v0.8 assumption semantics: queue rule,
  latency, hidden fraction, impact rule. A pairwise direction is HISTORICALLY_IDENTIFIED only when the policy-cost
  intervals separate on the inferential unit.
- **Inferential units.** Instrument-days for real data, with windows nested in days (paired block or hierarchical
  bootstrap), and world-seed clusters for synthetic data. Event rows are never treated as replicates.

## 7. Candidate hypotheses H1–H36 (operationalization)

All 36 are registered as CANDIDATE in `configs/v09/hypotheses-candidates.json`. Each candidate has these fields:
- endpoint;
- estimator;
- inferential unit;
- direction;
- practical threshold;
- statistical family;
- fidelity;
- required capability;
- evidence level.

All of them are fixed before any exploratory access. Exploration only estimates variance, effect and
feasibility. `hypothesis-promotion-v09` then applies the registered gate mechanically: power ≥ 0.80 at the
registered threshold, within the confirmatory sample ceiling and the compute budget, with capability and
measurement checks. The output freezes `configs/v09/hypotheses-confirmatory.json` before any confirmatory day is
read.

**Execution conclusions for the reality ladder.** H29–H34 and H36 range over a registered set of candidate
execution conclusions, C1–C6: policy pairs × mandate classes × endpoints. Promotion selects which are carried; the
final vault tests every carried conclusion once, with Holm across the carried set.

## 8. Module map (M0–M89)

| Group | Milestones | Module (`lob/v09/…`) |
|---|---|---|
| Governance | M0–M9 | `protocol.py`, `ledger.py`, `promotion.py`, `evidence.py`, `runtime.py` (fingerprint, SBOM, lock) |
| Data | M10–M19 | `datasets.py`, `capabilities.py`, `quality.py`, `data/{sources,bybit,binance,okx,normalize,clocks}.py` |
| Replay | M20–M32 | `data/reconstruct.py`, `data/replay.py`, `queue/{reconstruction,hidden}.py`, `replay/{bounded,latency,races,validity}.py` |
| Multi-fidelity | M33–M37 | `twin/{api,capabilities,runner,uncertainty,consensus}.py` |
| Representations | M38–M43 | `representations/{features,encoders,contrastive,graph,transfer}.py` |
| Generators | M44–M50 | `generators/{base,classical,point_process,sequence,latent,flow,hybrid,strategic}.py` |
| Realism | M51–M55 | `realism/{distributions,temporal,conditional,path,execution,tails,gates,benchmark}.py` |
| Inference | M56–M62 | `inference/{sbi,amortized,sequential,hierarchical,ppc,identifiability,active}.py` |
| Liquidity, fills, markouts | M63–M70 | `queue/{hidden,survival,fills}.py`, `markout.py` |
| Impact | M71–M74 | `impact/{kernels,decomposition,state_dependent,cross_impact,endogenous}.py` |
| Causal | M75–M78 | `causal/{estimands,dag,sensitivity,bounds}.py` |
| Decisions | M79–M84 | `execution/{dro,mpc,selector,fallback}.py`, `rl/*`, `uncertainty/{conformal,abstention}.py` |
| Stress | M85 | `stress/{evt,rare_event,adversarial,counterexample}.py` |
| Transfer, portfolio, synthesis | M86–M88 | `transfer.py`, `portfolio.py`, `agreement.py`, `reality_gap.py`, `voi.py` |
| Final | M89 | `studies/final.py`, `reporting.py`, `claims.py`, `verify.py` |

Earlier versions are imported and never edited. v0.9 reuses these v0.8 parts unchanged:
- the execution simulator, policies and aggregate-queue book;
- the counterfactual replay semantics;
- the statistics primitives;
- the deterministic runner.

The protocol, ledger v2, run schema v3, checkpoint v3 and claim audit are new.

## 9. Statistical rules

- **Confirmatory families** are registered with Holm FWER, α = 0.05, two-sided intervals.
- **Real data** uses paired block or hierarchical bootstrap over instrument-days.
- **Proportions** use Wilson intervals.
- **Equivalence and non-inferiority** are tested with preregistered margins (TOST).
- **Exploratory discovery** uses BH and is never mixed into confirmatory families.

ESTABLISHED requires all of the following:
- the registered decision rule;
- the practical threshold;
- the capability and evidence requirements;
- valid sealed evidence;
- no unresolved engineering invalidity.

## 10. Resource contracts

The host has 32 GB RAM, 16 logical CPUs and an RTX 5070 Laptop GPU. The installed PyTorch is CPU-only, so GPU
use is optional and never required for correctness. Each expensive study registers:
- task count;
- RAM ceiling;
- worker default and maximum;
- disk estimate;
- checkpoint interval.

Raw archives are streamed, and day caches are stored as compressed arrays under `data/v09/` (gitignored, never
committed or bundled). At most one heavy study family runs at a time. Exceeding a registered ceiling yields
RESOURCE_EXHAUSTED; a budget increase requires an amendment before any outcome inspection.

## 11. Scope statements (where "feasible" applies)

- **Neural models** (TPP, sequence, latent state-space, contrastive encoders, offline RL) are small, CPU-trainable
  research models with fixed seeds. They are compared with classical baselines, and none is assumed better.
- **Neural ODE/SDE and diffusion / flow matching** are implemented where the representation makes them
  meaningful:
  - a Neural SDE for the mid/spread latent path;
  - a conditional flow for the event-mark distribution.
  
  If a method fails its numerical-stability or positive-control checks, it is reported BLOCKED with the evidence,
  never presented as working.
- **Offline RL** uses behaviour data generated by registered stochastic behaviour policies in simulation, so that
  propensities exist. On historical data no propensities exist, and IPS/SNIPS/DR are NOT_AVAILABLE by
  construction.
- **Causal claims** are limited to registered estimands with DAG-based identification checks. Observational impact
  estimates are ASSUMPTION_DEPENDENT unless a design identifies them.

## 12. Phases

| Phase | Content |
|---|---|
| A | M0, protocol, ledger v2, promotion machinery, schemas, CLI skeleton, tests |
| B | data adapters, quality, reconstruction, replay (development day) |
| C–I | generators, realism, inference, queue/fill, impact, execution, RL, causal, stress, transfer, synthesis |
| J | exploratory pilots → promotion → confirmatory freeze |
| K | confirmatory studies → final vault → final analysis |
| L | reports, model/policy cards, datasheets, bundle, claim audit, verify |
| M–O | research PR and merge, then release PR, tag, GitHub Release, PyPI, public install, cleanup |
