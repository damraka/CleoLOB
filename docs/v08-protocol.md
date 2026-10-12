# v0.8 protocol

The v0.8 protocol is `configs/v08/protocol.json`, with its referenced configs:
- `hypotheses.json`
- `statistical-families.json`
- `dataset-registry.json`
- `compute-budget.json`
- `claim-registry.json`

It was frozen in the v0.8 ledger (`configs/v08/consumption-ledger.jsonl`) before any confirmatory seed
namespace was opened and before any v0.8 model was fitted to real data. `cleo protocol-v08 verify`
replays the ledger and checks three things:
- the protocol hash;
- the referenced-config hashes;
- the pinned v0.7 history (protocol hash, ledger length and head).

## What the protocol fixes

- **Hypotheses H1–H21.** Each has a question, domain, estimator, endpoint, sample unit,
  statistical family, threshold, mechanical status rule and failure semantics.
- **Dataset roles.**
  - Development: ETH 2020-10-01.
  - Calibration: ETH 2020-11-01.
  - Selection: ETH 2020-12-01.
  - Retrospective: the other consumed v0.5–v0.7 days.
  - Six fresh holdouts declared in the ledger:
    - 2021-02-01 for temporal, instrument and venue transfer;
    - 2021-03-01 for the final multi-axis holdout.
- **Acquisition rule for fresh holdouts.** Before any holdout request, a headers-only probe is sent to
  an already-consumed control file. If the probe fails:
  - an attempt with outcome NOT_AVAILABLE is ledgered;
  - no holdout byte is requested;
  - the holdout stays fresh.
- **Seed namespaces.**
  - `dev` is for development and pilots.
  - `confirm` is opened only by a sealed design.
  - `holdout` is the sealed synthetic holdout. It can be opened once.
- **The mandate.**
  - Buy 40 lots over 120 s, with decisions every 5 s and hard completion.
  - Taker fee 1 bps; maker rebate 0.25 bps.
  - The lot is 1/20 of the median touch size of the dataset.
- **Registered policies, world-set construction, stress and OOD scenarios.**
- **The generator rejection gate.**
- **Margins and thresholds.** Equivalence 0.5 bps, non-inferiority 3 bps, practical effect 0.25 bps.
- **Correction families, failure policies and evidence schemas.**

## Amendment 1 (before confirmatory data)

Amendment 1 adds a **registered secondary mandate of 400 lots**, reported beside each member.
- Hypothesis statuses are still decided on the frozen primary mandate.
- The reason: a pilot on development seeds, in worlds built from the registered fits, showed that under the
  40-lot mandate every aggressive policy pays roughly the half spread plus the taker fee. The decision
  problem was nearly degenerate.
- The amendment is ledgered (entry 157). No hypothesis, endpoint, margin, threshold, alpha, dataset role or
  primary mandate changed.

## M0 audit disclosure

During the repository audit, before the freeze, cached tapes of several consumed days were opened. The
only purpose was to read lot scales and event rates. No model was fitted to them and no outcome was
computed.

A headers-only probe of an already-consumed Tardis file returned HTTP 403 (a Cloudflare challenge).

## Designs

Every confirmatory study has a design in `configs/v08/designs/`. Each design:
- was sealed (ledger `seal_design`) before the study ran;
- pins its input runs by result hash;
- opens the `confirm` namespace (ledger `open_namespace`).

The final real holdout reader (`final-holdout-h20`) and the fresh transfer reader are sealed before any
access attempt.
