# Prior Linux research runs

The workflow at commits `a0ef372` and `c459dc6` ran the same frozen 20-model design
on every push. Both research jobs completed training, evaluation, and verification
on 2026-09-21. Their GitHub Actions artifact inventories are empty: the workflow
did not upload the models, training traces, or raw episode files before the
ephemeral runners were discarded.

This directory contains sanitized exports of the retrieved job logs and API
metadata, plus the unchanged JSON summaries printed in those logs. The former
branch label is replaced with `[previous branch]` in the exported logs and omitted
from the exported metadata. No scientific outcomes, commit IDs, or run IDs are
altered. The exports establish that the prior tests
were run. They cannot replace the missing sealed artifact package. The two runs
reuse the same seed blocks and must not be pooled as independent observations.
The subsequent local run retains a durable replication of the fixed design;
its test seeds cannot be described as previously unexposed.

| Run | Commit | Started UTC | Completed UTC | Final episodes |
| --- | --- | --- | --- | --- |
| [35649272649](https://github.com/damraka/CleoLOB/actions/runs/35649272649) | `a0ef372` | 2026-09-21 20:09:13 | 2026-09-21 21:15:53 | 704 / 704 |
| [35649544066](https://github.com/damraka/CleoLOB/actions/runs/35649544066) | `c459dc6` | 2026-09-21 20:11:50 | 2026-09-21 21:15:57 | 704 / 704 |

Both printed configuration hash
`ad27b6d96a527f36587a771b1935ed816a15b3aeaa6ec1669b3ef4dc990a65c9`.
`export-provenance.json` records the original download and published export
SHA-256 hashes for every sanitized file, and hashes the unchanged printed
summaries. Each run's `metadata.json` retains `log_sha256` as the hash of the
original downloaded log, before the explicit branch-label redaction.
No effect estimates from these logs were consulted to choose the local design,
configuration, or model checkpoint.
