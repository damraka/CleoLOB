# v0.4 evidence integrity

`lob.evidence` adds the `cleolob-research-v2` schema without rewriting legacy
seals. `register_evidence` creates a new directory with normalized configuration,
dataset/generator identities and hashes, adapter versions, roles, seed domains,
hypotheses, runtime/git provenance, a registration seal and exact source copies.
It runs before the experiment. Existing directories are refused.

`finalize_evidence` rejects changed implementation or registration bytes, validates
model identity/checkpoint hashes, binds the result to configuration/datasets/source
and registration, then creates a complete file checksum seal. Scientific failure
is a valid retained result. Results use explicit validity statuses. `verify_evidence`
checks file identity and semantic links, including stale checkpoints and a result
bound to the wrong configuration, without executing archived code.

Existing registered calibration and policy runners retain their specialized
selection/checkpoint/evaluation seals; their verifiers must be used on full runs.
`cleo verify-artifact` remains the portable byte verifier for compact exports.
An export lists omitted raw data, source/checkpoint/trace files and original hashes.
An export checksum does not assert that the omitted files were reproduced.

Checksums detect accidental changes, not an adversary who rewrites all files and
seals. Git commits and the review record provide external references, not a signed
scientific attestation. Reproduction creates a fresh run; it never replaces the
original evidence. Previously committed research source copies are intentional
immutable provenance, so packaging cleanup does not remove them.
