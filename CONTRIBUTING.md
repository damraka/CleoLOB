# Contributing to CleoLOB

Start with Python 3.11+ and `python tools/bootstrap.py`. Add `--rl` for PPO/DQN
work. Use a branch and keep implementation and compact empirical evidence in
separate commits. The Linux commands in [reproduction](docs/reproduction.md)
also work on Windows with `.venv/Scripts/python.exe`.

Run `python -m ruff check lob tests tools train_rl.py server.py`,
`python -m compileall -q lob tools train_rl.py server.py`, `python -m pytest -q`,
`python -m pip check`, and `python -m build --wheel` in the environment.
Include tests of economic semantics or failure modes when changing research code.

Research changes require a declared question, data identity, temporal/seed split,
observables, thresholds, comparison family and evaluation budget before reading
the final holdout. Preserve failed runs. A previously inspected holdout is consumed,
even when a new model has not been fitted to it. No exact FIFO or passive fills
may be invented from aggregate L2. Reconstructed trades are not hypothetical fills.

Submit a [reproduction report](docs/reproduction-report-template.md) with the
commit, runtime, command, dataset checksum, expected and observed outcome. Never
include credentials, personal absolute paths, restricted market data, model ZIPs,
large generated reports or raw CI logs in a pull request. Use external artifacts
with checksums for scientifically necessary large files. Community adoption and
independent review must only be described when independently evidenced.
