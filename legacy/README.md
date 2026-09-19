# Archived experiments and interfaces

These files preserve earlier interfaces and results. They are not evidence for
the current PPO–Almgren–Chriss study in `examples/studies/core`.

- `app.py`: Streamlit dashboard (`streamlit run legacy/app.py` from repository root).
- `evaluate.py`: former flat evaluation harness (`python -m legacy.evaluate`).
- `train_rl_original.py`: original training settings, including inspected diagnostic seeds.
- `baselines-100seeds/`: historical tables from earlier market mechanics.
- `README-original.md`: original claims and usage, retained as historical text.

The active entry points are `cleo` and `python train_rl.py`. The existing FastAPI
viewer remains available. Portfolio/FX modules are frozen: no additional product
or asset-class work is in scope until execution-model validity is established.
