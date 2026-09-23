"""Run the preregistered, configuration-matched PPO research workflow.

Register a frozen design before training, then evaluate once. The historical
single-model training script is archived in legacy/train_rl_original.py.

Examples::

    python train_rl.py register --config configs/core-study.json --out results/core-study
    python train_rl.py train --study results/core-study
    python train_rl.py evaluate --study results/core-study
"""

from lob.core_study import main


if __name__ == "__main__":
    main()
