"""M8 regime sensitivity: the frozen transfer classification within M6 regime labels.

The M8 design lists "regime sensitivity using the frozen M6 thresholds" among the
reported quantities. This supplementary analysis reads a sealed M8 run's episode
rows unchanged, labels each historical episode by the 5-minute block containing its
start (``lob.regimes`` with the development-only thresholds sealed by M6), and
repeats the frozen pairwise classification within every label that has at least six
episodes. It is descriptive: the interval level is the run's registered Bonferroni
alpha, the family is not enlarged, and a conclusion is regime-robust only if its
dataset-level classification recurs in every evaluable label (the M6 rule).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration_v2_study import _tape
from .experiments.registry import PROJECT_ROOT
from .policy_transfer import ALGORITHMS, CONTROLS, FILL_MODES, _interval, _m7, _paired, classify
from . import preregistration as pr
from .regimes import BLOCK_S, block_statistics, label
from .v05_evidence import LEDGER_PATH, PROTOCOL_PATH, finalize, new_run

MIN_EPISODES = 6


def episode_labels(starts: dict[int, float], window_s: float, blocks: list[dict],
                   thresholds: dict) -> dict[int, dict | None]:
    """Label of the complete block containing each episode's window midpoint; None if no valid block does."""
    out = {}
    for index, start in starts.items():
        middle = start + window_s / 2
        block = next((b for b in blocks if b["start_s"] <= middle < b["start_s"] + BLOCK_S), None)
        out[index] = None if block is None else label(block, thresholds)
    return out


def episode_starts(dataset_first_s: float, count: int) -> dict[int, float]:
    """Frozen M8 episode starts: first capture time + 300 s + k * 600 s."""
    return {k: dataset_first_s + 300.0 + 600.0 * k for k in range(count)}


def run(m8_dir: str | Path, m7_dir: str | Path, m3_select_dir: str | Path, m6_thresholds_dir: str | Path,
        out: str | Path, *, root: Path = PROJECT_ROOT) -> dict:
    protocol = pr.load_protocol(root / PROTOCOL_PATH)
    state = pr.replay_ledger(pr.read_ledger(root / LEDGER_PATH), protocol)
    for name in ("m6-design", "m8-design"):
        if name not in state.designs:
            raise ValueError(f"{name} must be sealed before M8 regime sensitivity")
    m8_dir = Path(m8_dir)
    m8 = json.loads((m8_dir / "result.json").read_text(encoding="utf-8"))
    rows = json.loads((m8_dir / "episodes.json").read_text(encoding="utf-8"))
    execution = _m7(Path(m7_dir))[1].config.execution
    window_sim_s = execution.warmup_seconds + execution.horizon + execution.settlement_timeout + 1.0
    window_s = window_sim_s * m8["mapping"]["clock_ratio"]
    scale = json.loads((Path(m3_select_dir) / "result.json").read_text(encoding="utf-8"))["scale_native_per_lot"]
    thresholds = json.loads((Path(m6_thresholds_dir) / "result.json").read_text(encoding="utf-8"))["thresholds"]
    datasets = list(m8["extraction"])
    pairs = [(a, c) for a in ALGORITHMS for c in CONTROLS]
    alpha = m8["alpha"]
    labels_by_dataset, analysis = {}, {}
    for dataset in datasets:
        tape, _, _, _ = _tape(dataset, scale, root, "M8 regime sensitivity")
        blocks = block_statistics(tape)
        count = m8["extraction"][dataset]["planned"]
        labels = episode_labels(episode_starts(float(tape.t[0]), count), window_s, blocks, thresholds)
        labels_by_dataset[dataset] = {str(k): v for k, v in labels.items()}
        by_comparison = {(c["dataset"], c["left"], c["right"]): c for c in m8["comparisons"]}
        dataset_result = {}
        for dimension in list(thresholds) + ["stress"]:
            for value in sorted({v[dimension] for v in labels.values() if v is not None}):
                keep = {k for k, v in labels.items() if v is not None and v[dimension] == value}
                key = f"{dimension}={value}"
                if len(keep) < MIN_EPISODES:
                    dataset_result[key] = {"episodes": len(keep), "status": "NOT_AVAILABLE",
                                           "reason": f"fewer than {MIN_EPISODES} episodes"}
                    continue
                subset = [r for r in rows if r["dataset"] == dataset and r["episode"] in keep]
                comparisons = {}
                for pair_index, (left, right) in enumerate(pairs):
                    synthetic = by_comparison[(dataset, left, right)]["synthetic"]
                    historical = {mode: _interval(_paired(subset, dataset, mode, left, right), alpha, 5000,
                                                  48901 + 10 * pair_index + mode_index)
                                  for mode_index, mode in enumerate(FILL_MODES)}
                    comparisons[f"{left}-{right}"] = {"historical": historical,
                                                      "classification": classify(synthetic, historical)}
                dataset_result[key] = {"episodes": len(keep), "comparisons": comparisons}
        robustness = {}
        for left, right in pairs:
            overall = by_comparison[(dataset, left, right)]["classification"]
            per_label = [v["comparisons"][f"{left}-{right}"]["classification"]
                         for v in dataset_result.values() if "comparisons" in v]
            robustness[f"{left}-{right}"] = {
                "dataset_classification": overall,
                "labels_evaluated": len(per_label),
                "same_in_every_label": bool(per_label) and all(c == overall for c in per_label),
                "label_classifications": {c: per_label.count(c) for c in sorted(set(per_label))}}
        analysis[dataset] = {"per_label": dataset_result, "regime_robustness": robustness,
                             "unlabelled_episodes": sum(v is None for v in labels.values())}
    result = {"source_run": {"binding_result_sha256": json.loads((m8_dir / "binding.json").read_text(
                  encoding="utf-8"))["result_sha256"]},
              "alpha": alpha, "min_episodes": MIN_EPISODES, "datasets": analysis, "episode_labels": labels_by_dataset,
              "interpretation": ("Descriptive regime split of the sealed M8 rows at the registered alpha; labels "
                                 "describe observable conditions only; no additional hypothesis is tested.")}
    out = new_run(out)
    finalize(out, analysis="m8-regime-sensitivity", dataset_ids=datasets,
             config={"m8_run": m8_dir.name, "window_s": window_s, "thresholds_from": Path(m6_thresholds_dir).name,
                     "min_episodes": MIN_EPISODES, "block_s": BLOCK_S},
             result=result, root=root)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m8", type=Path, required=True)
    parser.add_argument("--m7", type=Path, required=True)
    parser.add_argument("--m3-select", type=Path, required=True)
    parser.add_argument("--m6", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.m8, args.m7, args.m3_select, args.m6, args.out)
    print(json.dumps({d: {k: v["same_in_every_label"] for k, v in r["regime_robustness"].items()}
                      for d, r in result["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()
