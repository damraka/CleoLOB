"""Drive the v0.9 confirmatory, vault and final steps one sealed run at a time (resumable, stability first).

Usage (from a clean worktree of the repository, so the code cannot change while a run executes)::

    python tools/v09_pipeline.py --root ../CleoLOB seal
    python tools/v09_pipeline.py --root ../CleoLOB data
    python tools/v09_pipeline.py --root ../CleoLOB studies
    python tools/v09_pipeline.py --root ../CleoLOB transfer
    python tools/v09_pipeline.py --root ../CleoLOB close
    python tools/v09_pipeline.py --root ../CleoLOB vault
    python tools/v09_pipeline.py --root ../CleoLOB final --input gate=... --input counterexample=...

Every step is idempotent: a run directory that already holds a sealed, verifying run is skipped; a directory that
exists without a valid seal is a retained failed attempt and the next attempt uses ``<name>-2``, ``<name>-3``, ...
(lob.v09.batch ledgers the failed attempt; a nonzero child exit or a missing seal stops the step). One study runs at a time; before each, the driver waits until at
least ``--min-free-gb`` of memory is free.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # the worktree's lob package

CONFIRM_BASE = "results/v09/confirm"
FINAL_BASE = "results/v09/final"
PHASED = {"hybrid-twin-v09", "neural-point-process-v09", "sequence-generator-v09", "latent-dynamics-v09",
          "fill-study-v09", "survival-study-v09", "impact-study-v09", "inference-v09", "meta-adaptation-v09",
          "continual-calibration-v09", "conformal-study-v09", "representation-transfer-v09",
          "counterfactual-bounds-v09"}
ORDER = ["confirm-bounds", "confirm-bounds-late", "confirm-multifidelity-agreement"]   # inputs before consumers


def log(msg: str) -> None:
    print(f"=== {time.strftime('%H:%M:%S')} {msg}", flush=True)


def sealed_ok(root: Path, run: str) -> bool:
    from lob.v09.batch import sealed_ok as ok
    return ok(root, run)


def target(root: Path, base: str, name: str) -> str | None:
    """The run directory to use: None when already sealed, else the first unused attempt name."""
    k = 1
    while True:
        rel = f"{base}/{name}" + ("" if k == 1 else f"-{k}")
        if not (root / rel).exists():
            return rel
        if sealed_ok(root, rel):
            return None
        k += 1


def fresh(root: Path, base: str, name: str) -> str:
    """The first attempt name not yet used (each data-preparation round is a new sealed run)."""
    k = 1
    while (root / (rel := f"{base}/{name}" + ("" if k == 1 else f"-{k}"))).exists():
        k += 1
    return rel


def cli(root: Path, args: list[str], logfile: Path, min_free: float, out: str | None = None) -> int:
    """One fail-closed step (lob.v09.batch): 0 only if the child exits 0 and ``out`` is sealed and valid; a failure
    is ledgered as a retained attempt."""
    from lob.v09.batch import run_batch
    if out is None and "--out" in args:
        out = args[args.index("--out") + 1]
    step = {"name": Path(logfile).stem, "cli": args, "out": out}
    summary = run_batch(root, [step], logdir=Path(logfile).parent, min_free_gb=min_free, echo=lambda m: print(m, flush=True))
    return 0 if summary["valid"] else 1


def designs(root: Path) -> dict[str, dict]:
    from lob.v09 import confirm as CF, protocol as pr
    protocol, state = pr.state_of(root)
    out = {}
    for name in sorted(state.designs):
        if name.startswith("confirm-"):
            doc, _ = CF.substitute(pr.sealed_design(root, name), protocol, state)
            out[name] = doc
    return out


def needed_datasets(root: Path, docs: dict[str, dict]) -> list[str]:
    from lob.v09 import protocol as pr
    protocol = pr.load_protocol(root / pr.PROTOCOL_PATH)
    ids: list[str] = []

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and x not in ids:
            try:
                d = pr.dataset_declaration(protocol, x)
            except Exception:                                      # noqa: BLE001 - non-dataset strings
                return
            if d["role"] in pr.GUARDED_ROLES:
                ids.append(x)

    for name, doc in docs.items():
        if name != "confirm-data":
            walk({k: v for k, v in doc.items() if k != "reads"})
    return ids


def step_data(root: Path, ns, design: str = "confirm-data", namespace: str = "confirm", base: str = CONFIRM_BASE):
    from lob.v09 import protocol as pr
    from lob.v09.data import daycache
    for _ in range(8):
        ids = needed_datasets(root, designs(root)) if namespace == "confirm" else vault_datasets(root, design)
        _, state = pr.state_of(root)
        missing = [d for d in ids if not daycache.cache_path(root, d).is_file()]
        if not missing:
            log(f"all {len(ids)} datasets prepared")
            return
        out = fresh(root, base, "data-prep")
        args = ["validate-market-data-v09", "--namespace", namespace, "--design", design, "--out", out,
                "--workers", str(ns.workers)]
        for d in missing:
            args += ["--dataset", d]
        if cli(root, args, root / "data/v09" / f"{Path(out).name}.log", ns.min_free_gb) != 0:
            raise SystemExit(f"data preparation run {out} failed (ledgered); fix and rerun")
    raise SystemExit("data preparation did not converge after 8 rounds (inspect the ledger attempts)")


def vault_datasets(root: Path, design: str = "final-vault") -> list[str]:
    """The sealed vault design's datasets with the registered backup rule re-applied (failed slots -> backups)."""
    from lob.v09 import confirm as CF, protocol as pr
    doc, _ = CF.substitute(pr.sealed_design(root, design), *pr.state_of(root))
    return list(doc["datasets"])


def step_studies(root: Path, ns) -> None:
    docs = designs(root)
    names = [n for n in ORDER if n in docs] + sorted(n for n in docs if n not in ORDER and n != "confirm-data")
    for name in names:
        doc = docs[name]
        out = target(root, CONFIRM_BASE, name)
        if out is None:
            log(f"{name}: sealed")
            continue
        study = doc["study"]
        args = [study] + (["confirm"] if study in PHASED else []) + ["--namespace", "confirm", "--design", name,
                                                                     "--out", out, "--workers", str(ns.workers)]
        if study == "multifidelity-agreement-v09" and "confirm-bounds" in docs:
            args += ["--input", f"bounds={latest(root, CONFIRM_BASE, 'confirm-bounds')}"]
        rc = cli(root, args, root / "data/v09" / f"{Path(out).name}.log", ns.min_free_gb)
        if rc != 0:
            raise SystemExit(f"{name} failed (exit {rc}); record the attempt and rerun")


def latest(root: Path, base: str, name: str) -> str:
    k, found = 1, None
    while (root / (rel := f"{base}/{name}" + ("" if k == 1 else f"-{k}"))).exists():
        if sealed_ok(root, rel):
            found = rel
        k += 1
    if found is None:
        raise SystemExit(f"no sealed run of {name}")
    return found


def step_transfer(root: Path, ns) -> None:
    docs = designs(root)
    out = target(root, CONFIRM_BASE, "confirm-transfer")
    if out is None:
        return log("transfer: sealed")
    args = ["transfer-study-v09", "--namespace", "confirm", "--design", "confirm-bounds", "--out", out,
            "--input", f"bounds={latest(root, CONFIRM_BASE, 'confirm-bounds')}"]
    if "confirm-bounds-late" in docs:
        args += ["--input", f"late={latest(root, CONFIRM_BASE, 'confirm-bounds-late')}"]
    if cli(root, args, root / "data/v09" / f"{Path(out).name}.log", ns.min_free_gb) != 0:
        raise SystemExit("transfer failed")


def step_close(root: Path, ns) -> None:
    from lob.v09 import confirm as CF
    docs = designs(root)
    runs = {n: latest(root, CONFIRM_BASE, n) for n in docs if n != "confirm-data"}
    if "confirm-bounds" in docs:
        runs["confirm-transfer"] = latest(root, CONFIRM_BASE, "confirm-transfer")
    print(json.dumps(CF.close(root, runs)["index"]))


def step_vault(root: Path, ns) -> None:
    from lob.v09 import confirm as CF, protocol as pr
    _, state = pr.state_of(root)
    if "final-vault" not in state.designs:
        d = CF.final_design(root)
        pr.seal_design(root, "final-vault", d["document"], reads=d["reads"], namespaces=d["namespaces"],
                       reason="the single final-vault design (after close_confirmatory)")
        pr.open_namespace(root, "final", design="final-vault", reason="final seeds for the vault design")
    step_data(root, ns, design="final-vault", namespace="final", base=FINAL_BASE)
    out = target(root, FINAL_BASE, "vault-bounds")
    if out is None:
        return log("vault: sealed")
    args = ["counterfactual-bounds-v09", "confirm", "--namespace", "final", "--design", "final-vault", "--out", out,
            "--workers", str(ns.workers)]
    if cli(root, args, root / "data/v09" / f"{Path(out).name}.log", ns.min_free_gb) != 0:
        raise SystemExit("vault run failed")


def step_final(root: Path, ns) -> None:
    out = target(root, FINAL_BASE, "final-analysis")
    if out is None:
        return log("final: sealed")
    args = ["final-analysis-v09", "--namespace", "final", "--design", "final-vault", "--out", out,
            "--input", f"vault={latest(root, FINAL_BASE, 'vault-bounds')}"]
    for item in ns.input or []:
        args += ["--input", item]
    if cli(root, args, root / "data/v09" / f"{Path(out).name}.log", ns.min_free_gb) != 0:
        raise SystemExit("final analysis failed")


def dryrun_document(doc: dict, explore: dict[str, list[str]]) -> tuple[dict, list[str]]:
    """A small explore-day copy of a confirmatory design (engineering check of the confirm branch only)."""
    tr = {s: v[:2] for s, v in explore.items()}
    te = {s: v[-1:] for s, v in explore.items()}
    seq = {s: sorted(v[-3:], key=lambda d: d[-10:]) for s, v in explore.items()}
    out = {k: v for k, v in doc.items() if k not in ("test_datasets", "train_datasets", "plan", "sequences",
                                                     "datasets", "calibration_days", "reads")}
    if "test_datasets" in doc:
        out.update(test_datasets=[d for v in te.values() for d in v], train_datasets=[d for v in tr.values() for d in v])
    if "plan" in doc:
        out["plan"] = {s: {"train": tr[s], "test": te[s]} for s in explore}
    if "sequences" in doc:
        out["sequences"] = seq
    if "datasets" in doc and "train_datasets" in doc:
        out.update(datasets=[d for v in tr.values() for d in v] + [d for v in te.values() for d in v],
                   train_datasets=[d for v in tr.values() for d in v])
    elif "datasets" in doc:
        out["datasets"] = [d for v in te.values() for d in v]
    if "calibration_days" in doc:
        out["calibration_days"] = [v[0] for v in explore.values()]
    reads = sorted({d for v in explore.values() for d in v})
    return {**out, "dryrun": "engineering check of the confirm branch on explore days; never evidence"}, reads


def step_dryrun(root: Path, ns) -> None:
    from lob.v09 import confirm as CF, protocol as pr
    from lob.v09.artifacts import read_run
    protocol, state = pr.state_of(root)
    explore = CF.explore_days(protocol, state)
    registry = json.loads((root / pr.CONFIRMATORY_PATH).read_text(encoding="utf-8"))
    decisions = read_run(root / state.promotion["promotion_run"])["result"]["promotion"]["decisions"]
    built = {n: d["document"] for n, d in CF.build(root, registry, decisions).items()}
    failed: list[str] = []
    built_names = {"dryrun-" + n.removeprefix("confirm-") for n in built}
    for name, doc in built.items():
        if name in ("confirm-data",) or doc["study"] in ("transfer-study-v09",):
            continue
        dname = "dryrun-" + name.removeprefix("confirm-")
        ddoc, reads = dryrun_document(doc, explore)
        if dname not in pr.state_of(root)[1].designs:
            pr.seal_design(root, dname, ddoc, reads=reads, namespaces=[],
                           reason=f"engineering dry run of the {doc['study']} confirm branch on explore days")
        out = target(root, "results/v09/dryrun", dname)
        if out is None:
            log(f"{dname}: sealed")
            continue
        study = doc["study"]
        args = [study] + (["confirm"] if study in PHASED else []) + ["--namespace", "explore", "--design", dname,
                                                                     "--out", out, "--workers", str(ns.workers)]
        if study == "multifidelity-agreement-v09":
            args += ["--input", "bounds=results/v09/explore/bounds-pilot-2"]
        rc = cli(root, args, root / "data/v09" / f"{Path(out).name}.log", ns.min_free_gb)
        if rc != 0:
            failed.append(dname)
    tr = target(root, "results/v09/dryrun", "dryrun-transfer")
    if tr is not None and not failed:
        args = ["transfer-study-v09", "--namespace", "explore", "--design", "dryrun-bounds", "--out", tr,
                "--input", "bounds=" + latest(root, "results/v09/dryrun", "dryrun-bounds")]
        if "dryrun-bounds-late" in built_names:
            args += ["--input", "late=" + latest(root, "results/v09/dryrun", "dryrun-bounds-late")]
        if cli(root, args, root / "data/v09" / f"{Path(tr).name}.log", ns.min_free_gb) != 0:
            failed.append("dryrun-transfer")
    if failed:
        raise SystemExit(f"dry runs failed: {failed}; fix with a regression test and rerun before sealing")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--min-free-gb", type=float, default=8.0)
    p.add_argument("--input", action="append")
    p.add_argument("step", choices=("seal", "dryrun", "data", "studies", "transfer", "close", "vault", "final"))
    ns = p.parse_args()
    root = ns.root.resolve()
    if ns.step == "seal":
        from lob.v09 import confirm as CF
        print(json.dumps(CF.seal_all(root), indent=1))
    else:
        {"dryrun": step_dryrun, "data": step_data, "studies": step_studies, "transfer": step_transfer, "close": step_close,
         "vault": step_vault, "final": step_final}[ns.step](root, ns)


if __name__ == "__main__":
    main()
