"""Confirmatory designs (M2 -> K): built from the frozen registry, sealed before any confirmatory access.

After ``hypothesis-promotion-v09 freeze``, ``seal_all`` turns the frozen confirmatory registry into one sealed design
per confirmatory study plus the data-preparation design ``confirm-data``, all appended to the ledger in one batch
before a single confirmatory byte is read. Each design lists:

* ``reads`` - every primary slot it may use *and every registered backup slot of the same streams* (a failed slot is
  replaced only by the next unused backup in registered order; the firewall requires the design to list it);
* ``namespaces`` - ``["confirm"]`` (opened through the ledger right after sealing);
* the study-specific structure (``test_datasets`` / ``train_datasets``, ``plan``, ``sequences``, ``datasets``,
  ``calibration_days``) built from the registered rule: each promoted hypothesis reads the first ``confirm_units``
  primary slots of its pool (instrument-day units split evenly over the Bybit BTCUSDT / ETHUSDT streams; carried
  conclusions read BTCUSDT slots, H33 ETHUSDT, H34 the Binance L1 streams, H32 ``confirm_late`` BTCUSDT).

``substitute`` applies the backup rule at run time from the ledger (deterministic): within one design, the failed
primaries of a stream (in slot order) map to that stream's usable backups (in slot order) not already listed.
Training sets of confirmatory designs are the explore days (never confirmatory data).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from ..preregistration import ProtocolError
from . import protocol as pr

BYBIT = ("bybit-linear:BTCUSDT", "bybit-linear:ETHUSDT")
BINANCE = ("binance-um:BTCUSDT", "binance-um:ETHUSDT")

# study command -> (hypotheses, design kind)
STUDIES = {
    "hybrid-twin-v09": (("H1", "H12"), "folds"),
    "neural-point-process-v09": (("H2", "H5"), "folds"),
    "sequence-generator-v09": (("H3",), "folds"),
    "latent-dynamics-v09": (("H4",), "folds"),
    "fill-study-v09": (("H6",), "plan"),
    "survival-study-v09": (("H7",), "datasets"),
    "impact-study-v09": (("H10",), "datasets"),
    "inference-v09": (("H14",), "sequences"),
    "meta-adaptation-v09": (("H15",), "sequences"),
    "continual-calibration-v09": (("H16",), "sequences"),
    "conformal-study-v09": (("H17",), "folds"),
    "representation-transfer-v09": (("H35",), "folds"),
    "multifidelity-agreement-v09": (("H8", "H18", "H19", "H22", "H25", "H29", "H30", "H31"), "plan"),
    "strategic-response-v09": (("H11",), "synthetic"),
    "active-calibration-v09": (("H13",), "synthetic"),
    "abstention-study-v09": (("H20",), "synthetic"),
    "dro-study-v09": (("H21",), "synthetic"),
    "mpc-study-v09": (("H24",), "synthetic"),
    "rare-event-study-v09": (("H28",), "synthetic"),
    "offline-rl-v09": (("H26", "H27"), "synthetic"),
}
BOUNDS_HYPOTHESES = ("H9", "H23")             # conclusion C5 / C1 records on Bybit BTCUSDT + ETHUSDT days
TRANSFER = {"H32": ("confirm_late", ("bybit-linear:BTCUSDT",)), "H33": ("confirm", ("bybit-linear:ETHUSDT",)),
            "H34": ("confirm", BINANCE)}


def slots(protocol: dict, pool: str, stream: str) -> tuple[list[str], list[str]]:
    """(primaries, backups) of one stream of a pool, each in registered slot order."""
    venue, instrument = stream.split(":")
    rows = sorted((d for d in pr.datasets_by(protocol, pool=pool) if d["venue"] == venue
                   and d["instrument"] == instrument), key=lambda d: d["slot"])
    count = protocol["pools"][pool]["count"]
    return [d["id"] for d in rows if d["slot"] < count], [d["id"] for d in rows if d["slot"] >= count]


def first(protocol: dict, pool: str, stream: str, n: int) -> list[str]:
    primaries, _ = slots(protocol, pool, stream)
    if n > len(primaries):
        raise ProtocolError(f"{pool}/{stream}: {n} units exceed the {len(primaries)} registered primary slots")
    return primaries[:n]


def per_stream(units: int, streams=BYBIT) -> int:
    return int(math.ceil(units / len(streams)))


def explore_days(protocol: dict, state: pr.LedgerState, streams=BYBIT) -> dict[str, list[str]]:
    from .data import sources
    n = protocol["pools"]["explore"]["count"]
    return {s: sources.resolve(protocol, state, "explore", s, n)["datasets"] for s in streams}


def _with_backups(protocol: dict, pool: str, streams) -> list[str]:
    out = []
    for s in streams:
        p, b = slots(protocol, pool, s)
        out += p + b
    return out


def build(root: Path, registry: dict, decisions: dict | None = None) -> dict[str, dict]:
    """name -> {"document", "reads", "namespaces"} for every confirmatory design (not yet sealed).

    ``decisions`` (the promotion run's per-candidate decisions) supplies choices the pilots fixed for the
    confirmatory design: the survival model of H7 and the continual method of H16.
    """
    decisions = decisions or {}
    protocol, state = pr.state_of(root)
    explore = explore_days(protocol, state)
    explore_all = [d for s in BYBIT for d in explore[s]]
    promoted = registry["hypotheses"]
    units = registry.get("confirm_units", {})
    designs: dict[str, dict] = {}

    def add(name, document, pool, streams, used):
        reads = sorted(set(used) | set(_with_backups(protocol, pool, streams)))
        designs[name] = {"document": {**document, "pool": pool, "streams": list(streams),
                                      "rule": "first registered primary slots; failed slots -> ordered backups"},
                         "reads": reads, "namespaces": ["confirm"]}

    for study, (hyps, kind) in STUDIES.items():
        mine = [h for h in hyps if h in promoted]
        if not mine:
            continue
        n = max(int(units[h]) for h in mine)
        name = f"confirm-{study.removesuffix('-v09')}"
        doc = {"study": study, "hypotheses": mine, "confirm_units": {h: int(units[h]) for h in mine}}
        if kind == "synthetic":
            # synthetic world clusters: calibration days are explore days (cycled); fresh worlds come from the
            # confirm seed namespace, so no confirmatory market data are read
            cal = [explore_all[k % len(explore_all)] for k in range(n)]
            designs[name] = {"document": {**doc, "calibration_days": cal, "clusters": n,
                                          "rule": "one world cluster per index; calibration day cycles over the "
                                                  "explore days; seeds from the confirm namespace"},
                             "reads": [], "namespaces": ["confirm"]}
            continue
        k = per_stream(n) + (1 if kind == "sequences" else 0)
        test = {s: first(protocol, "confirm", s, k) for s in BYBIT}
        if kind == "folds":
            doc.update({"test_datasets": [d for s in BYBIT for d in test[s]], "train_datasets": explore_all})
        elif kind == "plan":
            doc["plan"] = {s: {"train": explore[s], "test": test[s]} for s in BYBIT}
        elif kind == "sequences":
            doc["sequences"] = {s: sorted(test[s], key=lambda d: d[-10:]) for s in BYBIT}    # chronological
        elif kind == "datasets":
            doc.update({"datasets": explore_all + [d for s in BYBIT for d in test[s]], "train_datasets": explore_all})
        if "H7" in mine:
            doc["model"] = decisions["H7"]["pilot"]["selected_model"]
        if "H16" in mine:
            doc["method"] = decisions["H16"]["pilot"]["selected_method"]
        add(name, doc, "confirm", BYBIT, [d for v in test.values() for d in v])

    # bounded replay: carried conclusions (BTCUSDT), H9/H23 (BTC+ETH), H33 (ETHUSDT), H34 (Binance L1)
    carried = registry.get("carried_conclusions", {})
    need = {s: 0 for s in BYBIT + BINANCE}
    if carried:
        need["bybit-linear:BTCUSDT"] = max(int(c["confirm_units"]) for c in carried.values())
    for h in BOUNDS_HYPOTHESES:
        if h in promoted:
            for s in BYBIT:
                need[s] = max(need[s], per_stream(int(units[h])))
    if "H33" in promoted:
        need["bybit-linear:ETHUSDT"] = max(need["bybit-linear:ETHUSDT"], int(units["H33"]))
    if "H34" in promoted:
        for s in BINANCE:
            need[s] = max(need[s], per_stream(int(units["H34"]), BINANCE))
    streams = [s for s in BYBIT + BINANCE if need[s] > 0]
    if streams:
        datasets = [d for s in streams for d in first(protocol, "confirm", s, need[s])]
        add("confirm-bounds", {"study": "counterfactual-bounds-v09", "datasets": datasets, "per_stream": need,
                               "carried": sorted(carried)}, "confirm", streams, datasets)
    if "H32" in promoted:
        s = "bybit-linear:BTCUSDT"
        datasets = first(protocol, "confirm_late", s, int(units["H32"]))
        add("confirm-bounds-late", {"study": "counterfactual-bounds-v09", "datasets": datasets,
                                    "per_stream": {s: len(datasets)}, "carried": sorted(carried)},
            "confirm_late", [s], datasets)
    data_reads = sorted({r for d in designs.values() for r in d["reads"]})
    if data_reads:
        designs["confirm-data"] = {"document": {"study": "validate-market-data-v09", "datasets": data_reads,
                                                "rule": "prepare the primaries every design uses; replace failures by "
                                                        "registered backups"},
                                   "reads": data_reads, "namespaces": ["confirm"]}
    return designs


def seal_all(root: Path) -> dict:
    """Seal every confirmatory design and open the confirm namespace for each (ledger batch, before any access)."""
    protocol, state = pr.state_of(root)
    if state.promotion is None:
        raise ProtocolError("confirmatory designs are sealed only after the promotion freeze")
    guarded = [d for d in state.first_access if pr.dataset_declaration(protocol, d)["role"] in pr.CONFIRM_ROLES]
    if guarded:
        raise ProtocolError(f"confirmatory data were accessed before the designs were sealed: {guarded[:3]}")
    registry = json.loads((root / pr.CONFIRMATORY_PATH).read_text(encoding="utf-8"))
    from .artifacts import read_run
    run = read_run(root / state.promotion["promotion_run"])
    designs = build(root, registry, run["result"]["promotion"]["decisions"])
    sealed = {}
    for name, d in designs.items():
        if name in state.designs:
            continue
        e = pr.seal_design(root, name, d["document"], reads=d["reads"], namespaces=d["namespaces"],
                           reason=f"confirmatory design for {d['document']['study']} (sealed before confirmatory access)")
        sealed[name] = e["index"]
    for name in designs:
        pr.open_namespace(root, "confirm", design=name, reason=f"confirm seeds for the sealed design {name}")
    return {"sealed": sealed, "designs": sorted(designs)}


def substitute(document: dict, protocol: dict, state: pr.LedgerState) -> tuple[dict, dict]:
    """Apply the registered backup rule to a sealed design document; returns (document, {failed: backup})."""
    failed = {a["dataset"] for a in state.attempts if a.get("outcome") == "NOT_AVAILABLE" and a.get("dataset")}
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
                pr.dataset_declaration(protocol, x)
            except ProtocolError:
                return
            ids.append(x)

    walk(document)
    present = set(ids)
    mapping: dict[str, str | None] = {}
    by_stream: dict[tuple, list[dict]] = {}
    for x in ids:
        if x in failed:
            d = pr.dataset_declaration(protocol, x)
            by_stream.setdefault((d["pool"], f"{d['venue']}:{d['instrument']}"), []).append(d)
    for (pool, stream), rows in by_stream.items():
        if pool == "explore":
            continue                              # explore days are resolved by the explore slot rule
        _, backups = slots(protocol, pool, stream)
        free = [b for b in backups if b not in failed and b not in present]
        for d in sorted(rows, key=lambda r: r["slot"]):
            mapping[d["id"]] = free.pop(0) if free else None

    def apply(x):
        if isinstance(x, dict):
            return {k: apply(v) for k, v in x.items()}
        if isinstance(x, list):
            return [apply(v) for v in x if not (isinstance(v, str) and v in mapping and mapping[v] is None)]
        return mapping.get(x, x) if isinstance(x, str) else x

    return (apply(document) if mapping else document), mapping


def close(root: Path, runs: dict[str, str], *, reason: str = "every confirmatory run sealed") -> dict:
    """Append close_confirmatory with the sealed confirmatory runs (design -> run directory)."""
    from .artifacts import verify_run
    protocol, state = pr.state_of(root)
    for design, run in runs.items():
        report = verify_run(root / run, root=root)
        if not report["valid"]:
            raise ProtocolError(f"{run} does not verify: {report['issues'][:3]}")
    return pr.append_event(root / pr.LEDGER_PATH, "close_confirmatory", reason=reason, protocol=protocol, root=root,
                           payload={"runs": runs})


def final_design(root: Path) -> dict:
    """The single vault design (H36): every final-pool slot of the three vault streams, after close_confirmatory."""
    protocol, state = pr.state_of(root)
    if state.closed is None:
        raise ProtocolError("the final vault design is sealed only after close_confirmatory")
    registry = json.loads((root / pr.CONFIRMATORY_PATH).read_text(encoding="utf-8"))
    streams = protocol["pools"]["final"]["streams"]
    n = protocol["pools"]["final"]["count"]
    datasets = [d for s in streams for d in first(protocol, "final", s, n)]
    reads = sorted(set(datasets) | set(_with_backups(protocol, "final", streams)))
    doc = {"study": "counterfactual-bounds-v09", "hypotheses": ["H36"], "datasets": datasets, "pool": "final",
           "streams": streams, "carried": sorted(registry.get("carried_conclusions", {})),
           "rule": "read once, after close_confirmatory, by this design only; failed slots -> ordered backups"}
    return {"document": doc, "reads": reads, "namespaces": ["final"]}
