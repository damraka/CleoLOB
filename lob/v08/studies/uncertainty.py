"""``uncertainty-study-v08``: uncertainty decomposition over the plausible world design (M18, H5).

Base worlds: the registered selection posterior draws (model family x parameter sample) plus calibration-window
variants (the Poisson family refitted on four disjoint blocks of development windows). Each base world gets
``variants`` copies with a balanced assignment of queue rule (uniform/front/back), latency (0/50/250 ms) and hidden
liquidity (0/0.5). All worlds share the same seeds. Per policy and per registered pair, the variance is split into
aleatoric (seed) and epistemic parts, and the epistemic part into first-order shares of model family, parameter
sample, calibration window, queue, latency and hidden liquidity. The regime source is not varied in this design
(reported as NOT_EVALUABLE here; regime effects are studied in the selection and drift benchmarks).
"""
from __future__ import annotations

from dataclasses import replace

from .. import realdata
from ..execution.simulator import LatencyProfile
from ..generators import PoissonGenerator
from ..inference.adaptive import concat
from ..uncertainty import summarize_policy_costs
from ..worlds import World
from . import common as C

DEFAULTS = {
    "tiny": {"worlds": 3, "variants": 1, "seeds": 2, "boot": 100, "windows": 4, "mandates": {"primary": {}}},
    "small": {"worlds": 8, "variants": 2, "seeds": 6, "boot": 300, "windows": 8, "mandates": {"primary": {}}},
    "medium": {"worlds": None, "variants": 3, "seeds": 16, "boot": 1000, "windows": 24,
               "mandates": {"primary": {}, "large": {"qty": 400}}},
}
POLICIES = ("twap", "mpc", "adaptive_twap", "heuristic", "pov")
PAIRS = (("twap", "mpc"), ("twap", "adaptive_twap"), ("twap", "heuristic"), ("twap", "pov"))
QUEUE, LATENCY_MS, HIDDEN = ("uniform", "front", "back"), (0, 50, 250), (0.0, 0.5)
SOURCES = ["model_family", "parameter", "calibration_window", "queue", "latency", "hidden_liquidity"]
ALPHA = 0.05 / len(PAIRS)


def _variant(world: World, k: int, v: int) -> tuple[World, dict]:
    q, lat, hid = QUEUE[(k + v) % 3], LATENCY_MS[(k + 2 * v) % 3], HIDDEN[(k + v) % 2]
    book = world.book.replace(cancel_position=q, hidden_fraction=hid)
    gd = world.generator.to_dict()
    gd["book"] = book.to_dict()
    from ..generators.base import MarketGenerator
    w = replace(world, world_id=f"{world.world_id}|v{v}", generator=MarketGenerator.from_dict(gd), book=book,
                latency=LatencyProfile.uniform(lat / 1000.0) if lat else None)
    return w, {"level_queue": q, "level_latency": str(lat), "level_hidden_liquidity": str(hid)}


def run(args) -> dict:
    root = C.root_of(args)
    d = C.design(args, DEFAULTS)
    out = C.start(args)
    wrun = C.input_run(args, "worlds")
    prun = C.input_run(args, "posterior")
    post = C.read_result(prun)
    base = C.load_world_set(wrun, "selection").worlds[1:]
    if d.get("worlds"):
        base = base[: d["worlds"]]
    levels: dict[str, dict] = {}
    for w in base:
        levels[w.world_id] = {"level_model_family": w.family, "level_parameter": f"{w.family}:{w.posterior_sample_id}",
                              "level_calibration_window": "posterior"}
    windows = realdata.registered_windows()[: d["windows"]]
    frames, _ = realdata.load_frames("deribit-eth-perp-2020-10-01", use="develop", design="uncertainty-study",
                                     purpose="calibration-window variants", root=root, windows=windows)
    book = C.generator_from(post["fits"]["poisson"]).book
    blocks = max(1, len(frames) // 4)
    for k in range(4):
        part = frames[k * blocks:(k + 1) * blocks]
        if not part:
            continue
        g = PoissonGenerator(book=book).fit(concat(part))
        w = World(f"poisson-window{k}", g, book, label="CALIBRATED", meta={"windows": [f.meta.get("window_start_s")
                                                                                   for f in part]})
        base.append(w)
        levels[w.world_id] = {"level_model_family": "poisson", "level_parameter": "point",
                              "level_calibration_window": f"block{k}"}
    worlds, extras = [], {}
    for k, w in enumerate(base):
        for v in range(d["variants"]):
            vw, lv = _variant(w, k, v)
            worlds.append(vw)
            extras[vw.world_id] = {**levels[w.world_id], **lv}
    registered = C.registered_policies(root)
    result = {"mandates": {}, "design": {"base_worlds": len(base), "worlds": len(worlds), "sources": SOURCES,
                                         "regime_source": "NOT_EVALUABLE in this design (not varied)"}}
    for mname, overrides in d["mandates"].items():
        m = C.mandate(root, **overrides)
        seeds = C.seeds(args, f"uncertainty.{mname}", d["seeds"])
        rows = C.evaluate(args, out, worlds, [(p, registered[p]) for p in POLICIES], m, seeds, tag=mname,
                          extras=extras)
        decomposition = summarize_policy_costs(rows, value="economic_cost_bps", policies=list(POLICIES),
                                               sources=SOURCES, n_boot=d["boot"], seed=args.seed, alpha=ALPHA,
                                               pairs=list(PAIRS))
        result["mandates"][mname] = {"decomposition": decomposition, "invalid_episodes": C.invalid_count(rows),
                                     "episodes": len(rows)}
    run_rel = out.relative_to(root).as_posix()
    realdata.mark_evaluated("deribit-eth-perp-2020-10-01", use="develop", design="uncertainty-study", run=run_rel,
                            root=root)
    result["labels"] = {"evidence": "SYNTHETIC", "pair_interval_alpha": ALPHA,
                        "meaning": "shares are design-dependent: they describe this registered world design"}
    return C.finish(args, out, analysis="uncertainty_study", config={"design": d}, result=C.json_safe(result),
                    datasets=["deribit-eth-perp-2020-10-01"],
                    inputs=[wrun.relative_to(root).as_posix(), prun.relative_to(root).as_posix()])
