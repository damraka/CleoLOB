"""Machine-readable v1.0 milestone registry (M0-M159) with mechanically computed status.

Every milestone names the module(s) that implement it, the test files and test-name prefixes that exercise it, the
documents and artifacts it must produce, its dependencies and its resource class. Status is computed, never
asserted:

* ``PLANNED``      - a declared module cannot be imported;
* ``IMPLEMENTED``  - every module imports;
* ``TESTED``       - additionally every declared test file exists and contains the declared test prefixes;
* ``COMPLETE``     - additionally every declared test passed in the supplied JUnit report, every declared artifact
  and document exists, and no dependency is BLOCKED or PLANNED;
* ``BLOCKED``      - an explicit, reasoned blocker (for example an external capability that is unavailable); a
  blocked milestone is never reported complete.

``cleo v1 milestones --junit report.xml`` prints the registry with computed statuses; CI runs it after the test
suite. Applicability notes distinguish, for example, "adapter validated against recorded data" from "adapter
validated against the live provider".
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
from pathlib import Path
import xml.etree.ElementTree as ET

STATUSES = ("PLANNED", "IMPLEMENTED", "TESTED", "COMPLETE", "BLOCKED")


@dataclass(frozen=True)
class Milestone:
    id: str
    title: str
    pillar: int
    modules: tuple[str, ...]
    tests: tuple[str, ...]                   # "tests/file.py::prefix" (prefix matches test function names)
    depends: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()          # repository-relative paths that must exist
    docs: tuple[str, ...] = ()
    resources: str = "smoke"                 # smoke | small | medium | full | external
    blocked: str | None = None
    applicability: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _m(i: int, title: str, pillar: int, modules, tests, depends=(), artifacts=(), docs=(), resources="smoke",
       blocked=None, applicability="") -> Milestone:
    t = lambda x: (x,) if isinstance(x, str) else tuple(x)  # noqa: E731
    return Milestone(f"M{i}", title, pillar, t(modules), t(tests), tuple(f"M{d}" for d in depends), t(artifacts),
                     t(docs), resources, blocked, applicability)


V = "lob.v1."
T = "tests/"

MILESTONES: tuple[Milestone, ...] = (
    # A. research integrity and governance (pillar 8 infrastructure, scientific integrity)
    _m(0, "v0.9 forensic audit and limitations catalog", 6, (V + "statistics", V + "generators.classical.book"),
       (T + "test_v1_book_calibration.py::test_", T + "test_v1_statistics.py::test_"),
       artifacts=("examples/studies/v1/retrospective/dependence.json", "tools/v1_retro_dependence.py"),
       docs=("docs/v1-forensic-audit.md",)),
    _m(1, "stable v1 architecture and backward-compatibility plan", 8, (V + "sdk", V + "compat"),
       (T + "test_v1_sdk.py::test_", T + "test_v1_compat.py::test_"), (0,), docs=("docs/v1-architecture.md",)),
    _m(2, "frozen protocol v1, schema and amendment system", 6, V + "protocol.document",
       T + "test_v1_protocol.py::test_protocol", (1,), docs=("docs/v1-scientific-protocol.md",)),
    _m(3, "exploratory/calibration/confirmatory/replication/vault isolation", 6, V + "protocol.namespaces",
       T + "test_v1_protocol.py::test_namespace", (2,)),
    _m(4, "immutable consumption ledger with hash verification", 6, V + "protocol.ledger",
       T + "test_v1_protocol.py::test_ledger", (2,)),
    _m(5, "source/artifact/dataset/environment provenance graph", 8, V + "evidence.provenance",
       T + "test_v1_evidence.py::test_provenance", (4,)),
    _m(6, "registered hypotheses, endpoints and promotion criteria", 6, V + "research.hypotheses",
       T + "test_v1_research.py::test_hypotheses", (2, 7)),
    _m(7, "pre-confirmatory power and effective-sample-size framework", 6, (V + "statistics", V + "research.power"),
       (T + "test_v1_statistics.py::test_power", T + "test_v1_research.py::test_power"), (0,)),
    _m(8, "independent reproducibility and deterministic random namespaces", 8, V + "protocol.seeds",
       T + "test_v1_protocol.py::test_seeds", (3,)),
    _m(9, "incident, contamination and amendment reporting", 6, V + "protocol.ledger",
       T + "test_v1_protocol.py::test_incident", (4,)),
    # B. market data acquisition
    _m(10, "universal data-source connector interface", 1, V + "data.connectors.base",
       T + "test_v1_data.py::test_connector", (1,), docs=("docs/v1-data-sources.md",)),
    _m(11, "Bybit historical L2/MBP adapter (ob500 and ob200 eras)", 1, V + "data.connectors.bybit",
       T + "test_v1_data.py::test_bybit", (10,)),
    _m(12, "Binance depth and trades adapter with capability labels", 1, V + "data.connectors.binance",
       T + "test_v1_data.py::test_binance", (10,)),
    _m(13, "independent spot market-data source support", 1, V + "data.connectors.bybit",
       T + "test_v1_data.py::test_bybit_spot", (11,)),
    _m(14, "genuine MBO/L3 source adapter", 1, (V + "data.connectors.bitstamp", V + "data.connectors.bitstamp_capture"),
       T + "test_v1_data.py::test_bitstamp", (10,),
       applicability="validated on prospectively recorded public Bitstamp order-level messages"),
    _m(15, "streaming and chunked compressed archive ingestion", 1, V + "data.ingest",
       T + "test_v1_data.py::test_stream", (10,)),
    _m(16, "data licences, access policies and provenance metadata", 1, V + "data.sources",
       T + "test_v1_data.py::test_licence", (10,)),
    _m(17, "SHA-256 raw integrity and content-addressed cache", 1, V + "data.store",
       T + "test_v1_data.py::test_store", (15,)),
    _m(18, "registered primary/backup acquisition and failover", 1, V + "data.acquire",
       T + "test_v1_data.py::test_acquire", (17, 4)),
    _m(19, "fresh-data eligibility and consumption firewall", 1, V + "data.firewall",
       T + "test_v1_data.py::test_firewall", (4, 18), artifacts=("configs/v1/consumed-inventory.json",)),
    # C. order book reconstruction
    _m(20, "canonical event schema for heterogeneous venues", 1, V + "data.schema",
       T + "test_v1_reconstruction.py::test_schema", (10,), docs=("docs/v1-market-reconstruction.md",)),
    _m(21, "deterministic MBP/L2 reconstruction", 1, V + "data.reconstruction.l2",
       T + "test_v1_reconstruction.py::test_l2", (20,)),
    _m(22, "order-id-aware MBO/L3 reconstruction", 1, V + "data.reconstruction.mbo",
       T + "test_v1_reconstruction.py::test_mbo", (20, 14)),
    _m(23, "snapshot/delta reconciliation and gap detection", 1, V + "data.reconstruction.l2",
       T + "test_v1_reconstruction.py::test_gap", (21,)),
    _m(24, "exchange-time versus receive-time alignment", 1, V + "data.reconstruction.timing",
       T + "test_v1_reconstruction.py::test_timing", (20,)),
    _m(25, "trade/book synchronization diagnostics", 1, V + "data.reconstruction.sync",
       T + "test_v1_reconstruction.py::test_sync", (21,)),
    _m(26, "queue-position observability classification", 1, V + "data.reconstruction.observability",
       T + "test_v1_reconstruction.py::test_observability", (21, 22)),
    _m(27, "hidden/iceberg liquidity uncertainty propagation", 1, V + "data.reconstruction.hidden",
       T + "test_v1_reconstruction.py::test_hidden", (21,)),
    _m(28, "multiple plausible reconstructions for ambiguous observations", 1, V + "data.reconstruction.ensemble",
       T + "test_v1_reconstruction.py::test_ensemble", (21, 27)),
    _m(29, "independent reconstruction validation against recorded snapshots", 1, V + "data.reconstruction.validate",
       T + "test_v1_reconstruction.py::test_validate", (21, 22)),
    # D. exchange mechanics
    _m(30, "matching engine v2 with deterministic price-time priority", 2, V + "engines.matching",
       T + "test_v1_engine.py::test_matching", (20,), docs=("docs/v1-engine-validation.md",)),
    _m(31, "configurable matching-rule adapters", 2, V + "engines.rules", T + "test_v1_engine.py::test_rules", (30,)),
    _m(32, "limit/market/cancel/replace/partial-fill semantics", 2, V + "engines.matching",
       T + "test_v1_engine.py::test_orders", (30,)),
    _m(33, "quantity, tick, lot and fee accounting", 2, V + "engines.accounting",
       T + "test_v1_engine.py::test_accounting", (30,)),
    _m(34, "queue advancement and cancellation behaviour", 2, V + "engines.matching",
       T + "test_v1_engine.py::test_queue", (30,)),
    _m(35, "latency, stale messages and cancel races", 2, V + "engines.latency", T + "test_v1_engine.py::test_latency",
       (30,)),
    _m(36, "inventory, outstanding orders and self-impact feedback", 2, V + "engines.agent",
       T + "test_v1_engine.py::test_inventory", (30, 33)),
    _m(37, "event-sourced snapshot/recovery", 2, V + "engines.journal", T + "test_v1_engine.py::test_journal", (30,)),
    _m(38, "slow reference-engine differential testing", 2, V + "engines.reference",
       T + "test_v1_engine.py::test_differential", (30,)),
    _m(39, "exchange invariants and property-based fuzzing", 2, V + "engines.invariants",
       T + "test_v1_engine.py::test_fuzz", (30, 38)),
    # E. performance
    _m(40, "typed, benchmarked Python reference core", 2, V + "engines.reference",
       T + "test_v1_performance.py::test_reference", (38,)),
    _m(41, "optional high-performance execution backend", 2, V + "engines.accelerated",
       T + "test_v1_performance.py::test_accelerated", (40,)),
    _m(42, "deterministic compatibility tests between engines", 2, V + "engines.accelerated",
       T + "test_v1_performance.py::test_compat", (41,)),
    _m(43, "columnar Arrow/Parquet data pipeline", 2, V + "data.columnar", T + "test_v1_performance.py::test_columnar",
       (20,)),
    _m(44, "memory-mapped and streaming replay", 2, V + "engines.replay", T + "test_v1_performance.py::test_mmap",
       (43,)),
    _m(45, "vectorized Monte Carlo episode scheduling", 2, V + "engines.batch", T + "test_v1_performance.py::test_batch",
       (41,)),
    _m(46, "optional GPU-accelerated batch simulation", 2, V + "engines.gpu", T + "test_v1_performance.py::test_gpu",
       (45,), resources="small", applicability="CUDA path exercised only where a CUDA device and torch build exist"),
    _m(47, "CPU fallback with reproducible numerical semantics", 2, V + "engines.gpu",
       T + "test_v1_performance.py::test_cpu_fallback", (46,)),
    _m(48, "multi-worker deterministic checkpoint/resume", 8, V + "experiments.runner",
       T + "test_v1_experiments.py::test_runner", (8,)),
    _m(49, "performance benchmark suite", 2, V + "engines.benchmark", T + "test_v1_performance.py::test_benchmark",
       (40, 41, 43), artifacts=("examples/studies/v1/benchmarks/engine-benchmark.json",)),
    # F. microstructure intelligence
    _m(50, "canonical microstructure feature store", 1, V + "microstructure.features",
       T + "test_v1_microstructure.py::test_features", (20,)),
    _m(51, "order-flow imbalance and multi-level pressure", 1, V + "microstructure.flow",
       T + "test_v1_microstructure.py::test_ofi", (50,)),
    _m(52, "spread, depth, resiliency and refill metrics", 1, V + "microstructure.liquidity",
       T + "test_v1_microstructure.py::test_resiliency", (50,)),
    _m(53, "liquidity droughts, shocks and cancel storms", 1, V + "microstructure.events",
       T + "test_v1_microstructure.py::test_shocks", (52,)),
    _m(54, "market state and regime ontology", 1, V + "regimes.ontology", T + "test_v1_regimes.py::test_ontology",
       (50,)),
    _m(55, "regime discovery and clustering", 1, V + "regimes.discovery", T + "test_v1_regimes.py::test_discovery",
       (54,)),
    _m(56, "structural-break and online drift detection", 1, V + "regimes.drift", T + "test_v1_regimes.py::test_drift",
       (50,)),
    _m(57, "multiscale dependence and long memory", 1, V + "microstructure.memory",
       T + "test_v1_microstructure.py::test_memory", (50,)),
    _m(58, "intraday seasonality and calendar controls", 1, V + "microstructure.seasonality",
       T + "test_v1_microstructure.py::test_seasonality", (50,)),
    _m(59, "microstructure fingerprint and similarity engine", 1, V + "microstructure.fingerprint",
       T + "test_v1_microstructure.py::test_fingerprint", (51, 52, 57)),
    # G. multi-fidelity simulation
    _m(60, "standardized fidelity API", 3, V + "twins.api", T + "test_v1_twins.py::test_api", (30,)),
    _m(61, "F0 observational evidence reader", 3, V + "twins.observational", T + "test_v1_twins.py::test_f0", (60,)),
    _m(62, "F1 exact-observation historical replay", 3, V + "twins.replay", T + "test_v1_twins.py::test_f1", (60, 21)),
    _m(63, "F2 bounded counterfactual replay", 3, V + "twins.bounded", T + "test_v1_twins.py::test_f2", (62,)),
    _m(64, "F3 hybrid historical twin", 3, V + "twins.hybrid", T + "test_v1_twins.py::test_f3", (63, 105)),
    _m(65, "F4 classical generative simulator", 3, V + "twins.classical", T + "test_v1_twins.py::test_f4", (60, 70)),
    _m(66, "F5 neural generative simulator", 3, V + "twins.neural", T + "test_v1_twins.py::test_f5", (60, 82)),
    _m(67, "F6 strategic multi-agent simulator", 3, V + "twins.strategic", T + "test_v1_twins.py::test_f6", (60, 100)),
    _m(68, "cross-fidelity scenario and policy comparison", 3, V + "twins.compare", T + "test_v1_twins.py::test_compare",
       (61, 62, 63, 64, 65, 66, 67)),
    _m(69, "fidelity-specific claim permissions and capability gates", 3, V + "twins.permissions",
       T + "test_v1_twins.py::test_permissions", (60, 133)),
    # H. classical generative models
    _m(70, "corrected state-dependent spread-improvement calibration", 3, V + "generators.classical.book",
       T + "test_v1_book_calibration.py::test_", (0,), docs=("docs/v1-generators.md",)),
    _m(71, "empirical event-intensity and size baseline", 3, V + "generators.classical.empirical",
       T + "test_v1_generators.py::test_empirical", (70,)),
    _m(72, "marked Hawkes process family", 3, V + "generators.point_process.hawkes",
       T + "test_v1_generators.py::test_marked_hawkes", (71,)),
    _m(73, "queue-reactive dynamics v2", 3, V + "generators.classical.queue_reactive",
       T + "test_v1_generators.py::test_queue_reactive", (70,)),
    _m(74, "Markov/semi-Markov regime switching", 3, V + "generators.classical.regime",
       T + "test_v1_generators.py::test_regime", (71,)),
    _m(75, "state-conditioned liquidity replenishment", 3, V + "generators.classical.replenishment",
       T + "test_v1_generators.py::test_replenishment", (70,)),
    _m(76, "cancellation and queue-survival dynamics", 3, V + "generators.classical.cancellation",
       T + "test_v1_generators.py::test_cancellation", (70,)),
    _m(77, "nonlinear state-dependent transition model", 3, V + "generators.classical.transition",
       T + "test_v1_generators.py::test_transition", (71,)),
    _m(78, "Bayesian classical generator ensemble", 3, V + "generators.classical.ensemble",
       T + "test_v1_generators.py::test_ensemble", (71, 72, 73, 74)),
    _m(79, "classical realism/identifiability benchmark", 3, V + "realism.benchmark",
       T + "test_v1_realism.py::test_classical_benchmark", (78, 132)),
    # I. neural generative models
    _m(80, "neural temporal point processes", 3, V + "generators.neural.tpp", T + "test_v1_neural.py::test_tpp", (71,),
       resources="small"),
    _m(81, "transformer Hawkes / marked event transformer", 3, V + "generators.neural.transformer",
       T + "test_v1_neural.py::test_transformer", (80,), resources="small"),
    _m(82, "causal autoregressive LOB event generator", 3, V + "generators.neural.autoregressive",
       T + "test_v1_neural.py::test_autoregressive", (80,), resources="small"),
    _m(83, "state-conditioned recurrent generator baseline", 3, V + "generators.neural.recurrent",
       T + "test_v1_neural.py::test_recurrent", (80,), resources="small"),
    _m(84, "neural state-space market dynamics", 3, V + "generators.neural.state_space",
       T + "test_v1_neural.py::test_state_space", (83,), resources="small"),
    _m(85, "latent SDE stochastic latent model", 3, V + "generators.neural.latent_sde",
       T + "test_v1_neural.py::test_latent_sde", (84,), resources="small"),
    _m(86, "conditional normalizing flow", 3, V + "generators.neural.flow", T + "test_v1_neural.py::test_flow", (80,),
       resources="small"),
    _m(87, "conditional flow-matching / diffusion", 3, V + "generators.neural.diffusion",
       T + "test_v1_neural.py::test_diffusion", (86,), resources="small"),
    _m(88, "neural hybrid replay response model", 3, V + "generators.neural.response",
       T + "test_v1_neural.py::test_response", (64,), resources="small"),
    _m(89, "neural validity, uncertainty and failure envelopes", 3, V + "generators.neural.envelope",
       T + "test_v1_neural.py::test_envelope", (80, 81, 82, 83, 84, 85, 86, 87)),
    # J. calibration and inference
    _m(90, "unified calibration objective and API", 4, V + "calibration.api", T + "test_v1_calibration.py::test_api",
       (71,), docs=("docs/v1-inference.md",)),
    _m(91, "realism-aware calibration with train/test separation", 4, V + "calibration.realism",
       T + "test_v1_calibration.py::test_realism", (90, 132)),
    _m(92, "sequential Bayesian inference / SBI", 4, V + "inference.sbi", T + "test_v1_calibration.py::test_sbi", (90,)),
    _m(93, "simulation-based calibration and rank diagnostics", 4, V + "inference.sbc",
       T + "test_v1_calibration.py::test_sbc", (92,)),
    _m(94, "hierarchical temporal and cross-market calibration", 4, V + "calibration.hierarchical",
       T + "test_v1_calibration.py::test_hierarchical", (90,)),
    _m(95, "active simulation / expected information gain", 4, V + "inference.active",
       T + "test_v1_calibration.py::test_active", (92,)),
    _m(96, "parameter and model-class identifiability", 4, V + "inference.identifiability",
       T + "test_v1_calibration.py::test_identifiability", (92,)),
    _m(97, "posterior predictive checks", 4, V + "inference.ppc", T + "test_v1_calibration.py::test_ppc", (92,)),
    _m(98, "calibration half-life and drift recovery", 4, V + "calibration.drift",
       T + "test_v1_calibration.py::test_half_life", (90, 56)),
    _m(99, "decision-focused calibration and uncertainty decomposition", 4, V + "calibration.decision",
       T + "test_v1_calibration.py::test_decision", (90, 110)),
    # K. agents, impact and liquidity
    _m(100, "multi-agent ecology runtime", 5, V + "ecology.runtime", T + "test_v1_ecology.py::test_runtime", (30,)),
    _m(101, "inventory-aware market makers", 5, V + "ecology.agents", T + "test_v1_ecology.py::test_market_maker",
       (100,)),
    _m(102, "momentum/mean-reversion/noise proxy agents", 5, V + "ecology.agents", T + "test_v1_ecology.py::test_proxy",
       (100,)),
    _m(103, "strategic cancellation and liquidity response", 5, V + "ecology.agents",
       T + "test_v1_ecology.py::test_strategic_cancel", (100,)),
    _m(104, "self-impact and endogenous feedback", 5, V + "ecology.runtime", T + "test_v1_ecology.py::test_feedback",
       (100, 36)),
    _m(105, "temporary/transient/persistent impact kernels", 5, V + "impact.kernels",
       T + "test_v1_impact.py::test_kernels", (50,)),
    _m(106, "state-dependent impact with uncertainty bounds", 5, V + "impact.state",
       T + "test_v1_impact.py::test_state", (105,)),
    _m(107, "cross-instrument impact", 5, V + "impact.cross", T + "test_v1_impact.py::test_cross", (105,)),
    _m(108, "queue survival and fill-hazard models", 5, V + "queue.survival", T + "test_v1_queue.py::test_survival",
       (26,), docs=("docs/v1-queue-and-fills.md",)),
    _m(109, "post-fill markout and adverse selection", 5, V + "queue.markout", T + "test_v1_queue.py::test_markout",
       (108,)),
    # L. constrained execution
    _m(110, "TWAP/VWAP/POV/Almgren-Chriss baselines", 5, V + "execution.baselines",
       T + "test_v1_execution.py::test_baselines", (30,), docs=("docs/v1-execution.md",)),
    _m(111, "completion-aware adaptive execution", 5, V + "execution.adaptive",
       T + "test_v1_execution.py::test_completion", (110,)),
    _m(112, "liquidity-sensitive and imbalance-aware policies", 5, V + "execution.adaptive",
       T + "test_v1_execution.py::test_liquidity", (110, 51)),
    _m(113, "stochastic MPC v3", 5, V + "optimization.mpc", T + "test_v1_execution.py::test_mpc", (110,)),
    _m(114, "chance-constrained MPC", 5, V + "optimization.mpc", T + "test_v1_execution.py::test_chance", (113,)),
    _m(115, "CVaR / tail-risk MPC", 5, V + "optimization.mpc", T + "test_v1_execution.py::test_cvar", (113,)),
    _m(116, "distributionally robust MPC", 5, V + "optimization.mpc", T + "test_v1_execution.py::test_dro", (113,)),
    _m(117, "multi-objective and lexicographic optimization", 5, V + "optimization.objectives",
       T + "test_v1_execution.py::test_lexicographic", (113,)),
    _m(118, "meta-policy, mixture-of-experts and contextual fallback", 5, V + "execution.meta",
       T + "test_v1_execution.py::test_meta", (110, 113)),
    _m(119, "adversarial stress and failure envelopes", 5, V + "stress.adversarial",
       T + "test_v1_execution.py::test_stress", (110, 113)),
    # M. learning
    _m(120, "Gymnasium-compatible research environment", 5, V + "rl.env", T + "test_v1_rl.py::test_env", (110,),
       docs=("docs/v1-learning.md",)),
    _m(121, "behaviour-cloning benchmark", 5, V + "rl.offline", T + "test_v1_rl.py::test_bc", (120,)),
    _m(122, "conservative offline RL baseline", 5, V + "rl.offline", T + "test_v1_rl.py::test_cql", (120,)),
    _m(123, "implicit Q-learning baseline", 5, V + "rl.offline", T + "test_v1_rl.py::test_iql", (120,)),
    _m(124, "decision-transformer baseline", 5, V + "rl.sequence", T + "test_v1_rl.py::test_decision_transformer",
       (120,), resources="small"),
    _m(125, "conservative policy-improvement gating", 5, V + "rl.gating", T + "test_v1_rl.py::test_gating",
       (121, 122, 126)),
    _m(126, "offline policy evaluation and support diagnostics", 5, V + "rl.ope", T + "test_v1_rl.py::test_ope", (120,)),
    _m(127, "continual-learning comparison framework", 5, V + "continual.framework",
       T + "test_v1_continual.py::test_framework", (90,)),
    _m(128, "meta-learning and rapid regime adaptation", 5, V + "continual.meta", T + "test_v1_continual.py::test_meta",
       (127, 55)),
    _m(129, "representation transfer and model-based planning", 5, V + "continual.representation",
       T + "test_v1_continual.py::test_representation", (127,)),
    # N. statistical science and independent validation
    _m(130, "clustered/day-block inference framework", 6, V + "statistics", T + "test_v1_statistics.py::test_", (0,)),
    _m(131, "positive, negative and falsification controls", 6, V + "research.controls",
       T + "test_v1_research.py::test_controls", (130,)),
    _m(132, "realism benchmark with conditional/temporal/path/tail metrics", 4, V + "realism.metrics",
       T + "test_v1_realism.py::test_metrics", (50,), docs=("docs/v1-realism.md",)),
    _m(133, "capability-specific realism acceptance", 4, V + "realism.gate", T + "test_v1_realism.py::test_gate",
       (132,)),
    _m(134, "synthetic-hybrid-historical ranking agreement", 6, V + "twins.agreement",
       T + "test_v1_twins.py::test_agreement", (68,)),
    _m(135, "historical counterfactual partial-identification engine", 6, V + "causal.bounds",
       T + "test_v1_causal.py::test_bounds", (63,), docs=("docs/v1-causal-bounds.md",)),
    _m(136, "causal estimand registry and identification diagnostics", 6, V + "causal.estimands",
       T + "test_v1_causal.py::test_estimands", (135,)),
    _m(137, "cross-time, cross-instrument and cross-venue transfer", 6, V + "transfer.study",
       T + "test_v1_transfer.py::test_", (130,), docs=("docs/v1-transfer.md",)),
    _m(138, "independent registered replication of v0.9 conclusions", 6, V + "research.replication",
       T + "test_v1_research.py::test_replication", (6, 137), docs=("docs/v1-independent-replication.md",),
       resources="full"),
    _m(139, "final untouched multi-axis replication vault", 6, V + "research.vault", T + "test_v1_research.py::test_vault",
       (138,), resources="full"),
    # O. product, SDK and studio
    _m(140, "stable Python public API and semver contract", 8, V + "sdk", T + "test_v1_sdk.py::test_",
       (1,), docs=("docs/v1-api-reference.md",)),
    _m(141, "unified CLI with backward-compatible v0.x commands", 8, V + "cli", T + "test_v1_cli.py::test_", (140,)),
    _m(142, "local read-only research API on localhost", 7, V + "api.server", T + "test_v1_api.py::test_", (151,)),
    _m(143, "CleoLAB Studio web research application", 7, V + "studio.app", T + "test_v1_studio.py::test_app",
       (142,), docs=("docs/v1-studio-guide.md",)),
    _m(144, "depth and liquidity heatmaps", 7, V + "studio.views", T + "test_v1_studio.py::test_heatmap", (143,)),
    _m(145, "event playback, timeline scrubbing and queue view", 7, V + "studio.views",
       T + "test_v1_studio.py::test_playback", (143,)),
    _m(146, "twin side-by-side comparison and scenario explorer", 7, V + "studio.views",
       T + "test_v1_studio.py::test_twin", (143, 68)),
    _m(147, "policy comparator with uncertainty charts", 7, V + "studio.views", T + "test_v1_studio.py::test_policy",
       (143,)),
    _m(148, "evidence dashboard with claim graph and run inspection", 7, V + "studio.views",
       T + "test_v1_studio.py::test_evidence", (143, 152)),
    _m(149, "exportable figures, reports and study manifests", 7, V + "reporting.export",
       T + "test_v1_reporting.py::test_export", (153,)),
    # P. publication and release
    _m(150, "experiment DAG planner, runner and resume", 8, V + "experiments.dag", T + "test_v1_experiments.py::test_dag",
       (48,)),
    _m(151, "evidence bundle v1 with verifiable manifest", 8, V + "evidence.bundle", T + "test_v1_evidence.py::test_bundle",
       (5,)),
    _m(152, "machine-readable claim registry", 8, V + "evidence.claims", T + "test_v1_evidence.py::test_claims",
       (151,)),
    _m(153, "automated paper/report/appendix generation", 8, V + "reporting.paper",
       T + "test_v1_reporting.py::test_paper", (152,), docs=("docs/v1-paper.md", "docs/v1-final-report.md")),
    _m(154, "datasheets, model cards and policy cards", 8, V + "reporting.cards", T + "test_v1_reporting.py::test_cards",
       (153,)),
    _m(155, "reproducible containers and environment lock", 8, V + "reporting.environment",
       T + "test_v1_reporting.py::test_environment", (1,), artifacts=("Dockerfile",)),
    _m(156, "cross-platform test/CI and wheel packaging", 8, V + "compat", T + "test_v1_compat.py::test_packaging",
       (141,), artifacts=(".github/workflows/ci.yml",)),
    _m(157, "backward compatibility, migration guide and stable API guarantees", 8, V + "compat",
       T + "test_v1_compat.py::test_legacy", (140, 141), docs=("docs/v1-migration.md",)),
    _m(158, "research PR and release PR integration", 8, V + "compat", T + "test_v1_compat.py::test_release_gate",
       (156,), resources="external"),
    _m(159, "annotated v1.0.0 tag, GitHub Release, trusted PyPI and public install", 8, V + "compat",
       T + "test_v1_compat.py::test_release_gate", (158,), resources="external"),
)

BY_ID = {m.id: m for m in MILESTONES}


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:  # noqa: BLE001 - any failure means "not implemented"
        return False


def _test_spec(spec: str) -> tuple[str, str]:
    path, _, prefix = spec.partition("::")
    return path, prefix or "test_"


def _tests_declared(root: Path, spec: str) -> bool:
    path, prefix = _test_spec(spec)
    p = root / path
    return p.exists() and f"def {prefix}" in p.read_text(encoding="utf-8")


def junit_passed(report: Path | None) -> dict[str, set[str]]:
    """{test file stem: {passed test names}} from a pytest JUnit XML report."""
    if report is None or not report.exists():
        return {}
    out: dict[str, set[str]] = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        if any(child.tag in ("failure", "error", "skipped") for child in case):
            continue
        stem = case.get("classname", "").split(".")[-1]
        out.setdefault(stem, set()).add(case.get("name", "").split("[")[0])
    return out


def _tests_passed(spec: str, passed: dict[str, set[str]]) -> bool:
    path, prefix = _test_spec(spec)
    names = passed.get(Path(path).stem, set())
    return any(n.startswith(prefix) for n in names)


@dataclass
class Evaluation:
    milestone: Milestone
    status: str
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"id": self.milestone.id, "title": self.milestone.title, "status": self.status,
                "missing": self.missing, "modules": list(self.milestone.modules),
                "tests": list(self.milestone.tests), "depends": list(self.milestone.depends),
                "resources": self.milestone.resources, "blocked": self.milestone.blocked,
                "applicability": self.milestone.applicability}


def evaluate(root: Path = Path("."), junit: Path | None = None) -> dict[str, Evaluation]:
    passed = junit_passed(junit)
    out: dict[str, Evaluation] = {}
    for m in MILESTONES:
        if m.blocked:
            out[m.id] = Evaluation(m, "BLOCKED", [m.blocked])
            continue
        missing = [f"module {x}" for x in m.modules if not _importable(x)]
        if missing:
            out[m.id] = Evaluation(m, "PLANNED", missing)
            continue
        missing = [f"tests {x}" for x in m.tests if not _tests_declared(root, x)]
        if missing:
            out[m.id] = Evaluation(m, "IMPLEMENTED", missing)
            continue
        missing = [f"passing {x}" for x in m.tests if not _tests_passed(x, passed)]
        missing += [f"artifact {a}" for a in m.artifacts if not (root / a).exists()]
        missing += [f"doc {d}" for d in m.docs if not (root / d).exists()]
        out[m.id] = Evaluation(m, "TESTED" if missing else "COMPLETE", missing)
    # a milestone is complete only if its dependencies are not planned/blocked
    for m in MILESTONES:
        e = out[m.id]
        if e.status == "COMPLETE":
            bad = [d for d in m.depends if out[d].status in ("PLANNED", "BLOCKED")]
            if bad:
                e.status = "TESTED"
                e.missing.append(f"dependencies not implemented: {', '.join(bad)}")
    return out


def summary(evaluations: dict[str, Evaluation]) -> dict:
    counts = {s: 0 for s in STATUSES}
    for e in evaluations.values():
        counts[e.status] += 1
    return {"milestones": len(evaluations), "counts": counts,
            "complete": sorted((k for k, e in evaluations.items() if e.status == "COMPLETE"), key=lambda k: int(k[1:]))}


def check_registry() -> list[str]:
    issues = []
    ids = [m.id for m in MILESTONES]
    if ids != [f"M{i}" for i in range(160)]:
        issues.append("milestones must be exactly M0..M159 in order")
    for m in MILESTONES:
        for d in m.depends:
            if d not in BY_ID:
                issues.append(f"{m.id}: unknown dependency {d}")
            elif int(d[1:]) == int(m.id[1:]):
                issues.append(f"{m.id}: self dependency")
        if not m.modules or not m.tests:
            issues.append(f"{m.id}: needs modules and tests")
    # dependency graph must be acyclic
    state: dict[str, int] = {}

    def visit(k: str) -> bool:
        if state.get(k) == 1:
            return False
        if state.get(k) == 2:
            return True
        state[k] = 1
        ok = all(visit(d) for d in BY_ID[k].depends if d in BY_ID)
        state[k] = 2
        return ok

    if not all(visit(k) for k in BY_ID):
        issues.append("dependency cycle")
    return issues


__all__ = ["Milestone", "MILESTONES", "BY_ID", "evaluate", "summary", "check_registry", "junit_passed", "STATUSES"]
