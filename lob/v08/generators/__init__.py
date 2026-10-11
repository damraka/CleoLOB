"""v0.8 generator families; importing this package registers every built-in family."""
from .base import REGISTRY, MarketGenerator, get, load_plugins, register  # noqa: F401
from .families import (HawkesGenerator, MixtureGenerator, PoissonGenerator, QueueReactiveGenerator,  # noqa: F401
                       RegimeSwitchingGenerator, StateDependentGenerator)
from .ecology import EcologyGenerator  # noqa: F401
from .endogenous import EndogenousGenerator  # noqa: F401
from .scaled import ScaledGenerator  # noqa: F401

CORE_FAMILIES = ("poisson", "hawkes", "queue_reactive", "regime_switching", "state_dependent", "mixture")
