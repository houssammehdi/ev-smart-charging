"""evcharge: smart-charging policies and a simulator for grid-constrained EV sites.

Typical use::

    from evcharge import scenarios, simulate, compute_metrics
    from evcharge.policies import ModelPredictiveControl

    sc = scenarios.workplace(n_sessions=40, seed=7, grid_limit_kw=60)
    result = simulate(sc, ModelPredictiveControl())
    print(compute_metrics(result))
"""

from __future__ import annotations

from evcharge.metrics import Metrics, compute_metrics, jain_index
from evcharge.model import (
    Charger,
    Horizon,
    Scenario,
    Session,
    Site,
    Tariff,
    ValidationError,
)
from evcharge.sim import SimulationResult, Violation, ViolationKind, simulate

__version__ = "0.2.0"

__all__ = [
    "Charger",
    "Horizon",
    "Metrics",
    "Scenario",
    "Session",
    "SimulationResult",
    "Site",
    "Tariff",
    "ValidationError",
    "Violation",
    "ViolationKind",
    "__version__",
    "compute_metrics",
    "jain_index",
    "simulate",
]
