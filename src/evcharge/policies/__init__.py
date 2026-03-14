"""Charging policies: real-time heuristics, the offline optimum and MPC."""

from __future__ import annotations

from collections.abc import Callable

from evcharge.policies.base import Observation, OnlinePolicy, Policy, SessionState, Setpoints
from evcharge.policies.heuristics import (
    EarliestDeadlineFirst,
    EqualShare,
    LeastLaxityFirst,
    PriceAware,
    Uncontrolled,
)
from evcharge.policies.optimal import ModelPredictiveControl, OptimalSchedule

POLICY_FACTORIES: dict[str, Callable[[], Policy]] = {
    "uncontrolled": Uncontrolled,
    "equal-share": EqualShare,
    "edf": EarliestDeadlineFirst,
    "llf": LeastLaxityFirst,
    "price-aware": PriceAware,
    "mpc": ModelPredictiveControl,
    "optimal": OptimalSchedule,
}
"""Registry of built-in policies by CLI name, in presentation order."""


def make_policy(name: str) -> Policy:
    """Instantiate a built-in policy by name (see :data:`POLICY_FACTORIES`)."""
    try:
        return POLICY_FACTORIES[name]()
    except KeyError:
        known = ", ".join(POLICY_FACTORIES)
        raise ValueError(f"unknown policy {name!r}; choose from: {known}") from None


__all__ = [
    "POLICY_FACTORIES",
    "EarliestDeadlineFirst",
    "EqualShare",
    "LeastLaxityFirst",
    "ModelPredictiveControl",
    "Observation",
    "OnlinePolicy",
    "OptimalSchedule",
    "Policy",
    "PriceAware",
    "SessionState",
    "Setpoints",
    "Uncontrolled",
    "make_policy",
]
