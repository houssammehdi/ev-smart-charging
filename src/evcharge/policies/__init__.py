"""Charging policies: real-time heuristics, the offline optimum and MPC variants."""

from __future__ import annotations

from collections.abc import Callable

from evcharge.forecast import ArrivalForecast
from evcharge.policies.adapters import PhaseBlind
from evcharge.policies.base import Observation, OnlinePolicy, Policy, SessionState, Setpoints
from evcharge.policies.forecast import ExpectedValueMPC, ReserveMPC, ScenarioMPC
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

FORECAST_POLICY_FACTORIES: dict[str, Callable[[ArrivalForecast], Policy]] = {
    "mpc-reserve": ReserveMPC,
    "mpc-ev": ExpectedValueMPC,
    "mpc-saa": ScenarioMPC,
}
"""Forecast-aware MPC variants by CLI name; each needs an :class:`ArrivalForecast`."""


def make_policy(name: str, forecast: ArrivalForecast | None = None) -> Policy:
    """Instantiate a built-in policy by name.

    Names are those of :data:`POLICY_FACTORIES` and
    :data:`FORECAST_POLICY_FACTORIES`; the latter need ``forecast``.
    """
    if name in FORECAST_POLICY_FACTORIES:
        if forecast is None:
            raise ValueError(f"policy {name!r} needs an arrival forecast (training days)")
        return FORECAST_POLICY_FACTORIES[name](forecast)
    try:
        return POLICY_FACTORIES[name]()
    except KeyError:
        known = ", ".join([*POLICY_FACTORIES, *FORECAST_POLICY_FACTORIES])
        raise ValueError(f"unknown policy {name!r}; choose from: {known}") from None


__all__ = [
    "FORECAST_POLICY_FACTORIES",
    "POLICY_FACTORIES",
    "EarliestDeadlineFirst",
    "EqualShare",
    "ExpectedValueMPC",
    "LeastLaxityFirst",
    "ModelPredictiveControl",
    "Observation",
    "OnlinePolicy",
    "OptimalSchedule",
    "PhaseBlind",
    "Policy",
    "PriceAware",
    "ReserveMPC",
    "ScenarioMPC",
    "SessionState",
    "Setpoints",
    "Uncontrolled",
    "make_policy",
]
