"""Policy protocol, observations and shared allocation helpers.

A policy is called once per time step with an :class:`Observation` of the
currently connected EVs and returns a power setpoint (kW) per session id.
Setpoints are *commands*: the simulator treats a non-zero setpoint below the
session's minimum power as a violation, and an EV stops drawing power as soon as
its requested energy is reached, so commanding ``p_min`` to finish the last few
hundred Wh is legitimate and physically accurate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from evcharge.model import ENERGY_TOL_KWH, POWER_TOL_KW, Scenario, Session

Setpoints = dict[str, float]
"""Mapping of session id to commanded power in kW."""


@dataclass(frozen=True)
class SessionState:
    """What the controller knows about one connected session at a given step.

    Attributes:
        session: The session as declared at plug-in (departure and energy request).
        delivered_kwh: Energy delivered into the battery so far.
        p_min_kw: Lowest non-zero power that can be commanded.
        p_max_kw: Highest power that can be commanded.
        steps_left: Steps until departure, including the current one.
        dt_h: Step length in hours.
    """

    session: Session
    delivered_kwh: float
    p_min_kw: float
    p_max_kw: float
    steps_left: int
    dt_h: float

    @property
    def id(self) -> str:
        """Session id."""
        return self.session.id

    @property
    def remaining_kwh(self) -> float:
        """Energy still to be delivered into the battery."""
        return max(0.0, self.session.energy_kwh - self.delivered_kwh)

    @property
    def is_satisfied(self) -> bool:
        """Whether the energy request has been met (within tolerance)."""
        return self.remaining_kwh <= ENERGY_TOL_KWH

    @property
    def finish_kw(self) -> float:
        """Grid-side average power that would complete the request in this step."""
        return self.remaining_kwh / (self.session.efficiency * self.dt_h)

    @property
    def desired_kw(self) -> float:
        """Useful power this step: ``min(p_max, finish_kw)``."""
        return min(self.p_max_kw, self.finish_kw)

    @property
    def laxity_h(self) -> float:
        """Slack in hours: time left minus time needed at full power."""
        needed_h = self.remaining_kwh / (self.session.efficiency * self.p_max_kw)
        return self.steps_left * self.dt_h - needed_h

    @property
    def delivered_fraction(self) -> float:
        """Share of the request delivered so far, in ``[0, 1]``."""
        return min(1.0, self.delivered_kwh / self.session.energy_kwh)


@dataclass(frozen=True)
class Observation:
    """Information revealed to the policy at the start of a step.

    Attributes:
        step: Index of the current step.
        sessions: Connected sessions (arrived and not yet departed).
        headroom_kw: Power available for EV charging this step
            (grid limit - base load + PV).
        peak_import_kw: Highest site import observed in earlier steps (>= 0).
    """

    step: int
    sessions: tuple[SessionState, ...]
    headroom_kw: float
    peak_import_kw: float

    def pending(self) -> list[SessionState]:
        """Return the connected sessions that still need energy."""
        return [s for s in self.sessions if not s.is_satisfied]


@runtime_checkable
class Policy(Protocol):
    """Interface every charging policy implements.

    The simulator calls :meth:`reset` once, then :meth:`decide` at every step.
    Policies with ``clairvoyant = True`` receive the full scenario (including
    future sessions) in :meth:`reset`; online policies receive it with the
    sessions removed, so they cannot peek at future arrivals.
    """

    @property
    def name(self) -> str:
        """Short identifier used in tables and the CLI."""
        ...

    @property
    def clairvoyant(self) -> bool:
        """Whether the policy needs perfect foresight of all sessions."""
        ...

    def reset(self, scenario: Scenario) -> None:
        """Prepare for a new simulation run."""
        ...

    def decide(self, obs: Observation) -> Mapping[str, float]:
        """Return power setpoints (kW) for connected sessions; omitted ids get 0."""
        ...


class OnlinePolicy:
    """Convenience base class for policies without foresight."""

    name = "online"
    clairvoyant = False

    def __init__(self) -> None:
        self._scenario: Scenario | None = None

    @property
    def scenario(self) -> Scenario:
        """Scenario passed to :meth:`reset` (sessions removed)."""
        if self._scenario is None:
            raise RuntimeError(f"policy {self.name!r} used before reset()")
        return self._scenario

    def reset(self, scenario: Scenario) -> None:
        """Store the (session-free) scenario for later steps."""
        self._scenario = scenario

    def decide(self, obs: Observation) -> Mapping[str, float]:
        """Return setpoints for the current step."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def command_bounds(state: SessionState) -> tuple[float, float]:
    """Return ``(lowest, useful)`` command for a session that should charge now.

    ``useful`` is the power that is actually useful (``desired_kw``), raised to
    ``p_min`` when finishing requires less than the minimum; the EV stops by
    itself once full, so the surplus is never drawn.
    """
    useful = max(state.desired_kw, state.p_min_kw)
    return state.p_min_kw, useful


def priority_fill(states: Iterable[SessionState], headroom_kw: float) -> Setpoints:
    """Serve sessions in the given order, each up to its useful power.

    A session is skipped (and the next one tried) if less than its minimum power
    is left, so the result always respects both the site headroom and the
    minimum-current rule.
    """
    remaining = max(0.0, headroom_kw)
    out: Setpoints = {}
    for st in states:
        if st.is_satisfied:
            continue
        lowest, useful = command_bounds(st)
        if remaining + POWER_TOL_KW < max(lowest, POWER_TOL_KW):
            continue
        cmd = min(useful, remaining)
        if cmd <= POWER_TOL_KW:
            continue
        out[st.id] = cmd
        remaining -= cmd
    return out


def water_fill(caps: Sequence[float], budget: float) -> list[float]:
    """Split ``budget`` as equally as possible subject to per-item caps.

    Returns allocations ``a_i = min(cap_i, level)`` with ``sum(a) = min(budget,
    sum(caps))`` (classic water-filling).
    """
    n = len(caps)
    alloc = [0.0] * n
    if n == 0 or budget <= 0:
        return alloc
    order = sorted(range(n), key=lambda i: caps[i])
    left = budget
    for rank, i in enumerate(order):
        share = left / (n - rank)
        a = min(caps[i], share)
        alloc[i] = a
        left -= a
    return alloc
