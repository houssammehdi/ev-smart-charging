"""Policy protocol, observations and shared allocation helpers.

A policy is called once per time step with an :class:`Observation` of the
currently connected EVs and returns a setpoint per session id, in the
session's control unit (:attr:`SessionState.control`): kW on aggregate sites,
amperes per phase on phase-aware sites. Setpoints are *commands*: the
simulator treats a non-zero setpoint below the session's minimum as a
violation, and an EV stops drawing power as soon as its requested energy is
reached, so commanding the minimum to finish the last few hundred Wh is
legitimate and physically accurate.

Capacity is described by :attr:`Observation.constraints`, the linear rows of
the step (one kW row on aggregate sites, plus one row per line on phase-aware
sites). The helpers below allocate against those rows, so every built-in
heuristic is phase-aware without special cases, and on one kW row they reduce
exactly to the original single-headroom arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol, runtime_checkable

import numpy as np

from evcharge.capacity import ROW_TOL, Allocation, StepConstraints, fit_to_rows
from evcharge.model import ENERGY_TOL_KWH, POWER_TOL_KW, Control, Scenario, Session

Setpoints = dict[str, float]
"""Mapping of session id to its setpoint (kW, or A per phase on phase-aware sites)."""


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
        control_spec: How the session is commanded; ``None`` means kW setpoints
            in ``[p_min_kw, p_max_kw]`` (the aggregate model).
    """

    session: Session
    delivered_kwh: float
    p_min_kw: float
    p_max_kw: float
    steps_left: int
    dt_h: float
    control_spec: Control | None = None

    @cached_property
    def control(self) -> Control:
        """Setpoint unit, range and resolution of the session."""
        if self.control_spec is not None:
            return self.control_spec
        return Control("kW", 1.0, self.p_min_kw, self.p_max_kw)

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
        constraints: Linear rows the setpoints must satisfy this step. ``None``
            means the single aggregate row ``sum of kW setpoints <= headroom_kw``.
    """

    step: int
    sessions: tuple[SessionState, ...]
    headroom_kw: float
    peak_import_kw: float
    constraints: StepConstraints | None = None

    def pending(self) -> list[SessionState]:
        """Return the connected sessions that still need energy."""
        return [s for s in self.sessions if not s.is_satisfied]

    @property
    def rows(self) -> StepConstraints:
        """The step's constraint rows (the aggregate kW row if none were given)."""
        if self.constraints is not None:
            return self.constraints
        return StepConstraints.single(self.headroom_kw)

    def allocation(self) -> Allocation:
        """Fresh :class:`Allocation` over this step's rows."""
        return Allocation(self.rows)


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
        """Return setpoints for connected sessions (their control unit); omitted ids get 0."""
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
    """Return ``(lowest, useful)`` setpoint for a session that should charge now.

    ``useful`` is the setpoint that finishes the request this step (rounded up
    to the resolution) or the maximum, raised to the minimum when finishing
    needs less; the EV stops by itself once full, so the surplus is never drawn.
    """
    ctl = state.control
    desired = min(ctl.charge_max, ctl.snap_up(state.finish_kw / ctl.kw_per_unit))
    return ctl.charge_min, max(desired, ctl.charge_min)


def priority_fill(states: Iterable[SessionState], capacity: float | Allocation) -> Setpoints:
    """Serve sessions in the given order, each up to its useful setpoint.

    Args:
        states: Sessions in priority order.
        capacity: An :class:`Allocation` over the step's rows (updated in place),
            or a kW headroom for the aggregate model.

    A session is skipped (and the next one tried) if less than its minimum is
    left on any row it loads, so the result always respects every row, the
    minimum-current rule and the setpoint resolution.
    """
    alloc = capacity if isinstance(capacity, Allocation) else Allocation.single(capacity)
    out: Setpoints = {}
    for st in states:
        if st.is_satisfied:
            continue
        lowest, useful = command_bounds(st)
        room = st.control.snap_down(alloc.headroom(st.id))
        if room + POWER_TOL_KW < max(lowest, POWER_TOL_KW):
            continue
        cmd = min(useful, room)
        if cmd <= POWER_TOL_KW:
            continue
        out[st.id] = cmd
        alloc.take(st.id, cmd)
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


def fair_fill(
    states: Sequence[SessionState], caps: Sequence[float], alloc: Allocation
) -> list[float]:
    """Max-min fair setpoints under the step's rows (progressive filling).

    All sessions rise at the same rate until a cap or a row binds; sessions on
    a binding row (or at their cap) stop and the rest continue. On the
    aggregate model's single row this is exactly :func:`water_fill`. The levels
    are rounded down to each session's resolution; ``alloc`` is not modified.
    """
    n = len(states)
    if alloc.is_scalar:
        return water_fill(caps, float(alloc.slack[0]))
    level = np.zeros(n)
    cap = np.asarray(caps, dtype=np.float64)
    a = np.array([alloc.constraints.coefficient(st.id) for st in states]).reshape(n, -1)
    slack = alloc.slack.copy()
    active = [i for i in range(n) if cap[i] > POWER_TOL_KW]
    while active:
        rate = a[active].sum(axis=0)
        grow = rate > 1e-12
        t_rows = float(np.min(np.maximum(slack[grow], 0.0) / rate[grow])) if grow.any() else np.inf
        t = max(0.0, min(t_rows, float(np.min(cap[active] - level[active]))))
        level[active] += t
        slack -= rate * t
        tight = grow & (slack <= 1e-9)
        still = [i for i in active if level[i] < cap[i] - 1e-9 and not np.any(a[i, tight] > 0)]
        if len(still) == len(active):
            break
        active = still
    return [st.control.snap_down(float(v)) for st, v in zip(states, level, strict=True)]


def finalize(obs: Observation, desired: Mapping[str, float]) -> Setpoints:
    """Turn continuous setpoints (e.g. from an LP) into feasible charger commands.

    Setpoints are clipped to their range, rounded down to the resolution and
    paused if that leaves them below the minimum; then they are moved toward
    zero until every row of the step holds, which also absorbs solver
    tolerances. Finally, a session whose setpoint finishes its request this
    step is rounded *up* if the rows allow it: the EV stops when full, so
    rounding up cannot over-deliver, and it avoids leaving a sliver of energy
    undelivered at the end of a session.
    """
    states = {st.id: st for st in obs.sessions}
    floors: Setpoints = {}
    finishing: dict[str, float] = {}
    for sid, value in desired.items():
        st = states.get(sid)
        if st is None or value <= POWER_TOL_KW:
            continue
        ctl = st.control
        v = min(float(value), ctl.charge_max)
        if v + POWER_TOL_KW < ctl.charge_min:
            continue
        if v >= st.finish_kw / ctl.kw_per_unit - 1e-9:
            finishing[sid] = min(ctl.charge_max, ctl.snap_up(v))
        down = ctl.snap_down(v)
        if down + POWER_TOL_KW < ctl.charge_min:
            continue
        floors[sid] = max(down, ctl.charge_min)
    controls = {sid: states[sid].control for sid in desired if sid in states}
    fitted, _ = fit_to_rows(floors, obs.rows, controls)
    alloc = obs.allocation()
    for sid, v in fitted.items():
        alloc.take(sid, v)
    for sid, up in finishing.items():
        extra = up - fitted.get(sid, 0.0)
        if extra > 0 and alloc.headroom(sid) + ROW_TOL >= extra:
            fitted[sid] = up
            alloc.take(sid, extra)
    return fitted
