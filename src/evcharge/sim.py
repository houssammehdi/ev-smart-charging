"""Discrete-time simulator that plays a policy against a scenario.

At every step the simulator

1. reveals the sessions that are connected (arrivals are revealed online),
2. asks the policy for setpoints (kW, or A per phase on phase-aware sites),
3. enforces physics and site rules on the commands, recording every correction
   as a :class:`Violation` (well-behaved policies produce none),
4. lets each EV draw ``min(command, power that finishes its request)``, and
5. records per-session power, the resulting site import and, on phase-aware
   sites, the current on every line.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from evcharge.capacity import StepConstraints, fit_to_rows
from evcharge.model import POWER_TOL_KW, FloatArray, Scenario, on_grid
from evcharge.policies.base import Observation, Policy, SessionState


class ViolationKind(StrEnum):
    """Categories of commands the simulator had to correct."""

    INVALID = "invalid"
    """Negative, NaN or infinite setpoint (applied as 0)."""
    NOT_CONNECTED = "not-connected"
    """Setpoint for an unknown or unplugged session (ignored)."""
    ABOVE_MAX = "above-max"
    """Setpoint above the charger/EV maximum (clipped)."""
    RESOLUTION = "resolution"
    """Setpoint not a multiple of the charger's resolution (rounded down)."""
    BELOW_MIN = "below-min"
    """Non-zero setpoint below the minimum current (charger pauses: 0)."""
    SITE_LIMIT = "site-limit"
    """Setpoints above the site's kW headroom (scaled down proportionally)."""
    LINE_LIMIT = "line-limit"
    """Setpoints above a line's current limit (scaled down proportionally)."""


@dataclass(frozen=True)
class Violation:
    """One corrected command.

    Attributes:
        step: Step index.
        kind: What went wrong.
        session_id: Affected session (``None`` for site- and line-level violations).
        requested: Commanded value (for row violations: the row's total).
        applied: Value after correction (for row violations: the row's limit).
        unit: Unit of ``requested`` and ``applied``: ``"kW"``, or ``"A"`` for
            per-phase setpoints and line rows.
    """

    step: int
    kind: ViolationKind
    session_id: str | None
    requested: float
    applied: float
    unit: str = "kW"

    @property
    def requested_kw(self) -> float:
        """``requested`` if it is in kW, else NaN (kept for the original kW API)."""
        return self.requested if self.unit == "kW" else math.nan

    @property
    def applied_kw(self) -> float:
        """``applied`` if it is in kW, else NaN (kept for the original kW API)."""
        return self.applied if self.unit == "kW" else math.nan


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """Trajectory of one simulation run.

    Attributes:
        scenario: The simulated scenario.
        policy_name: Name of the policy.
        setpoint_kw: Commands after enforcement in kW, shape ``(n_sessions, n_steps)``.
        power_kw: Average power actually drawn, shape ``(n_sessions, n_steps)``.
        delivered_kwh: Energy delivered into each battery.
        net_import_kw: Site import per step (negative values are export).
        violations: Corrected commands, in step order.
        runtime_s: Wall-clock time of the run, including policy computation.
        setpoint: Commands after enforcement in each session's control unit.
        line_current_a: Modelled current per line, shape ``(n_steps, 3)``, from
            the commands of sessions that drew power (``None`` on aggregate sites).
    """

    scenario: Scenario
    policy_name: str
    setpoint_kw: FloatArray
    power_kw: FloatArray
    delivered_kwh: FloatArray
    net_import_kw: FloatArray
    violations: tuple[Violation, ...]
    runtime_s: float
    setpoint: FloatArray | None = None
    line_current_a: FloatArray | None = None

    @property
    def import_kw(self) -> FloatArray:
        """Grid import per step (kW, >= 0)."""
        return np.maximum(self.net_import_kw, 0.0)

    @property
    def export_kw(self) -> FloatArray:
        """Grid export per step (kW, >= 0)."""
        return np.maximum(-self.net_import_kw, 0.0)

    @property
    def ev_power_kw(self) -> FloatArray:
        """Total EV charging power per step."""
        return np.asarray(self.power_kw.sum(axis=0), dtype=np.float64)

    @property
    def unmet_kwh(self) -> FloatArray:
        """Undelivered energy per session."""
        requested = np.array([s.energy_kwh for s in self.scenario.sessions], dtype=np.float64)
        return np.maximum(requested - self.delivered_kwh, 0.0)


def _enforce(
    step: int,
    raw: dict[str, float],
    states: dict[str, SessionState],
    rows: StepConstraints,
    violations: list[Violation],
) -> dict[str, float]:
    """Return corrected commands and append the corrections to ``violations``."""
    cmd: dict[str, float] = {}
    for sid, value in raw.items():
        st = states.get(sid)
        v = float(value)
        if st is None:
            if not math.isfinite(v) or abs(v) > POWER_TOL_KW:
                violations.append(Violation(step, ViolationKind.NOT_CONNECTED, sid, v, 0.0))
            continue
        ctl = st.control
        unit = ctl.unit
        if not math.isfinite(v) or v < -POWER_TOL_KW:
            violations.append(Violation(step, ViolationKind.INVALID, sid, v, 0.0, unit))
            continue
        if v <= POWER_TOL_KW:
            continue
        if v > ctl.charge_max + POWER_TOL_KW:
            top = ctl.charge_max
            violations.append(Violation(step, ViolationKind.ABOVE_MAX, sid, v, top, unit))
            v = top
        if not on_grid(v, ctl.step):
            rounded = ctl.snap_down(v)
            violations.append(Violation(step, ViolationKind.RESOLUTION, sid, v, rounded, unit))
            v = rounded
        if v < ctl.charge_min - POWER_TOL_KW:
            violations.append(Violation(step, ViolationKind.BELOW_MIN, sid, v, 0.0, unit))
            continue
        cmd[sid] = min(max(v, ctl.charge_min), ctl.charge_max)

    fitted, fixes = fit_to_rows(cmd, rows, {sid: states[sid].control for sid in cmd})
    for r, usage, limit in fixes:
        line = rows.kinds[r] == "line"
        kind = ViolationKind.LINE_LIMIT if line else ViolationKind.SITE_LIMIT
        violations.append(Violation(step, kind, None, usage, limit, "A" if line else "kW"))
    return fitted


def simulate(scenario: Scenario, policy: Policy) -> SimulationResult:
    """Run ``policy`` on ``scenario`` and return the full trajectory.

    Clairvoyant policies receive the complete scenario in ``reset``; all others
    receive it without sessions and learn about each EV only when it plugs in.
    """
    started = time.perf_counter()
    policy.reset(scenario if policy.clairvoyant else scenario.without_sessions())

    horizon = scenario.horizon
    n_t = horizon.n_steps
    dt = horizon.dt_h
    sessions = scenario.sessions
    n_s = len(sessions)
    index = {s.id: i for i, s in enumerate(sessions)}
    bounds = [scenario.power_bounds(s) for s in sessions]
    controls = [scenario.control(s) for s in sessions]
    coefficients = {s.id: scenario.row_coefficients(s) for s in sessions}
    rows = scenario.rows
    headroom = scenario.ev_headroom_kw
    net_base = scenario.net_base_kw

    setpoint = np.zeros((n_s, n_t))
    setpoint_kw = np.zeros((n_s, n_t))
    power = np.zeros((n_s, n_t))
    delivered = np.zeros(n_s)
    net_import = np.zeros(n_t)
    violations: list[Violation] = []
    peak = 0.0

    for t in range(n_t):
        states: dict[str, SessionState] = {}
        for i, s in enumerate(sessions):
            if s.is_connected(t):
                states[s.id] = SessionState(
                    session=s,
                    delivered_kwh=float(delivered[i]),
                    p_min_kw=bounds[i][0],
                    p_max_kw=bounds[i][1],
                    steps_left=s.departure_step - t,
                    dt_h=dt,
                    control_spec=controls[i],
                )
        step_rows = StepConstraints(
            rows.names, rows.kinds, rows.rhs[t], {sid: coefficients[sid] for sid in states}
        )
        obs = Observation(
            step=t,
            sessions=tuple(states.values()),
            headroom_kw=float(headroom[t]),
            peak_import_kw=peak,
            constraints=step_rows,
        )
        raw = dict(policy.decide(obs))
        cmd = _enforce(t, raw, states, step_rows, violations)
        for sid, c in cmd.items():
            i = index[sid]
            st = states[sid]
            c_kw = c * controls[i].kw_per_unit
            drawn = min(c_kw, st.finish_kw)
            setpoint[i, t] = c
            setpoint_kw[i, t] = c_kw
            power[i, t] = drawn
            eta = sessions[i].efficiency
            delivered[i] = min(sessions[i].energy_kwh, delivered[i] + eta * drawn * dt)
        net_import[t] = net_base[t] + power[:, t].sum()
        peak = max(peak, float(net_import[t]))

    line_current = None
    if scenario.site.phase_aware:
        line_current = scenario.line_currents_a(np.where(power > 0.0, setpoint, 0.0))
    return SimulationResult(
        scenario=scenario,
        policy_name=policy.name,
        setpoint_kw=setpoint_kw,
        power_kw=power,
        delivered_kwh=delivered,
        net_import_kw=net_import,
        violations=tuple(violations),
        runtime_s=time.perf_counter() - started,
        setpoint=setpoint,
        line_current_a=line_current,
    )
