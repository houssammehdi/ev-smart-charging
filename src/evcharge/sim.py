"""Discrete-time simulator that plays a policy against a scenario.

At every step the simulator

1. reveals the sessions that are connected (arrivals are revealed online),
2. asks the policy for setpoints,
3. enforces physics and site rules on the commands, recording every correction
   as a :class:`Violation` (well-behaved policies produce none),
4. lets each EV draw ``min(command, power that finishes its request)``, and
5. records per-session power and the resulting site import.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from evcharge.model import POWER_TOL_KW, FloatArray, Scenario
from evcharge.policies.base import Observation, Policy, SessionState


class ViolationKind(StrEnum):
    """Categories of commands the simulator had to correct."""

    INVALID = "invalid"
    """Negative, NaN or infinite setpoint (applied as 0)."""
    NOT_CONNECTED = "not-connected"
    """Setpoint for an unknown or unplugged session (ignored)."""
    ABOVE_MAX = "above-max"
    """Setpoint above the charger/EV maximum (clipped)."""
    BELOW_MIN = "below-min"
    """Non-zero setpoint below the minimum current (charger pauses: 0 kW)."""
    SITE_LIMIT = "site-limit"
    """Sum of setpoints above the site headroom (scaled down proportionally)."""


@dataclass(frozen=True)
class Violation:
    """One corrected command.

    Attributes:
        step: Step index.
        kind: What went wrong.
        session_id: Affected session (``None`` for site-level violations).
        requested_kw: Commanded value (sum of commands for site-level).
        applied_kw: Value after correction (headroom for site-level).
    """

    step: int
    kind: ViolationKind
    session_id: str | None
    requested_kw: float
    applied_kw: float


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """Trajectory of one simulation run.

    Attributes:
        scenario: The simulated scenario.
        policy_name: Name of the policy.
        setpoint_kw: Commands after enforcement, shape ``(n_sessions, n_steps)``.
        power_kw: Average power actually drawn, shape ``(n_sessions, n_steps)``.
        delivered_kwh: Energy delivered into each battery.
        net_import_kw: Site import per step (negative values are export).
        violations: Corrected commands, in step order.
        runtime_s: Wall-clock time of the run, including policy computation.
    """

    scenario: Scenario
    policy_name: str
    setpoint_kw: FloatArray
    power_kw: FloatArray
    delivered_kwh: FloatArray
    net_import_kw: FloatArray
    violations: tuple[Violation, ...]
    runtime_s: float

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
    headroom_kw: float,
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
        if not math.isfinite(v) or v < -POWER_TOL_KW:
            violations.append(Violation(step, ViolationKind.INVALID, sid, v, 0.0))
            continue
        if v <= POWER_TOL_KW:
            continue
        if v > st.p_max_kw + POWER_TOL_KW:
            violations.append(Violation(step, ViolationKind.ABOVE_MAX, sid, v, st.p_max_kw))
            v = st.p_max_kw
        if v < st.p_min_kw - POWER_TOL_KW:
            violations.append(Violation(step, ViolationKind.BELOW_MIN, sid, v, 0.0))
            continue
        cmd[sid] = min(max(v, st.p_min_kw), st.p_max_kw)

    total = sum(cmd.values())
    if total > headroom_kw + POWER_TOL_KW:
        violations.append(Violation(step, ViolationKind.SITE_LIMIT, None, total, headroom_kw))
        scale = max(0.0, headroom_kw) / total
        scaled: dict[str, float] = {}
        for sid, v in cmd.items():
            nv = v * scale
            # Scaling can push an EV below its minimum: the charger then pauses.
            if nv + POWER_TOL_KW >= states[sid].p_min_kw and nv > POWER_TOL_KW:
                scaled[sid] = nv
        cmd = scaled
    return cmd


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
    headroom = scenario.ev_headroom_kw
    net_base = scenario.net_base_kw

    setpoint = np.zeros((n_s, n_t))
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
                )
        obs = Observation(
            step=t,
            sessions=tuple(states.values()),
            headroom_kw=float(headroom[t]),
            peak_import_kw=peak,
        )
        raw = dict(policy.decide(obs))
        cmd = _enforce(t, raw, states, float(headroom[t]), violations)
        for sid, c in cmd.items():
            i = index[sid]
            st = states[sid]
            drawn = min(c, st.finish_kw)
            setpoint[i, t] = c
            power[i, t] = drawn
            eta = sessions[i].efficiency
            delivered[i] = min(sessions[i].energy_kwh, delivered[i] + eta * drawn * dt)
        net_import[t] = net_base[t] + power[:, t].sum()
        peak = max(peak, float(net_import[t]))

    return SimulationResult(
        scenario=scenario,
        policy_name=policy.name,
        setpoint_kw=setpoint,
        power_kw=power,
        delivered_kwh=delivered,
        net_import_kw=net_import,
        violations=tuple(violations),
        runtime_s=time.perf_counter() - started,
    )
