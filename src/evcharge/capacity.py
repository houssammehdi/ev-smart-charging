"""Per-step capacity: the linear rows that one step's setpoints must satisfy.

At every step the simulator hands policies a :class:`StepConstraints`: rows
``sum_s a[r, s] * x_s <= rhs[r]`` over the connected sessions' setpoints ``x_s``
(in each session's :class:`~evcharge.model.Control` unit). On aggregate sites
there is a single row (site import, kW); phase-aware sites add a row per line
(amperes). Heuristics allocate against an :class:`Allocation`; optimisation
policies and the simulator use :func:`fit_to_rows` to make a set of commands
feasible by moving them toward zero, which is always possible because zero EV
current is feasible (the scenario validates that).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from evcharge.model import Control, FloatArray

ROW_TOL = 1e-6
"""Tolerance on a row, in the row's unit (kW or A)."""

_UNIT_ROW = np.ones(1)


@dataclass(frozen=True, eq=False)
class StepConstraints:
    """Rows of one step over the connected sessions.

    Attributes:
        names: Row names, e.g. ``"L1 import"`` or ``"site import"``.
        kinds: ``"line"`` (amperes) or ``"site"`` (kW) per row.
        rhs: Right-hand side per row (never negative).
        coefficients: Session id to the coefficient of its setpoint in every row.
            Sessions not listed have coefficient 1 on every row, which is the
            aggregate model's single kW row.
    """

    names: tuple[str, ...]
    kinds: tuple[str, ...]
    rhs: FloatArray
    coefficients: Mapping[str, FloatArray] = field(default_factory=dict)

    @classmethod
    def single(cls, headroom_kw: float) -> StepConstraints:
        """One ``site import`` row: the sum of kW setpoints may not exceed the headroom."""
        return cls(("site import",), ("site",), np.array([max(0.0, headroom_kw)]))

    @property
    def n_rows(self) -> int:
        """Number of rows."""
        return len(self.names)

    @property
    def is_scalar(self) -> bool:
        """Whether this is a single row with unit coefficients (the aggregate kW model)."""
        return self.n_rows == 1 and all(
            a.shape == (1,) and a[0] == 1.0 for a in self.coefficients.values()
        )

    def coefficient(self, session_id: str) -> FloatArray:
        """Coefficient vector of a session's setpoint."""
        a = self.coefficients.get(session_id)
        if a is None:
            if self.n_rows != 1:
                raise KeyError(f"no row coefficients for session {session_id!r}")
            return _UNIT_ROW
        return a

    def usage(self, setpoints: Mapping[str, float]) -> FloatArray:
        """Left-hand side of every row for the given setpoints."""
        out = np.zeros(self.n_rows)
        for sid, x in setpoints.items():
            if x:
                out += self.coefficient(sid) * x
        return out


class Allocation:
    """Mutable row slack while a policy hands out setpoints within one step."""

    def __init__(self, constraints: StepConstraints) -> None:
        self.constraints = constraints
        self.slack = np.array(constraints.rhs, dtype=np.float64)

    @classmethod
    def single(cls, headroom_kw: float) -> Allocation:
        """Allocation over the aggregate model's single kW row."""
        return cls(StepConstraints.single(headroom_kw))

    @property
    def is_scalar(self) -> bool:
        """Whether this is the aggregate model's single row."""
        return self.constraints.is_scalar

    def headroom(self, session_id: str) -> float:
        """Largest additional setpoint for ``session_id`` that keeps every row feasible."""
        a = self.constraints.coefficient(session_id)
        pos = a > 0
        if not pos.any():
            return math.inf
        return max(0.0, float(np.min(self.slack[pos] / a[pos])))

    def take(self, session_id: str, amount: float) -> None:
        """Book ``amount`` of setpoint for ``session_id``."""
        self.slack -= self.constraints.coefficient(session_id) * amount


def fit_to_rows(
    setpoints: Mapping[str, float],
    constraints: StepConstraints,
    controls: Mapping[str, Control],
    *,
    tol: float = ROW_TOL,
) -> tuple[dict[str, float], list[tuple[int, float, float]]]:
    """Move setpoints toward zero until every row holds.

    The most violated row is repaired first: every setpoint with a positive
    coefficient in it is scaled down by the same factor, rounded down to its
    resolution, and paused if that leaves it below its minimum. Setpoints only
    ever shrink and zero is feasible, so this terminates. If scaling has not
    converged after a few rounds, the contributors to the most violated row are
    paused, one row at a time.

    Returns:
        The feasible setpoints and one ``(row, usage, rhs)`` tuple per repair.
    """
    x = {sid: v for sid, v in setpoints.items() if v}
    fixes: list[tuple[int, float, float]] = []
    rounds = 0
    while True:
        usage = constraints.usage(x)
        excess = usage - constraints.rhs
        r = int(np.argmax(excess)) if excess.size else 0
        if not excess.size or excess[r] <= tol:
            return x, fixes
        fixes.append((r, float(usage[r]), float(constraints.rhs[r])))
        contrib = {
            sid: float(constraints.coefficient(sid)[r] * v)
            for sid, v in x.items()
            if constraints.coefficient(sid)[r] * v > 0
        }
        rounds += 1
        if rounds > 4 * constraints.n_rows + 4:
            for sid in contrib:
                del x[sid]
            continue
        total = sum(contrib.values())
        allowed = max(0.0, float(constraints.rhs[r]) - (float(usage[r]) - total))
        scale = allowed / total
        for sid in contrib:
            ctl = controls[sid]
            v = ctl.snap_down(x[sid] * scale)
            if v + tol < ctl.charge_min or v <= tol:
                del x[sid]
            else:
                x[sid] = v


def trim_to_rows(
    setpoints: Mapping[str, float],
    constraints: StepConstraints,
    controls: Mapping[str, Control],
    *,
    rank: Callable[[str, float], float] | None = None,
    tol: float = ROW_TOL,
) -> dict[str, float]:
    """Lower setpoints just enough that every row holds, most flexible first.

    Unlike :func:`fit_to_rows`, which scales every contributor to a violated
    row (and so pauses everyone near the minimum), this takes the excess from
    one setpoint at a time: the contributor with the highest
    ``rank(session_id, current_setpoint)`` (default 0) and then the largest
    margin above its minimum is lowered by the fewest resolution steps that
    clear the excess, never below its minimum. Only when every contributor is
    at its minimum is one paused. Every iteration lowers or pauses a setpoint,
    so this terminates.
    """
    x = {sid: v for sid, v in setpoints.items() if v}

    def key(sid: str) -> tuple[float, float, str]:
        r = 0.0 if rank is None else rank(sid, x[sid])
        return (-r, -(x[sid] - controls[sid].charge_min), sid)

    while True:
        usage = constraints.usage(x)
        excess = usage - constraints.rhs
        if not excess.size:
            return x
        r = int(np.argmax(excess))
        if excess[r] <= tol:
            return x
        contributors = [sid for sid in x if constraints.coefficient(sid)[r] > 0]
        margin = [sid for sid in contributors if x[sid] - controls[sid].charge_min > tol]
        if not margin:
            del x[min(contributors, key=key)]
            continue
        sid = min(margin, key=key)
        ctl = controls[sid]
        need = float(excess[r]) / float(constraints.coefficient(sid)[r])
        cut = min(x[sid] - ctl.charge_min, ctl.snap_up(need))
        x[sid] = max(ctl.charge_min, round(x[sid] - cut, 9))
