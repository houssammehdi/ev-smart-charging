"""Linear / mixed-integer formulation of the site charging problem (HiGHS via SciPy).

The same formulation serves three purposes:

* the perfect-foresight optimum (:class:`evcharge.policies.OptimalSchedule`),
* the rolling-horizon controller (:class:`evcharge.policies.ModelPredictiveControl`),
* a certified lower bound on the cost of *any* feasible schedule
  (:func:`relaxation_bound`).

See ``docs/theory.md`` for the mathematical statement. Decision variables:

==========  =======================================================
``x+, x-``  charging / discharging setpoint of session *s* in step *t* (kW, or A
            per phase), only in its window; ``x-`` only for bidirectional sessions
``soc``     battery energy of a session with a battery model at the end of step *t*
``g, e``    site import / export in step *t* (kW)
``P``       peak import over the horizon (kW), ``P >= peak_floor``
``u``       unmet energy of session *s* (kWh), penalised
``o``       energy overshoot of session *s* (kWh); see below
``w``       charge a full battery refuses in step *t* (kWh, battery side; see below)
``y``       binary on/off of each setpoint in step *t* (MILP only)
==========  =======================================================

Capacity is a set of per-step rows ``sum_s a[r, s] x[s, t] <= rhs[t, r]``: one
kW row on aggregate sites, plus one row per line (amperes) on phase-aware sites,
plus export rows on sites with bidirectional chargers
(:class:`evcharge.model.RowModel`). A session's grid power is
``kw_per_unit * (x+ - x-)``, so the same LP handles kW and ampere setpoints.

Charge-only sessions are modelled by their total energy, ``sum eta x+ dt + u - o = E``.
Sessions with a battery model (V2G) are modelled by their state of charge:
``soc_t = soc_{t-1} + eta x+_t dt - x-_t dt / eta_d`` within ``[min, max]``, and
``soc`` at departure plus ``u`` must reach the target. Degradation is charged per
kWh of battery throughput in both directions, and a binary per direction with
``y+ + y- <= 1`` keeps a bidirectional charger from charging and discharging at
once (the LP relaxation may do both, which only weakens the bound).

Minimum-current (IEC 61851 6 A) steps are modelled as semi-continuous variables,
``x = 0`` or ``x_min <= x <= x_max``, using the binaries ``y``. Because a
charge-only EV stops drawing power once its request is met, commanding the
minimum for the final few hundred Wh is allowed; the model captures this with an
overshoot variable ``o <= eta * x_min * kw_per_unit * dt``: energy that is
commanded but never drawn. The plan pays for it at the import price, and in
addition at the most negative export price of the session's window (zero if no
export price is negative), so that undrawn energy can never look like revenue.
With that term the MILP objective is an upper bound on the cost the simulator
measures when the plan is replayed, and the LP relaxation (no overshoot) is a
lower bound. A battery with a V2G spec stops accepting charge at its ceiling in
the same way, so where the minimum is enforced its state equation may spill up
to one minimum step: ``soc_t = ... - w_t`` with ``0 <= w_t <= eta x+_t dt``,
priced like the overshoot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

from evcharge.model import POWER_TOL_KW, FloatArray, RowModel, Scenario, on_grid

DEFAULT_UNMET_PENALTY_EUR_PER_KWH = 100.0
"""Penalty on undelivered energy. It must exceed the marginal cost of delivering
one more kWh (``price + demand_charge / (eta * dt)``) so that the optimiser
never trades energy for money; 100 EUR/kWh comfortably does for realistic tariffs."""


class SolverError(RuntimeError):
    """Raised when HiGHS fails to return a usable solution."""


@dataclass(frozen=True)
class LPSession:
    """A session as seen by the optimiser, in kW (times are local to the LP horizon).

    This is the original aggregate-model input; :class:`FlexLoad` generalises it
    to other setpoint units and constraint rows.

    Attributes:
        id: Session id.
        start: First local step the EV is available.
        end: First local step the EV is gone (exclusive).
        energy_kwh: Energy still to be delivered into the battery.
        efficiency: Grid-to-battery efficiency.
        p_min_kw: Minimum non-zero power.
        p_max_kw: Maximum power.
    """

    id: str
    start: int
    end: int
    energy_kwh: float
    efficiency: float
    p_min_kw: float
    p_max_kw: float


@dataclass(frozen=True)
class FlexLoad:
    """A session as seen by the optimiser, in its own setpoint unit.

    Attributes:
        id: Session id.
        start: First local step the EV is available.
        end: First local step the EV is gone (exclusive).
        energy_kwh: Energy still to be delivered into the battery. With a
            ``battery`` model this is the required net change of the battery
            energy by ``end`` and may be negative.
        charge_max: Maximum setpoint.
        charge_min: Minimum non-zero setpoint (0: continuous).
        kw_per_unit: Grid power per setpoint unit (1 for kW setpoints).
        efficiency: Grid-to-battery efficiency.
        rows: Coefficient of the setpoint in every constraint row of the
            problem (``None``: 1 on every row, the aggregate model).
        step: Setpoint resolution (0: continuous); ``charge_min`` and
            ``charge_max`` must be multiples of it. Used by
            ``solve_schedule(..., round_to_grid=True)``.
        discharge_max: Maximum discharge magnitude (0: charge only).
        discharge_min: Minimum non-zero discharge magnitude.
        discharge_efficiency: Battery-to-grid efficiency.
        discharge_rows: Coefficient of the discharge magnitude in every row
            (required with ``rows`` when ``discharge_max > 0``).
        battery: ``(initial, min, max)`` battery energy in kWh. Set it to model
            the state of charge (required for discharging); the session then
            leaves with at least ``initial + energy_kwh``.
        degradation_eur_per_kwh: Cost per kWh of battery throughput (both directions).
    """

    id: str
    start: int
    end: int
    energy_kwh: float
    charge_max: float
    charge_min: float = 0.0
    kw_per_unit: float = 1.0
    efficiency: float = 1.0
    rows: tuple[float, ...] | None = None
    step: float = 0.0
    discharge_max: float = 0.0
    discharge_min: float = 0.0
    discharge_efficiency: float = 1.0
    discharge_rows: tuple[float, ...] | None = None
    battery: tuple[float, float, float] | None = None
    degradation_eur_per_kwh: float = 0.0

    def __post_init__(self) -> None:
        if self.discharge_max > 0.0 and self.battery is None:
            raise ValueError(f"session {self.id}: discharging needs a battery model")
        if self.discharge_max > 0.0 and self.rows is not None and self.discharge_rows is None:
            raise ValueError(f"session {self.id}: discharging needs discharge row coefficients")

    @classmethod
    def from_lp_session(cls, s: LPSession) -> FlexLoad:
        """The kW-model session as a :class:`FlexLoad`."""
        return cls(s.id, s.start, s.end, s.energy_kwh, s.p_max_kw, s.p_min_kw, 1.0, s.efficiency)


Strategy = Literal["exact", "relax-and-fix"]
"""How the minimum-power rule is solved: a full MILP, or the LP relaxation
followed by a small MILP over the entries the relaxation left fractional."""


@dataclass(frozen=True, eq=False)
class ScheduleProblem:
    """Input of :func:`solve_schedule`; all series have the same length ``T``.

    Attributes:
        dt_h: Step length in hours.
        price: Import price per step (EUR/kWh).
        export_price: Export price per step (EUR/kWh).
        net_base_kw: Base load minus PV per step (kW).
        grid_limit_kw: Import limit at the connection point (used when
            ``rows`` is ``None``).
        demand_charge: EUR per kW of peak import.
        sessions: Sessions to schedule.
        peak_floor_kw: Peak import already incurred (MPC); the peak cost is
            ``demand_charge * max(peak_floor_kw, future peak)``.
        unmet_penalty: EUR per kWh of undelivered energy.
        min_power_steps: Enforce the minimum-power rule exactly (binaries) in
            the first ``min_power_steps`` local steps; ``None`` means all steps
            and ``0`` gives a pure LP (minimum power relaxed).
        quick_charge_weight: Optional regulariser in EUR per kWh per hour of
            delay that favours charging early; 0 gives the pure cost objective.
        rows: Per-step capacity rows over the local horizon. ``None`` means the
            aggregate model's single row ``sum x <= grid_limit_kw - net_base_kw``.
    """

    dt_h: float
    price: FloatArray
    export_price: FloatArray
    net_base_kw: FloatArray
    grid_limit_kw: float
    demand_charge: float
    sessions: tuple[LPSession | FlexLoad, ...]
    peak_floor_kw: float = 0.0
    unmet_penalty: float = DEFAULT_UNMET_PENALTY_EUR_PER_KWH
    min_power_steps: int | None = None
    quick_charge_weight: float = 0.0
    rows: RowModel | None = None

    @property
    def n_steps(self) -> int:
        """Number of steps ``T``."""
        return int(self.price.size)


@dataclass(frozen=True, eq=False)
class ScheduleSolution:
    """Output of :func:`solve_schedule`.

    Attributes:
        power_kw: Commanded grid power, shape ``(n_sessions, T)``.
        unmet_kwh: Undelivered energy per session.
        import_kw: Site import per step.
        export_kw: Site export per step.
        peak_kw: Optimised peak variable ``P``.
        objective: Objective value (EUR, including penalties and regulariser).
        lower_bound: Proven lower bound on the optimal objective (LP relaxation
            or MILP dual bound); equals ``objective`` for pure LPs.
        status: ``"optimal"`` or ``"time_limit"`` (MILP with incumbent).
        n_binaries: Number of binary variables in the final solve.
        solve_time_s: Wall-clock time spent in the solver(s).
        setpoint: Commanded setpoints in each session's unit (equal to
            ``power_kw`` for kW setpoints).
        n_rounding: Integer variables of the rounding stage
            (``round_to_grid=True``), 0 otherwise.
        battery_kwh: Planned battery energy at the end of each step for sessions
            with a battery model (NaN elsewhere); ``None`` if there are none.
    """

    power_kw: FloatArray
    unmet_kwh: FloatArray
    import_kw: FloatArray
    export_kw: FloatArray
    peak_kw: float
    objective: float
    lower_bound: float
    status: str
    n_binaries: int
    solve_time_s: float
    setpoint: FloatArray
    n_rounding: int = 0
    battery_kwh: FloatArray | None = None

    @property
    def gap(self) -> float:
        """Relative gap between :attr:`objective` and :attr:`lower_bound`."""
        return max(0.0, self.objective - self.lower_bound) / max(1e-9, abs(self.objective))


ROUNDING_WINDOWS = (1, 0)
"""Neighbourhoods tried by the rounding stage: a setpoint may move to grid values
within this many steps beyond its floor or ceiling (1 first, then plain
floor-or-ceiling if HiGHS finds no incumbent in time)."""

ROUNDING_MIP_GAP = 0.01
"""Relative gap at which the rounding MILP stops. A gap (not a time limit) keeps
the result independent of machine speed; 1 % of the objective is well below
the resolution effects the stage removes."""

DEFAULT_ROUNDING_TIME_LIMIT_S = 60.0
"""Safety time limit of each rounding MILP (the gap normally stops it in seconds)."""

NUMERIC_ZERO = 1e-9
"""Coefficients smaller than this in magnitude are treated as exactly zero.

HiGHS can stall on MILPs whose data contains denormal-scale values (for example a
price of 1e-308); snapping them to zero changes the objective by a negligible amount.
"""


def _snap(values: FloatArray) -> FloatArray:
    return np.where(np.abs(values) < NUMERIC_ZERO, 0.0, values)


def _grid_floor(values: FloatArray, step: FloatArray) -> FloatArray:
    safe = np.where(step > 0.0, step, 1.0)
    return np.asarray(step * np.floor(values / safe + 1e-6), dtype=np.float64)


def _grid_ceil(values: FloatArray, step: FloatArray) -> FloatArray:
    safe = np.where(step > 0.0, step, 1.0)
    return np.asarray(step * np.ceil(values / safe - 1e-6), dtype=np.float64)


def _milp_bound(res: object) -> float:
    """Proven lower bound of a :func:`scipy.optimize.milp` result.

    HiGHS reports its dual bound as ``mip_dual_bound``. A bound of exactly 0.0
    is valid (and common), so only a missing or non-finite bound falls back to
    the incumbent objective.
    """
    bound = getattr(res, "mip_dual_bound", None)
    if bound is None or not np.isfinite(bound):
        return float(getattr(res, "fun"))  # noqa: B009 - duck-typed result object
    return float(bound)


def _flex_loads(problem: ScheduleProblem) -> tuple[list[FlexLoad], FloatArray, int]:
    """Sessions as :class:`FlexLoad`, the row right-hand sides ``(T, R)`` and ``R``."""
    loads = [
        s if isinstance(s, FlexLoad) else FlexLoad.from_lp_session(s) for s in problem.sessions
    ]
    if problem.rows is None:
        rhs = np.maximum(0.0, problem.grid_limit_kw - problem.net_base_kw)[:, None]
    else:
        rhs = np.asarray(problem.rows.rhs, dtype=np.float64)
        if rhs.shape[0] != problem.n_steps:
            raise ValueError(f"rows cover {rhs.shape[0]} steps, the problem {problem.n_steps}")
    n_rows = int(rhs.shape[1])
    for s in loads:
        if s.rows is not None and len(s.rows) != n_rows:
            raise ValueError(f"session {s.id}: {len(s.rows)} row coefficients for {n_rows} rows")
        if s.rows is None and n_rows != 1:
            raise ValueError(f"session {s.id}: row coefficients are required with {n_rows} rows")
        if s.discharge_rows is not None and len(s.discharge_rows) != n_rows:
            raise ValueError(
                f"session {s.id}: {len(s.discharge_rows)} discharge coefficients for {n_rows} rows"
            )
    if problem.rows is not None:
        rhs = _floor_line_rows(rhs, problem.rows.kinds, loads)
    return loads, rhs, n_rows


def _floor_line_rows(rhs: FloatArray, kinds: tuple[str, ...], loads: list[FlexLoad]) -> FloatArray:
    """Round line rows down to the setpoint grid of the sessions that load them.

    A line row sums per-phase currents with coefficients 0 or 1. If every
    session on the row has a resolution, the left-hand side of any executable
    schedule is a multiple of the finest one, so rounding the right-hand side
    down to that grid removes no executable schedule; it only tightens the LP
    (whose bound therefore stays valid) and keeps plans away from capacity no
    grid schedule can use.
    """
    out = rhs.copy()
    for r, kind in enumerate(kinds):
        if kind != "line":
            continue
        coefficients = [
            (s, a[r])
            for s in loads
            for a in (s.rows, s.discharge_rows if s.discharge_max > 0.0 else None)
            if a is not None and a[r] != 0.0
        ]
        if not coefficients or any(s.step <= 0.0 or abs(a) != 1.0 for s, a in coefficients):
            continue
        grid = min(s.step for s, _ in coefficients)
        if any(not on_grid(s.step, grid) for s, _ in coefficients):
            continue
        out[:, r] = grid * np.floor(out[:, r] / grid + 1e-6)
    return out


class _Model:
    """Sparse matrices of one :class:`ScheduleProblem`, reusable across solves.

    Setpoint columns come first: the charging entries of every session, then
    the discharging entries of bidirectional sessions. Charge-only problems
    therefore have exactly the layout of the original kW formulation.
    """

    def __init__(self, problem: ScheduleProblem) -> None:
        self.problem = problem
        n_t = problem.n_steps
        sessions, rhs, n_rows = _flex_loads(problem)
        n_s = len(sessions)
        dt = problem.dt_h
        for s in sessions:
            if not 0 <= s.start < s.end <= n_t:
                raise ValueError(f"session {s.id}: window [{s.start}, {s.end}) outside [0, {n_t})")
        length = np.array([s.end - s.start for s in sessions], dtype=np.int64)
        c_offset = np.concatenate([[0], np.cumsum(length)])[:-1]
        c_sess = np.repeat(np.arange(n_s), length)
        c_time = np.array([t for s in sessions for t in range(s.start, s.end)], dtype=np.int64)
        two_way = np.array([s.discharge_max > 0.0 for s in sessions], dtype=bool)
        d_sess = c_sess[two_way[c_sess]]
        d_time = c_time[two_way[c_sess]]
        n_c, n_d = int(c_sess.size), int(d_sess.size)
        n_p = n_c + n_d
        self.n_t, self.n_s, self.n_p, self.n_c = n_t, n_s, n_p, n_c
        self.p_sess = np.concatenate([c_sess, d_sess])
        self.p_time = np.concatenate([c_time, d_time])
        self.p_dir = np.concatenate([np.ones(n_c), -np.ones(n_d)])
        # complementarity partner of every setpoint column (-1: none)
        self.partner = np.full(n_p, -1, dtype=np.int64)
        starts = np.array([s.start for s in sessions], dtype=np.int64)
        charge_of_d = c_offset[d_sess] + (d_time - starts[d_sess])
        self.partner[n_c + np.arange(n_d)] = charge_of_d
        self.partner[charge_of_d] = n_c + np.arange(n_d)

        def per(values: list[float]) -> FloatArray:
            return np.asarray(values, dtype=np.float64)

        c_min = per([s.charge_min for s in sessions])
        unit_kw = per([s.kw_per_unit for s in sessions])
        eta = per([s.efficiency for s in sessions])
        eta_d = per([s.discharge_efficiency for s in sessions])
        energy = per([s.energy_kwh for s in sessions])
        wear = per([s.degradation_eur_per_kwh for s in sessions])
        self.unit_kw = unit_kw
        self.p_max = np.concatenate(
            [
                per([s.charge_max for s in sessions])[c_sess],
                per([s.discharge_max for s in sessions])[d_sess],
            ]
        )
        self.p_min = np.concatenate(
            [c_min[c_sess], per([s.discharge_min for s in sessions])[d_sess]]
        )
        self.step = per([s.step for s in sessions])[self.p_sess]
        k_c, k_d = unit_kw[c_sess], unit_kw[d_sess]

        tracked = np.array([s.battery is not None for s in sessions], dtype=bool)
        batteries = np.array(
            [s.battery if s.battery is not None else (0.0, 0.0, 0.0) for s in sessions]
        ).reshape(n_s, 3)
        soc_mask = tracked[c_sess]
        soc_sess, soc_time = c_sess[soc_mask], c_time[soc_mask]
        n_soc = int(soc_sess.size)
        self.soc_sess, self.soc_time = soc_sess, soc_time
        loose = np.flatnonzero(~tracked)
        held = np.flatnonzero(tracked)
        target = batteries[:, 0] + energy

        int_horizon = n_t if problem.min_power_steps is None else problem.min_power_steps
        early = self.p_time < int_horizon
        #: setpoint entries that need a binary: the minimum-current rule, or a
        #: bidirectional charger that must not charge and discharge at once
        self.eligible = early & ((self.p_min > POWER_TOL_KW) | two_way[self.p_sess])
        has_min = np.zeros(n_s, dtype=bool)
        c_elig = self.eligible[:n_c] & (self.p_min[:n_c] > POWER_TOL_KW)
        has_min[c_sess[c_elig]] = True
        has_min &= ~tracked

        # A battery at its ceiling stops accepting charge (the battery management
        # cuts off), so a minimum-current command may put less into it than
        # commanded. The spill w (battery side, per tracked entry) is that part;
        # like the overshoot it is only needed where the minimum is enforced.
        soc_charge_col = np.flatnonzero(soc_mask)
        spill_ub = np.where(c_elig[soc_charge_col], (eta * c_min * unit_kw)[soc_sess] * dt, 0.0)

        self.off_soc = n_p
        self.off_g = n_p + n_soc
        self.off_e = self.off_g + n_t
        self.off_peak = self.off_e + n_t
        self.off_u = self.off_peak + 1
        self.off_o = self.off_u + n_s
        self.off_w = self.off_o + n_s
        self.n_cont = self.off_w + n_soc

        c = np.zeros(self.n_cont)
        c[:n_c] = (
            problem.quick_charge_weight * c_time * dt * (k_c * dt)
            + wear[c_sess] * eta[c_sess] * k_c * dt
        )
        c[n_c:n_p] = wear[d_sess] * k_d * dt / eta_d[d_sess]
        c[self.off_g : self.off_e] = problem.price * dt
        c[self.off_e : self.off_peak] = -problem.export_price * dt
        c[self.off_peak] = problem.demand_charge
        c[self.off_u : self.off_o] = problem.unmet_penalty
        # Overshoot is paid for at the import price but never drawn. Undrawn grid
        # energy can raise the executed cost by at most the most negative export
        # price in the window (the cost of a step is convex in its consumption with
        # slopes between the export and the import price), so charge that here:
        # the objective then bounds the executed cost from above.
        if n_s:
            window_min_export = np.array(
                [float(problem.export_price[s.start : s.end].min()) for s in sessions]
            )
            c[self.off_o : self.off_w] = np.maximum(0.0, -window_min_export) / eta
            c[self.off_w :] = (np.maximum(0.0, -window_min_export) / eta)[soc_sess]
        self.c = _snap(c)

        lb = np.zeros(self.n_cont)
        ub = np.full(self.n_cont, np.inf)
        lb[self.off_soc : self.off_g] = batteries[soc_sess, 1]
        ub[self.off_soc : self.off_g] = batteries[soc_sess, 2]
        floor = problem.peak_floor_kw
        lb[self.off_peak] = floor if floor >= NUMERIC_ZERO else 0.0
        ub[self.off_u : self.off_o] = np.where(
            tracked, np.maximum(0.0, target - batteries[:, 1]), energy
        )
        # Commanding the minimum to finish a request is allowed: the EV stops by itself.
        ub[self.off_o : self.off_w] = np.where(has_min, eta * c_min * unit_kw * dt, 0.0)
        ub[self.off_w :] = spill_ub
        self.lb, self.ub = lb, ub

        cols_c = np.arange(n_c)
        cols_d = n_c + np.arange(n_d)
        t_idx = np.arange(n_t)
        # Equalities. Power balance (row t): sum k (x+ - x-) - g + e = -(base - pv).
        # Energy of charge-only sessions (row n_t + j): sum eta k x+ dt + u - o = E.
        # Battery of tracked sessions: soc_t - soc_{t-1} - eta k x+ dt + k x- dt / eta_d
        # + w_t = initial (first step) or 0.
        j_of = np.full(n_s, -1, dtype=np.int64)
        j_of[loose] = np.arange(loose.size)
        e_rows = c_sess[~tracked[c_sess]]
        soc_base = n_t + loose.size
        soc_row = soc_base + np.arange(n_soc)
        soc_col = self.off_soc + np.arange(n_soc)
        first = np.zeros(n_soc, dtype=bool)
        if n_soc:
            first[0] = True
            first[1:] = soc_sess[1:] != soc_sess[:-1]
        soc_row_of_c = np.full(n_c, -1, dtype=np.int64)
        soc_row_of_c[np.flatnonzero(soc_mask)] = soc_row
        d_soc_row = soc_row_of_c[charge_of_d]
        rows = np.concatenate(
            [
                c_time,
                d_time,
                t_idx,
                t_idx,
                n_t + j_of[e_rows],
                n_t + np.arange(loose.size),
                n_t + np.arange(loose.size),
                soc_row,
                soc_row[~first],
                soc_row_of_c[soc_mask],
                d_soc_row,
                soc_row,
            ]
        )
        cols = np.concatenate(
            [
                cols_c,
                cols_d,
                self.off_g + t_idx,
                self.off_e + t_idx,
                cols_c[~tracked[c_sess]],
                self.off_u + loose,
                self.off_o + loose,
                soc_col,
                soc_col[:-1][~first[1:]] if n_soc else soc_col,
                cols_c[soc_mask],
                cols_d,
                self.off_w + np.arange(n_soc),
            ]
        )
        vals = np.concatenate(
            [
                k_c,
                -k_d,
                -np.ones(n_t),
                np.ones(n_t),
                eta[e_rows] * k_c[~tracked[c_sess]] * dt,
                np.ones(loose.size),
                -np.ones(loose.size),
                np.ones(n_soc),
                -np.ones(int(np.count_nonzero(~first))),
                -eta[soc_sess] * unit_kw[soc_sess] * dt,
                k_d * dt / eta_d[d_sess],
                np.ones(n_soc),
            ]
        )
        n_eq = soc_base + n_soc
        self.a_eq = coo_matrix((vals, (rows, cols)), shape=(n_eq, self.n_cont)).tocsr()
        soc_rhs = np.where(first, batteries[soc_sess, 0], 0.0)
        self.b_eq = _snap(np.concatenate([-problem.net_base_kw, energy[loose], soc_rhs]))
        # Inequalities. Capacity (row t * R + r): sum a x <= rhs[t, r]; peak: g - P <= 0;
        # departure of tracked sessions: -soc_last - u <= -target; spill never exceeds
        # the commanded charge: w - eta k x+ dt <= 0.
        coef = np.array(
            [s.rows if s.rows is not None else (1.0,) * n_rows for s in sessions], dtype=np.float64
        ).reshape(n_s, n_rows)
        d_coef = np.array(
            [
                s.discharge_rows if s.discharge_rows is not None else (1.0,) * n_rows
                for s in sessions
            ],
            dtype=np.float64,
        ).reshape(n_s, n_rows)
        entry_coef = coef[c_sess]
        nz_entry, nz_row = np.nonzero(entry_coef)
        d_entry_coef = d_coef[d_sess]
        nz_d, nz_d_row = np.nonzero(d_entry_coef)
        n_cap = n_t * n_rows
        last = np.zeros(n_soc, dtype=bool)
        if n_soc:
            last[-1] = True
            last[:-1] = soc_sess[:-1] != soc_sess[1:]
        dep_row = n_cap + n_t + np.arange(held.size)
        spill = np.flatnonzero(spill_ub > 0.0)
        spill_row = n_cap + n_t + held.size + np.arange(spill.size)
        rows = np.concatenate(
            [
                c_time[nz_entry] * n_rows + nz_row,
                d_time[nz_d] * n_rows + nz_d_row,
                n_cap + t_idx,
                n_cap + t_idx,
                dep_row,
                dep_row,
                spill_row,
                spill_row,
            ]
        )
        cols = np.concatenate(
            [
                cols_c[nz_entry],
                cols_d[nz_d],
                self.off_g + t_idx,
                np.full(n_t, self.off_peak),
                soc_col[last],
                self.off_u + held,
                self.off_w + spill,
                soc_charge_col[spill],
            ]
        )
        vals = np.concatenate(
            [
                entry_coef[nz_entry, nz_row],
                d_entry_coef[nz_d, nz_d_row],
                np.ones(n_t),
                -np.ones(n_t),
                -np.ones(held.size),
                -np.ones(held.size),
                np.ones(spill.size),
                -(eta * unit_kw)[soc_sess[spill]] * dt,
            ]
        )
        n_ub = n_cap + n_t + held.size + spill.size
        self.a_ub = coo_matrix((vals, (rows, cols)), shape=(n_ub, self.n_cont)).tocsr()
        self.b_ub = np.concatenate(
            [_snap(rhs).reshape(-1), np.zeros(n_t), -target[held], np.zeros(spill.size)]
        )

    def solve(
        self,
        p_lb: FloatArray,
        p_ub: FloatArray,
        binaries: npt.NDArray[np.int64],
        *,
        time_limit_s: float | None,
        mip_rel_gap: float,
    ) -> tuple[FloatArray, float, float, str]:
        """Solve with the given setpoint bounds; ``binaries`` index semi-continuous entries.

        Returns ``(x, objective, lower_bound, status)``.
        """
        lb = self.lb.copy()
        ub = self.ub.copy()
        lb[: self.n_p] = p_lb
        ub[: self.n_p] = p_ub
        n_y = int(binaries.size)
        if n_y == 0:
            res = linprog(
                self.c,
                A_ub=self.a_ub,
                b_ub=self.b_ub,
                A_eq=self.a_eq,
                b_eq=self.b_eq,
                bounds=np.column_stack([lb, ub]),
                method="highs",
            )
            if res.status != 0 or res.x is None:
                raise SolverError(f"HiGHS LP failed: {res.message}")
            return np.asarray(res.x, dtype=np.float64), float(res.fun), float(res.fun), "optimal"

        # x_i - x_max y_i <= 0 and x_min y_i - x_i <= 0 for every binary entry i, and
        # y+ + y- <= 1 for the two directions of a bidirectional charger.
        n_var = self.n_cont + n_y
        y_cols = self.n_cont + np.arange(n_y)
        y_rows = np.arange(n_y)
        position = np.full(self.n_p, -1, dtype=np.int64)
        position[binaries] = np.arange(n_y)
        charge_y = np.flatnonzero((self.p_dir[binaries] > 0) & (self.partner[binaries] >= 0))
        pair_y = position[self.partner[binaries[charge_y]]]
        charge_y, pair_y = charge_y[pair_y >= 0], pair_y[pair_y >= 0]
        n_pair = int(charge_y.size)
        pair_rows = 2 * n_y + np.arange(n_pair)
        link = coo_matrix(
            (
                np.concatenate(
                    [
                        np.ones(n_y),
                        -self.p_max[binaries],
                        -np.ones(n_y),
                        self.p_min[binaries],
                        np.ones(2 * n_pair),
                    ]
                ),
                (
                    np.concatenate(
                        [y_rows, y_rows, n_y + y_rows, n_y + y_rows, pair_rows, pair_rows]
                    ),
                    np.concatenate(
                        [binaries, y_cols, binaries, y_cols, y_cols[charge_y], y_cols[pair_y]]
                    ),
                ),
            ),
            shape=(2 * n_y + n_pair, n_var),
        )
        pad = csr_matrix((self.a_eq.shape[0], n_y))
        a_eq = hstack([self.a_eq, pad]).tocsr()
        a_ub = vstack([hstack([self.a_ub, csr_matrix((self.a_ub.shape[0], n_y))]), link]).tocsr()
        b_ub = np.concatenate([self.b_ub, np.zeros(2 * n_y), np.ones(n_pair)])
        integrality = np.concatenate([np.zeros(self.n_cont), np.ones(n_y)])
        # HiGHS presolve is disabled for MILPs: on some tiny instances its postsolve
        # path re-runs the solver and prints debug output to stdout; the MILPs built
        # here are small (MPC) or already reduced by the LP (relax-and-fix).
        options: dict[str, float | bool] = {
            "disp": False,
            "presolve": False,
            "mip_rel_gap": mip_rel_gap,
        }
        if time_limit_s is not None:
            options["time_limit"] = time_limit_s
        res = milp(
            np.concatenate([self.c, np.zeros(n_y)]),
            constraints=[
                LinearConstraint(a_eq, self.b_eq, self.b_eq),
                LinearConstraint(a_ub, -np.inf, b_ub),
            ],
            integrality=integrality,
            bounds=Bounds(np.concatenate([lb, np.zeros(n_y)]), np.concatenate([ub, np.ones(n_y)])),
            options=options,
        )
        if res.x is None or res.status not in (0, 1):
            raise SolverError(f"HiGHS MILP failed: {res.message}")
        x = np.asarray(res.x, dtype=np.float64)
        on = x[self.n_cont :] > 0.5
        p = x[binaries]
        x[binaries] = np.where(on, np.maximum(p, self.p_min[binaries]), 0.0)
        bound = _milp_bound(res)
        return (
            x[: self.n_cont],
            float(res.fun),
            bound,
            "optimal" if res.status == 0 else "time_limit",
        )

    def round_to_grid(
        self, x: FloatArray, *, time_limit_s: float | None, mip_rel_gap: float
    ) -> tuple[FloatArray, float, str, int]:
        """Put every setpoint with a resolution on its grid, optimally near ``x``.

        ``x`` must already satisfy the minimum-current rule. Every setpoint that
        is on and has a resolution may move to a grid value within one step of
        its floor or ceiling (never outside its range): ``lo + step * n`` with
        integer ``n``; setpoints that are off stay off and continuous setpoints
        keep their on/off state. All rows, energy balances and the objective are
        unchanged, so the result is the cheapest grid schedule in that
        neighbourhood. If HiGHS returns no incumbent in the time limit, the
        plain floor-or-ceiling neighbourhood is tried, and as a last resort
        every setpoint is rounded down, which is always feasible (unmet energy
        is a slack).

        Returns ``(x, objective, status, n_integer_variables)``.
        """
        for window in ROUNDING_WINDOWS:
            found = self._round_window(x, window, time_limit_s, mip_rel_gap)
            if found is not None:
                return found
        p = x[: self.n_p]
        floor = np.where(self.step > 0.0, _grid_floor(p, self.step), p)
        on = p > POWER_TOL_KW
        p_lb = np.where(on, np.maximum(floor, self.p_min), 0.0)
        p_ub = np.where(on, np.maximum(floor, self.p_min), 0.0)
        cont = (self.step <= 0.0) & (on | (self.p_min <= POWER_TOL_KW))
        p_ub[cont] = self.p_max[cont]
        try:
            x_lp, obj, _, _ = self.solve(
                p_lb, p_ub, np.zeros(0, np.int64), time_limit_s=None, mip_rel_gap=mip_rel_gap
            )
        except SolverError:
            # rounding every setpoint down can break a battery bound; keep the plan
            # continuous and let the replay round it step by step
            return x, float(self.c @ x), "unrounded", 0
        return x_lp, obj, "time_limit", 0

    def _round_window(
        self, x: FloatArray, window: int, time_limit_s: float | None, mip_rel_gap: float
    ) -> tuple[FloatArray, float, str, int] | None:
        p = x[: self.n_p]
        step = self.step
        quantized = step > 0.0
        on = p > POWER_TOL_KW
        lo = np.where(quantized, _grid_floor(p, step) - window * step, p)
        hi = np.where(quantized, _grid_ceil(p, step) + window * step, p)
        lo = np.maximum(lo, self.p_min)
        hi = np.minimum(hi, self.p_max)
        p_lb = np.where(quantized, lo, np.where(on, self.p_min, 0.0))
        p_ub = np.where(quantized, hi, np.where(on | (self.p_min <= POWER_TOL_KW), self.p_max, 0.0))
        p_lb[~on] = 0.0
        p_ub[quantized & ~on] = 0.0
        var = np.flatnonzero(quantized & on & (hi - lo > POWER_TOL_KW))
        lb = self.lb.copy()
        ub = self.ub.copy()
        lb[: self.n_p] = p_lb
        ub[: self.n_p] = p_ub
        n_z = int(var.size)
        if n_z == 0:
            x_lp, obj, _, status = self.solve(
                p_lb, p_ub, np.zeros(0, np.int64), time_limit_s=None, mip_rel_gap=mip_rel_gap
            )
            return x_lp, obj, status, 0
        # x_j - step_j n_j = lo_j with integer n_j in [0, (hi_j - lo_j) / step_j]
        z_cols = self.n_cont + np.arange(n_z)
        z_rows = np.arange(n_z)
        link = coo_matrix(
            (
                np.concatenate([np.ones(n_z), -step[var]]),
                (np.concatenate([z_rows, z_rows]), np.concatenate([var, z_cols])),
            ),
            shape=(n_z, self.n_cont + n_z),
        )
        a_eq = vstack([hstack([self.a_eq, csr_matrix((self.a_eq.shape[0], n_z))]), link]).tocsr()
        b_eq = np.concatenate([self.b_eq, lo[var]])
        a_ub = hstack([self.a_ub, csr_matrix((self.a_ub.shape[0], n_z))]).tocsr()
        n_max = np.round((hi[var] - lo[var]) / step[var])
        options: dict[str, float | bool] = {
            "disp": False,
            "presolve": False,
            "mip_rel_gap": mip_rel_gap,
        }
        if time_limit_s is not None:
            options["time_limit"] = time_limit_s
        res = milp(
            np.concatenate([self.c, np.zeros(n_z)]),
            constraints=[
                LinearConstraint(a_eq, b_eq, b_eq),
                LinearConstraint(a_ub, -np.inf, self.b_ub),
            ],
            integrality=np.concatenate([np.zeros(self.n_cont), np.ones(n_z)]),
            bounds=Bounds(np.concatenate([lb, np.zeros(n_z)]), np.concatenate([ub, n_max])),
            options=options,
        )
        if res.x is None or res.status not in (0, 1):
            return None
        xr = np.asarray(res.x, dtype=np.float64)
        xr[var] = lo[var] + step[var] * np.round(xr[self.n_cont :])
        status = "optimal" if res.status == 0 else "time_limit"
        return xr[: self.n_cont], float(res.fun), status, n_z

    def to_solution(
        self, x: FloatArray, objective: float, bound: float, status: str, n_y: int, elapsed: float
    ) -> ScheduleSolution:
        p = np.clip(x[: self.n_p], 0.0, self.p_max)
        p[p < POWER_TOL_KW] = 0.0
        setpoint = np.zeros((self.n_s, self.n_t))
        np.add.at(setpoint, (self.p_sess, self.p_time), self.p_dir * p)
        battery = None
        if self.soc_sess.size:
            battery = np.full((self.n_s, self.n_t), np.nan)
            battery[self.soc_sess, self.soc_time] = x[self.off_soc : self.off_g]
        return ScheduleSolution(
            power_kw=setpoint * self.unit_kw[:, None],
            unmet_kwh=np.maximum(0.0, x[self.off_u : self.off_o]),
            import_kw=x[self.off_g : self.off_e],
            export_kw=x[self.off_e : self.off_peak],
            peak_kw=float(x[self.off_peak]),
            objective=objective,
            lower_bound=min(bound, objective),
            status=status,
            n_binaries=n_y,
            solve_time_s=elapsed,
            setpoint=setpoint,
            battery_kwh=battery,
        )


def solve_schedule(
    problem: ScheduleProblem,
    *,
    strategy: Strategy = "exact",
    time_limit_s: float | None = None,
    mip_rel_gap: float = 1e-6,
    round_to_grid: bool = False,
    rounding_time_limit_s: float | None = DEFAULT_ROUNDING_TIME_LIMIT_S,
) -> ScheduleSolution:
    """Solve the charging LP/MILP with HiGHS.

    Uses :func:`scipy.optimize.linprog` (``method="highs"``) when no binaries
    are needed and :func:`scipy.optimize.milp` otherwise.

    Args:
        problem: The problem instance.
        strategy: ``"exact"`` solves the full MILP. ``"relax-and-fix"`` solves
            the LP relaxation, fixes every entry the relaxation left at 0 (off)
            or at >= the minimum (on), and solves a small MILP over the
            remaining fractional entries. It is a heuristic, but the relaxation
            gives a certified :attr:`ScheduleSolution.lower_bound`.
        time_limit_s: HiGHS time limit for MILP solves.
        mip_rel_gap: Relative MIP gap at which HiGHS stops.
        round_to_grid: Afterwards put every setpoint of a session with a
            resolution (:attr:`FlexLoad.step`) on its grid with a rounding
            MILP (see :meth:`_Model.round_to_grid`).
        rounding_time_limit_s: Safety time limit of each rounding MILP; it
            normally stops at :data:`ROUNDING_MIP_GAP`.

    Raises:
        SolverError: if HiGHS reports failure or returns no solution.
    """
    started = time.perf_counter()
    model = _Model(problem)
    p_lb = np.zeros(model.n_p)
    p_ub = model.p_max.copy()
    eligible = np.flatnonzero(model.eligible)
    if eligible.size == 0 or strategy == "exact":
        x, obj, bound, status = model.solve(
            p_lb, p_ub, eligible, time_limit_s=time_limit_s, mip_rel_gap=mip_rel_gap
        )
        n_y = int(eligible.size)
    else:
        x_lp, bound, _, _ = model.solve(
            p_lb, p_ub, np.zeros(0, np.int64), time_limit_s=None, mip_rel_gap=mip_rel_gap
        )
        values = x_lp[: model.n_p]
        p_lp = values[eligible]
        partner = model.partner[eligible]
        other = np.where(partner >= 0, values[np.maximum(partner, 0)], 0.0)
        off = p_lp <= POWER_TOL_KW
        # on in its own direction, with the opposite direction (if any) off
        on = (p_lp >= model.p_min[eligible] - POWER_TOL_KW) & (other <= POWER_TOL_KW)
        p_ub[eligible[off]] = 0.0
        p_lb[eligible[on]] = model.p_min[eligible[on]]
        fractional = eligible[~off & ~on]
        try:
            x, obj, _, status = model.solve(
                p_lb, p_ub, fractional, time_limit_s=time_limit_s, mip_rel_gap=mip_rel_gap
            )
            n_y = int(fractional.size)
        except SolverError:
            # fixing can make a battery trajectory infeasible; fall back to the full MILP
            x, obj, _, status = model.solve(
                np.zeros(model.n_p),
                model.p_max.copy(),
                eligible,
                time_limit_s=time_limit_s,
                mip_rel_gap=mip_rel_gap,
            )
            n_y = int(eligible.size)
    n_z = 0
    if round_to_grid and np.any(model.step > 0.0):
        x, obj, round_status, n_z = model.round_to_grid(
            x, time_limit_s=rounding_time_limit_s, mip_rel_gap=ROUNDING_MIP_GAP
        )
        if round_status != "optimal":
            status = round_status
    solution = model.to_solution(x, obj, bound, status, n_y, time.perf_counter() - started)
    return replace(solution, n_rounding=n_z)


def flex_loads_from_scenario(scenario: Scenario) -> tuple[FlexLoad, ...]:
    """Every session of ``scenario`` as a :class:`FlexLoad` on the full horizon."""
    loads = []
    for s in scenario.sessions:
        ctl = scenario.control(s)
        v2g = s.v2g
        loads.append(
            FlexLoad(
                id=s.id,
                start=s.arrival_step,
                end=s.departure_step,
                energy_kwh=s.energy_kwh,
                charge_max=ctl.charge_max,
                charge_min=ctl.charge_min,
                kw_per_unit=ctl.kw_per_unit,
                efficiency=s.efficiency,
                rows=tuple(float(a) for a in scenario.row_coefficients(s)),
                step=ctl.step,
                discharge_max=ctl.discharge_max,
                discharge_min=ctl.discharge_min,
                discharge_efficiency=1.0 if v2g is None else v2g.discharge_efficiency,
                discharge_rows=tuple(float(a) for a in scenario.row_discharge_coefficients(s))
                if ctl.can_discharge
                else None,
                battery=None if v2g is None else (v2g.initial_kwh, v2g.min_kwh, v2g.ceiling_kwh),
                degradation_eur_per_kwh=0.0 if v2g is None else v2g.degradation_eur_per_kwh,
            )
        )
    return tuple(loads)


def problem_from_scenario(
    scenario: Scenario,
    *,
    min_power_steps: int | None = None,
    unmet_penalty: float = DEFAULT_UNMET_PENALTY_EUR_PER_KWH,
) -> ScheduleProblem:
    """Build the full-horizon, perfect-foresight problem of a scenario."""
    return ScheduleProblem(
        dt_h=scenario.horizon.dt_h,
        price=scenario.tariff.price_eur_per_kwh,
        export_price=scenario.tariff.export_price,
        net_base_kw=scenario.net_base_kw,
        grid_limit_kw=scenario.site.grid_limit_kw,
        demand_charge=scenario.tariff.demand_charge_eur_per_kw,
        sessions=flex_loads_from_scenario(scenario),
        unmet_penalty=unmet_penalty,
        min_power_steps=min_power_steps,
        rows=scenario.rows,
    )


def relaxation_bound(
    scenario: Scenario, *, unmet_penalty: float = DEFAULT_UNMET_PENALTY_EUR_PER_KWH
) -> float:
    """Lower bound on the penalised cost of any feasible schedule for ``scenario``.

    This is the perfect-foresight LP with the minimum-current rule (and the
    setpoint resolution) relaxed. Every schedule a policy can execute in the
    simulator is feasible for it, so its optimum bounds
    ``energy cost + demand charge + penalty * unmet`` from below.
    """
    problem = problem_from_scenario(scenario, min_power_steps=0, unmet_penalty=unmet_penalty)
    return solve_schedule(problem).objective


def slice_rows(rows: RowModel, start: int, stop: int) -> RowModel:
    """The rows of steps ``[start, stop)`` (for a rolling-horizon problem)."""
    return RowModel(rows.names, rows.kinds, rows.rhs[start:stop])
