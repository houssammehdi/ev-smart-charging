"""Linear / mixed-integer formulation of the site charging problem (HiGHS via SciPy).

The same formulation serves three purposes:

* the perfect-foresight optimum (:class:`evcharge.policies.OptimalSchedule`),
* the rolling-horizon controller (:class:`evcharge.policies.ModelPredictiveControl`),
* a certified lower bound on the cost of *any* feasible schedule
  (:func:`relaxation_bound`).

See the README for the mathematical statement. Decision variables per LP:

========  =======================================================
``p``     power drawn by session *s* in step *t* (kW), only inside its window
``g, e``  site import / export in step *t* (kW)
``P``     peak import over the horizon (kW), ``P >= peak_floor``
``u``     unmet energy of session *s* (kWh), penalised
``o``     energy overshoot of session *s* (kWh); see below
``y``     binary on/off of session *s* in step *t* (MILP only)
========  =======================================================

Minimum-power (IEC 61851 6 A) steps are modelled as semi-continuous variables,
``p = 0`` or ``p_min <= p <= p_max``, using the binaries ``y``. Because an EV
stops drawing power once its request is met, commanding ``p_min`` for the
final few hundred Wh is allowed; the model captures this with an overshoot
variable ``o <= eta * p_min * dt``: energy that is commanded but never drawn.
The plan pays for it at the import price, and in addition at the most negative
export price of the session's window (zero if no export price is negative), so
that undrawn energy can never look like revenue. With that term the MILP
objective is an upper bound on the cost the simulator measures when the plan
is replayed, and the LP relaxation (no overshoot) is a lower bound.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

from evcharge.model import POWER_TOL_KW, FloatArray, Scenario

DEFAULT_UNMET_PENALTY_EUR_PER_KWH = 100.0
"""Penalty on undelivered energy. It must exceed the marginal cost of delivering
one more kWh (``price + demand_charge / (eta * dt)``) so that the optimiser
never trades energy for money; 100 EUR/kWh comfortably does for realistic tariffs."""


class SolverError(RuntimeError):
    """Raised when HiGHS fails to return a usable solution."""


@dataclass(frozen=True)
class LPSession:
    """A session as seen by the optimiser (times are local to the LP horizon).

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
        grid_limit_kw: Import limit at the connection point.
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
    """

    dt_h: float
    price: FloatArray
    export_price: FloatArray
    net_base_kw: FloatArray
    grid_limit_kw: float
    demand_charge: float
    sessions: tuple[LPSession, ...]
    peak_floor_kw: float = 0.0
    unmet_penalty: float = DEFAULT_UNMET_PENALTY_EUR_PER_KWH
    min_power_steps: int | None = None
    quick_charge_weight: float = 0.0

    @property
    def n_steps(self) -> int:
        """Number of steps ``T``."""
        return int(self.price.size)


@dataclass(frozen=True, eq=False)
class ScheduleSolution:
    """Output of :func:`solve_schedule`.

    Attributes:
        power_kw: Commanded power, shape ``(n_sessions, T)``.
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

    @property
    def gap(self) -> float:
        """Relative gap between :attr:`objective` and :attr:`lower_bound`."""
        return max(0.0, self.objective - self.lower_bound) / max(1e-9, abs(self.objective))


NUMERIC_ZERO = 1e-9
"""Coefficients smaller than this in magnitude are treated as exactly zero.

HiGHS can stall on MILPs whose data contains denormal-scale values (for example a
price of 1e-308); snapping them to zero changes the objective by a negligible amount.
"""


def _snap(values: FloatArray) -> FloatArray:
    return np.where(np.abs(values) < NUMERIC_ZERO, 0.0, values)


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


class _Model:
    """Sparse matrices of one :class:`ScheduleProblem`, reusable across solves."""

    def __init__(self, problem: ScheduleProblem) -> None:
        self.problem = problem
        n_t = problem.n_steps
        sessions = problem.sessions
        n_s = len(sessions)
        dt = problem.dt_h
        for s in sessions:
            if not 0 <= s.start < s.end <= n_t:
                raise ValueError(f"session {s.id}: window [{s.start}, {s.end}) outside [0, {n_t})")
        p_sess = np.array(
            [i for i, s in enumerate(sessions) for _ in range(s.start, s.end)], dtype=np.int64
        )
        p_time = np.array([t for s in sessions for t in range(s.start, s.end)], dtype=np.int64)
        n_p = int(p_sess.size)
        self.n_t, self.n_s, self.n_p = n_t, n_s, n_p
        self.p_sess, self.p_time = p_sess, p_time
        p_min = np.array([s.p_min_kw for s in sessions], dtype=np.float64)
        self.p_max = np.array([s.p_max_kw for s in sessions], dtype=np.float64)[p_sess]
        eta = np.array([s.efficiency for s in sessions], dtype=np.float64)
        energy = np.array([s.energy_kwh for s in sessions], dtype=np.float64)

        int_horizon = n_t if problem.min_power_steps is None else problem.min_power_steps
        self.p_min = p_min[p_sess]
        #: power entries subject to the semi-continuous minimum-power rule
        self.eligible = (p_time < int_horizon) & (self.p_min > POWER_TOL_KW)
        has_min = np.zeros(n_s, dtype=bool)
        has_min[p_sess[self.eligible]] = True

        self.off_g = n_p
        self.off_e = self.off_g + n_t
        self.off_peak = self.off_e + n_t
        self.off_u = self.off_peak + 1
        self.off_o = self.off_u + n_s
        self.n_cont = self.off_o + n_s

        c = np.zeros(self.n_cont)
        c[:n_p] = problem.quick_charge_weight * p_time * dt * dt
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
            c[self.off_o :] = np.maximum(0.0, -window_min_export) / eta
        self.c = _snap(c)

        lb = np.zeros(self.n_cont)
        ub = np.full(self.n_cont, np.inf)
        floor = problem.peak_floor_kw
        lb[self.off_peak] = floor if floor >= NUMERIC_ZERO else 0.0
        ub[self.off_u : self.off_o] = energy
        # Commanding p_min to finish a request is allowed: the EV stops by itself.
        ub[self.off_o :] = np.where(has_min, eta * p_min * dt, 0.0)
        self.lb, self.ub = lb, ub

        cols_p = np.arange(n_p)
        t_idx = np.arange(n_t)
        s_idx = np.arange(n_s)
        # Power balance: sum_s p - g + e = -(base - pv); energy: sum eta p dt + u - o = E.
        rows = np.concatenate([p_time, t_idx, t_idx, n_t + p_sess, n_t + s_idx, n_t + s_idx])
        cols = np.concatenate(
            [
                cols_p,
                self.off_g + t_idx,
                self.off_e + t_idx,
                cols_p,
                self.off_u + s_idx,
                self.off_o + s_idx,
            ]
        )
        vals = np.concatenate(
            [
                np.ones(n_p),
                -np.ones(n_t),
                np.ones(n_t),
                eta[p_sess] * dt,
                np.ones(n_s),
                -np.ones(n_s),
            ]
        )
        self.a_eq = coo_matrix((vals, (rows, cols)), shape=(n_t + n_s, self.n_cont)).tocsr()
        self.b_eq = _snap(np.concatenate([-problem.net_base_kw, energy]))
        # Site headroom: sum_s p <= L - (base - pv); peak: g - P <= 0.
        rows = np.concatenate([p_time, n_t + t_idx, n_t + t_idx])
        cols = np.concatenate([cols_p, self.off_g + t_idx, np.full(n_t, self.off_peak)])
        vals = np.concatenate([np.ones(n_p), np.ones(n_t), -np.ones(n_t)])
        self.a_ub = coo_matrix((vals, (rows, cols)), shape=(2 * n_t, self.n_cont)).tocsr()
        headroom = _snap(np.maximum(0.0, problem.grid_limit_kw - problem.net_base_kw))
        self.b_ub = np.concatenate([headroom, np.zeros(n_t)])

    def solve(
        self,
        p_lb: FloatArray,
        p_ub: FloatArray,
        binaries: npt.NDArray[np.int64],
        *,
        time_limit_s: float | None,
        mip_rel_gap: float,
    ) -> tuple[FloatArray, float, float, str]:
        """Solve with the given power bounds; ``binaries`` index semi-continuous entries.

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

        # p_i - p_max y_i <= 0 and p_min y_i - p_i <= 0 for every binary entry i.
        n_var = self.n_cont + n_y
        y_cols = self.n_cont + np.arange(n_y)
        y_rows = np.arange(n_y)
        link = coo_matrix(
            (
                np.concatenate(
                    [np.ones(n_y), -self.p_max[binaries], -np.ones(n_y), self.p_min[binaries]]
                ),
                (
                    np.concatenate([y_rows, y_rows, n_y + y_rows, n_y + y_rows]),
                    np.concatenate([binaries, y_cols, binaries, y_cols]),
                ),
            ),
            shape=(2 * n_y, n_var),
        )
        pad = csr_matrix((self.a_eq.shape[0], n_y))
        a_eq = hstack([self.a_eq, pad]).tocsr()
        a_ub = vstack([hstack([self.a_ub, csr_matrix((self.a_ub.shape[0], n_y))]), link]).tocsr()
        b_ub = np.concatenate([self.b_ub, np.zeros(2 * n_y)])
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

    def to_solution(
        self, x: FloatArray, objective: float, bound: float, status: str, n_y: int, elapsed: float
    ) -> ScheduleSolution:
        p = np.clip(x[: self.n_p], 0.0, self.p_max)
        p[p < POWER_TOL_KW] = 0.0
        power = np.zeros((self.n_s, self.n_t))
        power[self.p_sess, self.p_time] = p
        return ScheduleSolution(
            power_kw=power,
            unmet_kwh=np.maximum(0.0, x[self.off_u : self.off_o]),
            import_kw=x[self.off_g : self.off_e],
            export_kw=x[self.off_e : self.off_peak],
            peak_kw=float(x[self.off_peak]),
            objective=objective,
            lower_bound=min(bound, objective),
            status=status,
            n_binaries=n_y,
            solve_time_s=elapsed,
        )


def solve_schedule(
    problem: ScheduleProblem,
    *,
    strategy: Strategy = "exact",
    time_limit_s: float | None = None,
    mip_rel_gap: float = 1e-6,
) -> ScheduleSolution:
    """Solve the charging LP/MILP with HiGHS.

    Uses :func:`scipy.optimize.linprog` (``method="highs"``) when no binaries
    are needed and :func:`scipy.optimize.milp` otherwise.

    Args:
        problem: The problem instance.
        strategy: ``"exact"`` solves the full MILP. ``"relax-and-fix"`` solves
            the LP relaxation, fixes every entry the relaxation left at 0 (off)
            or at >= ``p_min`` (on), and solves a small MILP over the remaining
            fractional entries. It is a heuristic, but the relaxation gives a
            certified :attr:`ScheduleSolution.lower_bound`.
        time_limit_s: HiGHS time limit for MILP solves.
        mip_rel_gap: Relative MIP gap at which HiGHS stops.

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
        return model.to_solution(
            x, obj, bound, status, int(eligible.size), time.perf_counter() - started
        )

    x_lp, bound, _, _ = model.solve(
        p_lb, p_ub, np.zeros(0, np.int64), time_limit_s=None, mip_rel_gap=mip_rel_gap
    )
    p_lp = x_lp[: model.n_p][eligible]
    off = p_lp <= POWER_TOL_KW
    on = p_lp >= model.p_min[eligible] - POWER_TOL_KW
    p_ub[eligible[off]] = 0.0
    p_lb[eligible[on]] = model.p_min[eligible[on]]
    fractional = eligible[~off & ~on]
    x, obj, _, status = model.solve(
        p_lb, p_ub, fractional, time_limit_s=time_limit_s, mip_rel_gap=mip_rel_gap
    )
    return model.to_solution(
        x, obj, bound, status, int(fractional.size), time.perf_counter() - started
    )


def problem_from_scenario(
    scenario: Scenario,
    *,
    min_power_steps: int | None = None,
    unmet_penalty: float = DEFAULT_UNMET_PENALTY_EUR_PER_KWH,
) -> ScheduleProblem:
    """Build the full-horizon, perfect-foresight problem of a scenario."""
    sessions = []
    for s in scenario.sessions:
        p_min, p_max = scenario.power_bounds(s)
        sessions.append(
            LPSession(
                id=s.id,
                start=s.arrival_step,
                end=s.departure_step,
                energy_kwh=s.energy_kwh,
                efficiency=s.efficiency,
                p_min_kw=p_min,
                p_max_kw=p_max,
            )
        )
    return ScheduleProblem(
        dt_h=scenario.horizon.dt_h,
        price=scenario.tariff.price_eur_per_kwh,
        export_price=scenario.tariff.export_price,
        net_base_kw=scenario.net_base_kw,
        grid_limit_kw=scenario.site.grid_limit_kw,
        demand_charge=scenario.tariff.demand_charge_eur_per_kw,
        sessions=tuple(sessions),
        unmet_penalty=unmet_penalty,
        min_power_steps=min_power_steps,
    )


def relaxation_bound(
    scenario: Scenario, *, unmet_penalty: float = DEFAULT_UNMET_PENALTY_EUR_PER_KWH
) -> float:
    """Lower bound on the penalised cost of any feasible schedule for ``scenario``.

    This is the perfect-foresight LP with the minimum-power rule relaxed. Every
    schedule a policy can execute in the simulator is feasible for it, so its
    optimum bounds ``energy cost + demand charge + penalty * unmet`` from below.
    """
    problem = problem_from_scenario(scenario, min_power_steps=0, unmet_penalty=unmet_penalty)
    return solve_schedule(problem).objective
