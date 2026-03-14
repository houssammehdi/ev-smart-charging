"""Optimisation-based policies: perfect-foresight optimum and rolling-horizon MPC."""

from __future__ import annotations

from collections.abc import Mapping

from evcharge.model import Scenario
from evcharge.optim import (
    DEFAULT_UNMET_PENALTY_EUR_PER_KWH,
    LPSession,
    ScheduleProblem,
    ScheduleSolution,
    Strategy,
    problem_from_scenario,
    solve_schedule,
)
from evcharge.policies.base import Observation, OnlinePolicy


class OptimalSchedule:
    """Offline optimum with perfect foresight of every arrival.

    Solves one problem over the whole horizon and replays it. Without minimum
    powers this is a pure LP and the result is exactly optimal. With the
    IEC 61851 minimum it is a MILP; the default ``"relax-and-fix"`` strategy
    solves it in well under a second on the built-in scenarios with a
    certified gap (see :attr:`solution`), while ``"exact"`` runs the full MILP.
    It is a benchmark, not a deployable controller: real sites do not know
    tomorrow's arrivals.

    Args:
        unmet_penalty: EUR per kWh of undelivered energy.
        strategy: MILP strategy, see :func:`evcharge.optim.solve_schedule`.
        time_limit_s: HiGHS time limit for MILP solves; the best incumbent is
            used if it is reached.
        mip_rel_gap: Relative optimality gap at which HiGHS stops.
    """

    name = "optimal"
    clairvoyant = True

    def __init__(
        self,
        *,
        unmet_penalty: float = DEFAULT_UNMET_PENALTY_EUR_PER_KWH,
        strategy: Strategy = "relax-and-fix",
        time_limit_s: float | None = 120.0,
        mip_rel_gap: float = 1e-6,
    ) -> None:
        self.unmet_penalty = unmet_penalty
        self.strategy: Strategy = strategy
        self.time_limit_s = time_limit_s
        self.mip_rel_gap = mip_rel_gap
        self._plan: dict[str, list[float]] = {}
        self._solution: ScheduleSolution | None = None

    @property
    def solution(self) -> ScheduleSolution:
        """Solver output of the last :meth:`reset` (status, gap, timing)."""
        if self._solution is None:
            raise RuntimeError("OptimalSchedule used before reset()")
        return self._solution

    def reset(self, scenario: Scenario) -> None:
        """Solve the full-horizon problem for ``scenario``."""
        problem = problem_from_scenario(scenario, unmet_penalty=self.unmet_penalty)
        solution = solve_schedule(
            problem,
            strategy=self.strategy,
            time_limit_s=self.time_limit_s,
            mip_rel_gap=self.mip_rel_gap,
        )
        self._solution = solution
        self._plan = {
            s.id: [float(v) for v in solution.power_kw[i]] for i, s in enumerate(scenario.sessions)
        }

    def decide(self, obs: Observation) -> Mapping[str, float]:
        """Replay the pre-computed plan."""
        out: dict[str, float] = {}
        for st in obs.sessions:
            value = self._plan[st.id][obs.step]
            if value > 0.0:
                out[st.id] = value
        return out

    def __repr__(self) -> str:
        return f"OptimalSchedule(strategy={self.strategy!r})"


DEFAULT_QUICK_CHARGE_WEIGHT = 0.002
"""Default early-charging regulariser of the MPC: 0.2 cent per kWh per hour of delay."""


class ModelPredictiveControl(OnlinePolicy):
    """Rolling-horizon controller that re-optimises at every step.

    At step ``k`` it knows only the sessions that have already arrived, with
    their declared departure time and energy request, plus the price, base-load
    and PV forecasts. It solves the scheduling problem from ``k`` to the last
    known departure and applies only the first step. The demand charge is
    priced against the peak already incurred and the forecast base-load peak
    after that departure, so only a *new* peak costs extra.

    Only the applied first step needs the exact minimum-power rule; later steps
    are relaxed, so each solve has at most one binary per connected EV and stays
    fast. The relaxed tail is re-planned exactly at the next step anyway.

    Because future arrivals are invisible, a pure cost objective tends to
    postpone charging into cheap slots that later arrivals will also need. The
    optional ``quick_charge_weight`` (EUR/kWh per hour of delay) counters this
    by favouring earlier charging, in the spirit of the "quick charge" term of
    the Caltech Adaptive Charging Network scheduler. The default (0.2 cent per
    kWh per hour) is small against typical intraday price spreads; set it to
    0 for the pure cost objective.

    Args:
        unmet_penalty: EUR per kWh of undelivered energy.
        quick_charge_weight: Early-charging regulariser, EUR/kWh per hour.
    """

    name = "mpc"

    def __init__(
        self,
        *,
        unmet_penalty: float = DEFAULT_UNMET_PENALTY_EUR_PER_KWH,
        quick_charge_weight: float = DEFAULT_QUICK_CHARGE_WEIGHT,
    ) -> None:
        super().__init__()
        if quick_charge_weight < 0:
            raise ValueError("quick_charge_weight must be >= 0")
        self.unmet_penalty = unmet_penalty
        self.quick_charge_weight = quick_charge_weight

    def decide(self, obs: Observation) -> Mapping[str, float]:
        """Solve the problem over the known sessions and apply the first step."""
        pending = obs.pending()
        if not pending:
            return {}
        sc = self.scenario
        k = obs.step
        end = max(st.session.departure_step for st in pending)
        # Steps after the last known departure only carry the (forecast) base
        # load; their import still sets a floor for the horizon's peak.
        later_base_peak = float(sc.net_base_kw[end:].max(initial=0.0))
        sessions = tuple(
            LPSession(
                id=st.id,
                start=0,
                end=st.steps_left,
                energy_kwh=st.remaining_kwh,
                efficiency=st.session.efficiency,
                p_min_kw=st.p_min_kw,
                p_max_kw=st.p_max_kw,
            )
            for st in pending
        )
        problem = ScheduleProblem(
            dt_h=sc.horizon.dt_h,
            price=sc.tariff.price_eur_per_kwh[k:end],
            export_price=sc.tariff.export_price[k:end],
            net_base_kw=sc.net_base_kw[k:end],
            grid_limit_kw=sc.site.grid_limit_kw,
            demand_charge=sc.tariff.demand_charge_eur_per_kw,
            sessions=sessions,
            peak_floor_kw=max(obs.peak_import_kw, later_base_peak),
            unmet_penalty=self.unmet_penalty,
            min_power_steps=1,
            quick_charge_weight=self.quick_charge_weight,
        )
        solution = solve_schedule(problem)
        return {
            st.id: float(solution.power_kw[i, 0])
            for i, st in enumerate(pending)
            if solution.power_kw[i, 0] > 0.0
        }

    def __repr__(self) -> str:
        return f"ModelPredictiveControl(quick_charge_weight={self.quick_charge_weight})"
