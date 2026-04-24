"""Forecast-aware MPC: plan for EVs that have not arrived yet.

Plain :class:`~evcharge.policies.ModelPredictiveControl` only sees connected
EVs, so under scarce capacity it postpones charging into slots that later
arrivals will also need. The variants here add an
:class:`~evcharge.forecast.ArrivalForecast` learned from training days:

* :class:`ExpectedValueMPC` (``mpc-ev``) adds the *expected* future fleet as
  "ghost" loads (fractional EVs, aggregated per arrival and departure cell)
  and solves one problem: the certainty-equivalent controller.
* :class:`ScenarioMPC` (``mpc-saa``) samples ``K`` future days and solves
  them jointly with one shared first-step decision (non-anticipativity): a
  sample-average approximation of the two-stage stochastic program.
* :class:`ReserveMPC` (``mpc-reserve``) keeps plain MPC but reserves the
  capacity that future EVs would use charging at their uniform rate.

Ghosts are continuous kW loads without a minimum current. They share the
site's rows with the expected per-kW coefficient of a future EV: on
phase-aware sites the average line incidence over the installed chargers for
its phase count. Their undelivered energy is penalised like a connected EV's.
Only the connected EVs' first step is ever applied.
"""

from __future__ import annotations

import numpy as np

from evcharge.electrical import WiringError, wiring
from evcharge.forecast import ArrivalForecast, Ghost
from evcharge.model import FloatArray, Scenario
from evcharge.optim import (
    DEFAULT_UNMET_PENALTY_EUR_PER_KWH,
    FlexLoad,
    ScheduleSolution,
    solve_scenarios,
    solve_schedule,
)
from evcharge.policies.base import Observation, SessionState
from evcharge.policies.optimal import ModelPredictiveControl


def ghost_rows(scenario: Scenario, phase_counts: tuple[int, ...] = (1, 3)) -> dict[int, FloatArray]:
    """Expected coefficient of one kW of a future EV in every row, by phase count.

    Import rows count the EV: kW rows with 1, line rows with the mean ampere
    per kW over the site's chargers that can host an EV with that phase count
    (a balanced three-phase EV if none can). Export rows give charging no
    credit (0).
    """
    rows = scenario.rows
    supply = scenario.site.supply
    out: dict[int, FloatArray] = {}
    for phases in sorted({*phase_counts, 3}, reverse=True):
        per_line = np.zeros(3)
        if supply is not None:
            found = 0
            for c in scenario.site.chargers:
                try:
                    wires = wiring(
                        supply.grid,
                        supply.voltage_v,
                        c.phases,
                        scenario.site.charger_lines(c),
                        phases,
                    )
                except WiringError:
                    continue
                per_line += np.asarray(wires.incidence()) / wires.kw_per_a
                found += 1
            if found:
                per_line /= found
            else:
                per_line = np.full(3, 1.0 / supply.balanced_kw_per_a)
        coef = np.zeros(rows.n_rows)
        for r, (name, kind) in enumerate(zip(rows.names, rows.kinds, strict=True)):
            if name.endswith("export"):
                continue
            coef[r] = 1.0 if kind == "site" else per_line[int(name[1]) - 1]
        out[phases] = coef
    return out


def ghost_loads(
    ghosts: tuple[Ghost, ...], now: int, rows: dict[int, FloatArray], prefix: str = "ghost"
) -> tuple[FlexLoad, ...]:
    """Ghosts as optimiser loads on the local horizon that starts at step ``now``."""
    return tuple(
        FlexLoad(
            id=f"{prefix}-{i}",
            start=g.arrival_step - now,
            end=g.departure_step - now,
            energy_kwh=g.energy_kwh,
            charge_max=g.max_power_kw,
            rows=tuple(float(a) for a in rows[g.phases]),
        )
        for i, g in enumerate(ghosts)
        if g.arrival_step > now and g.energy_kwh > 0.0
    )


class _ForecastMPC(ModelPredictiveControl):
    """Common set-up of the forecast-aware variants."""

    def __init__(
        self,
        forecast: ArrivalForecast,
        *,
        bin_steps: int = 4,
        unmet_penalty: float = DEFAULT_UNMET_PENALTY_EUR_PER_KWH,
        quick_charge_weight: float = 0.0,
    ) -> None:
        super().__init__(unmet_penalty=unmet_penalty, quick_charge_weight=quick_charge_weight)
        if bin_steps < 1:
            raise ValueError("bin_steps must be >= 1")
        self.forecast = forecast
        self.bin_steps = bin_steps
        self._rows: dict[int, FloatArray] = {}

    def reset(self, scenario: Scenario) -> None:
        """Check the forecast's clock and precompute the ghosts' row coefficients."""
        self.forecast.check(scenario)
        super().reset(scenario)
        self._rows = ghost_rows(scenario, self.forecast.phase_counts)


class ExpectedValueMPC(_ForecastMPC):
    """MPC with the expected future fleet as ghost loads (certainty equivalent).

    Args:
        forecast: Arrival forecast learned from training days.
        bin_steps: Aggregation cell of the ghosts, in steps.
        unmet_penalty: EUR per kWh of undelivered energy (connected or ghost).
        quick_charge_weight: Early-charging regulariser (default 0: the
            forecast replaces it).
    """

    name = "mpc-ev"

    def plan(self, obs: Observation, pending: list[SessionState]) -> ScheduleSolution:
        """Solve the local problem with the expected ghosts."""
        k = obs.step
        expected = self.forecast.expected_ghosts(k, bin_steps=self.bin_steps)
        ghosts = ghost_loads(expected, k, self._rows)
        end = k + max((g.end for g in ghosts), default=0)
        return solve_schedule(self.local_problem(obs, pending, future=ghosts, end=end))

    def __repr__(self) -> str:
        return f"ExpectedValueMPC(bin_steps={self.bin_steps})"


class ScenarioMPC(_ForecastMPC):
    """MPC over ``n_scenarios`` sampled futures with a shared first step (SAA).

    Args:
        forecast: Arrival forecast learned from training days.
        n_scenarios: Number of sampled future days per step.
        seed: Seed of the sampler (reset at every :meth:`reset`).
        bin_steps: Aggregation cell of each sample's ghosts, in steps.
        unmet_penalty: EUR per kWh of undelivered energy (connected or ghost).
        quick_charge_weight: Early-charging regulariser (default 0).
    """

    name = "mpc-saa"

    def __init__(
        self,
        forecast: ArrivalForecast,
        *,
        n_scenarios: int = 10,
        seed: int = 0,
        bin_steps: int = 4,
        unmet_penalty: float = DEFAULT_UNMET_PENALTY_EUR_PER_KWH,
        quick_charge_weight: float = 0.0,
    ) -> None:
        super().__init__(
            forecast,
            bin_steps=bin_steps,
            unmet_penalty=unmet_penalty,
            quick_charge_weight=quick_charge_weight,
        )
        if n_scenarios < 1:
            raise ValueError("n_scenarios must be >= 1")
        self.n_scenarios = n_scenarios
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, scenario: Scenario) -> None:
        """Re-seed the sampler, so every run is reproducible."""
        super().reset(scenario)
        self._rng = np.random.default_rng(self.seed)

    def plan(self, obs: Observation, pending: list[SessionState]) -> ScheduleSolution:
        """Solve the sampled futures jointly; the connected EVs' step 0 is shared."""
        k = obs.step
        futures = [
            ghost_loads(
                self.forecast.sample_ghosts(self._rng, k, bin_steps=self.bin_steps),
                k,
                self._rows,
            )
            for _ in range(self.n_scenarios)
        ]
        problems = [
            self.local_problem(obs, pending, future=f, end=k + max((g.end for g in f), default=0))
            for f in futures
        ]
        return solve_scenarios(problems, n_shared=len(pending))

    def __repr__(self) -> str:
        return f"ScenarioMPC(n_scenarios={self.n_scenarios}, seed={self.seed})"


class ReserveMPC(_ForecastMPC):
    """Plain MPC with capacity reserved for the expected future arrivals.

    The expected load of EVs arriving after the current step, each charging
    at its uniform rate over its window
    (:meth:`~evcharge.forecast.ArrivalForecast.expected_load_kw`), is added to
    the base load of later steps and removed from their row capacity, so the
    connected EVs are pushed to use capacity now. A heuristic: the reserve is
    fixed, not optimised.

    Args:
        forecast: Arrival forecast learned from training days.
        unmet_penalty: EUR per kWh of undelivered energy.
        quick_charge_weight: Early-charging regulariser (default 0).
    """

    name = "mpc-reserve"

    def __init__(
        self,
        forecast: ArrivalForecast,
        *,
        unmet_penalty: float = DEFAULT_UNMET_PENALTY_EUR_PER_KWH,
        quick_charge_weight: float = 0.0,
    ) -> None:
        super().__init__(
            forecast, unmet_penalty=unmet_penalty, quick_charge_weight=quick_charge_weight
        )

    def plan(self, obs: Observation, pending: list[SessionState]) -> ScheduleSolution:
        """Solve the local problem with the reserve taken out of later steps."""
        n_rows = self.scenario.rows.n_rows
        reserve = np.zeros(self.forecast.n_steps)
        usage = np.zeros((self.forecast.n_steps, n_rows))
        for phases in self.forecast.phase_counts:
            load = self.forecast.expected_load_kw(obs.step, phases=phases)
            reserve += load
            usage += load[:, None] * self._rows[phases][None, :]
        return solve_schedule(
            self.local_problem(obs, pending, reserve_kw=reserve, reserve_usage=usage)
        )

    def __repr__(self) -> str:
        return "ReserveMPC()"
