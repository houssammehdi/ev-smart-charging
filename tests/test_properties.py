"""Property-based tests on random feasible instances.

Only guaranteed properties are asserted:

* every policy keeps within the physical limits (site, charger/EV, window,
  minimum current, energy request) and triggers no violations;
* the LP relaxation bounds every executed schedule's penalised cost from below;
* the offline MILP objective bounds the cost of its own replay from above;
* without minimum powers the offline optimum is a pure LP, so its cost is no
  higher than any policy's;
* with perfect information (all EVs present from the start) and no
  regulariser, MPC recovers the offline optimum (Bellman's principle);
* on phase-aware sites (TN and IT, any rotation, 1/2/3-phase EVs, any
  resolution) no line and no site row is ever exceeded by any policy, and
  every setpoint is on the charger's grid and within its range;
* with bidirectional (V2G) sessions every battery stays within its bounds,
  every row (including export rows) holds, and the same bounds apply;
* the forecast-aware MPC variants are as safe as the other policies and
  bounded by the LP relaxation, whatever the forecast.
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from evcharge.electrical import GridType, Supply, three_phase_kw_per_a
from evcharge.forecast import ArrivalForecast
from evcharge.metrics import compute_metrics
from evcharge.model import V2G, Charger, Horizon, Scenario, Session, Site, Tariff, on_grid
from evcharge.optim import relaxation_bound
from evcharge.policies import (
    POLICY_FACTORIES,
    ModelPredictiveControl,
    OptimalSchedule,
)
from evcharge.policies.forecast import ExpectedValueMPC, ReserveMPC, ScenarioMPC
from evcharge.sim import simulate

TOL = 1e-6


def cost_tol(reference: float) -> float:
    """Tolerance for comparing penalised costs of two solver runs.

    HiGHS works to a feasibility tolerance of about 1e-7, and an unmet-energy error
    of that size is priced at the 100 EUR/kWh penalty, so allow 1e-4 EUR on top of
    a relative 1e-5.
    """
    return 1e-4 + 1e-5 * abs(reference)


PROPERTY_SETTINGS = settings(
    # raise locally for a deeper search, e.g. EVCHARGE_HYPOTHESIS_EXAMPLES=1000
    max_examples=int(os.environ.get("EVCHARGE_HYPOTHESIS_EXAMPLES", "60")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@st.composite
def scenarios(
    draw: st.DrawFn, *, allow_min_power: bool = True, all_at_start: bool = False
) -> Scenario:
    n_steps = draw(st.integers(3, 10))
    step_minutes = draw(st.sampled_from([15, 30, 60]))
    n_chargers = draw(st.integers(1, 4))
    chargers = []
    for i in range(n_chargers):
        p_max = draw(st.floats(3.0, 22.0))
        p_min = draw(st.sampled_from([0.0, 1.38, 4.14])) if allow_min_power else 0.0
        chargers.append(Charger(f"C{i}", p_max, min(p_min, p_max)))
    sessions = []
    for c in chargers:
        t = 0
        for _ in range(draw(st.integers(0, 2))):
            arrival = 0 if all_at_start else draw(st.integers(t, n_steps - 1))
            departure = draw(st.integers(arrival + 1, n_steps))
            ev_max = draw(st.floats(2.0, 22.0))
            ev_min = draw(st.none() | st.floats(0.0, 2.0))
            sessions.append(
                Session(
                    id=f"S{len(sessions)}",
                    charger_id=c.id,
                    arrival_step=arrival,
                    departure_step=departure,
                    energy_kwh=draw(st.floats(0.2, 60.0)),
                    max_power_kw=ev_max,
                    efficiency=draw(st.floats(0.8, 1.0)),
                    min_power_kw=None
                    if ev_min is None or not allow_min_power
                    else min(ev_min, ev_max),
                )
            )
            t = departure
            if t >= n_steps or all_at_start:
                break
    limit = draw(st.floats(3.0, 60.0))
    base = np.array(draw(st.lists(st.floats(0.0, 0.8), min_size=n_steps, max_size=n_steps))) * limit
    pv = np.array(draw(st.lists(st.floats(0.0, 15.0), min_size=n_steps, max_size=n_steps)))
    price = np.array(draw(st.lists(st.floats(-0.05, 0.5), min_size=n_steps, max_size=n_steps)))
    fee = np.array(draw(st.lists(st.floats(0.0, 0.1), min_size=n_steps, max_size=n_steps)))
    return Scenario(
        name="random",
        horizon=Horizon(datetime(2026, 1, 5), n_steps, step_minutes),
        site=Site(limit, tuple(chargers)),
        tariff=Tariff(price, price - fee, draw(st.floats(0.0, 3.0))),
        sessions=tuple(sessions),
        base_load_kw=base,
        pv_kw=pv,
    )


def all_policies() -> list[object]:
    return [factory() for factory in POLICY_FACTORIES.values()]


@PROPERTY_SETTINGS
@given(scenarios())
def test_every_policy_respects_physical_limits(sc: Scenario) -> None:
    headroom = sc.ev_headroom_kw
    for policy in all_policies():
        res = simulate(sc, policy)  # type: ignore[arg-type]
        assert res.violations == (), (policy, res.violations)
        # site limit (on commands, which bound the drawn power)
        assert np.all(res.setpoint_kw.sum(axis=0) <= headroom + TOL)
        assert np.all(res.net_import_kw <= sc.site.grid_limit_kw + TOL)
        for i, s in enumerate(sc.sessions):
            p_min, p_max = sc.power_bounds(s)
            cmd = res.setpoint_kw[i]
            drawn = res.power_kw[i]
            assert np.all(drawn >= 0.0)
            assert np.all(drawn <= cmd + TOL)
            assert np.all(cmd <= p_max + TOL)
            # minimum current: paused or at least p_min
            assert np.all((cmd == 0.0) | (cmd >= p_min - TOL))
            # availability window
            outside = np.ones(sc.horizon.n_steps, dtype=bool)
            outside[s.arrival_step : s.departure_step] = False
            assert np.all(cmd[outside] == 0.0)
            # never over-deliver
            assert res.delivered_kwh[i] <= s.energy_kwh + TOL
            energy = float(drawn.sum()) * sc.horizon.dt_h * s.efficiency
            assert energy <= s.energy_kwh + 1e-5


@PROPERTY_SETTINGS
@given(scenarios())
def test_relaxation_bounds_every_policy(sc: Scenario) -> None:
    bound = relaxation_bound(sc)
    for policy in all_policies():
        m = compute_metrics(simulate(sc, policy))  # type: ignore[arg-type]
        assert bound <= m.penalised_cost_eur + cost_tol(bound), policy


@PROPERTY_SETTINGS
@given(scenarios())
def test_optimal_plan_bounds_its_replay_from_above(sc: Scenario) -> None:
    # The replay can only draw less than planned (an EV stops when full); the plan
    # prices that undrawn energy conservatively, even at negative export prices.
    opt = OptimalSchedule()
    executed = compute_metrics(simulate(sc, opt)).penalised_cost_eur
    assert relaxation_bound(sc) <= executed + cost_tol(executed)
    assert executed <= opt.solution.objective + cost_tol(executed)


@PROPERTY_SETTINGS
@given(scenarios(allow_min_power=False))
def test_optimal_lp_is_no_worse_than_any_policy(sc: Scenario) -> None:
    optimal = compute_metrics(simulate(sc, OptimalSchedule())).penalised_cost_eur
    assert optimal <= relaxation_bound(sc) + cost_tol(optimal)
    for policy in all_policies():
        m = compute_metrics(simulate(sc, policy))  # type: ignore[arg-type]
        assert optimal <= m.penalised_cost_eur + cost_tol(optimal), policy


@PROPERTY_SETTINGS
@given(scenarios(allow_min_power=False, all_at_start=True))
def test_mpc_with_perfect_information_is_optimal(sc: Scenario) -> None:
    optimal = compute_metrics(simulate(sc, OptimalSchedule())).penalised_cost_eur
    mpc = compute_metrics(
        simulate(sc, ModelPredictiveControl(quick_charge_weight=0.0))
    ).penalised_cost_eur
    assert abs(mpc - optimal) <= cost_tol(optimal)


ROTATIONS_3PH = ["L1L2L3", "L2L3L1", "L3L1L2", "L1L3L2"]
ROTATIONS_1PH = {"TN": ["L1", "L2", "L3"], "IT": ["L1L2", "L2L3", "L1L3"]}


@st.composite
def phase_scenarios(draw: st.DrawFn) -> Scenario:
    grid = draw(st.sampled_from(["TN", "IT"]))
    n_steps = draw(st.integers(3, 8))
    step_minutes = draw(st.sampled_from([15, 30, 60]))
    limits = tuple(draw(st.floats(12.0, 80.0)) for _ in range(3))
    chargers = []
    for i in range(draw(st.integers(1, 4))):
        phases = draw(st.sampled_from([1, 3]))
        rotation = draw(st.sampled_from(ROTATIONS_3PH if phases == 3 else ROTATIONS_1PH[grid]))
        chargers.append(
            Charger(
                f"C{i}",
                22.0,
                phases=phases,
                rotation=rotation,
                max_current_a=draw(st.sampled_from([10.0, 16.0, 32.0])),
                current_step_a=draw(st.sampled_from([0.0, 0.1, 1.0])),
            )
        )
    sessions = []
    ev_phases = [1, 2, 3] if grid == "TN" else [1, 3]
    for c in chargers:
        t = 0
        for _ in range(draw(st.integers(0, 2))):
            arrival = draw(st.integers(t, n_steps - 1))
            departure = draw(st.integers(arrival + 1, n_steps))
            sessions.append(
                Session(
                    id=f"S{len(sessions)}",
                    charger_id=c.id,
                    arrival_step=arrival,
                    departure_step=departure,
                    energy_kwh=draw(st.floats(0.2, 40.0)),
                    max_power_kw=22.0,
                    efficiency=draw(st.floats(0.8, 1.0)),
                    phases=draw(st.sampled_from(ev_phases)),
                    max_current_a=draw(st.sampled_from([6.0, 13.0, 16.0, 32.0])),
                )
            )
            t = departure
            if t >= n_steps:
                break
    lim = np.array(limits)
    base = (
        np.array(
            draw(st.lists(st.floats(0.0, 0.7), min_size=3 * n_steps, max_size=3 * n_steps))
        ).reshape(n_steps, 3)
        * lim
    )
    pv = (
        np.array(
            draw(st.lists(st.floats(0.0, 0.6), min_size=3 * n_steps, max_size=3 * n_steps))
        ).reshape(n_steps, 3)
        * lim
    )
    kw_per_a = three_phase_kw_per_a(GridType(grid), 230.0)
    base_kw = base.mean(axis=1) * kw_per_a
    pv_kw = pv.mean(axis=1) * kw_per_a
    limit_kw = float(max(draw(st.floats(5.0, 120.0)), float((base_kw - pv_kw).max()) + 1.0))
    price = np.array(draw(st.lists(st.floats(-0.05, 0.5), min_size=n_steps, max_size=n_steps)))
    return Scenario(
        name="random-phase",
        horizon=Horizon(datetime(2026, 1, 5), n_steps, step_minutes),
        site=Site(limit_kw, tuple(chargers), Supply(limits, GridType(grid))),
        tariff=Tariff(price, price - 0.02, draw(st.floats(0.0, 3.0))),
        sessions=tuple(sessions),
        base_load_kw=base_kw,
        pv_kw=pv_kw,
        base_current_a=base,
        pv_current_a=pv,
    )


@PROPERTY_SETTINGS
@given(phase_scenarios())
def test_no_policy_overloads_a_line(sc: Scenario) -> None:
    rows = sc.rows
    coef = np.array([sc.row_coefficients(s) for s in sc.sessions]).reshape(
        len(sc.sessions), rows.n_rows
    )
    assert sc.site.supply is not None
    limits = np.array(sc.site.supply.line_limit_a)
    for policy in all_policies():
        res = simulate(sc, policy)  # type: ignore[arg-type]
        assert res.violations == (), (policy, res.violations)
        assert res.setpoint is not None
        assert res.line_current_a is not None
        # every line within its limit, and every row (lines and site kW) holds
        assert np.all(res.line_current_a <= limits[None, :] + TOL), policy
        usage = res.setpoint.T @ coef
        assert np.all(usage <= rows.rhs + TOL), policy
        assert np.all(res.net_import_kw <= sc.site.grid_limit_kw + TOL)
        for i, s in enumerate(sc.sessions):
            ctl = sc.control(s)
            x = res.setpoint[i]
            assert np.all((x == 0.0) | ((x >= ctl.charge_min - TOL) & (x <= ctl.charge_max + TOL)))
            assert all(on_grid(float(v), ctl.step) for v in x), (policy, x, ctl.step)
            outside = np.ones(sc.horizon.n_steps, dtype=bool)
            outside[s.arrival_step : s.departure_step] = False
            assert np.all(x[outside] == 0.0)
            assert res.delivered_kwh[i] <= s.energy_kwh + TOL


@PROPERTY_SETTINGS
@given(phase_scenarios())
def test_relaxation_bounds_every_policy_on_phase_sites(sc: Scenario) -> None:
    bound = relaxation_bound(sc)
    for policy in all_policies():
        m = compute_metrics(simulate(sc, policy))  # type: ignore[arg-type]
        assert bound <= m.penalised_cost_eur + cost_tol(bound), policy


@PROPERTY_SETTINGS
@given(phase_scenarios())
def test_optimal_plan_bounds_its_replay_on_phase_sites(sc: Scenario) -> None:
    # The rounded plan is on the chargers' grid, so the replay executes it exactly
    # (up to the EV stopping when full), and line rows floored to the grid keep
    # the LP a valid bound for every executable schedule.
    opt = OptimalSchedule()
    executed = compute_metrics(simulate(sc, opt)).penalised_cost_eur
    assert relaxation_bound(sc) <= executed + cost_tol(executed)
    assert executed <= opt.solution.objective + cost_tol(executed)


@st.composite
def v2g_scenarios(draw: st.DrawFn, *, phases: bool = False) -> Scenario:
    """Random sites where some chargers are bidirectional and some EVs have a V2G spec."""
    sc = draw(phase_scenarios() if phases else scenarios())
    chargers = tuple(
        Charger(
            c.id,
            c.max_power_kw,
            c.min_power_kw,
            c.phases,
            c.rotation,
            c.max_current_a,
            c.min_current_a,
            c.current_step_a,
            bidirectional=draw(st.booleans()),
        )
        for c in sc.site.chargers
    )
    sessions = []
    for s in sc.sessions:
        if draw(st.booleans()):
            capacity = draw(st.floats(10.0, 80.0))
            low = draw(st.floats(0.0, 0.3)) * capacity
            high = draw(st.floats(0.7, 1.0)) * capacity
            initial = draw(st.floats(low, high))
            energy = min(s.energy_kwh, high - initial)
            v2g = V2G(
                capacity_kwh=capacity,
                initial_kwh=initial,
                min_kwh=low,
                max_kwh=high,
                max_discharge_kw=None if phases else draw(st.none() | st.floats(1.0, 22.0)),
                max_discharge_current_a=draw(st.none() | st.floats(6.0, 32.0)) if phases else None,
                discharge_efficiency=draw(st.floats(0.8, 1.0)),
                degradation_eur_per_kwh=draw(st.sampled_from([0.0, 0.02, 0.1])),
            )
            sessions.append(
                Session(
                    s.id,
                    s.charger_id,
                    s.arrival_step,
                    s.departure_step,
                    max(0.0, energy),
                    s.max_power_kw,
                    s.efficiency,
                    s.min_power_kw,
                    s.phases,
                    s.max_current_a,
                    v2g=v2g,
                )
            )
        else:
            sessions.append(s)
    export = draw(st.none() | st.floats(0.5, 40.0))
    site = Site(sc.site.grid_limit_kw, chargers, sc.site.supply, export)
    return sc.with_site(site).with_sessions(tuple(sessions))


def check_v2g_run(sc: Scenario, policy: object) -> None:
    res = simulate(sc, policy)  # type: ignore[arg-type]
    assert res.violations == (), (policy, res.violations)
    assert res.setpoint is not None
    rows = sc.rows
    for t in range(sc.horizon.n_steps):
        usage = np.zeros(rows.n_rows)
        for i, s in enumerate(sc.sessions):
            x = float(res.setpoint[i, t])
            if x > 0:
                usage += sc.row_coefficients(s) * x
            elif x < 0:
                usage += sc.row_discharge_coefficients(s) * -x
        assert np.all(usage <= rows.rhs[t] + TOL), (policy, t, usage, rows.rhs[t])
    for i, s in enumerate(sc.sessions):
        ctl = sc.control(s)
        x = res.setpoint[i]
        assert np.all(x >= -ctl.discharge_max - TOL)
        assert all(on_grid(abs(float(v)), ctl.step) for v in x)
        if s.v2g is None:
            assert np.all(x >= 0.0)
            assert res.delivered_kwh[i] <= s.energy_kwh + TOL
        else:
            assert res.battery_kwh is not None
            window = res.battery_kwh[i, s.arrival_step : s.departure_step]
            assert np.all(window >= s.v2g.min_kwh - TOL), (policy, window)
            assert np.all(window <= s.v2g.ceiling_kwh + TOL), (policy, window)


@PROPERTY_SETTINGS
@given(v2g_scenarios())
def test_v2g_policies_keep_batteries_and_rows(sc: Scenario) -> None:
    for policy in all_policies():
        check_v2g_run(sc, policy)


@PROPERTY_SETTINGS
@given(v2g_scenarios(phases=True))
def test_v2g_on_phase_sites_keeps_batteries_and_lines(sc: Scenario) -> None:
    for policy in all_policies():
        check_v2g_run(sc, policy)


@PROPERTY_SETTINGS
@given(v2g_scenarios())
def test_v2g_bounds_sandwich_every_policy(sc: Scenario) -> None:
    bound = relaxation_bound(sc)
    for policy in all_policies():
        m = compute_metrics(simulate(sc, policy))  # type: ignore[arg-type]
        assert bound <= m.penalised_cost_eur + cost_tol(bound), policy
    opt = OptimalSchedule()
    executed = compute_metrics(simulate(sc, opt)).penalised_cost_eur
    assert executed <= opt.solution.objective + cost_tol(executed)


@PROPERTY_SETTINGS
@given(
    st.one_of(scenarios(), phase_scenarios(), v2g_scenarios(), v2g_scenarios(phases=True)),
    st.sampled_from([None, 0.0, 3.0]),
)
def test_forecast_mpc_is_safe_and_bounded(sc: Scenario, bandwidth: float | None) -> None:
    # the scenario's own sessions as the "history": any forecast must keep every limit
    fc = ArrivalForecast.fit([sc], bandwidth_steps=bandwidth)
    bound = relaxation_bound(sc)
    for policy in (ExpectedValueMPC(fc), ReserveMPC(fc), ScenarioMPC(fc, n_scenarios=2)):
        check_v2g_run(sc, policy)
        m = compute_metrics(simulate(sc, policy))
        assert bound <= m.penalised_cost_eur + cost_tol(bound), policy
