"""Property-based tests on random feasible instances.

Only guaranteed properties are asserted:

* every policy keeps within the physical limits (site, charger/EV, window,
  minimum current, energy request) and triggers no violations;
* the LP relaxation bounds every executed schedule's penalised cost from below;
* without minimum powers the offline optimum is a pure LP, so its cost is no
  higher than any policy's;
* with perfect information (all EVs present from the start) and no
  regulariser, MPC recovers the offline optimum (Bellman's principle).
"""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from evcharge.metrics import compute_metrics
from evcharge.model import Charger, Horizon, Scenario, Session, Site, Tariff
from evcharge.optim import relaxation_bound
from evcharge.policies import (
    POLICY_FACTORIES,
    ModelPredictiveControl,
    OptimalSchedule,
)
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
