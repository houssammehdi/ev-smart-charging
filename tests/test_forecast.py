"""Arrival forecasts and the forecast-aware MPC variants."""

from __future__ import annotations

import numpy as np
import pytest

from evcharge import scenarios
from evcharge.forecast import ArrivalForecast, Ghost
from evcharge.metrics import compute_metrics
from evcharge.model import Charger, Scenario, Session, ValidationError
from evcharge.policies import ModelPredictiveControl
from evcharge.policies.forecast import ExpectedValueMPC, ReserveMPC, ScenarioMPC, ghost_rows
from evcharge.sim import simulate

from .helpers import make_scenario


def training(n_days: int = 6) -> list[Scenario]:
    return [scenarios.generate("workplace", n_sessions=20, seed=100 + d) for d in range(n_days)]


def test_fit_keeps_every_session_and_checks_the_clock() -> None:
    days = training()
    fc = ArrivalForecast.fit(days)
    assert fc.n_days == 6
    assert fc.counts == (20,) * 6
    assert len(fc.samples) == 120
    assert fc.mean_sessions_per_day == 20.0
    assert 0.0 < fc.bandwidth_steps < 3.0  # Silverman on arrivals with an 0.8 h spread
    residential = scenarios.generate("residential", n_sessions=5)
    with pytest.raises(ValidationError, match="different clock"):
        ArrivalForecast.fit([days[0], residential])
    with pytest.raises(ValidationError, match="forecast's clock"):
        fc.check(residential)
    with pytest.raises(ValidationError, match="at least one"):
        ArrivalForecast.fit([])


def test_expected_ghosts_are_the_empirical_expectation() -> None:
    days = training()
    fc = ArrivalForecast.fit(days, bandwidth_steps=0.0)
    now = 32  # 08:00
    expected = sum(
        s.energy_kwh / s.efficiency for d in days for s in d.sessions if s.arrival_step > now
    ) / len(days)
    ghosts = fc.expected_ghosts(now, bin_steps=1)
    assert all(g.arrival_step > now for g in ghosts)
    assert sum(g.energy_kwh for g in ghosts) == pytest.approx(expected)
    # coarser cells keep the energy (windows are long enough here)
    assert sum(g.energy_kwh for g in fc.expected_ghosts(now)) == pytest.approx(expected)
    # uniform-rate load integrates to the same energy
    assert fc.expected_load_kw(now).sum() * fc.dt_h == pytest.approx(expected)
    # with a kernel, sessions just before "now" still count partly
    smooth = ArrivalForecast.fit(days, bandwidth_steps=2.0)
    assert sum(g.energy_kwh for g in smooth.expected_ghosts(now)) > 0.0


def test_sampled_days_are_reproducible_and_in_the_future() -> None:
    fc = ArrivalForecast.fit(training())
    a = fc.sample_ghosts(np.random.default_rng(3), 30)
    b = fc.sample_ghosts(np.random.default_rng(3), 30)
    assert a == b
    assert a
    for g in a:
        assert isinstance(g, Ghost)
        assert 30 < g.arrival_step < g.departure_step <= fc.n_steps
        assert g.energy_kwh <= g.max_power_kw * (g.departure_step - g.arrival_step) * fc.dt_h


def test_ghost_rows_follow_the_installation() -> None:
    kw = scenarios.generate("workplace", n_sessions=4)
    assert ghost_rows(kw)[3].tolist() == [1.0]
    tn = scenarios.generate("workplace", n_sessions=6, grid="TN")
    rows = ghost_rows(tn)
    assert tn.rows.names[:4] == ("L1 import", "L2 import", "L3 import", "site import")
    # a three-phase EV draws 1000 / (3 x 230) A per kW on every line
    np.testing.assert_allclose(rows[3][:4], [1000 / 690] * 3 + [1.0])
    # single-phase EVs on rotated chargers: on average a third of 1000 / 230 per line
    np.testing.assert_allclose(rows[1][:3].sum(), 1000 / 230)


def arrival_day() -> Scenario:
    """A connects at 0 and B at 1; both need 10 kWh by step 2 behind 10 kW; step 1 is cheap."""
    return make_scenario(
        [
            Session("A", "C1", 0, 2, 10.0, 10.0, efficiency=1.0),
            Session("B", "C2", 1, 2, 10.0, 10.0, efficiency=1.0),
        ],
        n_steps=3,
        grid_limit_kw=10.0,
        prices=[0.2, 0.1, 0.1],
        chargers=[Charger("C1", 10.0, 0.0), Charger("C2", 10.0, 0.0)],
    )


@pytest.mark.parametrize("make", [ExpectedValueMPC, ReserveMPC, ScenarioMPC])
def test_forecast_mpc_charges_ahead_of_a_known_rush(make: type[ExpectedValueMPC]) -> None:
    day = arrival_day()
    plain = compute_metrics(simulate(day, ModelPredictiveControl(quick_charge_weight=0.0)))
    assert plain.unmet_kwh == pytest.approx(10.0)
    forecast = ArrivalForecast.fit([day, day], bandwidth_steps=0.0)
    res = simulate(day, make(forecast))
    assert res.violations == ()
    np.testing.assert_allclose(res.power_kw[:, :2], [[10.0, 0.0], [0.0, 10.0]], atol=1e-7)
    assert compute_metrics(res).unmet_kwh == pytest.approx(0.0, abs=1e-7)


def test_forecast_policies_respect_phase_rows_and_are_reproducible() -> None:
    days = [
        scenarios.generate("workplace", n_sessions=6, seed=s, grid="IT", step_minutes=60)
        for s in (1, 2)
    ]
    fc = ArrivalForecast.fit(days)
    test_day = scenarios.generate("workplace", n_sessions=6, seed=9, grid="IT", step_minutes=60)
    for policy in (ExpectedValueMPC(fc), ReserveMPC(fc), ScenarioMPC(fc, n_scenarios=3)):
        first = simulate(test_day, policy)
        assert first.violations == ()
        np.testing.assert_array_equal(simulate(test_day, policy).power_kw, first.power_kw)
    with pytest.raises(ValidationError, match="forecast's clock"):
        simulate(scenarios.generate("workplace", n_sessions=3, grid="IT"), ReserveMPC(fc))


def test_forecast_policy_arguments_are_checked() -> None:
    fc = ArrivalForecast.fit([arrival_day()])
    with pytest.raises(ValueError, match="bin_steps"):
        ExpectedValueMPC(fc, bin_steps=0)
    with pytest.raises(ValueError, match="n_scenarios"):
        ScenarioMPC(fc, n_scenarios=0)
    assert repr(ScenarioMPC(fc, n_scenarios=4, seed=2)) == "ScenarioMPC(n_scenarios=4, seed=2)"
