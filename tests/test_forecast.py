"""Arrival forecasts learned from training days."""

from __future__ import annotations

import numpy as np
import pytest

from evcharge import scenarios
from evcharge.forecast import ArrivalForecast, Ghost
from evcharge.model import Scenario, ValidationError


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
