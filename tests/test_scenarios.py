from __future__ import annotations

import numpy as np
import pytest

from evcharge import scenarios
from evcharge.model import ValidationError


@pytest.mark.parametrize("kind", sorted(scenarios.PROFILES))
def test_generators_are_deterministic_and_valid(kind: str) -> None:
    a = scenarios.generate(kind, n_sessions=25, seed=3)
    b = scenarios.generate(kind, n_sessions=25, seed=3)
    c = scenarios.generate(kind, n_sessions=25, seed=4)
    assert a.sessions == b.sessions
    np.testing.assert_array_equal(a.tariff.price_eur_per_kwh, b.tariff.price_eur_per_kwh)
    assert a.sessions != c.sessions
    assert len(a.sessions) == 25
    assert a.horizon.n_steps == 96
    dt = a.horizon.dt_h
    for s in a.sessions:
        _, p_max = a.power_bounds(s)
        # every request is individually feasible
        assert s.energy_kwh <= p_max * s.efficiency * s.dwell_steps * dt + 1e-9
        assert 0.88 <= s.efficiency <= 0.94
    assert np.all(a.tariff.export_price <= a.tariff.price_eur_per_kwh)


def test_named_wrappers_match_generate() -> None:
    w = scenarios.workplace(n_sessions=5, seed=1)
    assert w.sessions == scenarios.generate("workplace", n_sessions=5, seed=1).sessions
    assert scenarios.depot(n_sessions=5).name == "depot"
    assert scenarios.residential(n_sessions=5).name == "residential"


def test_profiles_have_the_expected_timing() -> None:
    work = scenarios.workplace(n_sessions=200, seed=11)
    hours = np.array([work.horizon.clock_hours()[s.arrival_step] for s in work.sessions])
    assert 7.5 < float(np.median(hours)) < 9.5
    dwell = np.array([s.dwell_steps * work.horizon.dt_h for s in work.sessions])
    assert 7.0 < float(np.median(dwell)) < 9.5

    res = scenarios.residential(n_sessions=200, seed=11)
    assert res.horizon.start.hour == 12
    arrival_clock = res.horizon.clock_hours()[[s.arrival_step for s in res.sessions]]
    assert 16.0 < float(np.median(arrival_clock)) < 19.5

    dep = scenarios.depot(n_sessions=100, seed=11)
    assert float(np.median([s.energy_kwh for s in dep.sessions])) > 30.0


def test_price_shape_has_morning_and_evening_peaks() -> None:
    sc = scenarios.workplace(n_sessions=5, seed=2)
    price = sc.tariff.price_eur_per_kwh
    hours = sc.horizon.clock_hours()

    def mean_at(lo: float, hi: float) -> float:
        return float(price[(hours >= lo) & (hours < hi)].mean())

    night, morning, midday, evening = mean_at(2, 5), mean_at(7, 9), mean_at(12, 14), mean_at(17, 20)
    assert morning > max(night, midday)
    assert evening > max(night, midday)


def test_pv_profile() -> None:
    sc = scenarios.workplace(n_sessions=5, seed=2, pv_kwp=50.0)
    hours = sc.horizon.clock_hours()
    assert np.all(sc.pv[(hours < 6) | (hours > 21)] == 0.0)
    assert 10.0 < sc.pv.max() <= 0.75 * 50.0
    assert scenarios.workplace(n_sessions=5, seed=2).pv.max() == 0.0


def test_parameters_and_errors() -> None:
    sc = scenarios.depot(
        n_sessions=10,
        seed=5,
        grid_limit_kw=80.0,
        step_minutes=30,
        base_load_peak_kw=0.0,
        demand_charge_eur_per_kw=1.5,
    )
    assert sc.site.grid_limit_kw == 80.0
    assert sc.horizon.n_steps == 48
    assert sc.base_load.max() == 0.0
    assert sc.tariff.demand_charge_eur_per_kw == 1.5
    assert all(c.max_power_kw == 22.0 for c in sc.site.chargers)
    with pytest.raises(ValidationError, match="unknown scenario"):
        scenarios.generate("airport")
    with pytest.raises(ValidationError):
        scenarios.generate("workplace", n_sessions=0)
    with pytest.raises(ValidationError, match="divide 60"):
        scenarios.generate("workplace", step_minutes=7)
    with pytest.raises(ValidationError):
        scenarios.generate("workplace", pv_kwp=-1.0)
    with pytest.raises(ValidationError, match="exceeds the grid limit"):
        scenarios.generate("workplace", n_sessions=10, grid_limit_kw=5.0, base_load_peak_kw=10.0)
