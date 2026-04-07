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


@pytest.mark.parametrize("grid", ["TN", "IT"])
def test_phase_aware_generator(grid: str) -> None:
    kw = scenarios.generate("residential", n_sessions=30, seed=5)
    ph = scenarios.generate("residential", n_sessions=30, seed=5, grid=grid)
    assert ph.name == f"residential-{grid}"
    # phase options never change the random draws
    assert [(s.arrival_step, s.departure_step) for s in ph.sessions] == [
        (s.arrival_step, s.departure_step) for s in kw.sessions
    ]
    np.testing.assert_array_equal(ph.tariff.price_eur_per_kwh, kw.tariff.price_eur_per_kwh)
    np.testing.assert_array_equal(ph.base_load, kw.base_load)
    assert ph.site.supply is not None
    assert ph.site.supply.fuse_equivalent_kw == pytest.approx(ph.site.grid_limit_kw)
    assert [c.rotation for c in ph.site.chargers[:4]] == ["L1L2L3", "L2L3L1", "L3L1L2", "L1L2L3"]
    for s_kw, s_ph in zip(kw.sessions, ph.sessions, strict=True):
        ctl = ph.control(s_ph)
        assert ctl.unit == "A"
        assert ctl.charge_min == 6.0
        assert s_ph.phases == (1 if s_kw.max_power_kw == 3.7 else 3)
        # requests stay individually feasible with the real (grid-dependent) power
        p_max = ctl.charge_max * ctl.kw_per_unit
        assert s_ph.energy_kwh <= p_max * s_ph.efficiency * s_ph.dwell_steps * ph.horizon.dt_h
    if grid == "TN":
        assert [s.energy_kwh for s in ph.sessions] == [s.energy_kwh for s in kw.sessions]


def test_single_phase_share_and_rotation_options() -> None:
    sc = scenarios.generate(
        "depot", n_sessions=40, seed=2, grid="TN", single_phase_share=1.0, rotate_phases=False
    )
    assert all(s.phases == 1 for s in sc.sessions)
    assert all(c.rotation == "L1L2L3" for c in sc.site.chargers)
    none = scenarios.generate("workplace", n_sessions=40, seed=2, grid="IT", single_phase_share=0)
    assert all(s.phases == 3 for s in none.sessions)
    fused = scenarios.generate("workplace", n_sessions=10, seed=2, grid="IT", line_limit_a=40.0)
    assert fused.site.supply is not None
    assert fused.site.supply.line_limit_a == (40.0, 40.0, 40.0)
    with pytest.raises(ValidationError, match="phase-aware grid"):
        scenarios.generate("workplace", line_limit_a=40.0)
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        scenarios.generate("workplace", grid="TN", single_phase_share=1.5)


def test_v2g_share_adds_batteries_without_shifting_the_draws() -> None:
    plain = scenarios.generate("residential", n_sessions=40, seed=4)
    v2g = scenarios.generate(
        "residential", n_sessions=40, seed=4, v2g_share=0.5, degradation_eur_per_kwh=0.03
    )
    assert not plain.site.bidirectional
    assert all(s.v2g is None for s in plain.sessions)
    np.testing.assert_array_equal(v2g.tariff.price_eur_per_kwh, plain.tariff.price_eur_per_kwh)
    two_way = {c.id for c in v2g.site.chargers if c.bidirectional}
    n_v2g = 0
    for p, s in zip(plain.sessions, v2g.sessions, strict=True):
        assert (s.arrival_step, s.departure_step, s.charger_id) == (
            p.arrival_step,
            p.departure_step,
            p.charger_id,
        )
        if s.v2g is None:
            assert s.energy_kwh == p.energy_kwh
            continue
        n_v2g += 1
        assert s.charger_id in two_way
        b = s.v2g
        assert b.min_kwh == pytest.approx(0.2 * b.capacity_kwh)
        assert b.ceiling_kwh == pytest.approx(0.9 * b.capacity_kwh)
        assert 0.3 * b.capacity_kwh - 0.01 <= b.initial_kwh <= 0.7 * b.capacity_kwh + 0.01
        assert b.degradation_eur_per_kwh == 0.03
        # the request is capped so that the departure target fits under the ceiling
        assert s.energy_kwh <= p.energy_kwh
        assert s.target_kwh is not None
        assert s.target_kwh <= b.ceiling_kwh + 1e-9
        assert v2g.control(s).can_discharge
    assert 10 <= n_v2g <= 30
    everyone = scenarios.generate("depot", n_sessions=10, seed=4, v2g_share=1.0, grid="TN")
    assert all(s.v2g is not None for s in everyone.sessions)
    assert everyone.rows.names[-1] == "site export"
    with pytest.raises(ValidationError, match="v2g_share"):
        scenarios.generate("workplace", v2g_share=1.5)
