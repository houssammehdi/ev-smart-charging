from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from evcharge.electrical import Supply
from evcharge.model import (
    MIN_POWER_1PH_KW,
    MIN_POWER_3PH_KW,
    Charger,
    Horizon,
    Session,
    Site,
    Tariff,
    ValidationError,
    ac_power_kw,
)

from .helpers import START, make_scenario, session


def test_iec_minimum_powers() -> None:
    assert ac_power_kw(6.0) == pytest.approx(4.14)
    assert ac_power_kw(6.0, phases=1) == pytest.approx(1.38)
    assert ac_power_kw(16.0) == pytest.approx(11.04)
    assert pytest.approx(4.14) == MIN_POWER_3PH_KW
    assert pytest.approx(1.38) == MIN_POWER_1PH_KW
    with pytest.raises(ValidationError):
        ac_power_kw(6.0, phases=4)


def test_horizon_steps_and_rounding() -> None:
    hz = Horizon.spanning(START, 24, 15)
    assert hz.n_steps == 96
    assert hz.dt_h == 0.25
    assert hz.end == START + timedelta(hours=24)
    assert hz.time_of(4) == START + timedelta(hours=1)
    # arrivals round up, departures round down
    t = START + timedelta(minutes=37)
    assert hz.step_at_or_after(t) == 3
    assert hz.step_at_or_before(t) == 2
    assert hz.step_at_or_after(START + timedelta(minutes=30)) == 2
    assert hz.step_at_or_before(START + timedelta(minutes=30)) == 2
    np.testing.assert_allclose(hz.clock_hours()[:3], [0.0, 0.25, 0.5])


def test_horizon_validation() -> None:
    with pytest.raises(ValidationError, match="whole number"):
        Horizon.spanning(START, 1.1, 15)
    with pytest.raises(ValidationError):
        Horizon(START, 0)
    with pytest.raises(ValidationError):
        Horizon(START, 4, 0)
    hz = Horizon(START, 4)
    with pytest.raises(ValidationError, match="naive"):
        hz.step_at_or_after(datetime(2026, 1, 5, tzinfo=UTC))
    with pytest.raises(ValidationError):
        hz.time_of(5)


def test_charger_and_site_validation() -> None:
    with pytest.raises(ValidationError, match="min_power_kw"):
        Charger("C1", 3.0, 4.14)
    with pytest.raises(ValidationError):
        Charger("", 11.0)
    with pytest.raises(ValidationError, match="duplicate"):
        Site(50.0, (Charger("C1", 11), Charger("C1", 22)))
    with pytest.raises(ValidationError):
        Site(0.0, (Charger("C1", 11),))
    with pytest.raises(ValidationError, match="at least one"):
        Site(10.0, ())
    with pytest.raises(ValidationError, match="unknown charger"):
        Site(10.0, (Charger("C1", 11),)).charger("C9")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"arrival_step": 3, "departure_step": 3}, "shorter than one step"),
        ({"arrival_step": -1}, "arrival_step"),
        ({"energy_kwh": 0.0}, "energy_kwh"),
        ({"energy_kwh": float("nan")}, "finite"),
        ({"max_power_kw": 0.0}, "max_power_kw"),
        ({"efficiency": 1.2}, "efficiency"),
        ({"min_power_kw": 20.0}, "min_power_kw"),
    ],
)
def test_session_validation(kwargs: dict[str, float], match: str) -> None:
    base: dict[str, object] = {
        "id": "S1",
        "charger_id": "C1",
        "arrival_step": 0,
        "departure_step": 4,
        "energy_kwh": 10.0,
        "max_power_kw": 11.0,
    }
    base.update(kwargs)
    with pytest.raises(ValidationError, match=match):
        Session(**base)  # type: ignore[arg-type]


def test_tariff_validation() -> None:
    with pytest.raises(ValidationError, match="export price exceeds"):
        Tariff(np.array([0.1, 0.1]), np.array([0.05, 0.2]))
    with pytest.raises(ValidationError, match="NaN"):
        Tariff(np.array([0.1, np.nan]))
    with pytest.raises(ValidationError):
        Tariff(np.array([0.1]), demand_charge_eur_per_kw=-1.0)
    t = Tariff(np.array([0.1, 0.2]))
    np.testing.assert_array_equal(t.export_price, [0.0, 0.0])


def test_scenario_power_bounds() -> None:
    sc = make_scenario(
        [
            session("A", "C1", 0, 2, 5.0, max_kw=22.0),
            session("B", "C2", 0, 2, 5.0, max_kw=3.7, min_kw=1.38),
            session("C", "C3", 0, 2, 5.0, max_kw=3.0),
        ],
        charger_min_kw=4.14,
    )
    assert sc.power_bounds(sc.sessions[0]) == (4.14, 11.0)
    assert sc.power_bounds(sc.sessions[1]) == (1.38, 3.7)
    # EV cannot reach the charger minimum: on/off at its own maximum
    assert sc.power_bounds(sc.sessions[2]) == (3.0, 3.0)
    with pytest.raises(ValidationError, match="not part"):
        sc.power_bounds(session("X", "C1", 0, 1, 1.0))


def test_scenario_rejects_inconsistent_input() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        make_scenario([session("A", "C1", 0, 3, 5.0), session("B", "C1", 2, 4, 5.0)])
    with pytest.raises(ValidationError, match="duplicate session"):
        make_scenario([session("A", "C1", 0, 2, 5.0), session("A", "C2", 0, 2, 5.0)])
    with pytest.raises(ValidationError, match="after the horizon"):
        make_scenario([session("A", "C1", 0, 5, 5.0)])
    with pytest.raises(ValidationError, match="unknown charger"):
        make_scenario([session("A", "C9", 0, 2, 5.0)], chargers=[Charger("C1", 11.0)])
    with pytest.raises(ValidationError, match="exceeds the grid limit"):
        make_scenario([], grid_limit_kw=10.0, base_load=[5, 12, 5, 5])
    with pytest.raises(ValidationError, match="tariff has"):
        make_scenario([], prices=[0.1, 0.1])
    with pytest.raises(ValidationError, match="pv_kw"):
        make_scenario([], pv=[0, -1, 0, 0])


def test_back_to_back_sessions_are_allowed_and_headroom() -> None:
    sc = make_scenario(
        [session("A", "C1", 0, 2, 5.0), session("B", "C1", 2, 4, 5.0)],
        grid_limit_kw=20.0,
        base_load=[5, 5, 12, 5],
        pv=[0, 3, 0, 0],
    )
    np.testing.assert_allclose(sc.net_base_kw, [5, 2, 12, 5])
    np.testing.assert_allclose(sc.ev_headroom_kw, [15, 18, 8, 15])
    assert sc.energy_requested_kwh == 10.0
    assert sc.without_sessions().sessions == ()
    # base load above the limit is fine when PV covers it
    make_scenario([], grid_limit_kw=10.0, base_load=[12, 0, 0, 0], pv=[3, 0, 0, 0])


def phase_site(grid: str = "TN", *, limit_a: float = 32.0, chargers: list[Charger]) -> Site:
    return Site(100.0, tuple(chargers), supply=Supply.uniform(limit_a, grid=grid))


def test_phase_aware_controls_are_in_amperes() -> None:
    site = phase_site(
        chargers=[
            Charger("C1", 22.0, max_current_a=32.0),
            Charger("C2", 11.0, rotation="L2L3L1"),
            Charger("C3", 7.4, phases=1, rotation="L3", current_step_a=1.0),
        ]
    )
    sc = make_scenario(
        [
            Session("A", "C1", 0, 2, 5.0, 11.0, max_current_a=16.0),  # 3-phase 16 A EV
            Session("B", "C2", 0, 2, 5.0, 7.4, phases=1, max_current_a=32.0),
            Session("C", "C3", 0, 2, 5.0, 11.0, phases=3),
        ],
        chargers=list(site.chargers),
        supply=site.supply,
    )
    a, b, c = sc.sessions
    ca = sc.control(a)
    # every limit applies: 16 A x 3 x 230 V = 11.04 kW, so the declared 11.0 kW binds
    # first (15.94 A), rounded down to the 0.1 A resolution
    assert (ca.unit, ca.charge_min, ca.charge_max, ca.step) == ("A", 6.0, 15.9, 0.1)
    assert ca.kw_per_unit == pytest.approx(0.69)
    assert sc.power_bounds(a) == pytest.approx((4.14, 15.9 * 0.69))
    # 11 kW charger on one phase: the power cap binds at 11 / 0.23 = 47.8 A, the
    # EV's 7.4 kW at 32.17 A and its 32 A current limit below that
    cb = sc.control(b)
    assert cb.charge_max == pytest.approx(32.0)
    wb = sc.wiring(b)
    assert wb is not None
    assert wb.lines == (1,)
    # 7.4 kW single-phase charger on L3 with 1 A resolution: floor(7400 / 230) = 32 A
    cc = sc.control(c)
    assert (cc.charge_min, cc.charge_max, cc.step) == (6.0, 32.0, 1.0)
    assert sc.row_coefficients(c).tolist() == pytest.approx([0.0, 0.0, 1.0, 0.23])
    assert sc.rows.names == ("L1 import", "L2 import", "L3 import", "site import")
    assert sc.rows.kinds == ("line", "line", "line", "site")


def test_phase_minimum_is_a_current_not_a_power() -> None:
    # The aggregate model turns a 3.7 kW single-phase EV on a default charger into
    # an on/off 3.7 kW load; with phases the minimum is 6 A on one phase (1.38 kW).
    sessions = [Session("A", "C1", 0, 2, 2.0, 3.7)]
    assert make_scenario(sessions, charger_min_kw=4.14).power_bounds(sessions[0]) == (3.7, 3.7)
    one_phase = [Session("A", "C1", 0, 2, 2.0, 3.7, phases=1)]
    sc = make_scenario(one_phase, chargers=[Charger("C1", 11.0)], supply=Supply.uniform(25.0))
    assert sc.power_bounds(one_phase[0]) == pytest.approx((1.38, 3.68))
    with pytest.raises(ValidationError, match="cannot charge at the 6 A minimum"):
        make_scenario(sessions, chargers=[Charger("C1", 11.0)], supply=Supply.uniform(25.0))


def test_phase_rows_and_line_currents() -> None:
    chargers = [Charger("C1", 11.0), Charger("C2", 11.0, rotation="L2L3L1")]
    sessions = [
        Session("A", "C1", 0, 2, 5.0, 3.7, phases=1),
        Session("B", "C2", 0, 2, 5.0, 11.0),
    ]
    base = np.array([[10.0, 0.0, 5.0], [10.0, 0.0, 5.0]])
    pv = np.array([[0.0, 0.0, 0.0], [8.0, 8.0, 8.0]])
    tn = make_scenario(
        sessions,
        n_steps=2,
        chargers=chargers,
        supply=Supply.uniform(32.0),
        base_current_a=base,
        pv_current_a=pv,
    )
    # TN: PV current is credited (collinear, unity power factor)
    np.testing.assert_allclose(tn.rows.rhs[:, :3], [[22.0, 32.0, 27.0], [30.0, 40.0, 35.0]])
    x = np.array([[10.0, 16.0], [6.0, 6.0]])  # A on L1 (1-phase); B 3-phase
    np.testing.assert_allclose(tn.line_currents_a(x), [[26.0, 6.0, 11.0], [24.0, 2.0, 3.0]])
    it = make_scenario(
        sessions,
        n_steps=2,
        chargers=chargers,
        supply=Supply.uniform(32.0, grid="IT"),
        base_current_a=base,
        pv_current_a=pv,
    )
    # IT: no PV credit, and the single-phase EV loads L1 and L2
    np.testing.assert_allclose(it.rows.rhs[:, :3], [[22.0, 32.0, 27.0], [22.0, 32.0, 27.0]])
    assert it.row_coefficients(it.sessions[0]).tolist()[:3] == [1.0, 1.0, 0.0]
    np.testing.assert_allclose(it.line_currents_a(x), [[26.0, 16.0, 11.0], [32.0, 22.0, 11.0]])


def test_phase_validation() -> None:
    chargers = [Charger("C1", 11.0)]
    with pytest.raises(ValidationError, match="L2 limit at step 1"):
        make_scenario(
            [],
            n_steps=2,
            chargers=chargers,
            supply=Supply.uniform(16.0),
            base_current_a=np.array([[1.0, 1.0, 1.0], [1.0, 17.0, 1.0]]),
        )
    with pytest.raises(ValidationError, match="shape"):
        make_scenario([], chargers=chargers, supply=Supply.uniform(16.0), base_current_a=[1.0])
    with pytest.raises(ValidationError, match="phase-aware"):
        make_scenario([], chargers=chargers, base_current_a=np.zeros((4, 3)))
    with pytest.raises(ValidationError, match="two-phase"):
        make_scenario(
            [Session("A", "C1", 0, 2, 5.0, 11.0, phases=2)],
            chargers=chargers,
            supply=Supply.uniform(32.0, grid="IT"),
        )
    with pytest.raises(ValidationError, match="charger C1: a single-phase IT charger"):
        Site(10.0, (Charger("C1", 7.4, phases=1, rotation="L1"),), Supply.uniform(32, "IT"))
    with pytest.raises(ValidationError, match="phases must be 1 or 3"):
        Charger("C1", 11.0, phases=2)
    with pytest.raises(ValidationError, match="phases must be 1, 2 or 3"):
        Session("A", "C1", 0, 2, 5.0, 11.0, phases=4)
    with pytest.raises(ValidationError, match="line currents need"):
        make_scenario([session("A", "C1", 0, 2, 5.0)]).line_currents_a(np.zeros((1, 4)))
    # balanced default: 6.9 kW of base load is 10 A per line on TN
    sc = make_scenario([], chargers=chargers, supply=Supply.uniform(16.0), base_load=[6.9] * 4)
    np.testing.assert_allclose(sc.base_line_current_a, 10.0)
    assert sc.without_sessions().base_current_a is not None
