from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

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
