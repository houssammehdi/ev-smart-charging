from __future__ import annotations

import numpy as np
import pytest

from evcharge.metrics import compute_metrics, jain_index
from evcharge.policies import Uncontrolled
from evcharge.sim import simulate

from .helpers import make_scenario, session


@pytest.mark.parametrize(
    ("values", "expected"),
    [([1, 1, 1, 1], 1.0), ([1, 0, 0, 0], 0.25), ([0.5, 1.0], 0.9), ([], 1.0), ([0, 0], 1.0)],
)
def test_jain_index(values: list[float], expected: float) -> None:
    assert jain_index(np.asarray(values, dtype=float)) == pytest.approx(expected)


def test_metrics_hand_computed() -> None:
    sc = make_scenario(
        [session("A", "C1", 0, 2, 10.0), session("B", "C2", 2, 4, 4.0)],
        grid_limit_kw=20.0,
        prices=[0.1, 0.2, 0.3, 0.4],
        export_prices=[0.05, 0.1, 0.15, 0.2],
        demand_charge=2.0,
        base_load=[1, 1, 1, 1],
        pv=[0, 0, 7, 0],
    )
    res = simulate(sc, Uncontrolled())
    np.testing.assert_allclose(res.net_import_kw, [11.0, 1.0, -2.0, 1.0])
    m = compute_metrics(res, unmet_penalty=50.0)
    assert m.policy == "uncontrolled"
    assert m.energy_requested_kwh == 14.0
    assert m.energy_delivered_kwh == pytest.approx(14.0)
    assert m.delivered_pct == pytest.approx(100.0)
    assert m.unmet_kwh == pytest.approx(0.0)
    assert m.sessions_completed_pct == 100.0
    # 0.1*11 + 0.2*1 + 0.4*1 - 0.15*2
    assert m.energy_cost_eur == pytest.approx(1.4)
    assert m.peak_import_kw == pytest.approx(11.0)
    assert m.demand_charge_eur == pytest.approx(22.0)
    assert m.total_cost_eur == pytest.approx(23.4)
    assert m.penalised_cost_eur == pytest.approx(23.4)
    assert m.jain_fairness == pytest.approx(1.0)
    # EV energy 14 kWh over headroom 19 + 19 + 26 + 19 kWh (EVs connected in every step)
    assert m.capacity_utilisation_pct == pytest.approx(100 * 14 / 83)
    assert m.load_factor_pct == pytest.approx(100 * 3.25 / 11)
    assert m.violations == 0
    assert m.as_dict()["policy"] == "uncontrolled"


def test_metrics_with_unmet_energy() -> None:
    sc = make_scenario(
        [session("A", "C1", 0, 1, 30.0), session("B", "C2", 0, 1, 5.0)],
        n_steps=2,
        prices=[0.1, 0.1],
    )
    m = compute_metrics(simulate(sc, Uncontrolled()), unmet_penalty=10.0)
    assert m.energy_delivered_kwh == pytest.approx(16.0)
    assert m.unmet_kwh == pytest.approx(19.0)
    assert m.delivered_pct == pytest.approx(100 * 16 / 35)
    assert m.sessions_completed_pct == 50.0
    assert m.total_cost_eur == pytest.approx(1.6)
    assert m.penalised_cost_eur == pytest.approx(1.6 + 190.0)
    assert m.jain_fairness == pytest.approx(jain_index(np.array([11 / 30, 1.0])))
    # only step 0 has EVs connected: 16 kWh of 100 kWh available
    assert m.capacity_utilisation_pct == pytest.approx(16.0)


def test_metrics_without_sessions() -> None:
    m = compute_metrics(simulate(make_scenario([], base_load=[2, 2, 2, 2]), Uncontrolled()))
    assert m.delivered_pct == 100.0
    assert m.jain_fairness == 1.0
    assert m.capacity_utilisation_pct == 0.0
    assert m.energy_cost_eur == pytest.approx(0.8)
