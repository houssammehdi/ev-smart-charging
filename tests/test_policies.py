from __future__ import annotations

import numpy as np
import pytest

from evcharge.metrics import compute_metrics
from evcharge.policies import (
    POLICY_FACTORIES,
    EarliestDeadlineFirst,
    EqualShare,
    LeastLaxityFirst,
    ModelPredictiveControl,
    OptimalSchedule,
    Policy,
    PriceAware,
    Uncontrolled,
    make_policy,
)
from evcharge.policies.base import SessionState, priority_fill, water_fill
from evcharge.sim import simulate

from .helpers import make_scenario, session


def state(sid: str, remaining: float, *, p_min: float = 0.0, p_max: float = 11.0) -> SessionState:
    return SessionState(
        session=session(sid, "C1", 0, 4, remaining),
        delivered_kwh=0.0,
        p_min_kw=p_min,
        p_max_kw=p_max,
        steps_left=4,
        dt_h=1.0,
    )


def test_water_fill() -> None:
    assert water_fill([10, 10, 10], 15) == pytest.approx([5, 5, 5])
    assert water_fill([2, 10, 10], 15) == pytest.approx([2, 6.5, 6.5])
    assert water_fill([2, 3], 100) == pytest.approx([2, 3])
    assert water_fill([], 5) == []
    assert water_fill([4, 4], 0) == [0.0, 0.0]


def test_priority_fill_respects_minimum_and_skips() -> None:
    states = [state("A", 40, p_min=4.14), state("B", 40, p_min=4.14), state("C", 40, p_min=1.38)]
    out = priority_fill(states, 14.0)
    # A takes 11, 3 kW left: too little for B (4.14), enough for C (1.38)
    assert out == pytest.approx({"A": 11.0, "C": 3.0})


def test_priority_fill_commands_minimum_to_finish() -> None:
    out = priority_fill([state("A", 1.0, p_min=4.14)], 20.0)
    assert out == pytest.approx({"A": 4.14})


def test_uncontrolled_is_first_come_first_served() -> None:
    sc = make_scenario(
        [session("late", "C1", 1, 4, 30.0), session("early", "C2", 0, 4, 30.0)],
        grid_limit_kw=16.0,
    )
    res = simulate(sc, Uncontrolled())
    np.testing.assert_allclose(res.power_kw[:, 1], [5.0, 11.0])


def test_equal_share_splits_and_redistributes() -> None:
    sc = make_scenario(
        [
            session("A", "C1", 0, 4, 40.0),
            session("B", "C2", 0, 4, 40.0),
            session("C", "C3", 0, 4, 2.0),
        ],
        grid_limit_kw=20.0,
    )
    res = simulate(sc, EqualShare())
    # C only needs 2 kW; the other 18 kW are split equally
    np.testing.assert_allclose(res.power_kw[:, 0], [9.0, 9.0, 2.0])


def test_equal_share_admits_least_served_under_minimum_current() -> None:
    sessions = [session(s, f"C{i}", 0, 4, 40.0) for i, s in enumerate("ABC")]
    sc = make_scenario(sessions, grid_limit_kw=10.0, charger_min_kw=4.14)
    res = simulate(sc, EqualShare())
    on = res.power_kw > 0
    assert on.sum(axis=0).tolist() == [2, 2, 2, 2]  # only two fit at >= 4.14 kW
    np.testing.assert_allclose(res.power_kw[on], 5.0)
    # rotation: every EV gets a turn
    assert on.any(axis=1).all()
    assert res.violations == ()


def test_edf_and_llf_orderings_differ() -> None:
    sessions = [
        # leaves first but needs little: plenty of slack
        session("soon", "C1", 0, 3, 5.0),
        # leaves later but needs a lot: no slack
        session("later", "C2", 0, 4, 38.0),
    ]
    sc = make_scenario(sessions, grid_limit_kw=11.0)
    edf = simulate(sc, EarliestDeadlineFirst())
    llf = simulate(sc, LeastLaxityFirst())
    assert edf.power_kw[0, 0] == pytest.approx(5.0)  # EDF serves "soon" first
    assert edf.power_kw[1, 0] == pytest.approx(6.0)
    assert llf.power_kw[1, 0] == pytest.approx(11.0)  # LLF serves "later" first
    assert llf.power_kw[0, 0] == 0.0


def test_price_aware_uses_cheapest_slots() -> None:
    sc = make_scenario([session("A", "C1", 0, 4, 20.0)], prices=[0.3, 0.1, 0.2, 0.4])
    res = simulate(sc, PriceAware())
    np.testing.assert_allclose(res.power_kw[0], [0.0, 11.0, 9.0, 0.0])


def test_price_aware_values_pv_surplus_at_export_price() -> None:
    sc = make_scenario(
        [session("A", "C1", 0, 4, 8.0)],
        prices=[0.1, 0.3, 0.3, 0.3],
        export_prices=[0.05, 0.05, 0.05, 0.05],
        pv=[0, 0, 10, 0],
    )
    res = simulate(sc, PriceAware())
    np.testing.assert_allclose(res.power_kw[0], [0.0, 0.0, 8.0, 0.0])


def test_price_aware_books_capacity_in_laxity_order() -> None:
    sc = make_scenario(
        [session("flex", "C1", 0, 4, 11.0), session("tight", "C2", 0, 2, 22.0)],
        prices=[0.3, 0.1, 0.2, 0.2],
        grid_limit_kw=11.0,
    )
    res = simulate(sc, PriceAware())
    # "tight" must use both of its steps; "flex" takes the cheapest free slot
    np.testing.assert_allclose(res.power_kw[1], [11.0, 11.0, 0.0, 0.0])
    np.testing.assert_allclose(res.power_kw[0], [0.0, 0.0, 11.0, 0.0])


def test_optimal_lp_hand_checked() -> None:
    sc = make_scenario([session("A", "C1", 0, 4, 15.0)], prices=[0.3, 0.1, 0.2, 0.4])
    opt = OptimalSchedule()
    res = simulate(sc, opt)
    np.testing.assert_allclose(res.power_kw[0], [0.0, 11.0, 4.0, 0.0], atol=1e-7)
    assert opt.solution.status == "optimal"
    assert opt.solution.n_binaries == 0
    assert compute_metrics(res).energy_cost_eur == pytest.approx(1.9)


def test_optimal_flattens_peak_under_demand_charge() -> None:
    sc = make_scenario(
        [session("A", "C1", 0, 4, 20.0)], prices=[0.1, 0.1, 0.1, 0.1], demand_charge=10.0
    )
    res = simulate(sc, OptimalSchedule())
    np.testing.assert_allclose(res.power_kw[0], [5.0, 5.0, 5.0, 5.0], atol=1e-7)


def test_optimal_respects_minimum_power_with_both_strategies() -> None:
    sessions = [session(s, f"C{i}", 0, 4, 6.0) for i, s in enumerate("ABC")]
    sc = make_scenario(sessions, grid_limit_kw=9.0, charger_min_kw=4.14, demand_charge=1.0)
    costs = []
    for strategy in ("exact", "relax-and-fix"):
        opt = OptimalSchedule(strategy=strategy)
        res = simulate(sc, opt)
        p = res.setpoint_kw
        assert np.all((p == 0) | (p >= 4.14 - 1e-9))
        assert res.violations == ()
        assert opt.solution.lower_bound <= opt.solution.objective + 1e-9
        costs.append(compute_metrics(res).penalised_cost_eur)
    assert costs[0] <= costs[1] + 1e-6  # exact MILP is never worse


def test_overshoot_is_not_fictitious_revenue_at_negative_prices() -> None:
    # Regression: the overshoot slack (energy commanded at p_min but never drawn)
    # was only paid for at the import price. At a negative price that is revenue,
    # so the MILP planned 5.14 kW for a 1 kWh request and reported -2.57 EUR while
    # the replay costs -0.50 EUR. The plan must bound its own execution from above.
    sc = make_scenario(
        [session("A", "C1", 0, 1, 1.0, min_kw=4.14)],
        n_steps=1,
        prices=[-0.5],
        export_prices=[-0.5],
    )
    opt = OptimalSchedule()
    executed = compute_metrics(simulate(sc, opt)).penalised_cost_eur
    assert executed == pytest.approx(-0.5)
    assert opt.solution.objective >= executed - 1e-9
    assert opt.solution.objective == pytest.approx(-0.5)


def test_mpc_reoptimises_online() -> None:
    # B arrives unannounced at step 2 and needs the whole remaining capacity
    sc = make_scenario(
        [session("A", "C1", 0, 4, 11.0), session("B", "C2", 2, 4, 22.0)],
        prices=[0.3, 0.3, 0.1, 0.1],
        grid_limit_kw=11.0,
    )
    mpc = simulate(sc, ModelPredictiveControl(quick_charge_weight=0.0))
    opt = simulate(sc, OptimalSchedule())
    # Without foresight, MPC postpones A into the cheap slots B will need.
    assert compute_metrics(mpc).unmet_kwh == pytest.approx(11.0)
    assert compute_metrics(opt).unmet_kwh == pytest.approx(0.0)
    assert mpc.violations == ()


def test_mpc_quick_charge_weight_prefers_early_charging() -> None:
    sc = make_scenario([session("A", "C1", 0, 4, 11.0)], prices=[0.1, 0.1, 0.1, 0.1])
    res = simulate(sc, ModelPredictiveControl(quick_charge_weight=0.01))
    np.testing.assert_allclose(res.power_kw[0], [11.0, 0.0, 0.0, 0.0], atol=1e-7)
    with pytest.raises(ValueError, match="quick_charge_weight"):
        ModelPredictiveControl(quick_charge_weight=-1.0)


def test_mpc_does_not_pay_twice_for_an_existing_peak() -> None:
    # base load already peaks at 20 kW late in the day, so charging up to that level is free
    sc = make_scenario(
        [session("A", "C1", 0, 2, 16.0)],
        base_load=[2, 2, 2, 20],
        prices=[0.1, 0.1, 0.1, 0.1],
        demand_charge=5.0,
        grid_limit_kw=30.0,
    )
    mpc = compute_metrics(simulate(sc, ModelPredictiveControl(quick_charge_weight=0.0)))
    opt = compute_metrics(simulate(sc, OptimalSchedule()))
    assert mpc.peak_import_kw == pytest.approx(20.0)
    assert mpc.total_cost_eur == pytest.approx(opt.total_cost_eur)


def test_registry() -> None:
    assert list(POLICY_FACTORIES) == [
        "uncontrolled",
        "equal-share",
        "edf",
        "llf",
        "price-aware",
        "mpc",
        "optimal",
    ]
    for name in POLICY_FACTORIES:
        p = make_policy(name)
        assert p.name == name
        assert isinstance(p, Policy)
        assert repr(p)
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("magic")
    with pytest.raises(RuntimeError):
        _ = Uncontrolled().scenario
    with pytest.raises(RuntimeError):
        _ = OptimalSchedule().solution
