from __future__ import annotations

import numpy as np
import pytest

from evcharge.capacity import StepConstraints
from evcharge.electrical import Supply
from evcharge.forecast import ArrivalForecast
from evcharge.metrics import compute_metrics
from evcharge.model import Charger, Control, Session
from evcharge.optim import relaxation_bound
from evcharge.policies import (
    FORECAST_POLICY_FACTORIES,
    POLICY_FACTORIES,
    EarliestDeadlineFirst,
    EqualShare,
    LeastLaxityFirst,
    ModelPredictiveControl,
    OptimalSchedule,
    PhaseBlind,
    Policy,
    PriceAware,
    Uncontrolled,
    make_policy,
)
from evcharge.policies.base import Observation, SessionState, finalize, priority_fill, water_fill
from evcharge.sim import ViolationKind, simulate

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
    forecast = ArrivalForecast.fit([make_scenario([session("A", "C1", 0, 4, 5.0)])])
    for name in FORECAST_POLICY_FACTORIES:
        assert make_policy(name, forecast).name == name
        with pytest.raises(ValueError, match="needs an arrival forecast"):
            make_policy(name)
    with pytest.raises(RuntimeError):
        _ = Uncontrolled().scenario
    with pytest.raises(RuntimeError):
        _ = OptimalSchedule().solution


def test_phase_aware_heuristics_never_overload_a_line() -> None:
    # Three single-phase EVs on L1 and one three-phase EV, 20 A fuse per line.
    chargers = [Charger(f"C{i}", 7.4, phases=1, rotation="L1") for i in range(3)]
    chargers.append(Charger("C3", 11.0))
    sessions = [Session(f"S{i}", f"C{i}", 0, 4, 30.0, 7.4, phases=1) for i in range(3)]
    sessions.append(Session("S3", "C3", 0, 4, 30.0, 11.0))
    sc = make_scenario(sessions, chargers=chargers, supply=Supply.uniform(20.0))
    for policy in (Uncontrolled(), EqualShare(), EarliestDeadlineFirst(), LeastLaxityFirst()):
        res = simulate(sc, policy)
        assert res.violations == (), policy
        assert res.line_current_a is not None
        assert res.line_current_a.max() <= 20.0 + 1e-9, policy
    # FCFS: S0 (earliest id, a 32 A EV) takes all 20 A of L1. Nothing is left on L1,
    # so the other single-phase EVs wait and the three-phase EV cannot start either,
    # although L2 and L3 are idle: the imbalance a kW-only limit would not see.
    first = simulate(sc, Uncontrolled())
    assert first.setpoint is not None
    np.testing.assert_allclose(first.setpoint[:, 0], [20.0, 0.0, 0.0, 0.0])


def test_equal_share_is_max_min_fair_across_lines() -> None:
    chargers = [
        Charger("A", 7.4, phases=1, rotation="L1"),
        Charger("B", 7.4, phases=1, rotation="L1"),
        Charger("C", 7.4, phases=1, rotation="L2"),
    ]
    sessions = [Session(c.id, c.id, 0, 4, 40.0, 7.4, phases=1) for c in chargers]
    sc = make_scenario(sessions, chargers=chargers, supply=Supply.uniform(20.0))
    res = simulate(sc, EqualShare())
    assert res.setpoint is not None
    # A and B share L1 (10 A each); C is alone on L2 and gets its 32 A cap limited by 20 A
    np.testing.assert_allclose(res.setpoint[:, 0], [10.0, 10.0, 20.0])
    assert res.violations == ()


def test_price_aware_books_line_capacity() -> None:
    chargers = [Charger("A", 7.4, phases=1, rotation="L1"), Charger("B", 7.4, phases=1)]
    sessions = [
        Session("A", "A", 0, 4, 7.36, 7.4, phases=1, efficiency=1.0),
        Session("B", "B", 0, 2, 3.68, 7.4, phases=1, efficiency=1.0),
    ]
    sc = make_scenario(
        sessions,
        chargers=chargers,
        supply=Supply.uniform(16.0),
        prices=[0.3, 0.1, 0.2, 0.4],
    )
    res = simulate(sc, PriceAware())
    assert res.setpoint is not None
    # Both EVs are on L1. B (less laxity) books its cheaper step (1) at 16 A = 3.68 kWh;
    # A then takes the cheapest steps L1 still has free: 2 (0.2) and 0 (0.3)
    np.testing.assert_allclose(res.setpoint[1], [0.0, 16.0, 0.0, 0.0])
    np.testing.assert_allclose(res.setpoint[0], [16.0, 0.0, 16.0, 0.0])
    assert res.violations == ()


def test_optimal_and_mpc_respect_line_limits_in_amperes() -> None:
    # Two single-phase EVs on L1 behind a 20 A fuse: at most 20 A together, so
    # 2 x 16 A is impossible although 7.4 kW is far below any kW limit.
    chargers = [Charger("A", 7.4, phases=1, rotation="L1"), Charger("B", 7.4, phases=1)]
    sessions = [Session(c.id, c.id, 0, 4, 6.0, 7.4, phases=1, efficiency=1.0) for c in chargers]
    sc = make_scenario(sessions, chargers=chargers, supply=Supply.uniform(20.0))
    for policy in (OptimalSchedule(), ModelPredictiveControl()):
        res = simulate(sc, policy)
        assert res.violations == (), policy
        assert res.line_current_a is not None
        assert res.line_current_a[:, 0].max() <= 20.0 + 1e-9
        m = compute_metrics(res)
        # 12 kWh need 52.2 A-hours at 230 V; L1 offers 4 x 20 A-hours: feasible
        assert m.unmet_kwh == pytest.approx(0.0, abs=1e-6), policy


def test_optimal_replay_with_whole_amps_delivers_everything() -> None:
    # 1 A resolution: the continuous plan (e.g. 7.43 A) is rounded down each step,
    # the shortfall is carried forward and the finishing step rounds up.
    chargers = [Charger(f"C{i}", 11.0, current_step_a=1.0) for i in range(3)]
    sessions = [
        Session(f"S{i}", f"C{i}", 0, 8, 7.0 + i * 1.37, 11.0, efficiency=0.9) for i in range(3)
    ]
    sc = make_scenario(
        sessions,
        n_steps=8,
        step_minutes=15,
        chargers=chargers,
        supply=Supply.uniform(25.0),
        prices=[0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.1, 0.4],
        demand_charge=0.5,
    )
    res = simulate(sc, OptimalSchedule())
    assert res.violations == ()
    assert res.setpoint is not None
    assert np.allclose(res.setpoint, np.round(res.setpoint))
    assert compute_metrics(res).unmet_kwh == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_kw_model_is_the_phase_model_with_idle_lines(seed: int) -> None:
    # Three-phase EVs, continuous currents and line limits that never bind: the
    # phase-aware site must reproduce the aggregate kW model schedule by schedule.
    rng = np.random.default_rng(seed)
    n = 6
    arrivals = rng.integers(0, 6, n)
    sessions_a = []
    sessions_kw = []
    for i in range(n):
        arr = int(arrivals[i])
        dep = int(min(16, arr + rng.integers(4, 11)))
        energy = float(rng.uniform(2.0, 14.0))
        sessions_a.append(Session(f"S{i}", f"C{i}", arr, dep, energy, 22.0, max_current_a=16.0))
        sessions_kw.append(Session(f"S{i}", f"C{i}", arr, dep, energy, 16 * 0.69))
    kwargs = {
        "n_steps": 16,
        "step_minutes": 15,
        "grid_limit_kw": 20.0,
        "prices": list(rng.uniform(0.05, 0.3, 16)),
        "demand_charge": 0.4,
        "base_load": list(rng.uniform(0.0, 6.0, 16)),
    }
    phase = make_scenario(
        sessions_a,
        chargers=[Charger(f"C{i}", 22.0, current_step_a=0.0) for i in range(n)],
        supply=Supply.uniform(1000.0),
        **kwargs,  # type: ignore[arg-type]
    )
    aggregate = make_scenario(
        sessions_kw,
        chargers=[Charger(f"C{i}", 22.0, 4.14) for i in range(n)],
        **kwargs,  # type: ignore[arg-type]
    )
    for name in ("uncontrolled", "equal-share", "edf", "llf", "price-aware"):
        a = simulate(phase, make_policy(name))
        b = simulate(aggregate, make_policy(name))
        assert a.violations == b.violations == ()
        np.testing.assert_allclose(a.power_kw, b.power_kw, atol=1e-9, err_msg=name)
    # The LPs are degenerate (ties between steps), so the solver may pick another
    # optimal vertex; compare what is invariant: the bound and the MILP optimum.
    assert relaxation_bound(phase) == pytest.approx(relaxation_bound(aggregate), rel=1e-7)
    exact_phase, exact_kw = OptimalSchedule(strategy="exact"), OptimalSchedule(strategy="exact")
    simulate(phase, exact_phase)
    simulate(aggregate, exact_kw)
    assert exact_phase.solution.objective == pytest.approx(exact_kw.solution.objective, rel=1e-5)
    assert simulate(phase, ModelPredictiveControl()).violations == ()


def amp_state(
    sid: str, remaining_kwh: float, steps_left: int, *, step: float = 1.0, k: float = 0.23
) -> SessionState:
    return SessionState(
        session=session(sid, "C1", 0, steps_left, max(remaining_kwh, 1e-3), max_kw=7.4),
        delivered_kwh=0.0,
        p_min_kw=6 * k,
        p_max_kw=32 * k,
        steps_left=steps_left,
        dt_h=1.0,
        control_spec=Control("A", k, 6.0, 32.0, step),
    )


def amp_obs(states: list[SessionState], line_a: float) -> Observation:
    rows = StepConstraints(
        ("L1 import",), ("line",), np.array([line_a]), {st.id: np.ones(1) for st in states}
    )
    return Observation(0, tuple(states), 100.0, 0.0, rows)


def test_finalize_rounds_to_grid_and_respects_rows() -> None:
    a, b = amp_state("A", 10.0, 4), amp_state("B", 10.0, 4)
    # nearest: 7.6 -> 8 fits; 7.2 -> 7 (remainder below one half)
    assert finalize(amp_obs([a, b], 30.0), {"A": 7.6, "B": 7.2}) == {"A": 8.0, "B": 7.0}
    # rows win over rounding: only 15 A are available
    assert finalize(amp_obs([a, b], 15.0), {"A": 7.6, "B": 7.4}) == {"A": 8.0, "B": 7.0}
    # below the 6 A minimum after rounding: paused
    assert finalize(amp_obs([a], 30.0), {"A": 5.4}) == {}


def test_finalize_gives_the_last_step_priority() -> None:
    # 49 A on the line; three sessions whose continuous setpoints sum to 48.45 A all
    # finish now, but only the last-step one cannot finish later.
    early = amp_state("A", 20.768 * 0.23, 2)
    other = amp_state("B", 12.154 * 0.23, 7)
    last = amp_state("Z", 15.531 * 0.23, 1)
    out = finalize(amp_obs([early, other, last], 49.0), {"A": 20.768, "B": 12.154, "Z": 15.531})
    assert out["Z"] == 16.0  # rounded up: it finishes in its last step
    assert sum(out.values()) <= 49.0 + 1e-9


def test_finalize_makes_room_for_an_urgent_round_up() -> None:
    # Z must round 15.5 -> 16 in its last step; the line is full after flooring,
    # so the flexible session A gives up one ampere (it has later steps).
    flexible = amp_state("A", 30.0, 6)
    last = amp_state("Z", 15.5 * 0.23, 1)
    out = finalize(amp_obs([flexible, last], 30.0), {"A": 15.0, "Z": 15.5}, room_later={"A": 60.0})
    assert out == {"A": 14.0, "Z": 16.0}


def test_finalize_is_the_identity_on_continuous_kw_setpoints() -> None:
    st = state("A", 40.0, p_min=4.14)
    obs = Observation(0, (st,), 20.0, 0.0)
    assert finalize(obs, {"A": 7.123456}) == {"A": 7.123456}
    # solver noise just under the minimum is lifted to it; far below pauses
    assert finalize(obs, {"A": 4.14 - 1e-8}) == {"A": 4.14}
    assert finalize(obs, {"A": 3.0}) == {}


def test_phase_blind_controller_overloads_a_shared_line() -> None:
    # Six single-phase 32 A EVs, all on L1 (no rotation), behind 3 x 63 A: the kW
    # limit (fuse equivalent, 43.5 kW) looks fine, one line does not.
    chargers = [Charger(f"C{i}", 7.4, phases=1, rotation="L1") for i in range(6)]
    sessions = [Session(f"S{i}", f"C{i}", 0, 4, 20.0, 7.4, phases=1) for i in range(6)]
    supply = Supply.uniform(63.0)
    sc = make_scenario(
        sessions, chargers=chargers, supply=supply, grid_limit_kw=supply.fuse_equivalent_kw
    )
    kw_view = sc.aggregate()
    assert kw_view.site.supply is None
    assert [kw_view.power_bounds(s) for s in kw_view.sessions] == [
        sc.power_bounds(s) for s in sc.sessions
    ]
    # what the kW-only plan does to the lines, before any protection acts
    plan = simulate(kw_view, LeastLaxityFirst())
    ctl = sc.control(sc.sessions[0])
    currents = sc.line_currents_a(plan.power_kw / ctl.kw_per_unit)
    # the kW limit is 3 x 63 A x 230 V, and all of it lands on L1: three times its fuse
    assert currents[0, 0] == pytest.approx(3 * 63.0)
    # enforced on the phase-aware site: the simulator cuts L1 back to 63 A
    blind = simulate(sc, PhaseBlind(LeastLaxityFirst()))
    assert blind.policy_name == "llf (kW only)"
    assert {v.kind for v in blind.violations} == {ViolationKind.LINE_LIMIT}
    assert blind.line_current_a is not None
    assert blind.line_current_a.max() <= 63.0 + 1e-9
    aware = simulate(sc, LeastLaxityFirst())
    assert aware.violations == ()
    assert repr(PhaseBlind(LeastLaxityFirst())) == "PhaseBlind(LeastLaxityFirst())"
