from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from evcharge.model import RowModel
from evcharge.optim import (
    FlexLoad,
    LPSession,
    ScheduleProblem,
    _milp_bound,
    problem_from_scenario,
    relaxation_bound,
    solve_scenarios,
    solve_schedule,
)

from .helpers import make_scenario, session


def problem(sessions: tuple[LPSession, ...], **kwargs: object) -> ScheduleProblem:
    n = 4
    defaults: dict[str, object] = {
        "dt_h": 1.0,
        "price": np.array([0.3, 0.1, 0.2, 0.4]),
        "export_price": np.zeros(n),
        "net_base_kw": np.zeros(n),
        "grid_limit_kw": 20.0,
        "demand_charge": 0.0,
        "sessions": sessions,
    }
    defaults.update(kwargs)
    return ScheduleProblem(**defaults)  # type: ignore[arg-type]


def lp_session(energy: float, *, p_min: float = 0.0, start: int = 0, end: int = 4) -> LPSession:
    return LPSession("A", start, end, energy, 1.0, p_min, 11.0)


def test_lp_fills_cheapest_steps() -> None:
    sol = solve_schedule(problem((lp_session(15.0),)))
    np.testing.assert_allclose(sol.power_kw[0], [0, 11, 4, 0], atol=1e-8)
    assert sol.objective == pytest.approx(0.1 * 11 + 0.2 * 4)
    assert sol.lower_bound == pytest.approx(sol.objective)
    assert sol.gap == pytest.approx(0.0)
    assert sol.n_binaries == 0


def test_unmet_energy_is_penalised_not_infeasible() -> None:
    sol = solve_schedule(problem((lp_session(100.0),), unmet_penalty=10.0))
    np.testing.assert_allclose(sol.power_kw[0], [11, 11, 11, 11], atol=1e-8)
    assert sol.unmet_kwh[0] == pytest.approx(56.0)


def test_semicontinuous_minimum_power() -> None:
    # 6 kWh with p_min 4.14: the LP would put 6 kW in the cheapest step anyway,
    # but 2 x 3 kW is forbidden; make splitting attractive with a demand charge.
    relaxed = solve_schedule(
        problem((lp_session(6.0, p_min=4.14),), demand_charge=1.0, min_power_steps=0)
    )
    exact = solve_schedule(problem((lp_session(6.0, p_min=4.14),), demand_charge=1.0))
    assert relaxed.power_kw.max() < 4.14  # fractional in the relaxation
    p = exact.power_kw[0]
    assert np.all((p == 0) | (p >= 4.14 - 1e-9))
    assert exact.n_binaries == 4
    assert exact.objective >= relaxed.objective - 1e-9


def test_minimum_power_can_finish_small_requests() -> None:
    # 1 kWh is below one step at p_min: commanding p_min is allowed (overshoot).
    sol = solve_schedule(problem((lp_session(1.0, p_min=4.14),)))
    assert sol.unmet_kwh[0] == pytest.approx(0.0, abs=1e-9)
    assert sol.power_kw[0, 1] == pytest.approx(4.14)


def test_relax_and_fix_matches_or_bounds_exact() -> None:
    sessions = tuple(
        LPSession(f"S{i}", 0, 4, e, 1.0, 4.14, 11.0) for i, e in enumerate([6.0, 9.0, 13.0])
    )
    kw = {"demand_charge": 0.5, "grid_limit_kw": 15.0}
    exact = solve_schedule(problem(sessions, **kw), strategy="exact")
    fixed = solve_schedule(problem(sessions, **kw), strategy="relax-and-fix")
    assert exact.objective <= fixed.objective + 1e-7
    assert fixed.lower_bound <= exact.objective + 1e-7
    assert fixed.n_binaries <= exact.n_binaries


def test_peak_floor_and_export() -> None:
    sol = solve_schedule(
        problem(
            (lp_session(8.0),),
            net_base_kw=np.array([0.0, -5.0, 0.0, 0.0]),
            export_price=np.array([0.0, 0.05, 0.0, 0.0]),
            demand_charge=1.0,
            peak_floor_kw=7.0,
        )
    )
    assert sol.peak_kw == pytest.approx(7.0)
    # the PV surplus in step 1 (worth only 0.05) is used first
    assert sol.power_kw[0, 1] >= 5.0 - 1e-8
    assert sol.export_kw[1] == pytest.approx(0.0, abs=1e-8)


def test_window_validation() -> None:
    with pytest.raises(ValueError, match="outside"):
        solve_schedule(problem((lp_session(1.0, start=2, end=6),)))


def test_scenario_helpers() -> None:
    sc = make_scenario([session("A", "C1", 1, 3, 5.0)], prices=[0.3, 0.1, 0.2, 0.4])
    prob = problem_from_scenario(sc)
    assert prob.sessions[0].start == 1
    assert prob.sessions[0].end == 3
    assert relaxation_bound(sc) == pytest.approx(0.5)


def test_denormal_coefficients_do_not_stall_the_solver() -> None:
    # Regression: found by hypothesis. With a price of 2.2e-308 HiGHS' MILP never
    # returned; such coefficients are now snapped to zero before solving.
    prob = ScheduleProblem(
        dt_h=1.0,
        price=np.array([2.2250738585072014e-308]),
        export_price=np.array([-0.03888369]),
        net_base_kw=np.array([-3.24667145]),
        grid_limit_kw=4.3521814144702144,
        demand_charge=1e-9,
        sessions=(
            LPSession("S0", 0, 1, 52.015842698016804, 0.9447376711126765, 1.38, 2.5452732418727146),
            LPSession("S1", 0, 1, 43.58103384472968, 0.99999, 1.38, 7.418988493042832),
        ),
        peak_floor_kw=4.107320700251546,
        min_power_steps=1,
        quick_charge_weight=0.002,
    )
    sol = solve_schedule(prob)
    assert sol.status == "optimal"
    assert sol.power_kw.sum() == pytest.approx(4.3521814144702144 + 3.24667145)


def test_a_zero_dual_bound_is_kept() -> None:
    # Regression: ``mip_dual_bound or fun`` treated a proven bound of exactly 0.0 as
    # missing and reported the incumbent objective as the bound (gap 0 %).
    assert _milp_bound(SimpleNamespace(mip_dual_bound=0.0, fun=5.0)) == 0.0
    assert _milp_bound(SimpleNamespace(mip_dual_bound=-2.5, fun=5.0)) == -2.5
    assert _milp_bound(SimpleNamespace(mip_dual_bound=None, fun=5.0)) == 5.0
    assert _milp_bound(SimpleNamespace(mip_dual_bound=float("nan"), fun=5.0)) == 5.0
    assert _milp_bound(SimpleNamespace(fun=5.0)) == 5.0


def test_flex_loads_in_amperes_share_a_line_row() -> None:
    # Two single-phase EVs (0.23 kW/A) on one line of 20 A, plus a 10 kW site row.
    rows = RowModel(("L1 import", "site import"), ("line", "site"), np.tile([20.0, 10.0], (4, 1)))
    loads = (
        FlexLoad("A", 0, 4, 4.6, charge_max=16.0, kw_per_unit=0.23, rows=(1.0, 0.23)),
        FlexLoad("B", 0, 4, 4.6, charge_max=16.0, kw_per_unit=0.23, rows=(1.0, 0.23)),
    )
    sol = solve_schedule(problem(loads, rows=rows, price=np.array([0.1, 0.2, 0.3, 0.4])))
    # each needs 20 A-steps at 1 h; the cheapest step holds only 20 A in total
    assert sol.setpoint[:, 0].sum() == pytest.approx(20.0)
    assert sol.setpoint.sum() == pytest.approx(40.0)
    np.testing.assert_allclose(sol.power_kw, sol.setpoint * 0.23)
    assert sol.unmet_kwh.sum() == pytest.approx(0.0, abs=1e-9)
    assert (sol.setpoint.sum(axis=0) <= 20.0 + 1e-9).all()


def test_row_inputs_are_validated() -> None:
    rows = RowModel(("a", "b"), ("line", "site"), np.ones((4, 2)))
    with pytest.raises(ValueError, match="row coefficients are required"):
        solve_schedule(problem((lp_session(1.0),), rows=rows))
    bad = FlexLoad("A", 0, 4, 1.0, charge_max=5.0, rows=(1.0,))
    with pytest.raises(ValueError, match="1 row coefficients for 2 rows"):
        solve_schedule(problem((bad,), rows=rows))
    short = RowModel(("a",), ("site",), np.ones((3, 1)))
    with pytest.raises(ValueError, match="cover 3 steps"):
        solve_schedule(problem((lp_session(1.0),), rows=short))


KNOWN = FlexLoad("A", 0, 2, 10.0, 10.0)


def two_step(*future: FlexLoad) -> ScheduleProblem:
    """A known EV needing 10 kWh in two 1 h steps behind 10 kW; step 1 is cheaper."""
    return ScheduleProblem(
        dt_h=1.0,
        price=np.array([0.2, 0.1]),
        export_price=np.zeros(2),
        net_base_kw=np.zeros(2),
        grid_limit_kw=10.0,
        demand_charge=0.0,
        sessions=(KNOWN, *future),
        min_power_steps=1,
    )


def test_one_scenario_is_the_plain_problem() -> None:
    p = two_step(FlexLoad("B", 1, 2, 4.0, 10.0))
    single = solve_schedule(p)
    saa = solve_scenarios([p], n_shared=1)
    np.testing.assert_allclose(saa.setpoint, single.setpoint, atol=1e-9)
    assert saa.objective == pytest.approx(single.objective)


def test_non_anticipativity_hedges_against_a_likely_arrival() -> None:
    # A second EV arrives at step 1 with probability 1/2 and needs all 10 kW then.
    arrival = FlexLoad("B", 1, 2, 10.0, 10.0)
    # Without it the cheap step wins: charge A at step 1.
    assert solve_schedule(two_step()).setpoint[0, 0] == pytest.approx(0.0)
    # Sample average over "arrives" and "does not": 5 kWh of A left for step 1
    # would be unmet with probability 1/2 at 100 EUR/kWh, so all 10 kWh go now.
    saa = solve_scenarios([two_step(arrival), two_step()], n_shared=1)
    assert saa.setpoint[0, 0] == pytest.approx(10.0)
    assert saa.objective == pytest.approx(0.5 * (2.0 + 1.0) + 0.5 * 2.0)
    # The certainty equivalent (half an EV) only moves half of A forward.
    half = FlexLoad("B", 1, 2, 5.0, 5.0)
    assert solve_schedule(two_step(half)).setpoint[0, 0] == pytest.approx(5.0)
    # Weights are scenario probabilities: an unlikely arrival is not worth 0.1 EUR/kWh.
    rare = solve_scenarios([two_step(arrival), two_step()], n_shared=1, weights=[1e-4, 1.0])
    assert rare.setpoint[0, 0] == pytest.approx(0.0)


def test_solve_scenarios_validates_its_input() -> None:
    other = replace(two_step(), sessions=(replace(KNOWN, energy_kwh=5.0),))
    with pytest.raises(ValueError, match="same 1 sessions"):
        solve_scenarios([two_step(), other], n_shared=1)
    with pytest.raises(ValueError, match="weights"):
        solve_scenarios([two_step()], n_shared=1, weights=[0.5, 0.5])
    with pytest.raises(ValueError, match="at least one"):
        solve_scenarios([], n_shared=0)
