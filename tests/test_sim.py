from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest

from evcharge import scenarios
from evcharge.electrical import Supply
from evcharge.model import Charger, Scenario, Session
from evcharge.policies import OnlinePolicy, OptimalSchedule, Uncontrolled, make_policy
from evcharge.policies.base import Observation
from evcharge.sim import ViolationKind, first_step, simulate

from .helpers import make_scenario, session


class Scripted(OnlinePolicy):
    """Replays fixed commands and records what it observed."""

    name = "scripted"

    def __init__(self, commands: Mapping[int, Mapping[str, float]]) -> None:
        super().__init__()
        self.commands = commands
        self.observations: list[Observation] = []

    def decide(self, obs: Observation) -> Mapping[str, float]:
        self.observations.append(obs)
        return self.commands.get(obs.step, {})


def test_uncontrolled_hand_computed() -> None:
    sc = make_scenario([session("A", "C1", 0, 4, 20.0)])
    res = simulate(sc, Uncontrolled())
    np.testing.assert_allclose(res.power_kw[0], [11.0, 9.0, 0.0, 0.0])
    np.testing.assert_allclose(res.delivered_kwh, [20.0])
    np.testing.assert_allclose(res.net_import_kw, [11.0, 9.0, 0.0, 0.0])
    assert res.violations == ()
    assert res.policy_name == "uncontrolled"


def test_efficiency_losses_are_drawn_from_the_grid() -> None:
    sc = make_scenario([session("A", "C1", 0, 4, 20.0, efficiency=0.9)])
    res = simulate(sc, Uncontrolled())
    np.testing.assert_allclose(res.power_kw[0], [11.0, 11.0, 0.2 / 0.9, 0.0])
    assert res.delivered_kwh[0] == pytest.approx(20.0)
    assert res.power_kw.sum() * 0.9 == pytest.approx(20.0)


def test_ev_stops_drawing_when_full() -> None:
    sc = make_scenario([session("A", "C1", 0, 4, 2.0)], charger_min_kw=4.14)
    res = simulate(sc, Scripted({0: {"A": 11.0}, 1: {"A": 11.0}}))
    assert res.setpoint_kw[0, 0] == 11.0
    assert res.power_kw[0, 0] == pytest.approx(2.0)
    assert res.power_kw[0, 1] == 0.0
    assert res.delivered_kwh[0] == pytest.approx(2.0)
    assert res.violations == ()


def test_online_reveal_and_observation_contents() -> None:
    sc = make_scenario(
        [session("A", "C1", 0, 2, 10.0), session("B", "C2", 2, 4, 5.0)],
        base_load=[3, 3, 3, 3],
        grid_limit_kw=30.0,
    )
    policy = Scripted({0: {"A": 11.0}, 2: {"B": 5.0}})
    simulate(sc, policy)
    ids = [[s.id for s in o.sessions] for o in policy.observations]
    assert ids == [["A"], ["A"], ["B"], ["B"]]
    first = policy.observations[0].sessions[0]
    assert first.steps_left == 2
    assert first.remaining_kwh == 10.0
    assert policy.observations[0].headroom_kw == 27.0
    # peak so far: 0 before any step, then 3 + 10 = 13 kW from step 0 on
    assert [o.peak_import_kw for o in policy.observations] == [0.0, 13.0, 13.0, 13.0]
    assert policy.observations[1].sessions[0].is_satisfied


def test_online_policies_do_not_see_future_sessions() -> None:
    seen: dict[str, int] = {}

    class Probe(Scripted):
        def reset(self, scenario: Scenario) -> None:
            super().reset(scenario)
            seen[self.name] = len(scenario.sessions)

    sc = make_scenario([session("A", "C1", 1, 3, 5.0)])
    simulate(sc, Probe({}))
    assert seen["scripted"] == 0
    opt = OptimalSchedule()
    simulate(sc, opt)  # clairvoyant: planned the session before it arrived
    assert opt.solution.power_kw.shape == (1, 4)


def test_violations_are_corrected_and_recorded() -> None:
    sc = make_scenario(
        [session("A", "C1", 0, 4, 40.0), session("B", "C2", 1, 4, 40.0)],
        charger_min_kw=4.14,
    )
    policy = Scripted(
        {
            0: {"A": -1.0, "B": 5.0, "ghost": 3.0},
            1: {"A": 15.0, "B": 2.0},
            2: {"A": float("nan"), "B": 0.0},
        }
    )
    res = simulate(sc, policy)
    kinds = [(v.step, v.kind, v.session_id) for v in res.violations]
    assert kinds == [
        (0, ViolationKind.INVALID, "A"),
        (0, ViolationKind.NOT_CONNECTED, "B"),
        (0, ViolationKind.NOT_CONNECTED, "ghost"),
        (1, ViolationKind.ABOVE_MAX, "A"),
        (1, ViolationKind.BELOW_MIN, "B"),
        (2, ViolationKind.INVALID, "A"),
    ]
    np.testing.assert_allclose(res.power_kw[:, :3], [[0.0, 11.0, 0.0], [0.0, 0.0, 0.0]])


def test_site_limit_scales_down_and_pauses_below_minimum() -> None:
    sessions = [session("A", "C1", 0, 2, 40.0), session("B", "C2", 0, 2, 40.0)]
    sc = make_scenario(sessions, n_steps=2, grid_limit_kw=10.0, charger_min_kw=4.14)
    res = simulate(sc, Scripted({0: {"A": 11.0, "B": 11.0}}))
    np.testing.assert_allclose(res.power_kw[:, 0], [5.0, 5.0])
    assert res.violations[0].kind is ViolationKind.SITE_LIMIT
    assert res.violations[0].requested_kw == pytest.approx(22.0)

    sc = make_scenario(sessions, n_steps=2, grid_limit_kw=6.0, charger_min_kw=4.14)
    res = simulate(sc, Scripted({0: {"A": 11.0, "B": 11.0}}))
    np.testing.assert_allclose(res.power_kw[:, 0], [0.0, 0.0])
    assert len(res.violations) == 1


def test_export_and_import_split() -> None:
    sc = make_scenario([session("A", "C1", 0, 1, 2.0)], base_load=[1, 1, 1, 1], pv=[0, 5, 0, 0])
    res = simulate(sc, Uncontrolled())
    np.testing.assert_allclose(res.net_import_kw, [3.0, -4.0, 1.0, 1.0])
    np.testing.assert_allclose(res.import_kw, [3.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(res.export_kw, [0.0, 4.0, 0.0, 0.0])
    np.testing.assert_allclose(res.ev_power_kw, [2.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(res.unmet_kwh, [0.0])


def one_phase_site(limit_a: float = 20.0, step_a: float = 0.1) -> dict[str, object]:
    chargers = [
        Charger("C1", 7.4, phases=1, rotation="L1", current_step_a=step_a),
        Charger("C2", 7.4, phases=1, rotation="L1", current_step_a=step_a),
        Charger("C3", 22.0, current_step_a=step_a),
    ]
    return {"chargers": chargers, "supply": Supply.uniform(limit_a), "grid_limit_kw": 100.0}


def test_line_limit_is_enforced_in_amperes() -> None:
    sessions = [
        Session("A", "C1", 0, 2, 20.0, 7.4, phases=1),
        Session("B", "C2", 0, 2, 20.0, 7.4, phases=1),
        Session("C", "C3", 0, 2, 20.0, 11.0),
    ]
    sc = make_scenario(sessions, n_steps=2, **one_phase_site())  # type: ignore[arg-type]
    res = simulate(sc, Scripted({0: {"A": 16.0, "B": 16.0, "C": 10.0}, 1: {"A": 12.34}}))
    kinds = [(v.step, v.kind, v.unit) for v in res.violations]
    assert kinds == [
        (0, ViolationKind.LINE_LIMIT, "A"),
        (1, ViolationKind.RESOLUTION, "A"),
    ]
    over = res.violations[0]
    # L1 carries A + B + C = 42 A against 20 A: A and B shrink, C (on all lines) too
    assert over.requested == pytest.approx(42.0)
    assert over.applied == pytest.approx(20.0)
    assert math.isnan(over.requested_kw)
    assert res.line_current_a is not None
    assert res.line_current_a[0, 0] <= 20.0 + 1e-9
    assert res.setpoint is not None
    np.testing.assert_allclose(res.setpoint[:, 1], [12.3, 0.0, 0.0])
    np.testing.assert_allclose(res.setpoint_kw[0, 1], 12.3 * 0.23)
    # every applied setpoint is on the 0.1 A grid
    assert np.allclose(res.setpoint * 10, np.round(res.setpoint * 10))


def test_phase_aware_physics_and_line_currents() -> None:
    sessions = [Session("A", "C1", 0, 2, 1.0, 7.4, phases=1, efficiency=1.0)]
    sc = make_scenario(sessions, n_steps=2, **one_phase_site())  # type: ignore[arg-type]
    res = simulate(sc, Scripted({0: {"A": 16.0}, 1: {"A": 16.0}}))
    # 16 A x 230 V = 3.68 kW for up to an hour; the EV stops after 1 kWh, and the
    # command of step 1 is recorded although the full EV draws nothing
    np.testing.assert_allclose(res.setpoint_kw[0], [3.68, 3.68])
    np.testing.assert_allclose(res.power_kw[0], [1.0, 0.0])
    assert res.line_current_a is not None
    np.testing.assert_allclose(res.line_current_a, [[16.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert res.violations == ()


@pytest.mark.parametrize("name", ["llf", "equal-share", "mpc", "optimal"])
def test_first_step_is_the_first_step_of_a_simulation(name: str) -> None:
    sc = scenarios.generate("residential", n_sessions=8, seed=3, grid="TN")
    sc = sc.with_sessions(
        tuple(replace(s, arrival_step=0) for s in sc.sessions if s.arrival_step < 30)
    )
    cmd, violations = first_step(sc, make_policy(name))
    res = simulate(sc, make_policy(name))
    assert violations == ()
    assert res.setpoint is not None
    expected = {s.id: float(res.setpoint[i, 0]) for i, s in enumerate(sc.sessions)}
    assert {sid: v for sid, v in expected.items() if v} == pytest.approx(cmd)
