from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from evcharge.model import Scenario
from evcharge.policies import OnlinePolicy, OptimalSchedule, Uncontrolled
from evcharge.policies.base import Observation
from evcharge.sim import ViolationKind, simulate

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
