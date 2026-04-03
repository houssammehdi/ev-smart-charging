"""Bidirectional charging (V2G): model, simulator and metrics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from evcharge.electrical import Supply
from evcharge.metrics import compute_metrics
from evcharge.model import V2G, Charger, Scenario, Session, ValidationError
from evcharge.policies import LeastLaxityFirst, OnlinePolicy
from evcharge.policies.base import Observation
from evcharge.sim import ViolationKind, simulate

from .helpers import make_scenario

PRICES = [0.1, 0.5, 0.1, 0.5]


def arbitrage(degradation: float = 0.0, *, bidirectional: bool = True) -> Scenario:
    """One EV, 10 kW both ways, battery 2..18 kWh, arrives and must leave with 10 kWh."""
    v2g = V2G(
        capacity_kwh=20.0,
        initial_kwh=10.0,
        min_kwh=2.0,
        max_kwh=18.0,
        discharge_efficiency=1.0,
        degradation_eur_per_kwh=degradation,
    )
    return make_scenario(
        [Session("A", "C1", 0, 4, 0.0, 10.0, efficiency=1.0, v2g=v2g)],
        chargers=[Charger("C1", 10.0, 0.0, bidirectional=bidirectional)],
        prices=PRICES,
        export_prices=PRICES,
        grid_limit_kw=20.0,
    )


class Scripted(OnlinePolicy):
    name = "scripted"

    def __init__(self, commands: Mapping[int, Mapping[str, float]]) -> None:
        super().__init__()
        self.commands = commands

    def decide(self, obs: Observation) -> Mapping[str, float]:
        return self.commands.get(obs.step, {})


CYCLE = {0: {"A": 10.0}, 1: {"A": -10.0}, 2: {"A": 10.0}, 3: {"A": -8.0}}


def test_battery_follows_the_commands_and_metrics_count_throughput() -> None:
    # charging stops at the 18 kWh ceiling (8 kWh drawn), then 10 out, 10 in, 8 out
    res = simulate(arbitrage(0.1), Scripted(CYCLE))
    assert res.violations == ()
    np.testing.assert_allclose(res.power_kw[0], [8.0, -10.0, 10.0, -8.0])
    assert res.battery_kwh is not None
    np.testing.assert_allclose(res.battery_kwh[0], [18.0, 8.0, 18.0, 10.0])
    m = compute_metrics(res)
    assert m.energy_cost_eur == pytest.approx(-7.2)
    assert m.discharged_kwh == pytest.approx(18.0)
    # 0.1 EUR per kWh of throughput, 18 kWh in and 18 kWh out
    assert m.degradation_cost_eur == pytest.approx(3.6)
    assert m.total_cost_eur == pytest.approx(-7.2 + 3.6)
    assert m.unmet_kwh == pytest.approx(0.0, abs=1e-9)
    assert m.delivered_pct == 100.0


def test_charger_without_bidirectionality_refuses_discharge() -> None:
    res = simulate(arbitrage(bidirectional=False), Scripted(CYCLE))
    assert [(v.step, v.kind) for v in res.violations] == [
        (1, ViolationKind.INVALID),
        (3, ViolationKind.INVALID),
    ]
    assert compute_metrics(res).discharged_kwh == 0.0


def test_unidirectional_view_and_heuristics_never_discharge() -> None:
    uni = arbitrage().unidirectional()
    assert not uni.site.bidirectional
    assert not uni.control(uni.sessions[0]).can_discharge
    assert uni.sessions[0].v2g is not None
    # heuristics are unidirectional by design on bidirectional sites too
    assert compute_metrics(simulate(arbitrage(), LeastLaxityFirst())).discharged_kwh == 0.0


def test_simulator_enforces_battery_limits() -> None:
    v2g = V2G(capacity_kwh=20.0, initial_kwh=10.0, min_kwh=8.0, max_kwh=17.0)
    sessions = [
        Session("A", "C1", 0, 4, 0.0, 10.0, efficiency=1.0, v2g=v2g),
        Session("B", "C2", 0, 4, 5.0, 10.0),
    ]
    sc = make_scenario(
        sessions,
        chargers=[Charger("C1", 10.0, 0.0, bidirectional=True), Charger("C2", 10.0, 0.0)],
    )
    res = simulate(sc, Scripted({0: {"A": -10.0, "B": -3.0}, 1: {"A": 10.0}, 2: {"A": 10.0}}))
    kinds = [(v.step, v.kind, v.session_id) for v in res.violations]
    # A holds 2 kWh above its floor: at 0.9 discharge efficiency it may send 1.8 kWh;
    # B's charger is not bidirectional
    assert kinds == [(0, ViolationKind.SOC_LIMIT, "A"), (0, ViolationKind.INVALID, "B")]
    assert res.violations[0].applied == pytest.approx(-1.8)
    assert res.battery_kwh is not None
    np.testing.assert_allclose(res.battery_kwh[0, :3], [8.0, 17.0, 17.0])
    # charging stops by itself at the 17 kWh ceiling (no violation)
    np.testing.assert_allclose(res.power_kw[0, :3], [-1.8, 9.0, 0.0])


def test_export_limit_caps_discharge() -> None:
    v2g = V2G(capacity_kwh=40.0, initial_kwh=20.0, min_kwh=0.0, discharge_efficiency=1.0)
    sc = make_scenario(
        [Session("A", "C1", 0, 4, 0.0, 10.0, efficiency=1.0, v2g=v2g)],
        chargers=[Charger("C1", 10.0, 0.0, bidirectional=True)],
        base_load=[1.0] * 4,
    )
    sc = sc.with_site(type(sc.site)(sc.site.grid_limit_kw, sc.site.chargers, None, 3.0))
    assert sc.rows.names == ("site import", "site export")
    res = simulate(sc, Scripted({0: {"A": -10.0}}))
    assert [v.kind for v in res.violations] == [ViolationKind.SITE_LIMIT]
    # export is capped at 3 kW: the EV may cover the 1 kW base load plus 3 kW export
    assert res.net_import_kw[0] == pytest.approx(-3.0)


def line_scenario(grid: str) -> Scenario:
    """18 A of base load on L1 behind 20 A; A charges and B discharges on that line."""
    rotation = "L1" if grid == "TN" else "L1L2"
    chargers = [
        Charger("A", 7.4, phases=1, rotation=rotation),
        Charger("B", 7.4, phases=1, rotation=rotation, bidirectional=True),
    ]
    v2g = V2G(capacity_kwh=40.0, initial_kwh=20.0)
    sessions = [
        Session("A", "A", 0, 2, 20.0, 7.4, phases=1),
        Session("B", "B", 0, 2, 0.0, 7.4, phases=1, v2g=v2g),
    ]
    base = np.array([[18.0, 0.0, 0.0], [18.0, 0.0, 0.0]])
    return make_scenario(
        sessions,
        n_steps=2,
        chargers=chargers,
        supply=Supply.uniform(20.0, grid=grid),
        base_current_a=base,
    )


@pytest.mark.parametrize(("grid", "cut"), [("TN", False), ("IT", True)])
def test_discharge_offsets_line_current_on_tn_only(grid: str, cut: bool) -> None:
    # B discharges 10 A on the line A charges on: on TN (collinear currents)
    # that frees 10 A for A; on IT it gives no credit.
    res = simulate(line_scenario(grid), Scripted({0: {"A": 12.0, "B": -10.0}}))
    assert res.line_current_a is not None
    if cut:
        assert [v.kind for v in res.violations] == [ViolationKind.LINE_LIMIT]
    else:
        assert res.violations == ()
        assert res.line_current_a[0, 0] == pytest.approx(20.0)


def test_v2g_validation() -> None:
    with pytest.raises(ValidationError, match="min_kwh <= initial_kwh"):
        V2G(capacity_kwh=40.0, initial_kwh=5.0, min_kwh=8.0)
    with pytest.raises(ValidationError, match="discharge_efficiency"):
        V2G(capacity_kwh=40.0, initial_kwh=5.0, discharge_efficiency=1.5)
    v2g = V2G(capacity_kwh=40.0, initial_kwh=30.0, max_kwh=36.0)
    with pytest.raises(ValidationError, match="exceeds max_kwh"):
        Session("A", "C1", 0, 4, 10.0, 10.0, v2g=v2g)
    # a V2G session may request nothing; a charge-only one may not
    Session("A", "C1", 0, 4, 0.0, 10.0, v2g=v2g)
    with pytest.raises(ValidationError, match="energy_kwh must be > 0"):
        Session("A", "C1", 0, 4, 0.0, 10.0)
    assert v2g.ceiling_kwh == 36.0
    assert Session("A", "C1", 0, 4, 2.0, 10.0, v2g=v2g).target_kwh == 32.0
