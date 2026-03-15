from __future__ import annotations

import math

from evcharge.experiment import compare, describe_scenario, format_table
from evcharge.policies import EarliestDeadlineFirst, OptimalSchedule

from .helpers import make_scenario, session


def test_compare_and_table() -> None:
    sc = make_scenario([session("A", "C1", 0, 4, 15.0)], prices=[0.3, 0.1, 0.2, 0.4])
    cmp = compare(sc, [EarliestDeadlineFirst(), OptimalSchedule()])
    assert [m.policy for m in cmp.metrics] == ["edf", "optimal"]
    assert cmp.lower_bound_eur == cmp.metrics[1].penalised_cost_eur
    assert math.isclose(cmp.gap_pct(cmp.metrics[1]), 0.0, abs_tol=1e-6)
    # EDF charges at once: 11 kWh at 0.3 + 4 kWh at 0.1 = 3.7 EUR vs 1.9 EUR
    assert math.isclose(cmp.gap_pct(cmp.metrics[0]), 100 * (3.7 - 1.9) / 1.9)
    table = format_table(cmp).splitlines()
    assert table[0].split()[:3] == ["policy", "delivered", "%"]
    assert set(table[1]) == {"-", " "}
    assert table[2].startswith("edf")
    assert table[-1].startswith("lower bound")
    assert "1 sessions on 1 chargers" in describe_scenario(sc)


def test_gap_is_nan_when_the_bound_is_zero() -> None:
    sc = make_scenario([session("A", "C1", 0, 4, 5.0)], prices=[0.0] * 4)
    cmp = compare(sc, [EarliestDeadlineFirst()])
    assert cmp.lower_bound_eur == 0.0
    assert math.isnan(cmp.gap_pct(cmp.metrics[0]))
    assert "nan" in format_table(cmp)
