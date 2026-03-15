from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evcharge.io import load_scenario, parse_series, scenario_from_dict
from evcharge.model import Horizon, ValidationError

from .helpers import START

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "office.json"


def base_doc() -> dict[str, Any]:
    return {
        "horizon": {"start": "2026-04-15T00:00:00", "hours": 2, "step_minutes": 30},
        "site": {"grid_limit_kw": 20, "chargers": [{"id": "C1", "max_power_kw": 11}]},
        "tariff": {"price_eur_per_kwh": 0.1},
        "sessions": [
            {
                "id": "S1",
                "charger": "C1",
                "arrival": "2026-04-15T00:10:00",
                "departure": "2026-04-15T01:50:00",
                "energy_kwh": 5,
                "max_power_kw": 11,
            }
        ],
    }


def test_example_file_loads() -> None:
    sc = load_scenario(EXAMPLE)
    assert sc.name == "office-example"
    assert sc.horizon.n_steps == 96
    assert len(sc.sessions) == 8
    assert sc.tariff.price_eur_per_kwh[0:4].tolist() == [0.092] * 4
    s3 = next(s for s in sc.sessions if s.id == "EV-03")
    assert sc.power_bounds(s3) == (1.38, 3.7)
    assert sc.pv.max() == pytest.approx(7.6)


def test_times_are_rounded_conservatively() -> None:
    sc = scenario_from_dict(base_doc())
    s = sc.sessions[0]
    assert (s.arrival_step, s.departure_step) == (1, 3)  # 00:10 -> 00:30, 01:50 -> 01:30
    assert s.efficiency == 0.9
    assert sc.site.chargers[0].min_power_kw == pytest.approx(4.14)
    assert sc.name == "custom"


def test_series_formats() -> None:
    hz = Horizon(START, 4, 30)
    np.testing.assert_array_equal(parse_series(2.5, hz, "x"), [2.5] * 4)
    np.testing.assert_array_equal(parse_series([1, 2, 3, 4], hz, "x"), [1, 2, 3, 4])
    np.testing.assert_array_equal(
        parse_series({"resolution_minutes": 60, "values": [1, 2]}, hz, "x"), [1, 1, 2, 2]
    )
    with pytest.raises(ValidationError, match="per-step"):
        parse_series([1, 2], hz, "x")
    with pytest.raises(ValidationError, match="multiple"):
        parse_series({"resolution_minutes": 45, "values": [1]}, hz, "x")
    with pytest.raises(ValidationError, match="cover"):
        parse_series({"resolution_minutes": 60, "values": [1]}, hz, "x")
    with pytest.raises(ValidationError, match=r"x\[1\]"):
        parse_series([1, "a", 3, 4], hz, "x")


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("horizon", "start"), "yesterday", "ISO 8601"),
        (("site", "grid_limit_kw"), "big", "site.grid_limit_kw"),
        (("sessions", 0, "charger"), "C9", "unknown charger"),
        (("sessions", 0, "departure"), "2026-04-15T00:20:00", r"sessions\[0\].*shorter"),
        (("sessions", 0, "energy_kwh"), True, "expected a number"),
        (("tariff", "price_eur_per_kwh"), [0.1, 0.2], "per-step"),
    ],
)
def test_errors_name_the_offending_field(path: tuple[Any, ...], value: Any, match: str) -> None:
    doc = copy.deepcopy(base_doc())
    target: Any = doc
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError, match=match):
        scenario_from_dict(doc)


def test_missing_fields_and_bad_json(tmp_path: Path) -> None:
    doc = base_doc()
    del doc["site"]
    with pytest.raises(ValidationError, match="missing required field 'site'"):
        scenario_from_dict(doc)
    with pytest.raises(ValidationError, match="expected an object"):
        scenario_from_dict([])
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValidationError, match="invalid JSON"):
        load_scenario(bad)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(base_doc()), encoding="utf-8")
    assert len(load_scenario(good).sessions) == 1
