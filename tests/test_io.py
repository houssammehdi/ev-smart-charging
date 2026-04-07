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


GARAGE = Path(__file__).resolve().parents[1] / "examples" / "garage-it.json"


def test_phase_aware_example_loads() -> None:
    sc = load_scenario(GARAGE)
    assert sc.site.supply is not None
    assert sc.site.supply.grid.value == "IT"
    assert sc.site.supply.line_limit_a == (80.0, 80.0, 80.0)
    p06 = sc.site.charger("P06")
    assert (p06.phases, p06.rotation, p06.current_step_a) == (1, "L3L1", 1.0)
    assert p06.max_power_kw == pytest.approx(3.68)  # derived: 1 x 230 V x 16 A
    ev6 = next(s for s in sc.sessions if s.id == "EV-06")
    wires = sc.wiring(ev6)
    assert wires is not None
    assert wires.lines == (2, 0)  # line-to-line on L3-L1
    assert sc.base_line_current_a[0].tolist() == [26.0, 20.0, 22.0]


def phase_doc() -> dict[str, Any]:
    doc = base_doc()
    doc["site"]["supply"] = {"grid": "tn", "line_limit_a": [32, 25, 32]}
    doc["site"]["chargers"][0].update({"phases": 3, "rotation": "STR", "max_current_a": 16})
    doc["sessions"][0].update({"phases": 1, "max_current_a": 16})
    return doc


def test_phase_fields_and_defaults() -> None:
    doc = phase_doc()
    del doc["site"]["grid_limit_kw"]
    doc["pv_current_a"] = {"L2": 3.0}
    sc = scenario_from_dict(doc)
    assert sc.site.supply is not None
    assert sc.site.supply.line_limit_a == (32.0, 25.0, 32.0)
    # without a kW limit the fuse-equivalent power applies: 3 x 230 V x 25 A
    assert sc.site.grid_limit_kw == pytest.approx(17.25)
    s = sc.sessions[0]
    wires = sc.wiring(s)
    assert wires is not None
    assert wires.lines == (1,)  # STR puts the charger's first conductor on L2
    assert sc.pv_line_current_a[0].tolist() == [0.0, 3.0, 0.0]


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("site", "supply", "grid"), "TT", "'TN' or 'IT'"),
        (("site", "supply", "line_limit_a"), [32, 32], "3 values"),
        (("site", "supply", "line_limit_a"), -5, "site.supply"),
        (("site", "chargers", 0, "rotation"), "L1L2", "charger C1"),
        (("sessions", 0, "phases"), 4, r"sessions\[0\].*phases"),
        (("base_current_a",), {"L4": 1.0}, "unknown line"),
        (("base_current_a",), {"L1": 99.0}, "L1 limit"),
    ],
)
def test_phase_errors_name_the_field(path: tuple[Any, ...], value: Any, match: str) -> None:
    doc = phase_doc()
    target: Any = doc
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError, match=match):
        scenario_from_dict(doc)


def test_v2g_fields() -> None:
    doc = base_doc()
    doc["site"]["export_limit_kw"] = 5
    doc["site"]["chargers"][0]["bidirectional"] = True
    doc["sessions"][0]["energy_kwh"] = 0
    doc["sessions"][0]["v2g"] = {
        "capacity_kwh": 60,
        "initial_kwh": 30,
        "min_kwh": 12,
        "max_kwh": 54,
        "max_discharge_kw": 7,
        "degradation_eur_per_kwh": 0.04,
    }
    sc = scenario_from_dict(doc)
    assert sc.site.export_limit == 5.0
    assert sc.site.bidirectional
    s = sc.sessions[0]
    assert s.v2g is not None
    assert (s.v2g.min_kwh, s.v2g.ceiling_kwh, s.v2g.discharge_efficiency) == (12.0, 54.0, 0.9)
    assert s.target_kwh == 30.0
    ctl = sc.control(s)
    assert (ctl.discharge_min, ctl.discharge_max) == (4.14, 7.0)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("site", "chargers", 0, "bidirectional"), "yes", "true or false"),
        (("site", "export_limit_kw"), -1, "export_limit_kw"),
        (("sessions", 0, "v2g"), {"capacity_kwh": 60}, r"v2g: missing required field"),
        (("sessions", 0, "v2g"), {"capacity_kwh": 60, "initial_kwh": 5, "soc": 1}, "unknown"),
        (("sessions", 0, "v2g"), {"capacity_kwh": 60, "initial_kwh": 58}, "exceeds max_kwh"),
    ],
)
def test_v2g_errors_name_the_field(path: tuple[Any, ...], value: Any, match: str) -> None:
    doc = base_doc()
    target: Any = doc
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError, match=match):
        scenario_from_dict(doc)
