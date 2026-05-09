"""The ACN-Data loader, on the hand-made sample file and on edge cases."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from evcharge.acn import acn_scenario, load_acn, parse_records
from evcharge.cli import main
from evcharge.model import ValidationError

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "acn-sample.json"


def sample() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(SAMPLE.read_text(encoding="utf-8"))
    return data


def test_sample_loads_with_delivered_energy_and_actual_departures() -> None:
    loaded = load_acn(SAMPLE, grid_limit_kw=15.0)
    sc = loaded.scenario
    # the 10-minute session cannot hold one 15-minute step
    assert loaded.dropped == ("sample-session-06",)
    assert loaded.capped == ()
    assert sc.horizon.start == datetime(2026, 4, 15, 15, 0, tzinfo=UTC)
    first = sc.sessions[0]
    # 15:05 is rounded up to 15:15 (step 1), 23:40 down to 23:30 (step 34)
    assert (first.id, first.arrival_step, first.departure_step) == ("sample-session-01", 1, 34)
    assert first.energy_kwh == 14.2
    assert first.efficiency == 1.0  # ACN meters at the station
    assert {c.id for c in sc.site.chargers} >= {s.charger_id for s in sc.sessions}
    assert sc.site.charger("sample-station-1").max_power_kw == 6.6


def test_requested_energy_and_departure() -> None:
    sc = load_acn(SAMPLE, grid_limit_kw=15.0, energy="requested", departure="requested").scenario
    by_id = {s.id: s for s in sc.sessions}
    assert by_id["sample-session-01"].energy_kwh == 16.0
    assert by_id["sample-session-01"].departure_step == 34  # requested 23:30
    # a requested departure after the actual one is cut at the actual one (22:00 -> 21:30)
    assert by_id["sample-session-04"].departure_step == 26
    # no user input: delivered energy and actual departure
    assert by_id["sample-session-02"].energy_kwh == 9.8


def test_energy_is_capped_at_what_fits_the_rounded_window() -> None:
    data = sample()
    data["_items"][1]["kWhDelivered"] = 40.0  # 17 steps x 6.6 kW x 0.25 h = 28.05 kWh
    loaded = acn_scenario(parse_records(data), grid_limit_kw=15.0)
    assert loaded.capped == ("sample-session-02",)
    capped = next(s for s in loaded.scenario.sessions if s.id == "sample-session-02")
    assert capped.energy_kwh == pytest.approx(28.05)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("connectionTime", None, "missing 'connectionTime'"),
        ("connectionTime", "yesterday", "cannot parse"),
        ("connectionTime", "2026-04-15T15:05:00", "no time zone"),
        ("kWhDelivered", "a lot", "expected a number"),
        ("userInputs", {"kWhRequested": 3}, "expected a list"),
    ],
)
def test_bad_records_name_the_field(key: str, value: object, match: str) -> None:
    data = copy.deepcopy(sample())
    data["_items"][0][key] = value
    with pytest.raises(ValidationError, match=match):
        parse_records(data)


def test_iso_timestamps_and_plain_lists_are_accepted() -> None:
    items = sample()["_items"]
    items[0]["connectionTime"] = "2026-04-15T15:05:00Z"
    records = parse_records(items)
    assert records[0].connection == datetime(2026, 4, 15, 15, 5, tzinfo=UTC)
    with pytest.raises(ValidationError, match="no sessions"):
        acn_scenario([], grid_limit_kw=10.0)
    with pytest.raises(ValidationError, match="'_items'"):
        parse_records({"sessions": []})


def test_cli_runs_acn_data(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["run", "--acn", str(SAMPLE), "--grid-limit", "15", "--policies", "llf", "optimal"]
    assert main(args) == 0
    captured = capsys.readouterr()
    assert "7 sessions on 5 chargers" in captured.out
    assert "1 sessions shorter than one step left out" in captured.err
    assert main(["run", "--acn", str(SAMPLE), "--policies", "llf"]) == 2
    assert "needs --grid-limit" in capsys.readouterr().err
