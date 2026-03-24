from __future__ import annotations

import math

import pytest

from evcharge.electrical import (
    GridType,
    Supply,
    WiringError,
    balanced_line_current_a,
    parse_rotation,
    three_phase_kw_per_a,
    wiring,
)

TN, IT = GridType.TN, GridType.IT


@pytest.mark.parametrize(
    ("text", "phases", "grid", "expected"),
    [
        (None, 3, TN, (0, 1, 2)),
        ("L1L2L3", 3, TN, (0, 1, 2)),
        ("L2L3L1", 3, TN, (1, 2, 0)),
        ("l3l1l2", 3, IT, (2, 0, 1)),
        ("L1L3L2", 3, TN, (0, 2, 1)),
        # OCPP ConnectorPhaseRotation: R, S, T are the grid's L1, L2, L3
        ("RST", 3, TN, (0, 1, 2)),
        ("STR", 3, TN, (1, 2, 0)),
        ("TRS", 3, TN, (2, 0, 1)),
        ("RTS", 3, TN, (0, 2, 1)),
        (None, 1, TN, (0,)),
        ("L3", 1, TN, (2,)),
        (None, 1, IT, (0, 1)),
        ("L2L3", 1, IT, (1, 2)),
    ],
)
def test_parse_rotation(
    text: str | None, phases: int, grid: GridType, expected: tuple[int, ...]
) -> None:
    assert parse_rotation(text, phases, grid) == expected


@pytest.mark.parametrize(
    ("text", "phases", "grid", "match"),
    [
        ("L1L2L4", 3, TN, "unknown line"),
        ("L1L1L2", 3, TN, "repeats"),
        ("L1L2", 3, TN, "3 line"),
        ("L1L2", 1, TN, "1 line"),
        ("L1", 1, IT, "2 line"),
        ("XYZ", 3, TN, "cannot parse"),
        ("L1XXL2", 3, TN, "unknown line"),
        ("L1L", 3, TN, "cannot parse"),
        (None, 2, TN, "1 or 3"),
    ],
)
def test_parse_rotation_errors(text: str | None, phases: int, grid: GridType, match: str) -> None:
    with pytest.raises(WiringError, match=match):
        parse_rotation(text, phases, grid)


def test_tn_wiring_loads_one_line_per_phase() -> None:
    rot = (1, 2, 0)  # L2L3L1
    one = wiring(TN, 230.0, 3, rot, 1)
    assert one.lines == (1,)
    assert one.incidence() == (0.0, 1.0, 0.0)
    assert one.kw_per_a == pytest.approx(0.23)
    two = wiring(TN, 230.0, 3, rot, 2)
    assert two.lines == (1, 2)
    assert two.kw_per_a == pytest.approx(0.46)
    three = wiring(TN, 230.0, 3, rot, 3)
    assert three.incidence() == (1.0, 1.0, 1.0)
    assert three.kw_per_a * 16 == pytest.approx(11.04)  # the familiar 11 kW at 16 A
    # a three-phase EV on a single-phase charger uses one phase
    assert wiring(TN, 230.0, 1, (2,), 3).lines == (2,)


def test_it_wiring_loads_two_lines_for_a_single_phase_ev() -> None:
    one = wiring(IT, 230.0, 3, (0, 1, 2), 1)
    assert one.lines == (0, 1)  # line-to-line: both lines carry the current
    assert one.phases == 1
    assert one.kw_per_a * 16 == pytest.approx(3.68)
    three = wiring(IT, 230.0, 3, (2, 0, 1), 3)
    assert three.lines == (2, 0, 1)
    assert three.kw_per_a == pytest.approx(math.sqrt(3) * 0.23)
    assert three.kw_per_a * 16 == pytest.approx(6.37, abs=0.01)
    assert wiring(IT, 230.0, 1, (1, 2), 3).lines == (1, 2)
    with pytest.raises(WiringError, match="two-phase"):
        wiring(IT, 230.0, 3, (0, 1, 2), 2)
    with pytest.raises(WiringError, match="1, 2 or 3"):
        wiring(TN, 230.0, 3, (0, 1, 2), 4)


def test_supply() -> None:
    tn = Supply.uniform(63.0)
    assert tn.line_limit_a == (63.0, 63.0, 63.0)
    assert tn.grid is TN
    assert tn.fuse_equivalent_kw == pytest.approx(3 * 230 * 63 / 1000)
    it = Supply((125.0, 100.0, 125.0), grid="IT")  # type: ignore[arg-type]
    assert it.grid is IT
    assert it.fuse_equivalent_kw == pytest.approx(math.sqrt(3) * 230 * 100 / 1000)
    assert three_phase_kw_per_a(TN, 230.0) == pytest.approx(0.69)
    assert balanced_line_current_a(6.9, TN, 230.0) == pytest.approx(10.0)
    with pytest.raises(WiringError, match="3 values"):
        Supply((63.0, 63.0))  # type: ignore[arg-type]
    with pytest.raises(WiringError, match="> 0"):
        Supply((63.0, 0.0, 63.0))
    with pytest.raises(WiringError, match="voltage"):
        Supply.uniform(63.0, voltage_v=0.0)
    with pytest.raises(ValueError, match="XX"):
        Supply.uniform(63.0, grid="XX")
