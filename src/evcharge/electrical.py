"""Phase-aware electrical model: grid types, phase rotation and line wiring.

A three-phase site is fed by lines L1, L2 and L3, each protected by a main fuse
(or breaker, cable rating or contractual limit) of ``line_limit_a`` amperes.
AC chargers are commanded in amperes per phase, and what a command does to the
lines depends on the grid type, the charger's phase connection and the EV:

* **TN** (230/400 V with neutral, the usual European system): a single-phase EV
  is connected line-to-neutral and loads one line; a two- or three-phase EV
  loads two or three lines. Power per ampere of per-phase current is
  ``n * 230 V`` for ``n`` phases (3 x 230 V x 16 A = 11 kW).
* **IT** (230 V between lines, no neutral; common in Norwegian homes): a
  "single-phase" EV is connected line-to-line and loads **two** lines with the
  same current; a three-phase EV loads all three. Power per ampere is 230 V for
  one phase and sqrt(3) x 230 V for three (16 A gives 3.7 kW or 6.4 kW).

A charger's *rotation* lists the site line each of its conductors is connected
to, e.g. ``"L2L3L1"``: an EV that uses one phase draws on the charger's first
conductor, so rotating chargers spreads single-phase EVs across the lines. On IT
grids a single-phase EV is supplied from the charger's first two conductors.
The OCPP ``ConnectorPhaseRotation`` notation (``RST``, ``STR``, ``TRS``,
``RTS``, ``SRT``, ``TSR``) is accepted for three-phase chargers.

The line-current model is linear: a session with per-phase setpoint ``I``
contributes ``I`` to every line in :attr:`Wiring.lines`. For loads this is a
conservative bound on the phasor sum (triangle inequality) and exact when the
currents on a line are in phase; see ``docs/theory.md`` for the treatment of
injected current (PV, V2G) and the exact statements.

This module is pure (no dependency on the rest of the package).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

LINES: tuple[str, str, str] = ("L1", "L2", "L3")
"""Names of the three site lines, in index order."""

IEC_61851_MIN_CURRENT_A = 6.0
"""Lowest current an AC charger may signal to the EV (IEC 61851-1, 10 % PWM duty cycle)."""

NOMINAL_VOLTAGE_V = 230.0
"""Voltage across a single-phase load: line-to-neutral on TN, line-to-line on IT."""

_OCPP_ROTATION = {"R": "L1", "S": "L2", "T": "L3"}


class GridType(StrEnum):
    """Earthing system of the low-voltage supply."""

    TN = "TN"
    """230/400 V with neutral; single-phase loads are line-to-neutral."""
    IT = "IT"
    """230 V line-to-line without neutral; single-phase loads are line-to-line."""


class WiringError(ValueError):
    """Raised for an impossible charger/EV/grid combination."""


@dataclass(frozen=True)
class Supply:
    """Phase-aware description of a site's grid connection.

    Attributes:
        line_limit_a: Current limit of L1, L2 and L3 in amperes (main fuse,
            breaker or contractual limit). Use :meth:`uniform` for equal limits.
        grid: :class:`GridType` of the supply.
        voltage_v: Voltage across a single-phase load: line-to-neutral on TN
            (230 V in a 230/400 V system), line-to-line on IT (230 V).
    """

    line_limit_a: tuple[float, float, float]
    grid: GridType = GridType.TN
    voltage_v: float = NOMINAL_VOLTAGE_V

    def __post_init__(self) -> None:
        limits = tuple(float(v) for v in self.line_limit_a)
        if len(limits) != 3:
            raise WiringError(f"line_limit_a needs 3 values (L1, L2, L3), got {len(limits)}")
        if not all(math.isfinite(v) and v > 0 for v in limits):
            raise WiringError(f"line limits must be finite and > 0, got {limits}")
        if not (math.isfinite(self.voltage_v) and self.voltage_v > 0):
            raise WiringError(f"voltage_v must be finite and > 0, got {self.voltage_v}")
        object.__setattr__(self, "line_limit_a", limits)
        object.__setattr__(self, "grid", GridType(self.grid))

    @classmethod
    def uniform(
        cls, limit_a: float, grid: GridType | str = GridType.TN, voltage_v: float = 230.0
    ) -> Supply:
        """Supply with the same limit on all three lines."""
        return cls((limit_a, limit_a, limit_a), GridType(grid), voltage_v)

    @property
    def balanced_kw_per_a(self) -> float:
        """Power of a balanced three-phase load per ampere of line current."""
        return three_phase_kw_per_a(self.grid, self.voltage_v)

    @property
    def fuse_equivalent_kw(self) -> float:
        """Largest balanced three-phase power the line limits allow.

        This is the number a kW-only controller would use as the site limit:
        ``3 x 230 V x I`` on TN, ``sqrt(3) x 230 V x I`` on IT.
        """
        return self.balanced_kw_per_a * min(self.line_limit_a)


def three_phase_kw_per_a(grid: GridType, voltage_v: float) -> float:
    """Return the kW per ampere of line current of a balanced three-phase load."""
    if grid is GridType.TN:
        return 3.0 * voltage_v / 1000.0
    return math.sqrt(3.0) * voltage_v / 1000.0


def parse_rotation(text: str | None, phases: int, grid: GridType) -> tuple[int, ...]:
    """Parse a charger's phase connection into site-line indices.

    Args:
        text: ``"L1L2L3"``-style list of site lines (or OCPP ``"RST"``-style for
            three-phase chargers). ``None`` gives the identity connection.
        phases: Phases the charger provides (1 or 3).
        grid: Grid type; a single-phase charger is connected to one line on TN
            and to two lines on IT.

    Returns:
        Site-line index (0 = L1) of each charger conductor, in conductor order.

    Raises:
        WiringError: for unknown lines, repeated lines or the wrong count.
    """
    if phases not in (1, 3):
        raise WiringError(f"a charger provides 1 or 3 phases, got {phases}")
    needed = 3 if phases == 3 else (1 if grid is GridType.TN else 2)
    if text is None:
        return tuple(range(needed))
    raw = text.strip().upper()
    if len(raw) == 3 and set(raw) == set(_OCPP_ROTATION):
        raw = "".join(_OCPP_ROTATION[c] for c in raw)
    if len(raw) % 2 or not raw:
        raise WiringError(f"cannot parse phase rotation {text!r}; use e.g. 'L2L3L1'")
    names = [raw[i : i + 2] for i in range(0, len(raw), 2)]
    if any(n not in LINES for n in names):
        raise WiringError(f"unknown line in phase rotation {text!r}; lines are L1, L2, L3")
    if len(set(names)) != len(names):
        raise WiringError(f"phase rotation {text!r} repeats a line")
    if len(names) != needed:
        kind = "three-phase" if phases == 3 else f"single-phase {grid.value}"
        raise WiringError(
            f"a {kind} charger is connected to {needed} line(s), rotation {text!r} names "
            f"{len(names)}"
        )
    return tuple(LINES.index(n) for n in names)


@dataclass(frozen=True)
class Wiring:
    """How one EV on one charger loads the site lines.

    Attributes:
        lines: Site lines (0 = L1) that carry the per-phase setpoint current.
        phases: Phases the EV actually uses (1, 2 or 3).
        kw_per_a: Grid power per ampere of per-phase setpoint.
    """

    lines: tuple[int, ...]
    phases: int
    kw_per_a: float

    def incidence(self) -> tuple[float, float, float]:
        """Amperes on L1, L2 and L3 per ampere of setpoint (each 0 or 1)."""
        return (
            float(0 in self.lines),
            float(1 in self.lines),
            float(2 in self.lines),
        )


def wiring(
    grid: GridType,
    voltage_v: float,
    charger_phases: int,
    charger_lines: tuple[int, ...],
    ev_phases: int,
) -> Wiring:
    """Line connection and power factor of an EV with ``ev_phases`` on a charger.

    The EV uses ``min(ev_phases, charger_phases)`` phases, starting with the
    charger's first conductor.

    Raises:
        WiringError: for a two-phase EV on an IT grid (the line currents then
            depend on how the charger wires the Type 2 neutral pin, which varies
            between products; model the EV as single- or three-phase instead).
    """
    if ev_phases not in (1, 2, 3):
        raise WiringError(f"an EV uses 1, 2 or 3 phases, got {ev_phases}")
    used = min(ev_phases, charger_phases)
    if grid is GridType.TN:
        return Wiring(charger_lines[:used], used, used * voltage_v / 1000.0)
    if used == 2:
        raise WiringError(
            "two-phase on-board chargers are not supported on IT grids: their line currents "
            "depend on the charger's neutral-pin wiring; model the EV as 1- or 3-phase"
        )
    if used == 1:
        return Wiring(charger_lines[:2], 1, voltage_v / 1000.0)
    return Wiring(charger_lines[:3], 3, math.sqrt(3.0) * voltage_v / 1000.0)


def balanced_line_current_a(power_kw: float, grid: GridType, voltage_v: float) -> float:
    """Line current of a balanced three-phase load of ``power_kw`` at unity power factor."""
    return power_kw / three_phase_kw_per_a(grid, voltage_v)
